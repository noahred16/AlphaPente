# ParallelMCTS segfaults (GPF) under sustained load as the arena fills up

## Where
- `include/Arena.hpp` — the underlying bump allocator.
- `src/ParallelMCTS.cpp` — per-thread slab allocator (`setupSlabs`, `refillSlab`,
  `allocateFromSlab`) and the node lifecycle (`allocateNode`, `initNodeChildren`,
  `expand`, `select`).

## Symptom
`renju` (and presumably `pente`/`gomoku`/`keryopente` — same `ParallelMCTS`
code path) segfaults instead of returning a move, when a search runs long
enough for the tree arena to approach capacity. Found via `web_scrape`, whose
`home.py` intentionally requests a very large iteration count (900,000,000)
for every game so the engine searches for "as many iterations as possible" —
it does not expect to ever complete that many iterations, and relies on
`ParallelMCTS`'s arena-exhaustion handling to stop the search early and
return the best move found so far. Real failure from that repo:

```
game_runner_2026-08-05_15-27.sh: line 1: 197505 Segmentation fault (core dumped)
  ./renju "1.  H8 H7  2.  H6 J7  3.  G7 J9  4.  J5 K4  5.  F8 E9  6.  G8 J8
  7.  J10 H9  8.  G9 G6  9.  F5 G10  10.  K7 F6  11.  K9 E10  12.  E8 D8 " 900000000 -n
```

## Reproduction

**1. Native repro (real crash, real timing).** Ran the exact failing command
with a lower iteration count (40,000,000, just to hit the wall sooner) on the
same move sequence. Crashed after ~5.5 minutes. `journalctl -k` confirms a
genuine fault, not an OOM-kill:

```
kernel: traps: renju[199024] general protection fault ip:... in renju[...]
```

RSS was climbing steadily toward the arena's configured capacity
(`ARENA_SIZE_GB` ≈ 29GB on this machine) right as it died.

**2. Under gdb, it does *not* crash.** Same repro, wrapped in
`gdb -batch -ex run`, does not segfault — it hits the intended
graceful-exhaustion path instead:

```
Arena: 28859.2 MB used / 29.0 GB capacity (97.2%) -- EXHAUSTED: search stopped early (raise ARENA_SIZE_GB)
```
...and exits normally with a valid move. The fact that gdb (which perturbs
thread scheduling via ptrace) hides the crash was the first strong signal
this is a data race rather than a deterministic exhaustion bug.

**3. TSan, small-arena repro (fast, deterministic-ish).** Built a second
CMake tree (`build-tsan/`, `-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON`)
and a standalone harness, [`repro_arena.cpp`](./parallel-mcts-arena-race/repro_arena.cpp),
that constructs a `ParallelMCTS` directly with a tiny arena (8MB instead of
29GB) and `maxIterations = 50,000,000`, so exhaustion happens in well under a
second instead of ~5 minutes, and loops over many trials in one process.
This turned a "wait 5 minutes and hope" repro into "crashes/races on
essentially every trial." Run it with:

```bash
cd build-tsan
g++ -std=c++20 -fsanitize=thread -fno-omit-frame-pointer -g -O1 \
    -DPROJECT_ROOT=\"$(pwd)/..\" -I../include -fopenmp \
    ../issues/parallel-mcts-arena-race/repro_arena.cpp \
    -L. -lpente_core -lpthread -o repro_arena
# setarch -R works around a TSan/PIE shadow-memory mapping conflict on this machine
ARENA_SIZE_GB=1 NUM_THREADS=4 setarch "$(uname -m)" -R ./repro_arena 10 8388608 4
```

10 trials (arena=8MB, 4 threads) produced **16 distinct data-race
signatures / 26 total reports** — full output in
[`tsan_race_reports.log`](./parallel-mcts-arena-race/tsan_race_reports.log)
(one deduplicated block per unique race site).

## Root cause: the arena/slab allocator double-issues an address near exhaustion

The most serious race TSan caught (reproduced twice with full stacks) is
between two *different* worker threads operating on two *different* nodes,
both landing on the **same byte address**:

- Thread A, inside `expand()` (holding *that node's* `nodeSubtreeLock`),
  writes a field on an existing, already-allocated node —
  `node->value = 1.0f` (ParallelMCTS.cpp:898) or
  `node->solvedStatus.store(SOLVED_WIN, ...)` (ParallelMCTS.cpp:899).
- Thread B, inside `select()` → `allocateNode()` (holding a *different*
  node's `nodeSubtreeLock`), placement-news a brand-new `ThreadSafeNode` —
  ParallelMCTS.cpp:1033.
- **TSan reports both writes at the identical address**, e.g.
  `0x7ffff359cae8`, with two distinct mutexes (M0, M1 — confirmed different
  nodes) held by each thread.

This can only happen if the arena/slab bump allocator (`allocateFromSlab` /
`refillSlab` / `setupSlabs`, `include/Arena.hpp`'s `Arena::allocate`) handed
the same bytes to two simultaneously-live nodes. Once that happens, Thread
B's placement-new (which default-constructs a `ThreadSafeNode`, resetting
`visits`, `totalValue`, `solvedStatus`, `children`, the embedded mutex, etc.)
stomps a node Thread A is actively using — textbook memory corruption, and
exactly the kind of bug that produces a general-protection-fault the moment
any corrupted pointer gets dereferenced afterward. This matches the native
crash signature exactly.

**This is not a rare edge case for `web_scrape`'s workload.** `home.py`
deliberately requests far more iterations (900M) than any practically-sized
arena can hold — on this machine the 29GB arena only has room for roughly
20–30M tree nodes for this game (~1.2–1.6KB/node) — so *every* real game run
is expected to drive the arena to exhaustion, which is exactly the condition
that exposes this race. It's the common case for this caller, not a corner
case.

### A second, related bug in the same neighborhood

`select()` reads `node->children[bestIndex]` on the fast path **without
holding a lock** (ParallelMCTS.cpp:839), while the slow path writes it
**under `nodeSubtreeLock`** (ParallelMCTS.cpp:849) — classic
double-checked-locking without the memory fences it needs. TSan flags this
too. `children` should either be an array of `std::atomic<ThreadSafeNode*>`
with acquire/release ordering, or the fast-path read needs an explicit
acquire load — as written, a reader can observe a non-null child pointer
before that child's constructor has finished publishing its fields.

I have not yet determined whether this is a separate bug or a contributing
cause of the address-aliasing bug above (e.g., a torn/reordered read feeding
back into allocator bookkeeping) — see Plan below.

## Impact
Crashes the whole engine process mid-search with no output, instead of the
designed "stop early at exhaustion, return the best move found so far"
behavior — which is the behavior `web_scrape` actually depends on, since it
intentionally over-requests iterations. Currently this makes long/high-`NUM_THREADS`
searches unreliable: some fraction of them will hard-crash rather than
gracefully truncate.

## Plan / next steps
1. **Narrow the aliasing bug.** Audit `setupSlabs()` / `refillSlab()` /
   `allocateFromSlab()` for a boundary condition that lets two allocations
   land on overlapping bytes once a worker's pre-carved slab *and* the
   shared fallback pool are both near/at exhaustion. Use
   `repro_arena.cpp` + TSan (sub-second per trial) to bisect — it's cheap to
   add temporary logging of every `(thread, address range)` pair handed out
   and assert non-overlap, which should isolate the exact call path fast.
2. **Fix the unsynchronized `children[bestIndex]` fast-path read** in
   `select()` (ParallelMCTS.cpp:839) — make `children` atomic-with-ordering,
   or add an explicit acquire load, so a reader can never see a child
   pointer before its constructor's writes are visible.
3. **Add a cheap TSan job** (even just documenting the recipe above, or a
   small `ctest` target) that runs a tiny-arena stress search under TSan, so
   exhaustion races like this get caught in seconds instead of requiring a
   5-minute native crash to notice.
4. Once fixed, re-verify against the *original* native repro (40M+
   iterations, real ~29GB arena, no tiny-arena trick) to confirm the GPF is
   gone, then re-run the exact failing game from `web_scrape`'s
   `game_runner_2026-08-05_15-27.sh` at the real 900,000,000-iteration
   count end to end.

Per discussion with Noah: `home.py`'s huge iteration count is intentional
(the intent is to run as many iterations as the time budget allows), so the
fix belongs here, in `ParallelMCTS`'s exhaustion-handling path — not in
lowering the iteration count on the `web_scrape` side.

## Root cause (confirmed): not the allocator — the unsynchronized `children[]` publish

Re-examined the "arena double-issues an address" theory against the code and
against a fresh, faster TSan run (`repro_arena 3 8388608 4`, 24 warnings in
~2s). It doesn't hold up: every real write to `arena_`'s shared bump offset
(`setupSlabs()`, `refillSlab()`, and `allocateFromSlab()`'s fallback path) is
already serialized under `arenaMutex_`, and each worker's `SlabView` is a
`thread_local` pointer into its own, unique element of `workerSlabs_` — two
threads can never share a slab or race on the arena's offset. The "two
different mutexes (M0, M1) at the identical address" observation in the
original writeup is fully explained by an entirely different, much simpler
bug: **M1 is always the *parent* node's lock (held in `select()` while a new
child is being constructed) and M0 is the *child's own* lock (held later in
`expand()`)** — different mutexes are expected for a parent/child pair, so
that was never evidence of address aliasing between unrelated nodes.

The actual bug: `ThreadSafeNode **children` was a plain pointer array.
`select()` published a newly-allocated child via a bare, non-atomic
`node->children[bestIndex] = child` (ParallelMCTS.cpp:849, no release
fence), while the fast path (ParallelMCTS.cpp:839) and `selectBestMoveIndex()`
(ParallelMCTS.cpp:995) read it back with a bare, non-atomic load (no
acquire fence). Nothing enforced that a reader observing the non-null child
pointer also observed the child's *constructor* having finished — including
`ThreadSafeNode`'s embedded `std::mutex`. A second worker could legitimately
discover the same brand-new child (this is normal, intentional lock-free
sharing) and immediately `lock()` its `nodeSubtreeLock` or read its fields
inside `expand()` while the placement-new in `allocateNode()` was still
completing on the allocating thread — a textbook missing-acquire-release bug,
and exactly the kind of thing that corrupts an object (including a live
mutex) and later segfaults on a bogus pointer deref.

### Fix
`children` is now `std::atomic<ThreadSafeNode *> *`. The publish in
`select()` uses `memory_order_release`; every read (`select()`'s fast path,
the re-read under lock, `selectBestMoveIndex()`, and the sibling scan in
`backpropagate()`) uses `memory_order_acquire`. `initNodeChildren()`
placement-constructs each atomic slot instead of `memset`-zeroing raw
pointer bytes.

### Verification
- `repro_arena 30 8388608 8` under TSan (`build-tsan/`): **0 warnings**
  (previously 24 warnings in just 3 trials of the same harness).
- `./unit_tests`: 82/82 test cases, 705/705 assertions passing.
- Real `renju` binary (optimized build, no sanitizer), same move sequence
  as the original native crash, `ARENA_SIZE_GB=1` (forces exhaustion in
  ~11s instead of ~5.5min) — **5/5 runs** (4 and 8 threads) hit
  `EXHAUSTED: search stopped early` and returned a valid move, no crash.

## Status
Fixed. Root cause was a missing acquire/release fence on the lazily-created
`ThreadSafeNode::children[]` slots (item 2 in the original plan), not the
arena/slab allocator itself (item 1) — the allocator was audited and found
correctly synchronized. Not yet re-verified at the full original scale
(40M+ iterations, real ~29GB arena, 900,000,000-iteration
`web_scrape` `game_runner_2026-08-05_15-27.sh` end to end) — recommended
before closing out entirely, but TSan is clean and the release binary now
survives repeated real exhaustion at reduced arena size that reliably
crashed before the fix.
