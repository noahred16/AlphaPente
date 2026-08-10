# Weakly solving 5x5 Pente with PNS/df-pn

## Where
`include/PositionKey.hpp`/`src/PositionKey.cpp`, `include/PNS.hpp`/`src/PNS.cpp`,
`include/PositionBook.hpp`/`src/PositionBook.cpp`, `apps/Solve5x5.cpp`.

## Goal
`pente_5x5_book` — a persisted, resumable database of proven WIN/LOSS/DRAW
(+ distance-to-result) covering as much of the 5x5 Pente game tree as a
time/memory budget allows, focused on opening moves. See
`issues/mcts-draw-not-proven-through-tree.md` for why MCTS/PUCT alone
couldn't close this out (100M+ single-threaded / 50M parallel iterations,
still unsolved - one branch absorbs nearly the whole budget). This doc tracks
the PNS-based follow-up (a full phased plan exists as a Claude Code plan
artifact from the `solve/5x5` branch's planning session; this doc is the
living, in-repo record of status and calibration results).

## Status: Phases 1-5 complete, calibration in progress

Built and unit-tested (`./unit_tests`, 141/141 passing throughout):
1. **`PositionKey`** - exact (bijective, not hashed) 59-bit packing of a
   position (2 bits/cell x25 + side-to-move + 4-bit black/white capture
   counts) + D4-symmetry canonicalization. No collision risk, unlike reusing
   `Zobrist`'s statistical hash for a definitive proof database at scale.
2. **`PNS`** - real df-pn implementation (previously `include/PNS.hpp` was
   comments-only and `src/PNS.cpp` was a 0-byte file, not even compiled).
   3-outcome (WIN/LOSS/DRAW) proof/disproof numbers, canonical transposition
   DAG, mate-distance depth tracking, `maxNodes`/`maxSeconds` budgets.
   Exhaustive move enumeration (not `PenteGame::getLegalMoves()`, which is
   heuristically incomplete - see `PNS.hpp`'s class comment) for soundness.
   Requires `tournamentRule`/`renjuForbiddenMoves` off (asserted in `solve()`).

   **Real bug found and fixed during this phase**: child moves were stored in
   physical board coordinates from whichever parent first expanded a
   (transposition-shared) node - wrong for any other parent reached via a
   different physical orientation. Applying them without un-rotating
   corrupted the DAG's proof numbers (observed as root `pn` growing
   unboundedly across millions of `mid()` calls instead of converging).
   Fixed by storing canonical coordinates and un-rotating per-visit via
   `Zobrist::applyInverseSym`, mirroring `MCTS::select()`'s identical
   handling of the same hazard.
3. **`PositionBook`** - flat binary file checkpoint/reload of resolved
   positions. Deliberately not an embedded KV store (RocksDB/LevelDB):
   `PositionKey`'s exact packing removes the collision-safety argument for
   one, and ~11 bytes/record keeps even 10^8 resolved positions around 1GB,
   comfortably in RAM. Revisit only if a real run's live working set doesn't
   fit in memory.
4. **`apps/solve5x5`** - CLI driver (`-B` board size, `-N`/`-t` node/time
   budgets, `-o`/`-i` checkpoint save/load). Known limitation, stated
   plainly rather than overclaimed: `-i` does not resume a genuinely
   interrupted mid-search (PNS only exports *resolved* nodes, not the full
   in-progress proof tree) - it only skips re-solving if the exact root was
   already fully resolved in a prior run. Real mid-search resume would need
   PNS to accept a pre-seeded transposition table, deferred.

## Calibration results

**The 4x4 rows below are invalidated by the even-boardSize key-corruption
bug found later in this doc - see "Two real bugs found chasing that longer
run." Left here only as a historical record of the investigation; don't
cite these numbers.** The 3x3 and 5x5 rows are odd-sized and unaffected.

Single-threaded, one machine (8-core, per `nproc`), Release build.

| Board | Budget | Nodes created | mid() calls | Nodes/sec | Resolved | Root status |
|---|---|---|---|---|---|---|
| 3x3 (either rules) | unbounded | 246 | 963 | - | 86 | **DRAW**, depth 9 (solved in 2ms) |
| 4x4 (Pente, captures on) | 50K nodes | 50,000 | 176,669 | ~68K/s | 0 | UNKNOWN, pn=24 dn=1 |
| 4x4 (Pente, captures on) | 300s | 10,246,742 | 96,534,528 | ~34K/s | 337,396 | UNKNOWN, pn=24 dn=1 |
| 5x5 (Pente, captures on) | 60s / 2M nodes | 2,000,000 | 4,708,999 | ~68K/s | 725 | UNKNOWN, pn=24 dn=1 |

3x3 matches MCTS's already-verified DRAW result (662 node visits there vs.
963 `mid()` calls here - same ballpark, different search strategies).

Both 4x4 and 5x5 show the same shape as the MCTS issue's finding: most of
the tree resolves quickly, with one narrow remaining branch absorbing the
rest of the budget (`dn=1` at the 4x4/300s mark means the search has
narrowed to essentially one unresolved line - encouraging, looks close to a
full proof rather than stuck, though not yet confirmed).

## Longer 4x4 run (2026-08-09) - also invalidated, see below

`./solve5x5 -B 4 -t 3600 -o book4x4_1hr.bin`. Actually ran only **715s**, not
the intended hour - `-N` wasn't raised alongside `-t`, so the default
20,000,000-node cap bound first. Fix for next time: always pass an `-N` large
enough that `-t` is the real limiter, e.g. `-N 500000000 -t 3600`.

| | |
|---|---|
| nodesCreated | 20,000,000 |
| midCalls | 241,942,301 |
| wall | 714.9s (~28.0K nodes/sec, ~338K mid()calls/sec) |
| transpositionHits | 69,702,092 (3.5x nodesCreated) |
| resolved | 955,054 (4.8% of touched nodes) |
| root | UNKNOWN, **pn=24, dn=1** |

**`pn=24, dn=1` is bit-for-bit identical to the 50K-node and 300s/10.2M-node
runs before it** - unchanged across a 400x increase in node budget. Worth
being suspicious of on its face (it's exactly the pattern the child-move
un-rotation bug produced earlier in Phase 2 - see there), so this was
checked rather than waved off:

- Root has exactly one child (Black's move 0 is forced to center), so the
  printed root pn/dn *is* that single child's pn/dn - i.e. this is really
  asking why White's-reply-node hasn't changed, not the root itself.
- The earlier real bug's signature was node count staying **flat** (383,
  forever) while pn grew **unboundedly** - a true non-terminating loop
  revisiting the same handful of nodes with no new positions ever created.
- Here, node count grew steadily and continuously across all three budgets
  (50K -> 10.2M -> 20M, hitting whatever cap was set each time) - real,
  new, distinct positions keep getting created throughout, which is not
  consistent with a livelock cycling a fixed small set of nodes.
- df-pn's threshold selection can legitimately spend enormous, deepening
  effort along one narrow, many-plies-deep line without that propagating
  back up to change a shallow ancestor's aggregate pn/dn at all, until the
  whole deep subtree finally resolves - this is exactly what the MCTS issue
  doc already observed ("H12 absorbed 98M of 100M visits... the other 15
  root replies were barely touched") and is the expected shape of a hard,
  narrow proof, not evidence of a stall.

Conclusion: most likely genuine (if slow) progress deep in one line, not a
new bug - but flagged here explicitly rather than silently assumed, since
the surface symptom (frozen top-level numbers under heavy compute) is
exactly what a real bug would also look like. If a much longer run *still*
shows bit-identical root pn/dn while node count keeps climbing, that would
be the point to actually distrust this conclusion and instrument further
(e.g. dump the specific child/grandchild chain currently absorbing effort).

## Two real bugs found chasing that longer run (2026-08-09)

Asked to actually let a long run go: it crashed (SIGSEGV) the first time,
and even after fixing that, the resulting "successful" run turned out to
also be invalid. Both are fixed now; **all 4x4 calibration data above and
below predating this section should be considered unreliable** - it was
unknowingly exercising a corrupted key space (see bug 2). The real 5x5/3x3
targets are odd-sized and were never affected by either bug.

**Bug 1 - stack overflow (SIGSEGV, exit 139).** `mid()` recurses once per
ply and takes a full `PenteGame` by value each level - `sizeof(PenteGame)`
is 8128 bytes, almost entirely its embedded `std::mt19937` (5000 bytes) -
and df-pn's threshold-driven descent can legitimately commit very deep into
one narrow line. A default ~8MB thread stack only had headroom for a few
hundred levels. Fixed: `PNS::Config::maxRecursionDepth` (new safety valve,
stops the whole search gracefully rather than crashing) plus
`apps/Solve5x5.cpp` raising `RLIMIT_STACK` and sizing the depth budget to
match. See the commit for full detail.

**Bug 2 - even boardSize silently corrupted the key space.** Re-running
after the stack fix "succeeded" (exit 0) but hit the *new* depth cap at
depth=53,687 - and that number is itself impossible: on a 4x4 board,
`capturesToWin=10` bounds total captured stones to ≤18 and stonesOnBoard to
≤16, so `moveCount = stonesOnBoard + totalCaptured ≤ 34` for any genuinely
non-terminal position. A depth 1500x past that theoretical ceiling meant
the DAG wasn't representing real game states faithfully.

Root cause: `PenteGame::minIdx()`/`maxIdx()` truncate asymmetrically for
even `boardSize` (`BOARD_SIZE=19` is odd, so `(19-boardSize)/2` truncates) -
`boardSize=4` actually yields a **5-wide** window (`minIdx=7, maxIdx=12`),
not 4. `PositionKey::packSym` packed against the requested
`config().boardSize` instead of the true window width
(`maxIdx()-minIdx()`), silently dropping a whole row/column from the key.
Two physically distinct positions differing only in that dropped
row/column collapsed to the *same* key - a genuine collision defeating the
entire "exact, collision-free" design premise, and the actual explanation
for the impossible depth: the DAG could "progress" the underlying game
indefinitely along the un-keyed dimension without the (corrupted) key ever
registering it as a new/different position, so termination logic never
kicked in as it should have. `PositionKeyTests.cpp` had never exercised an
even `boardSize` (only 3 and 5, both odd) - a real test-coverage gap that
let this ship undetected through all of Phase 1-5.

Fixed: `PositionKey::packSym`/`unpack` now compute the window from
`maxIdx()-minIdx()` rather than trusting `config().boardSize`. Regression
test added (`PositionKeyTests.cpp`) packing a stone into the specific cell
the bug used to drop and confirming it now changes the key. The actual
5x5/3x3 targets are odd, where `(19-boardSize)/2` never truncates - no
discrepancy, unaffected by either bug.

All prior 4x4 numbers/artifacts in this doc and the scratchpad were deleted
rather than "corrected" - they don't describe a real 4x4 game and aren't
worth preserving even as a labeled caveat.

## Fixed PenteGame's window sizing, then a 3rd bug, then 4x4 actually solved (2026-08-09)

Asked to fix "4x4" properly rather than route around it. Two changes, in order:

1. **`PenteGame::maxIdx()`** was `BOARD_SIZE - minIdx()`, which only equals
   `minIdx() + boardSize` for odd `boardSize` - for even sizes it silently
   widened the window by one cell (`boardSize=4` produced the *exact same*
   5-wide window as `boardSize=5` - there was no genuinely-smaller-than-5x5
   target at all before this). Fixed to `minIdx() + config_.boardSize`
   directly. Confirmed via grep that nothing else in the repo relied on the
   old even-size behavior. The forced-center-move rule needed no changes -
   physical `(9,9)` is still in-bounds for a genuine 4-wide window
   `{7,8,9,10}`, just not exactly centered (nothing can be, for an even
   width) - exactly the semi-arbitrary rounding this was expected to need.
   Also fixed a matching cosmetic bug in `GameUtils::printBoard` (it had its
   own stale inline copy of the old formula).

2. That surfaced a **third** bug, more subtle: a genuine 4-wide window sits
   at physical `[7,11)`, not centered on the full 19x19 grid's center (9) -
   true center of a 4-wide range is 8.5. `PNS` bridged canonical<->physical
   move coordinates via `Zobrist::applySymToMove`/`applyInverseSym`, which
   is defined relative to the *grid's* center - only correct when the window
   happens to be grid-centered, true for every odd `boardSize` (never
   exercised against an even one before). This produced wrong physical
   coordinates fed straight into `PenteGame::makeMove()` (no occupancy check
   of its own, by design), corrupting the game in a way that looked exactly
   like the "impossible" depth=53687 finding above, but this time with
   `blackCap=0, whiteCap=0` - impossible even by the simplest possible bound
   (16 cells, no captures, so moveCount can't exceed 16 without something
   placing moves on already-occupied cells). Fixed by adding
   `PositionKey::applySymToPhysical`/`applyInverseSymToPhysical` - a
   window-relative transform guaranteed self-consistent with
   `pack()`/`canonical()` by construction - and switching `PNS` to use those
   instead of `Zobrist`'s grid-relative ones.

**4x4 Pente (real rules, captures on) is now fully, validatedly solved:
root is DRAW, depth 16** (board-full - sane, since five-in-a-row can't fit
on a 4-wide board, and apparently neither side can force `capturesToWin=10`
first). 1,487,913 nodes, 13,973,801 `mid()` calls, 40.4s single-threaded,
75,524 positions resolved. Reproduced identically twice (deterministic
search, as expected). Checkpoint and log kept in the scratchpad
(`book4x4_solved.bin`), not committed (large binary, not repo material).

This is the first genuinely trustworthy calibration data point past the
trivial 3x3 case, and validates the whole PNS/df-pn approach end-to-end on
a real (if small) Pente-with-captures board.

## 4x4 fully solved and shipped: complete book + web UI (2026-08-10)

Asked for the complete picture (every position's result + moves-to-result,
not just enough to prove the root), and to wire it into the web UI.

**`PNS::solveExhaustive()`** (new, alongside `solve()`): a full memoized
postorder traversal - visit every child, no proof-number pruning - instead
of df-pn's threshold-guided descent. Reused `expandNode()`/`updatePnDn()`/
`resolveOutcome()` completely unchanged (provable: once every child is
resolved, a node's pn/dn are always exactly 0 in whichever direction
`resolveOutcome()` already expects), so this was a new traversal strategy,
not new combine logic. `solve5x5 -B 4 -x` result: **6,136,048 distinct
positions, every single one resolved**, root DRAW at depth 16 - exactly
matching `solve()`'s own root result, a real cross-check.

**Web UI**: 4x4 is now a board-size option, backed by the book instead of
live MCTS search. `wasm/PenteWasm.cpp`'s `WasmGame` loads the book
(preloaded into the WASM virtual FS) when `boardSize==4`; `computeAIMove()`
becomes an instant exact lookup (enumerate every legal reply, rank by
outcome then depth) rather than a search, and `getTopMoves()` shows every
legal reply's real result, not just MCTS's top-10-by-visits. Along the way,
fixed a pre-existing, unrelated WASM build breakage (`scripts/build_wasm.sh`
was missing `RenjuForbiddenPointFinder.cpp`/`RenjuRules.cpp`, which
`PenteGame.cpp` has depended on since Renju support was added - the WASM
build had apparently not been run since).

Verified: a native program mirroring `WasmGame`'s exact lookup/ranking logic
against the real book confirms full coverage (all 15 of White's replies to
Black's forced center are book-covered) and a real cross-check (White's best
achievable result is DRAW, consistent with the whole game being proven a
draw). Could not test actual browser rendering in-session (no browser tool
available) - worth a manual visual check.

The book ships as `docs/wasm/pente.data` (67MB, embedded via
`--preload-file`; `docs/data/book4x4.bin` is gitignored - cheaply
regenerable, not worth duplicating in git).

## Next

5x5 is ~25/16 times the cell count of 4x4 and combinatorially much larger
than that ratio suggests - no valid extrapolation from 4x4's solve time
exists yet, exhaustive or otherwise. Next real step: a long single-threaded
5x5 run (`solve()`, not `solveExhaustive()` - a full 5x5 book is very
plausibly far too large to be practical, even if the root itself resolves)
to see how far it gets, now that the engine itself is trustworthy end to
end. Decide whether Phase 6 (multithreaded df-pn, reusing `ParallelMCTS`'s
worker-pool/sharded-table/slab-allocator patterns - a real chunk of
separate work) is warranted once that lands.

<!-- Update below as longer runs complete. -->
