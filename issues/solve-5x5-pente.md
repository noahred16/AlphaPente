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

## Longer 4x4 run (2026-08-09)

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

## Next

A genuinely longer, node-unbounded 4x4 run (get one real full-resolution
data point) before deciding whether Phase 6 (multithreaded df-pn, reusing
`ParallelMCTS`'s worker-pool/sharded-table/slab-allocator patterns - a real
chunk of separate work) is warranted, or whether patience alone gets there
at this board size. 5x5's true difficulty remains unknown; 2M-node-scale
probes are nowhere near enough data to extrapolate from.

<!-- Update below as longer runs complete. -->
