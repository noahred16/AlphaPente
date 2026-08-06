# MCTS doesn't prove draws through the tree — root can't be solved as a tie

## Where
`src/MCTS.cpp` — `MCTS::search`'s main loop and `MCTS::backpropagate`.
`include/MCTS.hpp` — `SolvedStatus` enum.

## Current status
A leaf position with zero legal moves (board full, no winner) is marked
`SolvedStatus::SOLVED_DRAW` and backpropagates a neutral value (0.0). That
much works and is tested (see the WASM/CLI small-board work: `wasm/PenteWasm.cpp`,
`apps/Pente.cpp -B <size>`).

What does **not** happen: `SOLVED_DRAW` never propagates upward past that one
leaf. `MCTS::backpropagate`'s minimax bubbling only handles two outcomes:

```cpp
if (current->solvedStatus == SolvedStatus::SOLVED_WIN) {
    parent->solvedStatus = SolvedStatus::SOLVED_LOSS;
}
if (current->solvedStatus == SolvedStatus::SOLVED_LOSS) {
    parent->unprovenCount--;
    if (parent->unprovenCount == 0) {
        parent->solvedStatus = SolvedStatus::SOLVED_WIN;
    }
}
```

`SOLVED_DRAW` matches neither `if`, so it's silently ignored by an ancestor's
proving logic — a drawn child neither triggers the immediate "opponent has a
winning reply" short-circuit, nor counts toward "all my opponent's replies
lose for them." A node with a mix of proven-LOSS and proven-DRAW children
(no WIN among them) currently just stays `UNSOLVED` forever, no matter how
much of the tree under it gets fully explored.

## Goal
Be able to fully solve a 5x5 board: search until the **root** itself becomes
`SOLVED_DRAW` (a proven tie with best play), the same way it can already
become `SOLVED_WIN`/`SOLVED_LOSS` today.

## Why this should be reachable
The existing WIN-proving path already requires *every* legal move at a node
to be individually resolved before `unprovenCount` hits 0 — `select()`'s
progressive widening keeps advancing into untried moves via PUCT once the
promising ones are resolved, so full-width coverage per node is already an
implicit requirement of the current code, not something new that draw-proving
would need to introduce. On a 5x5 board (25 cells, small legal-move counts
once the opening is past) this is plausibly within reach of the existing
iteration budgets — nothing here is fundamentally a new kind of search, just
a 2-outcome→3-outcome extension of bookkeeping that already exists.

## What it would take
Extend the minimax step in `backpropagate()` from a binary WIN/LOSS proof to
a 3-way one:

1. **Understand the current convention first** (worth writing down somewhere
   since it isn't obvious from the code alone): `current->player` is whoever
   made the move that produced `current`, i.e. `parent`'s opponent-to-move.
   So `current == SOLVED_WIN` means "parent's opponent has a winning reply
   available" → parent is an immediate proven loss, regardless of parent's
   other children (correct to short-circuit). `current == SOLVED_LOSS` means
   "this specific opponent reply loses for them" → only informative in
   aggregate, hence `unprovenCount`.

2. **Add an analogous counter for draws**, or repurpose `unprovenCount` to
   decrement on *any* resolved child (WIN/LOSS/DRAW) rather than only LOSS,
   while keeping the immediate WIN short-circuit as-is (it doesn't need to
   wait for full resolution — one winning reply is enough). Track alongside
   it whether any resolved child was a DRAW.

3. **Decision once all children are resolved** (and no WIN short-circuit
   fired): if any child was DRAW → parent = `SOLVED_DRAW`; else (all
   children proven LOSS-for-opponent) → parent = `SOLVED_WIN` (existing
   behavior, now reached via the same generalized counter).

4. Update `isTerminal()` call sites and anything that pattern-matches
   `SolvedStatus` by `==` (already enumerated in the `SOLVED_DRAW` display
   commit — `printBestMoves`, `printMovesFromNode`, `getTopMoves`,
   `wasm/PenteWasm.cpp`) to make sure `SOLVED_DRAW` parents behave sanely
   in `getBestMove()`'s move-selection preferences too (right now it only
   special-cases `SOLVED_WIN` and excludes `SOLVED_LOSS`; a `SOLVED_DRAW`
   root would need `getBestMove()` to pick a child that actually preserves
   the draw rather than one that happens to have high visits but is a
   proven loss).

5. **Test carefully** — this touches the same proving logic
   `ParallelMCTS.cpp` has its own separate (and already more defensive,
   see `select()`'s `childCapacity == 0` handling) implementation of; that
   class isn't in scope here but has the identical WIN/LOSS-only gap if it
   ever gets the same treatment.

## Related: PNS is unimplemented
`include/PNS.hpp` is design notes only — `src/PNS.cpp` is a 0-byte empty
file. PNS is not a working alternative today; this MCTS-side extension is
the realistic near-term path to a solved 5x5 root. Worth noting for later:
PNS.hpp's own sketch ("Loss/Draw pn=inf dn=0") also lumps draw and loss
together from the root player's perspective, so a real PNS implementation
would face the same win/draw/loss-vs-win/not-win distinction this issue is
about — this isn't just an MCTS quirk to route around by switching to PNS.

## Status
Fixed. `backpropagate()`'s LOSS-only decrement became a LOSS-or-DRAW decrement
(a new `Node::hasDrawChild` bool tracks whether any resolved child was a
DRAW), and `getPUCTValue()` now treats `SOLVED_DRAW` like `SOLVED_LOSS`
(-infinity) so a proven-drawn subtree stops eating simulation budget once
found — the existing all-children-resolved fallback in `selectBestMoveIndex`
(originally written for the LOSS-only case) turned out to already handle the
DRAW case for free once DRAW also returns -infinity there, so no new
"already counted" bookkeeping was needed to avoid double-decrementing a
child that gets reselected. `getBestMove()` needed no changes: by the time a
root is fully resolved, every child is LOSS or DRAW, and the existing
exclude-LOSS + max-visits logic already lands on a DRAW child correctly.

Verified end to end:
- `tests/MCTSTests.cpp` — a 3x3 gomoku board (five-in-a-row is physically
  impossible there, so only the DRAW path can ever prove anything) fully
  solves the root as `SOLVED_DRAW` in 662 of a 50,000-iteration budget, and a
  second `search()` call does zero further work, confirming the root is
  truly terminal.
- Native CLI: `./pente -B 3 -s -n "1. K10" 200000` — real Pente rules
  (captures on), 3x3 board — prints `Solved status: SOLVED_DRAW - Best play
  leads to a draw`, matching the unit test's 662-visit count exactly.
- WASM: same 3x3 scenario through `Module.Game`, `getTopMoves()` shows
  `DRAW` on all 8 replies.
- Tried the actual target (5x5, full Pente rules) at 300k-500k iterations,
  then 10M single-threaded: still unsolved. Proving a 3x3 board takes ~662
  visits; 5x5 with captures enabled is a much larger game and evidently
  needs a materially bigger budget (or a stronger evaluator) to fully
  resolve — that's a search-scale question, not a correctness bug, and out
  of scope for this fix.
- `./unit_tests`: 83/85 pass (2 pre-existing skipped flaky, unrelated).

## Follow-up: ported the same fix to ParallelMCTS, still can't reach 5x5
`ParallelMCTS` (the multi-threaded engine the default `pente`/`gomoku`/etc.
binaries use) is a **separate implementation** with its own `SolvedStatus`
enum and atomic/CAS-based `backpropagate()` — the fix above only touched the
single-threaded `MCTS` class. Before this follow-up, `ParallelMCTS` had no
`SOLVED_DRAW` concept at all (enum was `{UNSOLVED, SOLVED_WIN, SOLVED_LOSS}`)
and didn't even detect "board full, no legal moves" as terminal, so a drawn
leaf just sat there permanently `UNSOLVED` with a stale evaluator value.

Ported the same design (`src/ParallelMCTS.cpp`, `include/ParallelMCTS.hpp`):
- `expand()`: when the evaluator's policy comes back empty (keyed off the
  pre-arena-exhaustion `capacity`, not the post-collapse `actualCap`, so an
  exhausted arena is never mis-marked as a draw) and the game isn't already
  over, mark `SOLVED_DRAW` with value 0.0f.
- `backpropagate()`: overrides the propagated value to 0.0f whenever the
  leaf's own `solvedStatus` is `SOLVED_DRAW`, in one place at the top of the
  function — covers first resolution, re-selection of an already-solved
  leaf, and the queue-mode Phase 2 round trip uniformly, without touching
  each call site. The LOSS-only full-rescan-under-lock aggregation (this
  class's equivalent of `unprovenCount`) now fires on LOSS-or-DRAW and
  tracks `anyDraw` to decide `SOLVED_WIN` vs `SOLVED_DRAW`. This scan-based
  design turned out to be inherently idempotent (unlike a decrementing
  counter) — re-running it for an already-resolved parent is a no-op via the
  `compare_exchange_strong(UNSOLVED, ...)` — so no extra "already counted"
  guard was needed here, unlike what a naive counter-based port might have
  required.
- `select()`'s `bestIndex < 0` fallback and `selectBestMoveIndex()`'s
  exploitation scoring both now treat `SOLVED_DRAW` like `SOLVED_LOSS`
  (-infinity, stop spending visits there) — same rationale as the
  single-threaded port. This is what made the `bestIndex < 0` fallback need
  a genuine fix, not just an extension: it used to assume "-inf on every
  child" could only mean all-LOSS and unconditionally marked `SOLVED_WIN`,
  which would have been outright wrong (should be `SOLVED_DRAW`) once DRAW
  also returns -inf there.
- `printBestMoves()`/`printStats()`: DRAW display text, matching the
  single-threaded versions.

New test in `tests/MCTSTests.cpp` mirrors the single-threaded one against
`ParallelMCTS` (4 workers, inline eval mode) on the same 3x3 board — passed
reliably across 5 repeated runs, needing ~127K iterations to fully resolve
(vs. 662 single-threaded — `ParallelMCTS` has no progressive widening or
transposition-table/symmetry sharing, so 4 workers redundantly explore
symmetric-equivalent subtrees from scratch instead of sharing them like the
single-threaded engine's canonical-hash transposition table does).

Verified against the real target: `./pente -B 5 -n "1. K10" <N>` (real
Pente, captures on, default parallel engine, 8 cores) —
- 5M iterations: 19s, unsolved, visits spread thinly across ~5 candidate moves.
- 100M iterations: 6m20s (264K iters/sec, ~3.3x single-threaded's
  throughput), 63.4M nodes, arena at 52% of 29GB — still unsolved. One move
  (H12) had absorbed 98M of 100M visits; the other 15 root replies were
  barely touched.

Conclusion: the ParallelMCTS port is correct (verified on 3x3), but full 5x5
Pente is evidently much too large a game to brute-force to a proven root
result with either engine at these budgets. The missing transposition/
symmetry sharing in `ParallelMCTS` (noted above) is likely the biggest lever
for closing this gap without just burning more raw compute — single-threaded
MCTS's canonical-hash transposition table is presumably why it needed far
fewer visits per node than the parallel engine's un-deduplicated tree, even
though it has far less raw throughput. That would be a separate, sizable
piece of work (making lock-free/atomic transposition insertion race-safe
across worker threads), not something to fold into this fix.

## Status (updated)
Draw-bubbling is fixed and verified on both engines. A fully solved 5x5
root remains out of reach at the budgets tried so far (up to 100M parallel
iterations, ~6.3 minutes on this 8-core machine) — this is now a search-
scale/algorithmic-improvement question (most plausibly: add transposition/
symmetry sharing to `ParallelMCTS`, or use a stronger evaluator to prune
harder), not a correctness bug, and needs explicit direction before any
further compute or engineering time goes into it.
