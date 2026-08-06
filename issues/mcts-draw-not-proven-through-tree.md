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
- Tried the actual target (5x5, full Pente rules) at 300k-500k iterations:
  still unsolved. Proving a 3x3 board takes ~662 visits; 5x5 with captures
  enabled is a much larger game and evidently needs a materially bigger
  budget (or a stronger evaluator) to fully resolve — that's a search-scale
  question, not a correctness bug, and out of scope for this fix.
- `./unit_tests`: 83/85 pass (2 pre-existing skipped flaky, unrelated).
