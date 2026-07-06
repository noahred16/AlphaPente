# Why `nn@800` underperforms raw policy — investigation report

**Date:** 2026-07-06
**Trigger:** Overnight self-play training showed little improvement across checkpoints `model_iter0006`–`model_iter0017`. `reports/pente/benchmark.csv` showed the MCTS-augmented benchmark (`nn@800`, 800 simulations) consistently scoring *below* the zero-search raw-policy benchmark (`nn`) on `tests/open-three-suite.json`:

```
2026-07-05T13:28:16,checkpoints/pente/model_iter0016.pt,nn,open-three-suite,250,250,100.0
2026-07-05T13:29:04,checkpoints/pente/model_iter0016.pt,nn@800,open-three-suite,202,250,80.8
```

This pattern held across every checkpoint sampled (iter6 through iter17): `nn` scored 98–100%, `nn@800` scored 71–95%, with no upward trend over training. Since search is supposed to only ever *refine* the raw policy, this is backwards, and the self-play training loop uses the same search machinery to generate its games — so a search bug here plausibly explains the stalled training too.

---

## 1. Ruled out

Two hypotheses were tested and disproved before the real mechanism was found. Recorded here because both were plausible enough to have been the answer, and disproving them is what narrowed the search.

### 1.1 Virtual-loss leak at end of search
`ParallelMCTS::search()` calls `drainEvalQueue()`/`drainBackpropQueue()` after stopping the worker/eval pools, and discards whatever they return. Any evaluation request still in flight at that moment would never reach `backpropagate()`, permanently orphaning the virtual loss added for it in `select()`.

Checked directly: after a full search, scanned every root child's `virtualLosses` counter. In every run, across every worker/eval-thread configuration tested: **0 leaked virtual loss**, and `root.visits == sum(childVisits) == maxIterations` exactly. The iteration accounting is clean. This theory is wrong.

### 1.2 Batched NN evaluation diverges from single-item evaluation
`NNEvaluator::evaluateBatch()` (used by the async eval-thread path) could in principle produce different numbers than `NNEvaluator::evaluate()` (used by the inline/synchronous path) for the same position, e.g. via a batch-norm or indexing bug.

Checked directly: ran both paths on 10 identical positions. Max absolute difference: `5.96e-8` (float rounding only). No batching bug.

---

## 2. Confirmed mechanism: First-Play-Urgency (FPU) pathology

### 2.1 Isolating the trigger

The same position/model/iteration-budget was run through four search configurations:

| config | numWorkerThreads | numEvalThreads | result on this case |
|---|---|---|---|
| matches `Benchmark.cpp` suite | 6 | 1 | broken |
| matches `Generate.cpp` self-play | 1 | 6 | broken |
| inline, single-threaded, deterministic | 1 | 0 | correct |
| inline, 6-way concurrent | 6 | 0 | correct |

Only the two **async/queued** eval configs (`numEvalThreads > 0`) showed the pathology — including the fully-serial, deterministic inline run, which reused the *identical* PUCT formula and priors and got it right every time. This ruled out a generic tree-traversal data race and pointed at something specific to how visit counts accumulate over the *async* pipeline.

### 2.2 Staged replay of one failing case

Test case: `1. K10 O8 2. N9 E19 3. M9 J7 4. L9` (suite expects `K9` or `O9` — the two moves that block Black's open three). Raw NN policy: `O9 p=0.306`, `K9 p=0.185`, everything else ≤0.05 — correctly confident. Raw NN value at root: `0.9466` (in the "previous-mover" convention this means Black, who just moved, is heavily favored — consistent with the tactical threat).

Re-ran the search (6 workers / 1 eval, `explorationConstant=1.7`, `dirichletAlpha=0`, `seed=42`) in checkpointed stages, using `prepareRoot()`'s tree-reuse-on-same-hash behavior to snapshot progress at growing cumulative iteration counts:

| cumulative iters | O9 | K9 | other |
|---|---|---|---|
| 200 | 36 | 67 | J11=42 |
| 400 | **204** (clearly dominant) | 67 | J11=42, K11=33 |
| 800 | **204 (unchanged)** | 67 (unchanged) | G9=94, P11=89, M12=75, L7=71, L11=64 — *all newly touched since iter 400* |

O9 received **zero** of the 400 additional iterations between the 400 and 800 checkpoints. Every one of them went to a move that had never been tried before iteration 400 — each of which, once tried, evaluated at roughly -1.0, matching what its 1-ply raw NN value already said.

### 2.3 Root cause

`ParallelMCTS::selectBestMoveIndex` (`src/ParallelMCTS.cpp`) assigned a flat `exploitation = 0.0` to any never-visited child:

```cpp
if (child == nullptr) {
    effectiveVisits = 0;
    exploitation    = 0.0;      // <- the bug
}
```

This is a reasonable default in a balanced position. But in a position where **every real move evaluates below -0.9** (exactly the shape of "block this or lose" tactical puzzles), an untried move's fictitious `0.0` will always outscore the correctly-identified best move once that move's own exploration bonus (`prior / (1 + visits)`) has decayed from repeated visits. The search keeps handing out "first tries" to every remaining untried sibling instead of reinforcing the move it already knows is best — and the imbalance **worsens with more search**, not less, because `sqrtParentVisits` (shared by every child's exploration term) keeps climbing while the leader's own denominator (`1 + visits`) grows even faster, so the *gap* between "0.0 for anyone untried" and "the leader's real, decaying bonus" only widens over time. This exactly matches the data: more simulation budget made the benchmark score worse, not better.

Since every candidate move in this suite is, by design, bad-to-very-bad for the side that has to respond to an open three, this suite is close to a worst case for the FPU=0 assumption — which is presumably why it, rather than other benchmarks, exposed the bug so clearly.

---

## 3. The fix, and why it wasn't sufficient on its own

### 3.1 Implementation

Standard fix (used in AlphaZero-family engines, e.g. KataGo's FPU reduction): seed an unvisited child's exploitation with the *parent's own* value estimate rather than a flat 0, so an untried move looks exactly as good/bad as "the position already looked" rather than artificially neutral.

Changes made (currently uncommitted in the working tree):
- `ParallelMCTS::expand()` now takes an explicit `value` parameter and stores it on the node (`node->value = value`) for non-terminal expansions. Previously this field was only ever populated for the root (via `prepareRoot()`) and for genuine terminal nodes (hardcoded `1.0f`) — every other node's `value` silently sat at its default-constructed `0.0f` forever, which is what made the old bug's `exploitation = 0.0` for unvisited children *doubly* wrong (it wasn't just a poor default, it was the *only* value ever available at those nodes).
- `selectBestMoveIndex` now computes `fpu = -node->value` once per call and uses it wherever exploitation would otherwise default to 0 (both null children and non-null children with `effectiveVisits == 0`).
- Sign check: `node->value` is stored in NNEvaluator's "previous-mover-perspective" convention (matches the documented convention used throughout `backpropagate()`), i.e. it reflects the position from the perspective of whoever moved *into* `node` — the opponent of `node`'s own mover. Negating it converts to "goodness for `node`'s own mover," the same convention already used by real children's `totalValue / visits`. Verified numerically: root value `0.9466` (good for Black) negates to `fpu = -0.9466` for root's children, matching the empirically observed real child values (-0.9 to -1.0) far better than the old `0.0`.

### 3.2 Result: a regression, not an improvement

Full-suite benchmark on `model_iter0016.pt`, `-s 800`:

| | before fix | after fix |
|---|---|---|
| `nn` (raw policy) | 250/250 (100.0%) | 250/250 (100.0%, unaffected as expected) |
| `nn@800` | 201/250 (80.4%) | **176/250 (70.4%)** |

Investigating the same traced case with the fix applied: the search now touches only 22–24 distinct children (down from 57) and commits early and decisively — 200+ visits — to a single branch. In three repeated runs it committed to `J11`, which is *not* an accepted answer. `J11`'s raw 1-ply NN value is -0.99 ("this doesn't block anything, you're losing"), but its *searched*, several-plies-deep average value climbed to around -0.5 — better-looking than O9's -0.6 to -0.7 or K9's -0.7 to -0.73.

Interpretation: the old bug's flat-0.0 default caused the search to spread thin and never fully commit to anything — which accidentally acted as a hedge, since O9/K9's much higher policy priors kept them in the lead often enough that the diffuse, directionless exploration never fully displaced them (confirmed non-deterministic: reruns of the *unfixed* code sometimes still landed on O9/K9 correctly, just unreliably). Fixing the FPU pathology removed that accidental hedge and let the search actually act on its own multi-ply value estimates — and at this simulation budget, those estimates aren't trustworthy enough yet to reward that trust.

### 3.3 The fix does work at higher budgets

Re-ran the same case with the fix applied at 50,000 iterations (60x the production budget):

```
Move    Visits     Prior   Avg Val      PUCT    Status
O9     19412     0.230    -0.966    -0.959         -
K9     18924     0.094    -0.962    -0.959         -
G9       344     0.021    -0.994    -0.959         -
K11      337     0.020    -0.994    -0.961         -
H8       331     0.021    -0.996    -0.962         -
```

O9/K9 are correctly and decisively dominant. So the fix is directionally sound — it just needs a much larger sample to let the real (but very small) signal in the value network's output win out over noise. That "very small signal buried in an extreme range" observation led to the deeper finding below.

Note also: even at 50,000 iterations, the `Status` column is `-` (`UNSOLVED`) for every move. The engine has a genuine proof-search path — a leaf that is an *actual* finished game (five-in-a-row / capture win) is hardcoded to `±1.0` and marked `SOLVED_WIN`/`SOLVED_LOSS`, which then overrides the value network entirely via `±infinity` in PUCT scoring and an immediate-return fast path in `getBestMove()`. That mechanism never fired anywhere in this tree at any budget tested. Proving "if White doesn't block, Black wins" requires walking the actual forcing sequence to a completed five-in-a-row several plies down *and* ruling out deviations at each branching point along the way — with a ~50-wide branching factor per ply, that's out of reach within these budgets. So every value backpropagated in this investigation, at 800 or 50,000 iterations, was the network's opinion, never ground truth.

---

## 4. Deeper root cause: the value head has never seen a non-extreme training target

### 4.1 The finding

```
$ ./inspect -b checkpoints/pente/buffer.pt -n 0
Buffer: 500000 positions
Value distribution:  +1=242056  -1=257944  0=0
```

**Every one of the 500,000 positions in the replay buffer has a value label of exactly +1 or -1.** Zero draws, and — critically — zero intermediate values anywhere in the training data.

Cause, from `src/SelfPlay.cpp`:

```cpp
// Convention: value = +1 if the player who MOVED INTO this position wins
// (previous-player perspective), matching HeuristicEvaluator and MCTS backprop.
ex.value = (winner == PenteGame::NONE) ? 0.0f
         : (ex.player == winner)        ? -1.0f
                                         :  1.0f;
```

The value target is purely "who eventually won the game" — applied identically to *every* position in the game, from the opening move to the last, with no discount for how many moves away the outcome was and no margin-of-victory signal. Pente essentially never draws, so in practice the label is a hard ±1 for every training example that has ever existed for this model.

### 4.2 Why this produces saturated, low-dynamic-range predictions

Training a network with MSE against a target that is *always* exactly ±1 gives it no training signal that ever rewards outputting something moderate. Once the network has even a weak feature-based lean toward one side, pushing the prediction toward that side's extreme reduces loss more than hedging toward zero would, because the observed target is never in between. Combined with the `tanh` output's saturation (gradient → 0 once the output is already near ±1, so there's little pressure to further *refine* the magnitude once it's roughly pointed the right way), the network jams almost everything into the -0.9-to-1.0 range and expresses whatever real discriminating signal it does have (confirmed present — O9/K9 do reliably outrank the rest once sampled enough) as a small residual wobble within that saturated band, rather than as a proportionate gap.

Corroborating data, `./inspect -m model_iter0016.pt` on a 3000-position sample of the same buffer:

```
mean pred:  -0.058   labels: -0.050
std  pred:   0.802   labels: 0.999
MAE:         0.360
Pearson r:   0.795

Prediction distribution:
  [-1.0,-0.6): 37.5%
  [-0.6,-0.2): 14.5%
  [-0.2, 0.2):  3.5%
  [ 0.2, 0.6): 12.9%
  [ 0.6, 1.0]: 31.5%
```

Pearson r=0.795 shows the network's *ranking* is meaningfully correlated with outcome — consistent with what the search evidence shows (O9/K9 do reliably win out, just fragilely). But the prediction histogram mirrors the label histogram's near-bimodal shape (69% of predictions land in the two extreme bins, only 3.5% near zero) — the network has learned to mimic the shape of its training targets rather than to output a calibrated, continuous estimate of position quality.

### 4.3 Why this matters beyond the FPU bug

This is very likely the deeper, more foundational issue, of which the FPU regression is a symptom:
- With real signal compressed into a ~0.03-0.05 gap out of a ±1 range, that signal is fragile against anything else happening in the search (exploration noise, virtual-loss dilution during concurrent selection, thread-scheduling variance) until a very large sample size averages it out — hence needing ~50,000 iterations instead of 800 for the FPU-fixed search to reliably resolve this one case.
- The *original* FPU=0 bug and the *new* FPU=-parent-value fix are both operating on this same badly-calibrated signal; neither can fully compensate for it. A better-calibrated value head — one that could say "-0.3, still worse for White but blocked" vs. "-0.95, actually losing" — would give the search a robust margin to work with regardless of which FPU strategy or exploration constant is used.
- Since self-play (`Generate.cpp`) uses this same search machinery to choose moves and generate the *next* round of training data, a value head that can only express near-extreme opinions plausibly contributes to the stalled training improvement that motivated this whole investigation, independent of the FPU bug.

---

## 5. Current repository state

As of this report, the FPU fix is implemented but **uncommitted** in the working tree:
- `include/ParallelMCTS.hpp`: `expand()` signature gained a `float value` parameter.
- `src/ParallelMCTS.cpp`: `expand()` stores `node->value` for non-terminal nodes; all three call sites updated; `selectBestMoveIndex` uses `fpu = -node->value` in place of the flat `0.0`.
- All existing unit tests pass (`68/68`, `586/586` assertions) — the fix does not break any existing invariant, it just isn't a net win on the 800-simulation benchmark as currently measured.

No decision has been made yet on whether to keep, revert, or further iterate on the fix (e.g. an FPU-reduction variant rather than the raw parent value), pending discussion of the value-head training issue above, since that may be the higher-leverage fix.

## 6. Suggested next steps (not yet actioned)

1. **Value target redesign** — the highest-leverage fix implied by this report. Options include: discounting the outcome by distance-to-game-end (TD(λ)-style bootstrapping), blending the raw game outcome with the search's own value estimate at the time the move was made (closer to what AlphaZero variants typically do), or otherwise avoiding a purely-binary regression target.
2. **Re-evaluate the FPU fix once the value head is better calibrated** — the fix is theoretically well-founded (matches standard practice) and did work at 50,000 iterations; it may only be net-negative *because of* the current label problem, not on its own merits.
3. **Check whether this generalizes beyond the open-three suite** — this investigation traced one suite and one representative case in depth; worth confirming the same saturation pattern (near-bimodal predictions, tiny real gaps) shows up on other benchmark suites (e.g. `value-suite.json`) before treating it as the universal explanation for stalled training.
