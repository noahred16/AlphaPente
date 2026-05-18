# NNEvaluator — Neural Network Evaluator for AlphaPente

## Overview

`NNEvaluator` is an AlphaZero-style neural network evaluator for Pente, Gomoku, and Keryo-Pente. It implements the same `Evaluator` interface as `UniformEvaluator` and `HeuristicEvaluator`, making it a drop-in replacement in both MCTS and ParallelMCTS. The model is a dual-head residual network (AlphaNet) that predicts a move policy distribution and a scalar value from a given board position.

The long-term goal is to use this evaluator to drive self-play training, accumulate game data, and iteratively improve the network until it surpasses the heuristic evaluator — following the AlphaZero paradigm.

---

## Architecture: AlphaNet

### Design rationale

AlphaZero (DeepMind, 2017) uses a deep ResNet with two output heads — one for move policy and one for position value. The key properties are:

- **Residual connections** prevent gradient vanishing in deep networks and allow the trunk to learn incrementally useful representations
- **Dual heads** share a common trunk so the network learns features useful for both policy and value simultaneously, which is more sample-efficient than training them separately
- **Policy as log-softmax over all board positions** makes the output a proper probability distribution that MCTS can use as a prior during PUCT selection
- **Value as tanh output** naturally bounds the value to `[-1.0, 1.0]`, matching the existing `Evaluator` convention

The scale (64 channels, 5 residual blocks) is intentionally smaller than AlphaZero's chess network (256 channels, 19 blocks). Pente/Gomoku are simpler games, and a smaller network trains faster during the initial self-play bootstrap.

### Input encoding

Three planes of shape `[19, 19]`, encoded from the **current player's perspective** (always player-to-move = plane 0, regardless of color):

| Plane | Content |
|-------|---------|
| 0 | Current player's stones (1.0 = occupied) |
| 1 | Opponent's stones (1.0 = occupied) |
| 2 | Empty squares (1.0 = unoccupied by either player) |

The current-player-perspective encoding means the network sees the same pattern regardless of whether it's playing Black or White. This halves the effective state space and dramatically improves sample efficiency.

Plane 2 (empty) is technically redundant — it equals `1 - (plane0 | plane1)` — but providing it explicitly gives the network a direct occupancy mask without requiring it to learn the complement.

Two additional **scalar inputs** are concatenated into the value head's fully-connected layer (not added as extra planes, since they carry no spatial information):

| Index | Content |
|-------|---------|
| 0 | Current player's captures, normalized: `captures / capturesToWin` |
| 1 | Opponent's captures, normalized: `captures / capturesToWin` |

Normalization to `[0, 1]` keeps the inputs in the same range as the binary planes.

### Network structure

```
Input: planes [B, 3, 19, 19]  +  captures [B, 2]

─── Trunk ───────────────────────────────────────────────────
Conv(3 → 64, 3×3, pad=1, no bias) → BN → ReLU
× 5 Residual Blocks:
    Conv(64 → 64, 3×3, pad=1, no bias) → BN → ReLU
    Conv(64 → 64, 3×3, pad=1, no bias) → BN
    + skip connection → ReLU

─── Policy Head ─────────────────────────────────────────────
Conv(64 → 2, 1×1, no bias) → BN → ReLU
Flatten → [B, 2×19×19 = 722]
Linear(722 → 361) → LogSoftmax(dim=1)
Output: log-probabilities over all 361 board positions [B, 361]

─── Value Head ──────────────────────────────────────────────
Conv(64 → 1, 1×1, no bias) → BN → ReLU
Flatten → [B, 361]
cat([B, 361], captures [B, 2]) → [B, 363]
Linear(363 → 256) → ReLU
Linear(256 → 1) → Tanh
Output: position value from current player's perspective [B, 1]
```

Total parameters at default scale (64 ch, 5 blocks): ~700K.

### How it differs from AlphaZero paper

| Property | AlphaZero (chess) | AlphaNet (AlphaPente) |
|---|---|---|
| Channels | 256 | 64 |
| Residual blocks | 19–39 | 5 |
| Input planes | 119 (8 time steps + constants) | 3 (single time step) |
| Capture scalars | N/A | 2, injected into value head |
| Games | Chess, Go, Shogi | Pente, Gomoku, Keryo-Pente |

The capture scalar injection into the value head (rather than as extra spatial planes) is a minor departure from the original paper. It's appropriate here because captures are a low-dimensional game-state quantity with no spatial meaning.

---

## Files Added / Modified

### New files

| File | Purpose |
|------|---------|
| `include/NNModel.hpp` | `AlphaNet` and `ResBlock` declarations (LibTorch `TORCH_MODULE` pattern) |
| `src/NNModel.cpp` | `AlphaNetImpl` and `ResBlockImpl` implementations |
| `src/NNEvaluator.cpp` | `NNEvaluator` implementation — board→tensor conversion, inference, policy filtering |
| `docs/NNEvaluator.md` | This file |

### Modified files

| File | Change |
|------|--------|
| `include/PenteGame.hpp` | Added `getBlackBitBoard()` and `getWhiteBitBoard()` const accessors returning `const BitBoard&` |
| `include/Evaluator.hpp` | Added `NNEvaluator` class declaration (guarded by `#ifdef WITH_TORCH`); added `<memory>` and `<string>` includes |
| `CMakeLists.txt` | Added `find_package(Torch QUIET)` block; conditionally compiles `NNModel.cpp` + `NNEvaluator.cpp` into `pente_core` and links LibTorch |
| `apps/Train.cpp` | Replaced placeholder MCTS search with model initialization: constructs `AlphaNet(64, 5)`, saves to `checkpoints/pente/best_model.pt`, prints parameter count |
| `apps/Pente.cpp` | Added `-N <path>` flag to load an `NNEvaluator` from a checkpoint; takes precedence over `-u` (uniform) and default (heuristic) |

---

## Build Instructions

LibTorch must be installed and findable by CMake. If `find_package(Torch)` fails, set `CMAKE_PREFIX_PATH` to your LibTorch installation:

```bash
# If installed via pip
TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cd build
cmake .. -DCMAKE_PREFIX_PATH="$TORCH_PATH"
make -j$(nproc)
```

If LibTorch is not found, the build proceeds without NN support (`WITH_TORCH` is not defined). All NN-related code is compiled out and the existing heuristic/uniform evaluators work as before.

---

## Usage

### Initialize a model checkpoint

```bash
cd build
./train
# Output:
# Saved initialized AlphaNet to checkpoints/pente/best_model.pt
# Channels: 64, ResBlocks: 5, Parameters: ~700000
```

This creates a randomly initialized model. The weights are garbage but the file is valid for loading and confirms the pipeline works end-to-end.

### Run pente with the NN evaluator

```bash
./pente "1. K10" 10000 -N checkpoints/pente/best_model.pt
```

The `-N` flag takes the path to a saved model. It can be combined with other flags:

```bash
./pente "1. K10" 10000 -N checkpoints/pente/best_model.pt -s   # serial MCTS
./pente "1. K10" 10000 -N checkpoints/pente/best_model.pt -n   # non-interactive
```

The `-N` flag overrides `-u` (uniform) and the default (heuristic).

### Policy inference details

The model outputs log-softmax probabilities over all 361 board positions. `NNEvaluator::evaluate` does the following post-processing:

1. `exp(log_policy)` → raw probabilities for all 361 positions
2. Filter to legal moves only (`game.getLegalMoves()` — the promising-moves set)
3. Renormalize filtered probabilities to sum to 1.0
4. Sort by probability (largest first), matching `HeuristicEvaluator` output format

If all legal-move probabilities are zero (degenerate case), falls back to uniform over legal moves.

---

## Per-game model organization

Each game variant uses a separate checkpoint directory. The architecture is shared — only the weights differ because each game has different tactical patterns and rules.

```
checkpoints/
  pente/
    best_model.pt
  gomoku/
    best_model.pt
  keryopente/
    best_model.pt
```

The `train` app currently only initializes the Pente model. Gomoku and Keryo-Pente initialization will follow the same pattern once self-play training is set up for each variant.

---

## Current Status (as of 2026-05-17)

**Complete:**
- `AlphaNet` model definition and LibTorch module registration
- `NNEvaluator` implementing the `Evaluator` interface
- Board → tensor encoding (3 planes, current-player perspective)
- Capture scalar normalization and injection into value head
- Optional LibTorch detection in CMake — build degrades gracefully without it
- `train` app initializes and saves a model checkpoint
- `pente` app loads a model via `-N <path>` flag

**Not yet working:**
- LibTorch `find_package` path not yet configured on this machine. Need to set `CMAKE_PREFIX_PATH` to the LibTorch installation. The code compiles correctly once found.
- Model is randomly initialized — outputs noise. The pipeline is structurally correct but results will be meaningless until training runs.

---

## Future Work

### 1. Wire up batched evaluation in ParallelMCTS

`ParallelMCTS` already has `numEvalThreads` and `evaluationBatchSize` fields, and a placeholder eval-thread queue. The current `NNEvaluator::evaluate` runs inference one position at a time. To use the GPU efficiently, the eval threads should:

1. Collect a batch of positions from the worker queue (up to `evaluationBatchSize`)
2. Stack them into a single `[B, 3, 19, 19]` tensor
3. Run a single forward pass
4. Fan out results back to waiting worker threads

This requires a `evaluateBatch(vector<const PenteGame*>)` method or a callback-based interface that ParallelMCTS drives directly. The single-position path in `evaluate()` stays for serial MCTS and testing.

### 2. Self-play data generation

The `train` app needs to:
1. Run MCTS with the current best model as evaluator
2. Record `(state, policy_target, value_target)` tuples from each game
3. Store to disk as a replay buffer (HDF5 or flat binary)

Policy target = MCTS visit-count distribution (not the raw NN policy prior).
Value target = game outcome from that player's perspective (+1 win, -1 loss, 0 draw).

### 3. Training loop

Given a replay buffer:
1. Sample minibatches
2. Compute cross-entropy loss on policy head (against MCTS visit distribution)
3. Compute MSE loss on value head (against game outcome)
4. Total loss = policy loss + value loss (equal weighting to start)
5. Adam optimizer, learning rate ~1e-3, weight decay ~1e-4

Standard AlphaZero uses a shared loss with equal weight. The value head often needs a separate scale factor because its target range is `[-1, 1]` while policy cross-entropy is unbounded.

### 4. Model evaluation and promotion

Before promoting a candidate model to `best_model.pt`:
- Play N games between candidate and current best (using MCTS with each as evaluator)
- Promote if candidate wins > 55% (AlphaZero used 55%)
- Save checkpoint with iteration number alongside `best_model.pt`

### 5. Gomoku and Keryo-Pente training

Same pipeline, separate checkpoint directories. Gomoku is simpler (no captures) so may converge faster — useful as a sanity check before running the full Pente/Keryo-Pente training.

### 6. Architecture scaling

Once training is running, scale up as needed:
- 128 channels, 10 blocks: ~2.8M parameters
- 256 channels, 19 blocks: ~23M parameters (AlphaZero scale for chess)

Pente on a 19×19 board is likely solvable with the smaller scale. Profile against the heuristic evaluator to determine when returns diminish.

### 7. TorchScript export (optional)

Once the model is trained, exporting to TorchScript (`torch::jit::script`) enables loading without requiring the `AlphaNetImpl` class definition at runtime. This simplifies deployment and is required for Python-side training if we move training to Python and inference back to C++.
