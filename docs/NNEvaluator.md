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
| `CMakeLists.txt` | Auto-detects LibTorch at `/opt/libtorch`; applies `TORCH_CXX_FLAGS`; skips separate OpenMP link when Torch is present to avoid dual `libgomp` conflict; defines `PROJECT_ROOT` and `WITH_TORCH` |
| `apps/Train.cpp` | Constructs `AlphaNet(64, 5)`, saves to `PROJECT_ROOT/checkpoints/pente/best_model.pt`, prints parameter count |
| `apps/Pente.cpp` | `-N` flag uses default checkpoint; `-p <path>` overrides; automatically sets `numEvalThreads=1` when NN is active |
| `src/ParallelMCTS.cpp` | Fixed `evalThreadMain` to call `evaluate()` (single forward pass) instead of `evaluateValue()` + `evaluatePolicy()` separately (was 2 forward passes per position) |

---

## Build

LibTorch is installed at `/opt/libtorch`. CMakeLists.txt auto-detects it — no extra flags needed:

```bash
cd build
cmake ..
make -j$(nproc)
```

If LibTorch is not found, the build proceeds without NN support (`WITH_TORCH` is not defined) and the existing heuristic/uniform evaluators work as before.

---

## Usage

### Initialize a model checkpoint

```bash
cd build
./train
# Saved initialized AlphaNet to /path/to/checkpoints/pente/best_model.pt
# Channels: 64, ResBlocks: 5, Parameters: ~700000
```

### Run pente with the NN evaluator

```bash
./pente "1. K10" 2000 -N -n          # default checkpoint
./pente "1. K10" 2000 -p /other.pt -n  # custom checkpoint path
```

`-N` is a boolean flag (no argument) that loads `checkpoints/pente/best_model.pt` via the `PROJECT_ROOT` compile-time constant. `-p <path>` accepts a custom path.

### Policy inference details

The model outputs log-softmax probabilities over all 361 board positions. `NNEvaluator::evaluate` post-processes them:

1. `exp(log_policy)` → raw probabilities for all 361 positions
2. Filter to legal moves only (`game.getLegalMoves()`)
3. Renormalize to sum to 1.0
4. Sort by probability descending, matching `HeuristicEvaluator` output format

If all filtered probabilities are zero (degenerate case), falls back to uniform over legal moves.

---

## Per-game model organization

Each game variant uses a separate checkpoint directory. The architecture is shared — only the weights differ.

```
checkpoints/
  pente/best_model.pt
  gomoku/best_model.pt        (future)
  keryopente/best_model.pt    (future)
```

---

## Current Status (as of 2026-05-17)

**Complete — pipeline validated end-to-end:**
- `AlphaNet` model (ResNet dual-head) compiles and runs with LibTorch
- `NNEvaluator` implementing the full `Evaluator` interface
- Board → tensor encoding (3 planes, current-player perspective) using `BitBoard::forEachSetBit`
- Capture scalar normalization injected into value head FC
- LibTorch auto-detected from `/opt/libtorch` — `cmake ..` with no extra args
- OpenMP/libgomp conflict resolved: Torch's bundled libgomp is used exclusively when Torch is present
- NNPACK warning suppressed via `at::globalContext().setUserEnabledNNPACK(false)`
- `train` initializes and saves a model to `checkpoints/pente/best_model.pt`
- `pente -N` loads and runs the model; eval serialized through one thread to avoid BLAS contention
- `evalThreadMain` fixed to use `evaluate()` (one forward pass per position, not two)

**Known limitation — CPU inference is slow without batching:**

Measured baseline on CPU (no GPU available on this machine):

| Mode | Throughput |
|------|-----------|
| Heuristic evaluator | ~500,000 iters/sec |
| NNEvaluator, unbatched CPU | ~130 iters/sec |

The bottleneck is one full ResNet forward pass per MCTS iteration. The eval thread serializes all inference through a single thread (correct for thread safety), but processes positions one at a time with no tensor batching.

---

## Future Work

### 1. True batched inference (highest priority)

The `evalThreadMain` already pops a batch from the queue but calls `evaluate()` per item. True batching means stacking the batch into a single `[B, 3, 19, 19]` tensor and running one `model->forward()` call. BLAS scales sublinearly with batch size — a batch of 16 costs ~2× a batch of 1, not 16×. Expected gain: **6–8× on CPU** (~800–1000 iters/sec), and a prerequisite for efficient GPU use.

Implementation plan:
1. Add `NNEvaluator::evaluateBatch(const vector<PenteGame>&)` that stacks all positions into one tensor, runs a single `forward()`, and returns a vector of `(policy, value)` pairs
2. Rework `evalThreadMain` to call `evaluateBatch()` on the full popped batch instead of looping over `evaluate()`

### 2. GPU support

No GPU is present on the development machine, but the code is structurally ready — input tensors are already moved to `impl_->device` before the forward pass. To enable:
1. Switch to a CUDA-enabled LibTorch build
2. In `NNEvaluator::Impl`: set `device = torch::kCUDA` if `torch::cuda::is_available()`
3. `model->to(device)` (already called at construction)

**Performance estimates (RTX 3060+ class GPU):**

| Mode | Throughput |
|------|-----------|
| Current: unbatched CPU | ~130 iters/sec |
| Batched CPU (batch=16) | ~800–1,000 iters/sec |
| Batched GPU (batch=16+) | ~5,000–25,000 iters/sec |

A modern GPU runs this network in ~0.1ms per forward pass vs ~6ms on CPU. With batching the gap widens further since GPU utilization improves with larger batches. Realistic expectation: **50–200× faster than current** once batching and GPU are both enabled.

### 3. Self-play data generation

The `train` app needs to run MCTS games and record `(state, policy_target, value_target)` tuples:
- Policy target = MCTS visit-count distribution (not the raw NN prior)
- Value target = game outcome from that player's perspective (+1 win, -1 loss, 0 draw)
- Store to disk as a replay buffer (flat binary or HDF5)

### 4. Training loop

Given a replay buffer:
1. Sample minibatches
2. Cross-entropy loss on policy head (vs. MCTS visit distribution)
3. MSE loss on value head (vs. game outcome)
4. Adam optimizer, lr ~1e-3, weight decay ~1e-4

### 5. Model evaluation and promotion

Before promoting a candidate to `best_model.pt`:
- Play N games between candidate and current best
- Promote if win rate > 55% (AlphaZero threshold)
- Archive checkpoints by iteration number

### 6. Gomoku and Keryo-Pente

Same pipeline, separate checkpoint dirs. Gomoku (no captures) is simpler — good sanity check before full Pente training.

### 7. Architecture scaling

Once training is running:
- 128 channels, 10 blocks: ~2.8M parameters
- 256 channels, 19 blocks: ~23M parameters

Profile improvement vs. heuristic evaluator to determine when to scale up.

### 8. TorchScript export (optional)

Exporting to TorchScript (`torch::jit::script`) enables loading without the `AlphaNetImpl` class definition at runtime. Required if training moves to Python and only inference stays in C++.
