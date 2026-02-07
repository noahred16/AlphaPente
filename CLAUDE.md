# AlphaPente

AlphaZero-inspired solver for Pente, Gomoku, and Keryo-Pente. The goal is to solve these games using MCTS with neural network guidance, then use Proof Number Search (PNS) with a trained network to prove optimal play.

## Quick Start

```bash
cd build
make           # rebuild after changes
./pente "1. K10 L9 2. K12 M10"   # analyze position
./unit_tests   # run tests
```

## Project Structure

- `include/` - Headers (BitBoard, PenteGame, MCTS, Evaluator, PNS)
- `src/` - Implementation files
- `apps/` - Executables (pente, gomoku, keryopente, play, train, test)
- `tests/` - Unit tests (doctest framework)
- `build/` - Build output

## Core Components

**PenteGame** - Game logic with configurable rules:
- `Config::pente()` - Standard Pente (10 captures to win)
- `Config::gomoku()` - Five-in-a-row only (no captures)
- `Config::keryoPente()` - 15 captures, 3-stone captures

**MCTS** - Monte Carlo Tree Search:
- UCB1 and PUCT selection policies
- Arena allocator for O(1) tree destruction
- Minimax backpropagation for solved positions

**Evaluator** - Policy and value prediction interface:
- `UniformEvaluator` - Baseline uniform policy
- `HeuristicEvaluator` - Hand-crafted heuristics
- (Planned) Neural network evaluator

**PNS** - Proof Number Search (in progress)

## Build

```bash
cd build
cmake ..       # initial setup
make           # build all targets
```

Executables: `pente`, `gomoku`, `keryopente`, `play`, `train`, `test`, `unit_tests`

## Tests

```bash
./unit_tests              # run all
./unit_tests -tc="MCTS*"  # filter by test name
./unit_tests -ltc         # list test cases
```

## Vision

1. Implement self-play training loop with NN
2. Train expert-level neural network
3. Use trained NN to guide PNS and solve each game mode
