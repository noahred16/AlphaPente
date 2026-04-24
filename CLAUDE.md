# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.



# Project: AlphaPente

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
- PUCT selection policies
- Arena allocator for O(1) tree destruction
- Minimax backpropagation for solved positions

**Evaluator** - Policy and value prediction interface:
- `UniformEvaluator` - Baseline uniform policy
- `HeuristicEvaluator` - Hand-crafted heuristics
- (Planned) Neural network evaluator

Executables: `pente`, `gomoku`, `keryopente`, `play`, `train`, `test`, `unit_tests`

## Tests

```bash
./unit_tests              # run all
./unit_tests -tc="MCTS*"  # filter by test name
./unit_tests -ltc         # list test cases
```

## Vision

1. Setup paralle MCTS for fast play and evaluation. 
2. Implement self-play training loop with NN.
3. Train expert-level neural network

