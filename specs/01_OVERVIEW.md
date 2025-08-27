# Fast MCTS Pente Implementation Design Document

## Project Overview
A high-performance Monte Carlo Tree Search implementation for Pente using bitboard representation, focused on speed and efficiency for self-play and game analysis.

## Performance Targets
- Simulation Rate: 10,000+ simulations per second
- Self-Play: Multiple complete games per second (40 moves/game)
- Move Generation: < 1 microsecond per move
- Tree Reuse: Preserve 50-80% of tree between moves

## Core Design Principles

### Single State with Undo (Critical for Speed)
Instead of copying game states for each simulation, maintain ONE game state and use make/undo operations:
- Eliminates thousands of deep copies per second
- Uses move deltas to perfectly reverse moves
- Orders of magnitude faster than state copying

### Precomputed Distance Rings
Calculate board geometry once at startup rather than repeatedly during search:
- 361 × 361 distance calculations done once (~50ms startup)
- O(1) lookup during MCTS instead of O(n) distance calculations
- Enables efficient progressive widening
