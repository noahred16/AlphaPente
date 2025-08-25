## Testing Strategy

### Unit Tests
1. BitBoard: set/remove/query operations
2. Undo: complete state restoration
3. Distance Rings: verify distances
4. Move Generation: correct ordering
5. Captures: proper detection

### Performance Benchmarks
1. Simulations per second (target: 10,000+)
2. Tree reuse percentage (target: 30%+)
3. Move generation speed (target: <1μs)
4. Self-play games per second

## File Structure
pente-mcts/
├── core/
│   ├── bitboard.py
│   ├── game_state.py
│   ├── move_generator.py
│   └── distance_rings.py
├── mcts/
│   ├── node.py
│   ├── engine.py
│   └── rollout.py
├── tests/
│   ├── test_bitboard.py
│   ├── test_game_state.py
│   └── benchmark.py
└── main.py

## Critical Implementation Notes
1. Single State is Essential - no copies!
2. Precompute Everything at startup
3. Progressive Widening for efficiency
4. Tree Reuse between moves
5. Profile Everything to find bottlenecks
