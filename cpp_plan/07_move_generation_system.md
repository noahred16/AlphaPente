7. Move Generation System
Files: include/core/move_generator.hpp, src/core/move_generator.cpp
Implementation:

Create MoveGenerator class using DistanceRings
Implement generate_ordered_moves() with distance limits
Add progressive widening logic based on visit counts
Focus moves around existing stones only

Tests:
cpp// tests/unit/test_move_generator.cpp
- Test empty board (should return center area)
- Test moves generated around placed stones
- Test distance limiting works
- Test move count limiting works
- Performance test: <1μs per generation