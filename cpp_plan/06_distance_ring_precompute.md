6. Distance Ring Precomputation
Files: include/core/distance_rings.hpp, src/core/distance_rings.cpp
Implementation:

Precompute distance relationships for all 361 positions
Store in efficient data structure for O(1) lookup
Implement get_positions_at_distance(center, distance)
Add get_ordered_moves_around_stones() method

Tests:
cpp// tests/unit/test_distance_rings.cpp
- Verify distance calculations are correct
- Test center positions vs edge positions
- Test performance (should be ~50ms initialization)
- Validate all distances up to maximum (18)