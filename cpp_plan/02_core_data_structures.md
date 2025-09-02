2. Core Data Structures
Files: include/core/move_delta.hpp, include/core/bitboard.hpp
Implementation:

Define Position struct with int8_t row/col
Implement MoveDelta struct with fixed-size arrays
Create BitBoard class with 6x uint64_t arrays per player
Implement core BitBoard methods: set_stone, remove_stone, get_stone, is_empty
Add optimized pos_to_bit_index calculation

Tests:
cpp// tests/unit/test_bitboard.cpp
- Test position to bit index conversion
- Test stone placement and removal
- Test edge cases (corners, boundaries)
- Test all 361 board positions work correctly