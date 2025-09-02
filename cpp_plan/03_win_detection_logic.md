3. Win Detection Logic
Files: Update include/core/bitboard.hpp, src/core/bitboard.cpp
Implementation:

Add check_five_in_a_row(row, col, player) method to BitBoard
Implement directional checking (horizontal, vertical, both diagonals)
Use efficient bit manipulation for consecutive stone detection

Tests:
cpp// tests/unit/test_win_detection.cpp
- Test horizontal wins (all positions)
- Test vertical wins (all positions) 
- Test diagonal wins (both directions)
- Test near-miss cases (4 in a row, gaps)
- Test edge/corner win conditions