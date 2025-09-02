4. Capture Detection System
Files: Update include/core/game_state.hpp, src/core/game_state.cpp
Implementation:

Create basic GameState class with BitBoard and move history
Implement detect_captures(Position, player) method
Add capture pattern recognition (opponent-opponent bracketing)
Implement capture removal and counting

Tests:
cpp// tests/unit/test_capture_detection.cpp
- Test basic capture patterns
- Test multiple captures in one move
- Test no false positives
- Test capture counting accuracy
- Test edge case captures