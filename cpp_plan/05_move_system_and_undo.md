5. Move System & Undo Logic
Files: Complete GameState class implementation
Implementation:

Implement make_move(Position) returning MoveDelta
Implement undo_move() using MoveDelta
Add player switching logic
Integrate capture detection with move making
Add terminal state detection (wins + capture limits)

Tests:
cpp// tests/unit/test_game_state.cpp
- Test move making and undoing
- Test capture integration with moves
- Test terminal state detection
- Test move history tracking
- Stress test: 1000 random move/undo cycles