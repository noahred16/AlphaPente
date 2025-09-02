10. MCTS Engine Core
Files: include/mcts/mcts_engine.hpp, src/mcts/mcts_engine.cpp
Implementation:

Create MCTSEngine with single GameState reference
Implement core MCTS loop (select, expand, simulate, backpropagate)
Add basic time management and early stopping
Implement move selection from tree

Tests:
cpp// tests/unit/test_mcts_engine.cpp
- Test complete MCTS iterations work
- Test time limits are respected
- Test early stopping logic
- Test move selection returns valid moves