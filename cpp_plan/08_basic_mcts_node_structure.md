8. Basic MCTS Node Structure
Files: include/mcts/mcts_node.hpp, src/mcts/mcts_node.cpp
Implementation:

Create MCTSNode class with visit/win statistics
Implement UCB1 calculation
Add child node management (expansion, selection)
Implement best child selection logic

Tests:
cpp// tests/unit/test_mcts_node.cpp
- Test UCB1 calculation accuracy
- Test child node creation and management
- Test selection logic with various visit counts
- Test memory management (no leaks)