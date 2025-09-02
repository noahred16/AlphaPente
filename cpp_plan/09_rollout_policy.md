9. Rollout Policy Implementation
Files: include/mcts/rollout.hpp, src/mcts/rollout.cpp
Implementation:

Create simple random rollout policy
Focus rollouts on moves near existing stones
Add rollout length limits and termination
Implement result evaluation (win/loss/draw)

Tests:
cpp// tests/unit/test_rollout.cpp
- Test rollouts reach terminal states
- Test rollout length limits work
- Test result evaluation accuracy
- Performance test: many rollouts per second