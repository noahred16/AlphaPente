### 1.5 MCTS with Single State and Tree Reuse

MCTSEngine key features:
- ONE shared GameState (never copied)
- Tree reuse via update_root()
- Progressive widening in tree policy
- Track moves_made list for undo

Search loop:
1. Start with empty moves_made list
2. Run tree_policy (modifies single state)
3. Run rollout (continues modifying)
4. Undo ALL moves using move history
5. Backpropagate results
6. Repeat until time limit

Tree reuse:
- After opponent move, find child node
- Make it new root
- Preserve entire subtree
- Garbage collection handles old branches

Critical: Never deep copy state!
