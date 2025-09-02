11. Tree Reuse System
Files: Update MCTSEngine class
Implementation:

Implement update_root(Position) method
Add subtree extraction and reparenting
Handle cases where opponent move wasn't explored
Add tree preservation statistics/logging

Tests:
cpp// tests/unit/test_tree_reuse.cpp
- Test successful tree reuse scenarios
- Test fallback when move not in tree
- Test tree structure remains valid after reuse
- Test memory management during tree operations