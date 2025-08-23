# Pente MCTS Performance Optimization Plan

## Major Performance Bottlenecks

### 1. **Excessive Game State Cloning** (Critical)
- **Location**: `mcts.py:11, 48, 66, 131, 158`
- **Issue**: Every MCTS node creates a full game clone, and simulations clone again
- **Impact**: O(board_size²) memory per node, massive memory overhead
- **Solution**: Implement move/undo pattern instead of cloning

### 2. **Inefficient Legal Move Generation** (High)
- **Location**: `pente.py:15-31`
- **Issue**: O(board_size²) scan for every move request
- **Impact**: Called thousands of times during MCTS
- **Solution**: Maintain incremental legal moves list

### 3. **Repeated Heuristic Object Creation** (High)
- **Location**: `mcts.py:18, 93, 140`
- **Issue**: New MoveHeuristic instance for every node/simulation
- **Impact**: Unnecessary object allocation overhead
- **Solution**: Reuse single heuristic instance

### 4. **Inefficient Win Detection** (Medium)
- **Location**: `pente.py:125-154`
- **Issue**: Full board scan for every terminal check
- **Impact**: O(board_size²) per check
- **Solution**: Incremental win tracking from last move

### 5. **Redundant Move Scoring** (Medium)
- **Location**: `move_heuristic.py:20-28`
- **Issue**: Each move evaluated independently with game cloning
- **Impact**: O(moves × complexity) per evaluation
- **Solution**: Batch evaluation with temporary move testing

## Specific Optimizations

### Memory Usage
- **Node storage**: Current ~361 bytes per node for 19×19 board
- **Clone overhead**: 1.4KB per game state clone
- **Recommendation**: Move/undo pattern could reduce memory by 80%

### Algorithm Efficiency
- **Legal moves**: Cache and update incrementally
- **Win detection**: Track from move position only, not full board
- **Heuristics**: Pre-compute static patterns, reuse evaluators

### Data Structures
- **Board representation**: Consider bitboards for capture detection
- **Move lists**: Use arrays instead of Python lists for hot paths
- **Transposition table**: Add caching for repeated positions

## Implementation Priority
1. **Replace cloning with move/undo** - Biggest impact
2. **Incremental legal moves** - High frequency operation  
3. **Reuse heuristic objects** - Simple but effective
4. **Optimize win detection** - Reduce terminal check cost
5. **Add transposition table** - Avoid duplicate work

## Expected Impact
These optimizations could improve MCTS performance by 3-5x while reducing memory usage significantly.