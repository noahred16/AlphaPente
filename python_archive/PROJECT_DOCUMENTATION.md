# AlphaPente Project Documentation

## Overview
AlphaPente is an advanced Pente game AI implementation featuring Monte Carlo Tree Search (MCTS) with sophisticated heuristics. The project includes both a command-line interface and a web-based UI for human vs AI gameplay.

## Project Structure

### Core Game Implementation
- **src/games/pente.py**: Complete Pente game engine with optimized performance
- **src/games/base_game.py**: Abstract base class for board games

### AI Components
- **src/mcts/mcts.py**: Monte Carlo Tree Search implementation with move ordering
- **src/mcts/move_heuristic.py**: Priority-based move evaluation system
- **src/players/**: Player implementations (human, MCTS AI)

### Web Interface
- **src/web/app.py**: Flask web application
- **src/web/game_controller.py**: Game session management
- **src/web/static/**: CSS and JavaScript assets
- **src/web/templates/**: HTML templates

### Legacy Implementation (v1/)
- Complete neural network training pipeline
- AlphaZero-style implementation with PyTorch
- Self-play training and evaluation systems

## Core Algorithms

### Pente Game Engine

**Key Features:**
- 19x19 board with configurable size
- Tournament rule support (2nd move ≥3 from center)
- Optimized legal move caching using sets and bitsets
- Capture detection and winner evaluation
- Move history with undo functionality

**Performance Optimizations:**
- Cached legal moves to avoid O(n²) scanning
- Winner detection only checks positions around recent moves
- Bitset representation for small boards (≤8x8)
- Lazy tournament rule cache evaluation
- Precomputed distance calculations for tournament rule and clustering

**Distance Calculations:**
The system uses Chebyshev distance (max of row/column differences) for spatial computations:
- Tournament rule: 2nd move must be ≥3 squares from center using `max(abs(row-center), abs(col-center))`
- Cluster detection: Moves within distance ≤2 of existing stones get medium priority
- Simulation fallback: Center-preferring moves using Manhattan distance for quick evaluation

### Monte Carlo Tree Search (MCTS)

**Algorithm Flow:**
1. **Selection**: Use UCB1 formula to traverse tree to leaf node
2. **Expansion**: Add one child node for best untried move
3. **Simulation**: Run epsilon-greedy rollout with heuristic guidance
4. **Backpropagation**: Update statistics up to root

**Key Features:**
- Move ordering using heuristic evaluation
- Scaled valuation system based on game length
- Critical move detection (wins/blocks) override MCTS statistics
- UCB1 exploration with configurable weight (default: 1.414)

### Move Heuristic System

**Priority Hierarchy:**
1. **Critical (100pts)**: Win now, block opponent win
2. **Very High (50pts)**: Capture stone pairs, create open 4
3. **High (20pts)**: Create open 3, block opponent open 3
4. **Medium (5pts)**: Move near existing stones (distance ≤2)
5. **Low (1pt)**: All other moves

**Evaluation Methods:**
- Pattern recognition for threats and captures
- Open-ended sequence detection
- Clustering analysis for strategic positioning

### Scaled Valuation System

**Formula:**
```
scale_factor = 0.2 + 0.8 * exp(-move_count / 50.0)

if winner == original_player:
    return scale_factor    # Win: +0.2 to +1.0
else:
    return -scale_factor   # Loss: -0.2 to -1.0
```

**Behavior:**
- Fast wins (10 moves): +0.85
- Slow wins (50 moves): +0.49
- Fast losses (10 moves): -0.85
- Slow losses (50 moves): -0.49
- All ties: 0.0

## Data Structures

### Game State
```cpp
class Pente {
    int board[19][19];           // Game board (-1, 0, 1)
    int current_player;          // 1 or -1
    int captures[2];             // Capture counts
    vector<pair<int,int>> move_history;
    set<pair<int,int>> legal_moves_cache;
    // Winner and tournament caches...
}
```

### MCTS Node
```cpp
class MCTSNode {
    Pente game_state;
    pair<int,int> move;
    MCTSNode* parent;
    vector<MCTSNode*> children;
    int visits;
    double wins;
    vector<pair<int,int>> untried_moves;
}
```

## Configuration

### Dependencies
- numpy >= 1.21.0
- pytest >= 6.0.0
- flask >= 2.0.0
- flask-session >= 0.4.0

### MCTS Parameters
- Default iterations: 1000
- Exploration weight: 1.414 (√2)
- Epsilon-greedy simulation: 20% random
- Minimum visits for reliable statistics: 1% of total iterations

### Game Rules
- Board size: 19x19 (configurable)
- Captures to win: 5 pairs (10 stones)
- Tournament rule: 2nd move ≥3 squares from center
- Win conditions: 5-in-a-row OR capture threshold

## Web API Endpoints

- `POST /api/new_game`: Initialize game with settings
- `POST /api/make_move`: Submit human player move
- `POST /api/ai_move`: Trigger AI move calculation  
- `GET /api/game_state`: Get current board state
- `GET /api/move_statistics`: Get MCTS analysis

## Testing Framework

**Comprehensive Test Suite:**
- Unit tests for game mechanics
- MCTS algorithm validation
- Heuristic evaluation correctness
- Performance optimization verification
- Integration tests for web API

**Key Test Files:**
- `tests/test_pente_legal_moves.py`: Legal move generation
- `tests/test_mcts_*.py`: MCTS algorithm components
- `tests/test_move_heuristic.py`: Heuristic evaluation
- `tests/test_ai_blocking_integration.py`: AI decision making

## C++ Implementation Considerations

### Memory Management
- Use smart pointers for MCTS tree nodes
- Implement copy-on-write for game state cloning
- Consider memory pool for frequent allocations

### Performance Optimizations
- Bitboard representation for legal moves
- Thread-safe MCTS for parallel simulations
- Template-based game interface for multiple variants
- **Precomputed distance tables**: Create lookup tables for all position pairs to eliminate runtime distance calculations

### Distance Precomputation
For optimal C++ performance, precompute distance matrices at initialization:
```cpp
class DistanceTable {
    int chebyshev[19][19][19][19];  // Chebyshev distance lookup
    int manhattan[19][19][19][19];  // Manhattan distance lookup
    
    void initialize() {
        for (int r1 = 0; r1 < 19; r1++) {
            for (int c1 = 0; c1 < 19; c1++) {
                for (int r2 = 0; r2 < 19; r2++) {
                    for (int c2 = 0; c2 < 19; c2++) {
                        chebyshev[r1][c1][r2][c2] = max(abs(r1-r2), abs(c1-c2));
                        manhattan[r1][c1][r2][c2] = abs(r1-r2) + abs(c1-c2);
                    }
                }
            }
        }
    }
};
```

### Recommended Libraries
- **Threading**: std::thread, std::async for parallel MCTS
- **Random**: std::random for simulation rollouts
- **Web**: cpp-httplib or similar for REST API
- **Testing**: Google Test framework
- **JSON**: nlohmann/json for web API serialization

### Architecture Suggestions
```cpp
// Core interfaces
class Game { /* abstract game interface */ };
class Player { /* abstract player interface */ };
class SearchAlgorithm { /* MCTS, minimax, etc. */ };

// Game-specific implementations  
class PenteGame : public Game { /* Pente rules */ };
class MCTSPlayer : public Player { /* AI player */ };
class HumanPlayer : public Player { /* human input */ };

// Web service
class GameServer { /* HTTP endpoints */ };
class SessionManager { /* game sessions */ };
```

This architecture provides a solid foundation for a high-performance C++ implementation while maintaining the algorithmic sophistication of the original Python version.