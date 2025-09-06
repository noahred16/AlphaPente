# AlphaPente C++ Implementation Plan

## TODOs
- [ ] More sophisticated evaluate_result function. 

## Overview
This document outlines the high-level plan for implementing the AlphaPente MCTS bitboard Pente program in C++ for maximum performance. Based on the Python implementation and detailed specifications, this C++ version targets 10,000+ simulations per second and complete games in under 100ms.

## Commands
- `make run`: run main.cpp
- `make test`: run all unit tests in tests/unit/

## Performance Goals
- **Simulation Rate**: 10,000+ MCTS simulations per second
- **Move Generation**: < 1 microsecond per move
- **Self-Play**: Multiple complete games per second (40 moves/game average)
- **Tree Reuse**: Preserve 50-80% of MCTS tree between moves
- **Memory Efficiency**: Minimize allocations during search

## Development Environment & Tools

### Build System & Dependencies
- **Build System**: Conan 2.x for dependency management
- **Build Tool**: Makefile with common commands (build, test, clean, benchmark)
- **Testing Framework**: GoogleTest for unit tests and benchmarks
- **Compiler**: Modern C++17/20 with optimization flags (-O3, -march=native)

### Project Structure
```
AlphaPente-cpp/
├── conanfile.txt           # Conan dependencies
├── Makefile               # Build commands
├── CMakeLists.txt         # CMake configuration
├── include/
│   ├── core/
│   │   ├── bitboard.hpp
│   │   ├── game_state.hpp
│   │   ├── move_delta.hpp
│   │   ├── move_generator.hpp
│   │   └── distance_rings.hpp
│   ├── mcts/
│   │   ├── mcts_node.hpp
│   │   ├── mcts_engine.hpp
│   │   └── rollout.hpp
│   └── utils/
│       ├── memory_pool.hpp
│       └── timer.hpp
├── src/
│   ├── core/
│   ├── mcts/
│   └── main.cpp
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
└── docs/
```

## Core Implementation Strategy

### 1. Single-State Architecture (Critical)
**Never copy game states during MCTS - this is the key performance optimization**

```cpp
struct Position {
    int8_t row, col;  // int8_t sufficient for 19x19 board
    
    bool operator==(const Position& other) const noexcept {
        return row == other.row && col == other.col;
    }
}
struct MoveDelta {
    Position move_pos;
    std::array<Position, 8> captured_stones; // For undo, the actual stones captured
    uint8_t capture_count = 0;               // How many stones captured THIS move
    std::array<int, 2> captures_before; // Both players' counts BEFORE this move
};

class GameState {
private:
    BitBoard board_;                    // Single bitboard instance
    std::vector<MoveDelta> move_history_;  // For efficient undo
    bool is_player1_turn_ = true;  // true = player 1, false = player -1
    std::array<int, 2> captures_ = {0, 0};
    
public:
    MoveDelta make_move(Position pos);     // Returns undo information
    void undo_move();                      // Uses last delta for perfect undo
    // No copy constructor/assignment - prevent accidental copies
    GameState(const GameState&) = delete;
    GameState& operator=(const GameState&) = delete;
    int current_player() const noexcept { 
        return is_player1_turn_ ? 1 : -1; 
    }
    
    void switch_player() noexcept { 
        is_player1_turn_ = !is_player1_turn_; 
    }
    
    bool is_terminal() const noexcept;  // Check win/capture conditions
    int get_winner() const noexcept;    // Return winner if terminal
};
```

### 2. High-Performance BitBoard
**Use 6x 64-bit integers per player for 19x19 board (361 positions)**

```cpp
class BitBoard {
private:
    std::array<uint64_t, 6> player1_bits_ = {0}; // Player 1 stones
    std::array<uint64_t, 6> player2_bits_ = {0}; // Player -1 stones
    
    inline std::pair<int, int> pos_to_bit_index(int row, int col) const noexcept {
        int bit_index = (row << 4) + (row << 1) + (row) + col; // row * 19 + col
        return {bit_index >> 6, bit_index & 63}; // / 64, % 64
    }
    
public:
    inline void set_stone(int row, int col, int player) noexcept;
    inline void remove_stone(int row, int col) noexcept;
    inline int get_stone(int row, int col) const noexcept;
    inline bool is_empty(int row, int col) const noexcept;
    bool check_five_in_a_row(int row, int col, int player) const noexcept; // Check win using last move
};
```

### 3. Precomputed Distance Rings
**O(1) distance lookups eliminate runtime calculations**

```cpp
class DistanceRings {
private:
    // Pre-allocated 2D arrays for maximum performance
    static constexpr int BOARD_SIZE = 19;
    static constexpr int MAX_POSITIONS = BOARD_SIZE * BOARD_SIZE;
    
    // rings_[position_id][distance] = vector of positions at that distance
    std::array<std::array<std::vector<Position>, BOARD_SIZE>, MAX_POSITIONS> rings_;
    
public:
    DistanceRings();  // Precompute all distances at startup (~50ms)
    
    const std::vector<Position>& get_positions_at_distance(
        Position center, int distance) const noexcept;
    
    std::vector<Position> get_ordered_moves_around_stones(
        const std::vector<Position>& stone_positions,
        int max_distance = 3) const;
};
```

### 4. Efficient Move Generation
**Distance-ordered moves with progressive widening**

```cpp
class MoveGenerator {
private:
    const DistanceRings& distance_rings_;
    
public:
    std::vector<Position> generate_ordered_moves(
        const GameState& state,
        int max_moves = 50,      // Progressive widening based on node visits
        int max_distance = 3     // Distance limit for efficiency
    ) const;
    
    // Progressive widening strategy:
    // < 10 visits: distance 1, 15 moves max
    // < 100 visits: distance 2, 30 moves max  
    // < 1000 visits: distance 3, 50 moves max
    // 1000+ visits: distance 5, 80 moves max
};
```

### 5. MCTS Engine with Tree Reuse
**Memory-efficient MCTS with sophisticated optimizations**

```cpp
class MCTSNode {
public:
    // Hot data (accessed every simulation)
    int visits_ = 0;
    double wins_ = 0.0;
    MCTSNode* parent_ = nullptr;
    
    // Cold data (accessed less frequently)
    Position move_; // Move that led to this state
    std::vector<std::unique_ptr<MCTSNode>> children_;
    std::vector<Position> untried_moves_;
    
    // UCB1 calculation
    double ucb1_value(double exploration_weight = 1.414) const noexcept;
    
    MCTSNode* select_best_child() const noexcept;
    MCTSNode* expand(GameState& state, const MoveGenerator& move_gen);
};

class MCTSEngine {
private:
    std::unique_ptr<MCTSNode> root_;
    GameState& shared_state_;               // Single shared state
    const MoveGenerator& move_generator_;
    std::mt19937 rng_;
    
public:
    Position search(int max_iterations = 1000, double time_limit_ms = 1000.0);
    void update_root(Position opponent_move);  // Tree reuse after opponent move, Subtree Extraction
    
private:
    double simulate(MCTSNode* leaf);          // Rollout phase: Random Near-Stone Moves using DistanceRings
    void backpropagate(MCTSNode* node, double result);
    
    // Smart time management
    bool should_stop_early() const;           // Stop if one move dominates after some threshold
};
```

## Advanced Optimizations

### 1. Memory Management
```cpp
// Memory pool for MCTSNode allocation
class NodePool {
private:
    std::vector<std::unique_ptr<MCTSNode[]>> chunks_;
    std::stack<MCTSNode*> free_nodes_;
    static constexpr size_t CHUNK_SIZE = 10000;
    
public:
    MCTSNode* allocate();
    void deallocate(MCTSNode* node);
};
```

### 2. Heuristic Evaluation
```cpp
class MoveHeuristic {
public:
    enum class Priority {
        CRITICAL = 100,    // Win now, block opponent win
        VERY_HIGH = 50,    // Capture stones, create open 4
        HIGH = 20,         // Create open 3, block opponent open 3  
        MEDIUM = 5,        // Move near existing stones (distance ≤2)
        LOW = 1            // All other moves
    };
    
    Priority evaluate_move(const GameState& state, Position move) const;
    std::vector<Position> order_moves_by_heuristic(
        const GameState& state, 
        const std::vector<Position>& moves) const;
};
```

### 3. Parallel MCTS (Future Enhancement)
```cpp
class ParallelMCTS {
private:
    std::vector<std::thread> worker_threads_;
    std::mutex tree_mutex_;                  // Protect tree updates
    std::atomic<bool> stop_search_ = false;
    
public:
    Position parallel_search(int num_threads = 4, double time_limit_ms = 1000.0);
};
```

## Key Performance Considerations

### Compiler Optimizations
```makefile
CXXFLAGS = -std=c++17 -O3 -march=native -flto -DNDEBUG
CXXFLAGS += -ffast-math -funroll-loops -ftree-vectorize
```

### Critical Performance Patterns
1. **Inline Hot Paths**: Mark bitboard operations and distance lookups as `inline noexcept`
2. **Avoid Allocations**: Use pre-allocated vectors and memory pools in hot loops  
3. **Cache-Friendly Data**: Structure data for sequential access patterns
4. **Branch Prediction**: Use `likely`/`unlikely` hints on conditional branches
5. **SIMD Opportunities**: Consider vectorization for bitboard operations

### Profiling Strategy
- **CPU Profiler**: Use `perf` to identify bottlenecks
- **Memory Profiler**: Ensure minimal allocations during search
- **Benchmarks**: Target-specific tests for each component

## Conan Dependencies
```ini
[requires]
gtest/1.14.0

[generators]
CMakeDeps
CMakeToolchain

[options]
gtest/*:shared=False
```

## Makefile Commands
```makefile
.PHONY: build test clean benchmark install

build:
	conan install . --output-folder=build --build=missing
	cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
	cd build && cmake --build . --config Release

test:
	cd build && ctest --output-on-failure

benchmark:
	cd build && ./bin/benchmark --benchmark_repetitions=3

clean:
	rm -rf build/

install: build
	cd build && cmake --install .
```

## Testing Strategy

### Unit Tests (GoogleTest)
- **BitBoard**: Set/remove/query operations, edge cases
- **MoveDelta**: Perfect undo functionality  
- **DistanceRings**: Distance calculations and ordering
- **GameState**: Move making, capture detection, win conditions
- **MoveGenerator**: Legal moves, progressive widening
- **MCTSEngine**: Tree operations, UCB1 calculations

### Integration Tests
- **Full Game Play**: Complete self-play games
- **Tree Reuse**: Verify tree preservation between moves
- **Performance Regression**: Ensure optimizations don't break correctness

### Benchmarks
- **Simulations/Second**: Target 10,000+ per second
- **Move Generation Speed**: Target < 1 microsecond
- **Memory Usage**: Monitor allocation patterns
- **Tree Reuse Efficiency**: Measure preservation percentage

## Migration Plan from Python

### Phase 1: Core Implementation (Week 1-2)
1. Set up build system (Conan, Makefile, GoogleTest)
2. Implement BitBoard with comprehensive tests
3. Implement MoveDelta and GameState with undo functionality
4. Port DistanceRings with precomputation

### Phase 2: MCTS Engine (Week 2-3)
1. Implement MCTSNode and basic tree operations
2. Port MCTSEngine with single-state architecture
3. Implement MoveGenerator with progressive widening
4. Add heuristic evaluation system

### Phase 3: Optimization & Testing (Week 3-4)
1. Add memory pooling and performance optimizations
2. Implement tree reuse between moves
3. Comprehensive benchmarking and profiling
4. Performance tuning and compiler optimization

### Phase 4: Advanced Features (Week 4+)
1. Parallel MCTS implementation
2. Web API interface (optional)
3. Tournament play and analysis tools
4. Additional optimizations based on profiling

## Expected Performance Gains

Based on the single-state architecture and precomputed optimizations:
- **10-50x faster** than Python implementation
- **Sub-millisecond** move generation
- **10,000+ simulations/second** on modern hardware
- **Efficient memory usage** with minimal allocations

This C++ implementation will provide the foundation for high-performance Pente AI capable of strong tournament play and rapid self-play for training purposes.