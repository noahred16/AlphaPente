# Parallel MCTS Design Document

## Overview

This document describes the implementation of a parallel Monte Carlo Tree Search (MCTS) system for the AlphaPente project. The goal is to enable multiple worker threads to collaboratively build a shared search tree, with neural network evaluation batched for efficiency.

## Architecture

### Core Components

#### 1. ThreadSafeNode
- **Purpose**: Represents nodes in the shared MCTS tree that can be safely accessed by multiple threads
- **Key Features**:
  - Atomic statistics: `visits`, `wins`, `totalValue` (lock-free updates)
  - Virtual loss tracking: `virtualLosses` atomic counter
  - Thread-safe expansion: `nodeSubtreeLock` mutex for critical sections
  - Standard MCTS fields: moves, priors, children, etc.

#### 2. Shared Arena
- **Purpose**: Single memory pool for all tree nodes
- **Benefits**:
  - All workers can access the same tree nodes
  - Efficient memory management (O(1) deallocation)
  - Pointers remain valid across threads

#### 3. Virtual Loss Manager
- **Purpose**: Prevents multiple workers from exploring the same promising paths
- **Mechanism**:
  - Adds virtual loss during selection to reduce PUCT values
  - Removes virtual loss after backpropagation
  - Effective visits = actual visits + virtual losses

#### 4. Evaluation Queue
- **Purpose**: Thread-safe queue for nodes awaiting neural network evaluation
- **Features**:
  - `tryPush()`: Non-blocking push with capacity limits
  - `popBatch()`: Returns batch of requests for efficient NN evaluation
  - Capacity-limited to prevent memory explosion

#### 5. Backpropagation Queue
- **Purpose**: Thread-safe queue for evaluation results ready for backpropagation
- **Features**:
  - `push()`: Add evaluation result
  - `popAll()`: Get all pending results for processing

#### 6. Worker Pool
- **Purpose**: Manages multiple worker threads for parallel tree search
- **Per-Worker Resources**:
  - Own `PenteGame` instance for move simulation
  - Own RNG for reproducible behavior
  - Local search path tracking

## Current Implementation Status

### ✅ Completed Components

#### Core Infrastructure
- **ParallelMCTS.hpp**: Complete header with all class definitions
- **ParallelMCTS.cpp**: Constructor, destructor, basic search orchestration
- **CMakeLists.txt**: Added ParallelMCTS.cpp to build system

#### Thread-Safe Data Structures
- **VirtualLossManager**: `addVirtualLoss()`, `removeVirtualLoss()`, `getEffectiveVisits()`
- **EvaluationQueue**: Thread-safe with mutex, capacity management, batch operations
- **BackpropagationQueue**: Thread-safe with mutex, bulk operations

#### Worker Thread System
- **WorkerPool**: Start/stop thread management, graceful shutdown
- **workerThreadMain()**: Main worker loop with selection and backpropagation phases

#### Selection Phase
- **select()**: Traverses tree using PUCT selection until leaf node
- **selectBestMoveIndex()**: PUCT formula with effective visits (virtual loss aware)
- **Virtual Loss Integration**: Adds virtual loss during selection traversal

#### Basic Search Orchestration
- **search()**: Initializes workers, runs brief search, collects results
- **Thread Safety**: Tree lock protects concurrent access during selection

### 🔄 Partially Implemented

#### Node Allocation
- **allocateNode()**: Basic arena allocation
- **initNodeChildren()**: TODO - needs implementation for child array setup

### ❌ Not Yet Implemented

#### Evaluation Thread
- **Missing**: Dedicated thread to process evaluation queue
- **Required**: Batch NN evaluation, push results to backpropagation queue

#### Expansion Phase
- **expand()**: Create children from evaluation results
- **Missing**: Policy application, child node creation, prior setting

#### Backpropagation Phase
- **backpropagate()**: Update statistics up the search path
- **Missing**: Atomic updates, virtual loss removal, minimax propagation

#### Search Orchestration
- **search()**: Currently just sleeps - needs proper iteration control
- **Missing**: Iteration counting, convergence detection, time limits

#### Tree Management
- **getBestMove()**: Select best child from root
- **reuseSubtree()**: Reuse tree for subsequent moves
- **undoSubtree()**: Rollback tree changes
- **clearTree()**: Proper tree cleanup

#### Advanced Features
- **Canonical Hashing**: Position symmetry handling
- **Transposition Table**: Node deduplication
- **Lazy Policy Loading**: On-demand prior computation
- **Tree Statistics**: Size counting, depth analysis

## Detailed Work Remaining

### Phase 1: Complete MCTS Cycle (High Priority)

#### 1.1 Evaluation Thread Implementation
**File**: `ParallelMCTS.cpp`
**Location**: Add to `WorkerPool` or as separate component
**Requirements**:
- Separate thread that monitors `evaluationQueue_`
- Batch processing: `popBatch(config_.evaluationBatchSize)`
- NN evaluation: `config_.evaluator->evaluate(gameState)`
- Result packaging: Create `EvaluationResult` with value, policy, searchPath
- Push to `backpropagationQueue_`

**Code Structure**:
```cpp
void evaluationThreadMain() {
    while (running) {
        auto batch = evaluationQueue_->popBatch(batchSize);
        if (!batch.empty()) {
            // Batch evaluate
            // Push results to backpropagation queue
        }
        std::this_thread::yield();
    }
}
```

#### 1.2 Expand Method
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Take `EvaluationResult` (node, value, policy, searchPath)
- Set node value: `node->value = result.value`
- Create children array using `initNodeChildren()`
- Set priors from policy
- Handle canonical coordinates if needed
- Mark node as expanded and evaluated

**Integration**: Called in worker backpropagation phase

#### 1.3 Backpropagation Method
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Update atomic statistics up the search path
- Remove virtual loss from nodes
- Handle terminal node detection
- Minimax propagation for solved positions

**Thread Safety**: Use atomic operations for statistics updates

#### 1.4 Search Orchestration
**File**: `ParallelMCTS.cpp`
**Current**: `std::this_thread::sleep_for(std::chrono::milliseconds(100))`
**Required**:
- Track total iterations across all workers
- Wait for completion or time limits
- Graceful worker shutdown
- Result aggregation

### Phase 2: Tree Management (Medium Priority)

#### 2.1 Best Move Selection
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Analyze root children statistics
- Select move with highest visit count
- Handle proven wins/losses
- Return physical coordinates

#### 2.2 Subtree Reuse
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Find child corresponding to played move
- Update root pointer
- Maintain tree structure
- Handle canonical coordinates

#### 2.3 Tree Statistics
**File**: `ParallelMCTS.cpp`
**Requirements**:
- `getTreeSize()`: Count total nodes
- `printStats()`: Comprehensive statistics
- `printBestMoves()`: Top move analysis
- Memory utilization tracking

### Phase 3: Advanced Features (Lower Priority)

#### 3.1 Canonical Hashing
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Position symmetry detection
- Canonical coordinate conversion
- Transposition table integration
- Memory efficiency improvements

#### 3.2 Lazy Policy Loading
**File**: `ParallelMCTS.cpp`
**Requirements**:
- On-demand prior computation
- Policy caching
- Memory optimization

#### 3.3 Robustness Features
**File**: `ParallelMCTS.cpp`
**Requirements**:
- Memory limit handling
- Timeout management
- Error recovery
- Configuration validation

## Integration Points

### With Existing MCTS
- **Evaluator Interface**: Reuse `Evaluator::evaluate()` and `evaluatePolicy()`
- **PenteGame**: Compatible move simulation and state management
- **Arena**: Same memory management system

### With Build System
- **CMakeLists.txt**: Already integrated
- **Dependencies**: OpenMP support (future parallelization)

## Testing Strategy

### Unit Tests
- Thread safety verification
- Queue operations
- Virtual loss mechanics
- Basic selection logic

### Integration Tests
- Single worker functionality
- Multi-worker coordination
- Memory usage validation
- Performance benchmarking

### Validation Tests
- Correctness vs sequential MCTS
- Convergence behavior
- Memory leak detection

## Performance Considerations

### Current Bottlenecks
- Tree lock contention during selection
- Single evaluation thread
- Atomic operations overhead

### Future Optimizations
- Lock-free data structures
- Multiple evaluation threads
- SIMD evaluation
- NUMA-aware allocation

## Configuration Options

### ParallelMCTS::Config
- `numWorkers`: Number of search threads (default: 4)
- `evaluationBatchSize`: NN batch size (default: 32)
- `queueCapacity`: Max queued evaluations (default: 10000)
- `arenaSize`: Memory allocation limit
- All standard MCTS parameters

## Usage Example

```cpp
ParallelMCTS::Config config;
config.numWorkers = 8;
config.maxIterations = 10000;

ParallelMCTS mcts(config);
PenteGame game;
PenteGame::Move bestMove = mcts.search(game);
```

## Conclusion

The parallel MCTS implementation provides a solid foundation with thread-safe data structures and basic selection mechanics. The remaining work focuses on completing the MCTS cycle (evaluation → expansion → backpropagation) and adding tree management features. The architecture supports efficient parallel search while maintaining correctness and memory safety.</content>
<parameter name="filePath">/Users/noahredsmith/modules/AlphaPente/ParallelDesign.md