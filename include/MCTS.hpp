#ifndef MCTS_HPP
#define MCTS_HPP

#include "PenteGame.hpp"
#include "Evaluator.hpp"
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// ============================================================================
// Arena Allocator for O(1) Tree Destruction
// ============================================================================

class MCTSArena {
public:
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024; // 256 MB
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 1.5; // 384 MB
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 2; // 512 MB
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 4; // 1 GB
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 8ull; // 2 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 12ull; // 3 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 16ull; // 4 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 20ull; // 5 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 24ull; // 6 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 32ull; // 8 GB (unsigned long long to avoid overflow)
    // static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 40ull; // 10 GB (unsigned long long to avoid overflow)
    static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024 * 48ull; // 12 GB (unsigned long long to avoid overflow)

    // static constexpr size_t MAX_SIZE = 16ULL * 1024 * 1024 * 1024; // 16 GB



    explicit MCTSArena(size_t size = DEFAULT_SIZE)
        : size_(size)
        , offset_(0)
        , memory_(nullptr) {
        memory_ = static_cast<char*>(std::aligned_alloc(64, size_)); // 64-byte alignment for cache lines
        if (!memory_) {
            throw std::bad_alloc();
        }
    }

    ~MCTSArena() {
        std::free(memory_);
    }

    // Non-copyable, non-movable
    MCTSArena(const MCTSArena&) = delete;
    MCTSArena& operator=(const MCTSArena&) = delete;

    // Allocate memory for type T with proper alignment
    template<typename T>
    T* allocate(size_t count = 1) {
        // Align to T's alignment requirement
        size_t alignment = alignof(T);
        size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        size_t totalBytes = sizeof(T) * count;

        if (alignedOffset + totalBytes > size_) {
            // Out of arena memory
            return nullptr;
        }

        T* ptr = reinterpret_cast<T*>(memory_ + alignedOffset);
        offset_ = alignedOffset + totalBytes;
        return ptr;
    }

    // O(1) tree destruction - just reset the offset
    void reset() {
        offset_ = 0;
    }

    // Swap internals with another arena (for subtree reuse)
    void swap(MCTSArena& other) {
        std::swap(size_, other.size_);
        std::swap(offset_, other.offset_);
        std::swap(memory_, other.memory_);
    }

    // Statistics
    size_t bytesUsed() const { return offset_; }
    size_t bytesRemaining() const { return size_ - offset_; }
    size_t totalSize() const { return size_; }
    double utilizationPercent() const { return 100.0 * offset_ / size_; }

private:
    size_t size_;
    size_t offset_;
    char* memory_;
};

// ============================================================================
// MCTS Class with Arena-Allocated Nodes
// ============================================================================

class MCTS {
public:
    // Solved status for minimax backpropagation (1 byte)
    enum class SolvedStatus : uint8_t {
        UNSOLVED = 0,  // Not proven yet
        SOLVED_WIN,    // Proven win for the player who made the move
        SOLVED_LOSS    // Proven loss for the player who made the move
    };
    enum class SearchMode { UCB1, PUCT };
    enum class HeuristicMode { UNIFORM, HEURISTIC, NEURAL_NET };
    enum class NodeState : uint8_t { UNEXPANDED = 0, EXPANDING, EXPANDED };

    // Configuration parameters
    struct Config {
        double explorationConstant;    // UCB1 exploration parameter
        int maxIterations = 10000;     // Number of MCTS iterations
        int maxSimulationDepth = 200;  // Max playout depth
        size_t arenaSize = MCTSArena::DEFAULT_SIZE; // Arena size in bytes
        
        SearchMode searchMode = SearchMode::UCB1;
        Evaluator* evaluator = nullptr; // For PUCT priors and value evaluation
        HeuristicMode heuristicMode = HeuristicMode::HEURISTIC;

        Config() : explorationConstant(std::sqrt(2.0)) {}
    };

    struct ParallelConfig {
        int numWorkers = 7;
        int batchSize = 32;           // Positions to batch for inference
        int batchTimeoutMs = 5;       // Max wait time for batch to fill
        bool useInferenceThread = true; // false = workers evaluate inline (CPU-only mode)
    };

    // Node in the MCTS tree - trivially destructible, ~64 bytes
    // All dynamic arrays are arena-allocated via raw pointers
    struct Node {
        // Move that led to this node (4 bytes)
        PenteGame::Move move;

        // Player who made the move (1 byte, uint8_t-backed enum)
        PenteGame::Player player;

        // Minimax proof status (atomic for parallel solver propagation)
        std::atomic<SolvedStatus> solvedStatus{SolvedStatus::UNSOLVED};

        // Child array metadata (2 bytes each = 4 bytes)
        uint16_t childCount = 0;
        uint16_t childCapacity = 0;

        // Untried moves metadata (atomic for parallel solver)
        std::atomic<int16_t> unprovenCount{0};

        // Statistics (16 bytes)
        std::atomic<int32_t> visits{0};
        std::atomic<int32_t> wins{0};
        std::atomic<double> totalValue{0.0};
        std::atomic<int32_t> virtualLoss{0};
        std::atomic<NodeState> state{NodeState::UNEXPANDED};

        float prior = -1.0f;
        float value = 0.0f;

        // Pointers (24 bytes)
        Node* parent = nullptr;
        Node** children = nullptr;            // Arena-allocated array of child pointers

        bool isFullyExpanded() const { return state.load(std::memory_order_acquire) == NodeState::EXPANDED; }
        bool isTerminal() const { return solvedStatus.load(std::memory_order_relaxed) != SolvedStatus::UNSOLVED; }
        double getUCB1Value(double explorationFactor) const;
        double getPUCTValue(double explorationFactor, int parentVisits) const;
    };

    // Constructor
    explicit MCTS(const Config& config = Config());
    ~MCTS();

    // Main search interface
    PenteGame::Move search(const PenteGame& game);
    PenteGame::Move parallelSearch(const PenteGame& game,
                                   const ParallelConfig& pconfig);

    // Get best move from current tree (no additional search)
    PenteGame::Move getBestMove() const;

    // Tree management
    void reset();
    void clearTree();
    void reuseSubtree(const PenteGame::Move& move);

    // Statistics and debugging
    int getTotalVisits() const;
    int getTreeSize() const;
    void printStats() const;
    void printBestMoves(int topN = 5) const;
    void printBranch(const char* moveStr, int topN = 5) const;
    void printBranch(int x, int y, int topN = 5) const;

    // Arena statistics
    size_t getArenaUsedBytes() const { return arena_.bytesUsed(); }
    double getArenaUtilization() const { return arena_.utilizationPercent(); }

    // Configuration
    void setConfig(const Config& config);
    const Config& getConfig() const;

private:
    // MCTS phases
    Node* select(Node* node, PenteGame& game);
    Node* expand(Node* node, PenteGame& game);
    double simulate(Node* node, PenteGame& game);
    void backpropagate(Node* node, double result);

    // Helper methods
    Node* selectBestChild(Node* node) const;
    void updateChildrenPriors(Node* node, const PenteGame& game);
    double evaluateTerminalState(const PenteGame& game, int depth = 0) const;

    // Arena allocation helpers
    Node* allocateNode();
    void initNodeChildren(Node* node, int capacity);
    void initializeNodePriors(Node* node, const std::vector<std::pair<PenteGame::Move, float>>& movePriors);

    // Tree reuse helpers (copies subtree to fresh arena)
    Node* copySubtree(Node* source, MCTSArena& destArena);
    void pruneTree(Node* keepNode);

    // Print helpers
    Node* findChildNode(Node* parent, int x, int y) const;
    void printMovesFromNode(Node* node, int topN) const;
    int countNodes(Node* node) const;

    // ---- Parallel search infrastructure ----

    struct InferenceRequest {
        Node* node;
        PenteGame gameState;
        int threadId;
    };

    struct InferenceResult {
        Node* node;
        std::vector<std::pair<PenteGame::Move, float>> movePriors;
        float value;
    };

    template<typename T>
    class ThreadSafeQueue {
    public:
        void push(T item) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
            cv_.notify_one();
        }

        bool tryPop(T& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return false;
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }

        bool waitPop(T& item, int timeoutMs) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                            [this] { return !queue_.empty(); })) {
                item = std::move(queue_.front());
                queue_.pop();
                return true;
            }
            return false;
        }

        size_t drainTo(std::vector<T>& out, size_t maxCount) {
            std::lock_guard<std::mutex> lock(mutex_);
            size_t count = 0;
            while (!queue_.empty() && count < maxCount) {
                out.push_back(std::move(queue_.front()));
                queue_.pop();
                count++;
            }
            return count;
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<T> queue_;
    };

    // Parallel worker/inference thread entry points
    void workerThread(
        int threadId,
        std::atomic<bool>& stopFlag,
        std::atomic<int>& iterationCount,
        ThreadSafeQueue<InferenceRequest>& inferenceQueue,
        ThreadSafeQueue<InferenceResult>& backpropQueue,
        const PenteGame& rootGame,
        const ParallelConfig& pconfig
    );

    void inferenceThread(
        std::atomic<bool>& stopFlag,
        ThreadSafeQueue<InferenceRequest>& inferenceQueue,
        ThreadSafeQueue<InferenceResult>& backpropQueue,
        int batchSize,
        int batchTimeoutMs
    );

    // Parallel MCTS phases
    Node* selectParallel(Node* node, PenteGame& game);
    void expandParallel(Node* node,
                       const std::vector<std::pair<PenteGame::Move, float>>& movePriors,
                       float value);
    void backpropWithVirtualLoss(Node* node, double result);

    // Virtual loss helpers
    void addVirtualLoss(Node* node);
    void removeVirtualLoss(Node* node);
    void cancelVirtualLoss(Node* node);

    // Parallel solver propagation (CAS-based, each node solved at most once)
    void solveNode(Node* node, SolvedStatus status);

    // Random rollout from a position (thread-safe, copies game)
    double rollout(PenteGame game) const;

    // Member variables
    std::mutex expansionMutex_;
    PenteGame game;
    Config config_;
    MCTSArena arena_;
    Node* root_ = nullptr;  // Raw pointer into arena
    mutable std::mt19937 rng_;

    // Statistics
    int totalSimulations_ = 0;
    double totalSearchTime_ = 0.0;
};

#endif // MCTS_HPP
