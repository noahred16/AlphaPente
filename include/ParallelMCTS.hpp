#ifndef PARALLEL_MCTS_HPP
#define PARALLEL_MCTS_HPP

#include "Arena.hpp"
#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Parallel MCTS with Virtual Loss and Worker Threads
// ============================================================================

class ParallelMCTS {
  public:
    // Reuse enums from single-threaded MCTS
    enum class SolvedStatus : uint8_t {
        UNSOLVED = 0,
        SOLVED_WIN,
        SOLVED_LOSS
    };
    enum class SearchMode { PUCT };
    enum class HeuristicMode { UNIFORM, HEURISTIC, NEURAL_NET };

    // Configuration parameters
    struct Config {
        double explorationConstant;
        int maxIterations = 10000;
        int maxSimulationDepth = 200;
        static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 6ull; // 6 GB
        size_t arenaSize = DEFAULT_ARENA_SIZE;

        SearchMode searchMode = SearchMode::PUCT;
        Evaluator *evaluator = nullptr;
        HeuristicMode heuristicMode = HeuristicMode::HEURISTIC;
        uint32_t seed = 0;
        int canonicalHashDepth = 10;

        // Parallel-specific config
        int numWorkerThreads = 4;        // Number of tree traversal threads
        int numEvalThreads = 1;          // Number of evaluation threads
        int evaluationBatchSize = 32;    // Batch size for neural network evaluation
        int queueCapacity = 10000;       // Max size of evaluation queue

        Config() : explorationConstant(std::sqrt(2.0)) {}
    };

    // Thread-safe node for parallel tree
    struct ThreadSafeNode {
        // Move and player info (immutable after creation)
        PenteGame::Move move;
        PenteGame::Player player;

        // Tree structure (immutable after creation)
        SolvedStatus solvedStatus = SolvedStatus::UNSOLVED;
        uint64_t positionHash = 0;

        // Move metadata (immutable after creation)
        uint16_t childCount = 0;
        uint16_t childCapacity = 0;
        uint16_t unprovenCount = 0;

        // Mutable statistics (protected by nodeLock)
        std::atomic<int32_t> visits{0};
        std::atomic<int32_t> wins{0};
        std::atomic<double> totalValue{0.0};

        // Move arrays (immutable after creation)
        PenteGame::Move *moves = nullptr;
        float *priors = nullptr;
        float value = 0.0f;
        int nextPriorIdx = 0;
        int8_t canonicalSym = -1;

        // Child pointers (immutable after creation)
        ThreadSafeNode **children = nullptr;
        std::atomic<bool> expanded{false};
        std::atomic<bool> evaluated{false};

        // Virtual loss tracking
        std::atomic<int32_t> virtualLosses{0};

        // Synchronization for thread-safe operations
        mutable std::mutex nodeSubtreeLock;  // Protects expansion and child creation

        bool isFullyExpanded() const { return expanded.load(); }
        bool isTerminal() const { return solvedStatus != SolvedStatus::UNSOLVED; }
    };

    // Evaluation request - node to be evaluated by the NN
    struct EvaluationRequest {
        ThreadSafeNode *node;
        PenteGame gameState;  // Game state at this node
        std::vector<ThreadSafeNode *> searchPath;  // Path from root to this node
    };

    // Evaluation result - after NN evaluation
    struct EvaluationResult {
        ThreadSafeNode *node;
        float value;
        std::vector<std::pair<PenteGame::Move, float>> policy;
        std::vector<ThreadSafeNode *> searchPath;
    };

    // Virtual loss manager - tracks virtual losses on nodes during search
    class VirtualLossManager {
      public:
        // Add virtual loss when a node is selected in parallel
        void addVirtualLoss(ThreadSafeNode *node, int32_t amount = 1);

        // Remove virtual loss after backpropagation
        void removeVirtualLoss(ThreadSafeNode *node, int32_t amount = 1);

        // Get effective visit count (accounting for virtual losses)
        int32_t getEffectiveVisits(ThreadSafeNode *node) const;

      private:
        mutable std::mutex statsLock;
    };

    // Thread-safe queue for evaluation requests
    class EvaluationQueue {
      public:
        explicit EvaluationQueue(size_t capacity);

        // Try to push evaluation request (non-blocking)
        bool tryPush(const EvaluationRequest &request);

        // Pop a batch of evaluation requests for NN evaluation
        std::vector<EvaluationRequest> popBatch(size_t batchSize);

        // Check if queue has items
        bool empty() const;

        size_t size() const;

      private:
        mutable std::mutex queueLock;
        std::deque<EvaluationRequest> queue;
        size_t capacity;
    };

    // Thread-safe queue for backpropagation results
    class BackpropagationQueue {
      public:
        // Push evaluation result for backpropagation
        void push(const EvaluationResult &result);

        // Pop all pending backpropagation results
        std::vector<EvaluationResult> popAll();

        bool empty() const;

      private:
        mutable std::mutex queueLock;
        std::deque<EvaluationResult> queue;
    };

    // Worker thread pool
    class WorkerPool {
      public:
        WorkerPool(int numWorkerThreads, ParallelMCTS *parent);
        ~WorkerPool();

        void start();
        void stop();
        bool isRunning() const;

      private:
        void workerThreadMain(int workerId);

        int numWorkerThreads;
        ParallelMCTS *parent;
        std::vector<std::thread> threads;
        std::atomic<bool> running{false};
        mutable std::mutex poolLock;
    };

    // Evaluation thread pool
    class EvalPool {
      public:
        EvalPool(int numEvalThreads, ParallelMCTS *parent);
        ~EvalPool();

        void start();
        void stop();
        bool isRunning() const;

      private:
        void evalThreadMain(int evalId);

        int numEvalThreads;
        ParallelMCTS *parent;
        std::vector<std::thread> threads;
        std::atomic<bool> running{false};
        mutable std::mutex poolLock;
    };

    // ========================================================================
    // Main ParallelMCTS Interface
    // ========================================================================

    explicit ParallelMCTS(const Config &config = Config());
    ~ParallelMCTS();

    // Main search interface
    PenteGame::Move search(const PenteGame &game);

    // Get best move from current tree
    PenteGame::Move getBestMove() const;

    // Tree management
    void reset();
    void clearTree();

    // Statistics and debugging
    int getTotalVisits() const;
    int getTreeSize() const;
    void printStats(double wallTime) const;

    // Configuration access
    void setConfig(const Config &config);
    const Config &getConfig() const;

    // Tree initialization
    void prepareRoot(const PenteGame &game);
    const ThreadSafeNode *getRoot() const;

    // Eval pool lifecycle (exposed for testing)
    void startEvalThreads();
    void stopEvalThreads();

    // Queue helpers (exposed for testing)
    void pushEvalRequest(const EvaluationRequest &request);
    std::vector<EvaluationResult> drainBackpropQueue();

  private:
    // MCTS phases
    ThreadSafeNode *select(ThreadSafeNode *node, PenteGame &game, std::vector<ThreadSafeNode *> &searchPath);
    ThreadSafeNode *expand(ThreadSafeNode *node, PenteGame &game);
    void backpropagate(ThreadSafeNode *node, float result, const std::vector<ThreadSafeNode *> &searchPath);

    // Helper methods
    int selectBestMoveIndex(ThreadSafeNode *node, const PenteGame &game) const;
    void updateChildrenPriors(ThreadSafeNode *node, const PenteGame &game);

    // Arena allocation
    ThreadSafeNode *allocateNode();
    void initNodeChildren(ThreadSafeNode *node, int capacity);

    // Member variables
    Config config_;
    std::unique_ptr<Arena> arena_;
    std::unique_ptr<VirtualLossManager> virtualLossManager_;
    std::unique_ptr<EvaluationQueue> evaluationQueue_;
    std::unique_ptr<BackpropagationQueue> backpropagationQueue_;
    std::unique_ptr<WorkerPool> workerPool_;
    std::unique_ptr<EvalPool> evalPool_;

    ThreadSafeNode *root_ = nullptr;
    std::unordered_map<uint64_t, ThreadSafeNode *> nodeTranspositionTable;
    mutable std::mutex treeLock;  // Protects root and transposition table
    mutable std::mt19937 rng_;

    // Search state
    PenteGame initialGame_;  // Initial game state for workers

    // Statistics
    std::atomic<int> totalIterations{0};
    std::atomic<int> nodeCount{0};
};

#endif // PARALLEL_MCTS_HPP