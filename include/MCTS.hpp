#ifndef MCTS_HPP
#define MCTS_HPP

#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <unordered_set>

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

    explicit MCTSArena(size_t size = DEFAULT_SIZE) : size_(size), offset_(0), memory_(nullptr) {
        memory_ = static_cast<char *>(std::aligned_alloc(64, size_)); // 64-byte alignment for cache lines
        if (!memory_) {
            throw std::bad_alloc();
        }
    }

    ~MCTSArena() { std::free(memory_); }

    // Non-copyable, non-movable
    MCTSArena(const MCTSArena &) = delete;
    MCTSArena &operator=(const MCTSArena &) = delete;

    // Allocate memory for type T with proper alignment
    template <typename T> T *allocate(size_t count = 1) {
        // Align to T's alignment requirement
        size_t alignment = alignof(T);
        size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        size_t totalBytes = sizeof(T) * count;

        if (alignedOffset + totalBytes > size_) {
            // Out of arena memory
            return nullptr;
        }

        T *ptr = reinterpret_cast<T *>(memory_ + alignedOffset);
        offset_ = alignedOffset + totalBytes;
        return ptr;
    }

    // O(1) tree destruction - just reset the offset
    void reset() { offset_ = 0; }

    // Swap internals with another arena (for subtree reuse)
    void swap(MCTSArena &other) {
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
    char *memory_;
};

// ============================================================================
// MCTS Class with Arena-Allocated Nodes
// ============================================================================

class MCTS {
  public:
    // Solved status for minimax backpropagation (1 byte)
    enum class SolvedStatus : uint8_t {
        UNSOLVED = 0, // Not proven yet
        SOLVED_WIN,   // Proven win for the player who made the move
        SOLVED_LOSS   // Proven loss for the player who made the move
    };
    enum class SearchMode { UCB1, PUCT };
    enum class HeuristicMode { UNIFORM, HEURISTIC, NEURAL_NET };

    // Configuration parameters
    struct Config {
        double explorationConstant;                 // UCB1 exploration parameter
        int maxIterations = 10000;                  // Number of MCTS iterations
        int maxSimulationDepth = 200;               // Max playout depth
        size_t arenaSize = MCTSArena::DEFAULT_SIZE; // Arena size in bytes

        SearchMode searchMode = SearchMode::UCB1;
        Evaluator *evaluator = nullptr; // For PUCT priors and value evaluation
        HeuristicMode heuristicMode = HeuristicMode::HEURISTIC;

        Config() : explorationConstant(std::sqrt(2.0)) {}
    };

    // Node in the MCTS tree - trivially destructible, ~64 bytes
    // All dynamic arrays are arena-allocated via raw pointers
    struct Node {
        // Move that led to this node (4 bytes)
        PenteGame::Move move;

        // Player who made the move (1 byte, uint8_t-backed enum)
        PenteGame::Player player;

        // Minimax proof status (1 byte)
        SolvedStatus solvedStatus = SolvedStatus::UNSOLVED;

        // Child array metadata (2 bytes each = 4 bytes)
        uint16_t childCount = 0;
        uint16_t childCapacity = 0;

        // Untried moves metadata (2 bytes each = 4 bytes)
        uint16_t unprovenCount = 0;

        // Statistics (16 bytes)
        int32_t visits = 0;
        int32_t wins = 0;
        double totalValue = 0.0;

        // moves and priors share the same size, and are accessed together
        PenteGame::Move *moves;
        float *priors;
        int moveCount = 0;

        float prior = -1.0f;
        float value = 0.0f;

        // Pointers (24 bytes)
        Node *parent = nullptr;
        Node **children = nullptr; // Arena-allocated array of child pointers
        bool expanded = false;
        bool evaluated = false;

        // Total: 4 + 1 + 1 + 4 + 4 + 16 + 24 = 54 bytes + padding = 56-64 bytes

        bool isFullyExpanded() const { return expanded; }                          // HMM
        bool isTerminal() const { return solvedStatus != SolvedStatus::UNSOLVED; } // HMM
        double getUCB1Value(double explorationFactor) const;
        double getPUCTValue(double explorationFactor, int parentVisits) const;
    };

    // Constructor
    explicit MCTS(const Config &config = Config());
    ~MCTS();

    // Main search interface
    PenteGame::Move search(const PenteGame &game);

    // Get best move from current tree (no additional search)
    PenteGame::Move getBestMove() const;

    // Tree management
    void reset();
    void clearTree();
    void reuseSubtree(const PenteGame::Move &move);
    bool undoSubtree();

    // Statistics and debugging
    int getTotalVisits() const;
    int getTreeSize() const;
    void printStats() const;
    void printBestMoves(int topN = 5) const;
    void printBranch(const char *moveStr, int topN = 5) const;
    void printBranch(int x, int y, int topN = 5) const;

    // Arena statistics
    size_t getArenaUsedBytes() const { return arena_.bytesUsed(); }
    double getArenaUtilization() const { return arena_.utilizationPercent(); }

    // Configuration
    void setConfig(const Config &config);
    const Config &getConfig() const;

  private:
    // MCTS phases
    Node *select(Node *node, PenteGame &game);
    Node *expand(Node *node, PenteGame &game);
    double simulate(Node *node, PenteGame &game);
    void backpropagate(Node *node, double result);

    // Helper methods
    int selectBestMoveIndex(Node *node, const PenteGame &game) const;
    void updateChildrenPriors(Node *node, const PenteGame &game);
    double evaluateTerminalState(const PenteGame &game, int depth = 0) const;

    // Arena allocation helpers
    Node *allocateNode();
    void initNodeChildren(Node *node, int capacity);

    // Tree reuse helpers (copies subtree to fresh arena)
    Node *copySubtree(Node *source, MCTSArena &destArena);
    void pruneTree(Node *keepNode);

    // Print helpers
    Node *findChildNode(Node *parent, int x, int y) const;
    void printMovesFromNode(Node *node, int topN) const;
    int countNodes(Node *node, std::unordered_set<Node *> &visited) const;

    // Member variables
    PenteGame game;
    Config config_;
    MCTSArena arena_;
    std::unordered_map<uint64_t, Node *> nodeTranspositionTable;
    Node *root_ = nullptr; // Raw pointer into arena
    mutable std::mt19937 rng_;

    // Statistics
    int totalSimulations_ = 0;
    double totalSearchTime_ = 0.0;
};

#endif // MCTS_HPP
