#ifndef MCTS_HPP
#define MCTS_HPP

#include "Arena.hpp"
#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <unordered_set>

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
    enum class SearchMode { PUCT };
    enum class HeuristicMode { UNIFORM, HEURISTIC, NEURAL_NET };

    // Configuration parameters
    struct Config {
        double explorationConstant;                 // PUCT exploration parameter
        int maxIterations = 10000;                  // Number of MCTS iterations
        int maxSimulationDepth = 200;               // Max playout depth
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 512; // 512 MB
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024; // 1 GB
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 2ull; // 2 GB
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 4ull; // 4 GB
        static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 6ull; // 6 GB
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 8ull; // 8 GB
        // static constexpr size_t DEFAULT_ARENA_SIZE = 1024 * 1024 * 1024 * 10ull; // 10 GB
        size_t arenaSize = DEFAULT_ARENA_SIZE;

        SearchMode searchMode = SearchMode::PUCT;
        Evaluator *evaluator = nullptr; // For PUCT priors and value evaluation
        HeuristicMode heuristicMode = HeuristicMode::HEURISTIC;
        uint32_t seed = 0; // 0 = non-deterministic (random_device), non-zero = deterministic seed
        int canonicalHashDepth = 10; // 0 = disabled; use canonical hash when getMoveCount() <= this value

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

        // hash
        uint64_t positionHash = 0;

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
        float value = 0.0f;
        int nextPriorIdx = 0;      // Index of next prior to explore for selection (used in PUCT)
        int8_t canonicalSym = -1;  // -1 = moves[] in physical coords; 0-7 = moves[] in canonical coords

        // Pointers (24 bytes)
        Node **children = nullptr; // Arena-allocated array of child pointers
        bool expanded = false;
        bool evaluated = false;

        // Total: 4 + 1 + 1 + 4 + 4 + 16 + 24 = 54 bytes + padding = 56-64 bytes

        bool isFullyExpanded() const { return expanded; }                          // HMM
        bool isTerminal() const { return solvedStatus != SolvedStatus::UNSOLVED; } // HMM
        double getPUCTValue(double explorationFactor, double sqrtParentVisits, float prior) const;
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
    void printStats(double wallTime, double cpuTime) const;
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
    Node *select(Node *node, PenteGame &game, std::vector<Node *> &searchPath);
    Node *expand(Node *node, PenteGame &game);
    double simulate(Node *node, PenteGame &game);
    void backpropagate(Node *node, double result, std::vector<Node *> &searchPath);

    // Helper methods
    int selectBestMoveIndex(Node *node, const PenteGame &game, int currentSym) const;
    void updateChildrenPriors(Node *node, const PenteGame &game);

    // Arena allocation helpers
    Node *allocateNode();
    void initNodeChildren(Node *node, int capacity);

    // Print helpers
    Node *findChildNode(Node *parent, int x, int y) const;
    void printMovesFromNode(Node *node, int topN) const;
    int countNodes(Node *node, std::unordered_set<Node *> &visited) const;

    // Member variables
    PenteGame game;
    Config config_;
    Arena arena_;
    std::unordered_map<uint64_t, Node *> nodeTranspositionTable;
    Node *root_ = nullptr;         // Raw pointer into arena
    std::vector<Node *> reusePath; // For subtree reuse during search
    mutable std::mt19937 rng_;


    // Statistics
    int startSimulations_ = 0;
    int totalSimulations_ = 0;
    double totalSearchTime_ = 0.0;
};


#endif // MCTS_HPP
