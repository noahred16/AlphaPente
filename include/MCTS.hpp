#ifndef MCTS_HPP
#define MCTS_HPP

#include "PenteGame.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>

class MCTS {
public:
    // Solved status for minimax backpropagation
    enum class SolvedStatus {
        UNSOLVED,      // Not proven yet
        SOLVED_WIN,    // Proven win for the player who made the move
        SOLVED_LOSS    // Proven loss for the player who made the move
    };

    // Configuration parameters
    struct Config {
        double explorationConstant;                   // UCB1 exploration parameter
        int maxIterations = 10000;                     // Number of MCTS iterations
        int maxSimulationDepth = 200;                  // Max playout depth
        bool useProgressiveBias = false;               // Enable domain knowledge
        int numThreads = 1;                            // Parallel MCTS support

        Config() : explorationConstant(std::sqrt(2.0)) {}
    };

    // Node in the MCTS tree
    struct Node {
        PenteGame::Move move;                          // Move that led to this node
        PenteGame::Player player;                      // Player who made the move

    int visits = 0;                                // Number of times visited
    double wins = 0.0;                             // Total wins (can be fractional)
    double totalValue = 0.0;                       // Sum of simulation results (for avg score)
        SolvedStatus solvedStatus = SolvedStatus::UNSOLVED; // Minimax proof status

        Node* parent = nullptr;                        // Parent node
        std::vector<std::unique_ptr<Node>> children;   // Child nodes
        std::vector<PenteGame::Move> untriedMoves;     // Legal moves not yet expanded

        bool isFullyExpanded() const;
        bool isTerminal() const;
        double getUCB1Value(double explorationConstant, int parentVisits) const;
    };

    // Constructor
    explicit MCTS(const Config& config = Config());
    ~MCTS();

    // Main search interface
    PenteGame::Move search(const PenteGame& game);
    PenteGame::Move searchWithTimeLimit(const PenteGame& game, double seconds);
    
    // Get best move from current tree (no additional search)
    PenteGame::Move getBestMove() const;
    
    // Tree management
    void reset();
    void clearTree();
    
    // Statistics and debugging
    int getTotalVisits() const;
    int getTreeSize() const;
    void printStats() const;
    void printBestMoves(int topN = 5) const;
    void printBranch(int x, int y, int topN = 5) const;
    
    // Configuration
    void setConfig(const Config& config);
    const Config& getConfig() const;

private:
    // MCTS phases
    Node* select(Node* node, PenteGame& game);
    Node* expand(Node* node, PenteGame& game);
    double simulate(const PenteGame& game);
    void backpropagate(Node* node, double result);

    // Minimax backpropagation helpers
    void updateSolvedStatus(Node* node);

    // Helper methods
    Node* selectBestChild(Node* node, bool useExploration) const;
    double evaluateTerminalState(const PenteGame& game, int depth = 0) const;
    PenteGame::Move selectSimulationMove(const PenteGame& game) const;
    
    // Tree reuse (for move ordering/pondering)
    void reuseSubtree(const PenteGame::Move& move);
    
    // Memory management
    void pruneTree(Node* keepNode);

    // Print helpers
    Node* findChildNode(Node* parent, int x, int y) const;
    void printMovesFromNode(Node* node, int topN) const;

    // Member variables
    Config config_;
    std::unique_ptr<Node> root_;
    std::mt19937 rng_;
    
    // Statistics
    int totalSimulations_ = 0;
    double totalSearchTime_ = 0.0;
    
    // Optional: Transposition table for zobrist hashing
    // std::unordered_map<uint64_t, Node*> transpositionTable_;
};

#endif // MCTS_HPP
