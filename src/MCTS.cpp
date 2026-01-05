#include "MCTS.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <iostream>
#include <iomanip>

// ============================================================================
// Node Implementation
// ============================================================================

bool MCTS::Node::isFullyExpanded() const {
    return untriedMoves.empty();
}

bool MCTS::Node::isTerminal() const {
    return children.empty() && untriedMoves.empty();
}

double MCTS::Node::getUCB1Value(double explorationConstant, int parentVisits) const {
    // Prioritize solved nodes
    if (solvedStatus == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (solvedStatus == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    if (visits == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double exploitation = totalValue / visits;
    double exploration = explorationConstant * std::sqrt(std::log(parentVisits) / visits);

    return exploitation + exploration;
}

// ============================================================================
// MCTS Constructor/Destructor
// ============================================================================

MCTS::MCTS(const Config& config) 
    : config_(config)
    , root_(nullptr)
    , rng_(std::random_device{}())
    , totalSimulations_(0)
    , totalSearchTime_(0.0) {
}

MCTS::~MCTS() {
    clearTree();
}

// ============================================================================
// Main Search Interface
// ============================================================================

PenteGame::Move MCTS::search(const PenteGame& game) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize root node
    reset();
    root_ = std::make_unique<Node>();
    root_->player = game.getCurrentPlayer();
    root_->untriedMoves = game.getLegalMoves();
    
    // Run MCTS iterations
    for (int i = 0; i < config_.maxIterations; i++) {
        // Clone game state for this iteration
        PenteGame gameClone = game.clone();
        
        // Selection: traverse tree to find node to expand
        Node* node = select(root_.get(), gameClone);
        
        // Expansion: add a new child node if not terminal
        if (!gameClone.isGameOver() && !node->untriedMoves.empty()) {
            node = expand(node, gameClone);
        }
        
        // Simulation: play out the game randomly
        double result = simulate(gameClone);
        
        // Backpropagation: update statistics
        backpropagate(node, result);
        
        totalSimulations_++;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    totalSearchTime_ = std::chrono::duration<double>(endTime - startTime).count();
    
    return getBestMove();
}

PenteGame::Move MCTS::searchWithTimeLimit(const PenteGame& game, double seconds) {
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::duration<double>(seconds);
    
    // Initialize root node
    reset();
    root_ = std::make_unique<Node>();
    root_->player = game.getCurrentPlayer();
    root_->untriedMoves = game.getLegalMoves();
    
    int iterations = 0;
    
    // Run MCTS iterations until time limit
    while (std::chrono::high_resolution_clock::now() < endTime) {
        // Clone game state for this iteration
        PenteGame gameClone = game.clone();
        
        // Selection
        Node* node = select(root_.get(), gameClone);
        
        // Expansion
        if (!gameClone.isGameOver() && !node->untriedMoves.empty()) {
            node = expand(node, gameClone);
        }
        
        // Simulation
        double result = simulate(gameClone);
        
        // Backpropagation
        backpropagate(node, result);
        
        totalSimulations_++;
        iterations++;
    }
    
    totalSearchTime_ = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    
    return getBestMove();
}

PenteGame::Move MCTS::getBestMove() const {
    if (!root_ || root_->children.empty()) {
        return PenteGame::Move(); // Invalid move
    }

    // First, check if any child is a proven win - always choose that
    for (const auto& child : root_->children) {
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            return child->move;
        }
    }

    // Select child with most visits (most robust), excluding proven losses
    Node* bestChild = nullptr;
    int maxVisits = -1;

    for (const auto& child : root_->children) {
        if (child->solvedStatus != SolvedStatus::SOLVED_LOSS && child->visits > maxVisits) {
            maxVisits = child->visits;
            bestChild = child.get();
        }
    }

    return bestChild ? bestChild->move : PenteGame::Move();
}

// ============================================================================
// MCTS Phases
// ============================================================================

MCTS::Node* MCTS::select(Node* node, PenteGame& game) {
    // Traverse tree using UCB1 until we find a node that's not fully expanded
    while (node->isFullyExpanded() && !node->children.empty()) {
        node = selectBestChild(node, true);
        
        // Apply move to game state
        if (node && node->move.isValid()) {
            game.makeMove(node->move.x, node->move.y);
        }
    }
    
    return node;
}

MCTS::Node* MCTS::expand(Node* node, PenteGame& game) {
    if (node->untriedMoves.empty()) {
        return node;
    }

    // Pick a random untried move
    std::uniform_int_distribution<size_t> dist(0, node->untriedMoves.size() - 1);
    size_t moveIndex = dist(rng_);
    PenteGame::Move move = node->untriedMoves[moveIndex];

    // Remove from untried moves
    node->untriedMoves.erase(node->untriedMoves.begin() + moveIndex);

    // Apply move to game
    game.makeMove(move.x, move.y);

    // Create new child node
    auto child = std::make_unique<Node>();
    child->move = move;
    child->player = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
    child->parent = node;

    // Get legal moves for this new state
    if (!game.isGameOver()) {
        child->untriedMoves = game.getLegalMoves();
    } else {
        // Terminal node - determine solved status
        // Note: node->player is the player who made the move (before this child)
        // child->player is the opponent (whose turn it would be, but game is over)
        PenteGame::Player winner = game.getWinner();
        if (winner == node->player) {
            // This move led to a win for the player who made it
            child->solvedStatus = SolvedStatus::SOLVED_WIN;
        } else if (winner != PenteGame::NONE) {
            // This move led to a loss for the player who made it
            child->solvedStatus = SolvedStatus::SOLVED_LOSS;
        }
        // If winner == NONE (draw), leave as UNSOLVED
    }

    Node* childPtr = child.get();
    node->children.push_back(std::move(child));

    return childPtr;
}

double MCTS::simulate(const PenteGame& game) {
    PenteGame simGame = game.clone();
    int depth = 0;
    
    // Play out game randomly until terminal or max depth
    while (!simGame.isGameOver() && depth < config_.maxSimulationDepth) {
        PenteGame::Move move = selectSimulationMove(simGame);
        
        if (!move.isValid()) {
            break;
        }
        
        simGame.makeMove(move.x, move.y);
        depth++;
    }
    
    // Evaluate with depth consideration
    return evaluateTerminalState(simGame, depth);
}

void MCTS::updateSolvedStatus(Node* node) {
    if (!node || !node->isFullyExpanded()) {
        return; // Can't determine solved status until fully expanded
    }

    if (node->children.empty()) {
        return; // Terminal node, status set during expansion
    }

    // Check for proven wins/losses based on children
    bool hasWinningChild = false;
    bool hasLosingChild = false;
    bool allChildrenLosses = true;
    bool allChildrenWins = true;

    for (const auto& child : node->children) {
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            hasWinningChild = true;
            allChildrenLosses = false;
        } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            hasLosingChild = true;
            allChildrenWins = false;
        } else {
            // Child is unsolved
            allChildrenLosses = false;
            allChildrenWins = false;
        }
    }

    // Apply minimax logic:
    // If ANY child is a SOLVED_WIN, this node is SOLVED_WIN (we can choose to go there)
    if (hasWinningChild) {
        node->solvedStatus = SolvedStatus::SOLVED_WIN;
    }
    // If ALL children are SOLVED_LOSS, this node is SOLVED_LOSS (all moves lose)
    else if (allChildrenLosses && !node->children.empty()) {
        node->solvedStatus = SolvedStatus::SOLVED_LOSS;
    }
}

void MCTS::backpropagate(Node* node, double result) {
    // Standard update: propagate result up the tree
    Node* current = node;
    double currentResult = result;

    while (current != nullptr) {
        current->visits++;
        current->wins += currentResult;
        current->totalValue += currentResult;

        // Update solved status based on children
        updateSolvedStatus(current);

        // Flip result for parent (opponent's perspective)
        currentResult = 1.0 - currentResult;
        current = current->parent;
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

MCTS::Node* MCTS::selectBestChild(Node* node, bool useExploration) const {
    if (!node || node->children.empty()) {
        return nullptr;
    }
    
    Node* bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();
    
    for (const auto& child : node->children) {
        double value;
        
        if (useExploration) {
            value = child->getUCB1Value(config_.explorationConstant, node->visits);
        } else {
            // Pure exploitation (for final move selection)
            value = child->visits > 0 ? (child->wins / child->visits) : 0.0;
        }
        
        if (value > bestValue) {
            bestValue = value;
            bestChild = child.get();
        }
    }
    
    return bestChild;
}

// TODO make better depth weight. scale between 0.1 and 1
double MCTS::evaluateTerminalState(const PenteGame& game, int depth) const {
    PenteGame::Player winner = game.getWinner();
    PenteGame::Player rootPlayer = root_ ? root_->player : PenteGame::NONE;
    
    if (winner == rootPlayer) {
        // Scale exponential to range [0.5, 1.0]
        // Win is always better than draw, but faster is much better
        double rawScore = std::exp(-depth / 20.0);
        return 0.5 + (rawScore * 0.5);  // Maps [0,1] -> [0.5,1.0]
    } else if (winner == PenteGame::NONE) {
        return 0.5;
    } else {
        // Loss: scale to [0.0, 0.5]
        double rawScore = std::exp(-depth / 20.0);
        return 0.5 - (rawScore * 0.5);  // Maps [0,1] -> [0.5,0.0]
    }
}


PenteGame::Move MCTS::selectSimulationMove(const PenteGame& game) const {
    // Simple random playout policy
    // Could be enhanced with domain knowledge
    return game.getRandomMove();
}

void MCTS::reuseSubtree(const PenteGame::Move& move) {
    if (!root_) {
        return;
    }
    
    // Find child matching the move
    for (auto& child : root_->children) {
        if (child->move.x == move.x && child->move.y == move.y) {
            // Detach this child and make it the new root
            child->parent = nullptr;
            root_ = std::move(child);
            return;
        }
    }
    
    // Move not in tree, clear everything
    clearTree();
}

void MCTS::pruneTree(Node* keepNode) {
    // Not implemented - would recursively delete all nodes except keepNode's subtree
    // For now, just clear everything
    clearTree();
}

// ============================================================================
// Tree Management
// ============================================================================

void MCTS::reset() {
    totalSimulations_ = 0;
    totalSearchTime_ = 0.0;
}

void MCTS::clearTree() {
    root_.reset();
}

// ============================================================================
// Statistics and Debugging
// ============================================================================

int MCTS::getTotalVisits() const {
    return root_ ? root_->visits : 0;
}

int MCTS::getTreeSize() const {
    if (!root_) {
        return 0;
    }
    
    // BFS to count nodes
    std::vector<Node*> queue;
    queue.push_back(root_.get());
    int count = 0;
    
    while (!queue.empty()) {
        Node* node = queue.back();
        queue.pop_back();
        count++;
        
        for (const auto& child : node->children) {
            queue.push_back(child.get());
        }
    }
    
    return count;
}

void MCTS::printStats() const {
    std::cout << "\n=== MCTS Statistics ===\n";
    std::cout << "Total simulations: " << totalSimulations_ << "\n";
    std::cout << "Total search time: " << std::fixed << std::setprecision(3) 
              << totalSearchTime_ << " seconds\n";
    std::cout << "Simulations/second: " << std::fixed << std::setprecision(0)
              << (totalSearchTime_ > 0 ? totalSimulations_ / totalSearchTime_ : 0) << "\n";
    std::cout << "Tree size: " << getTreeSize() << " nodes\n";
    std::cout << "Root visits: " << getTotalVisits() << "\n";
    
    if (root_) {
        std::cout << "Root win rate: " << std::fixed << std::setprecision(3)
                  << (root_->visits > 0 ? root_->wins / root_->visits : 0.0) << "\n";
    }
    
    std::cout << "=======================\n\n";
}

void MCTS::printBestMoves(int topN) const {
    if (!root_ || root_->children.empty()) {
        std::cout << "No moves analyzed yet.\n";
        return;
    }
    
    // Collect and sort children by visits
    std::vector<Node*> children;
    for (const auto& child : root_->children) {
        children.push_back(child.get());
    }
    
    std::sort(children.begin(), children.end(), 
        [](Node* a, Node* b) { return a->visits > b->visits; });
    
    int movesConsidered = root_->children.size();
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves of "
              << movesConsidered << " Considered ===\n";
    std::cout << std::setw(8) << "Move"
              << std::setw(10) << "Visits"
              << std::setw(12) << "Win Rate"
              << std::setw(14) << "Avg Score"
              << std::setw(12) << "UCB1"
              << std::setw(12) << "Status\n";
    std::cout << std::string(66, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double winRate = child->visits > 0 ? child->wins / child->visits : 0.0;
        double avgScore = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        double ucb1 = child->getUCB1Value(config_.explorationConstant, root_->visits);

        std::string moveStr = PenteGame::displayMove(child->move.x, child->move.y);

        std::string status;
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(8) << moveStr
                  << std::setw(10) << child->visits
                  << std::setw(12) << std::fixed << std::setprecision(3) << winRate
                  << std::setw(14) << std::fixed << std::setprecision(3) << avgScore
                  << std::setw(12) << std::fixed << std::setprecision(3) << ucb1
                  << std::setw(12) << status
                  << "\n";
    }

    std::cout << "===================\n\n";
}


MCTS::Node* MCTS::findChildNode(MCTS::Node* parent, int x, int y) const {
    if (!parent) {
        return nullptr;
    }
    
    for (const auto& child : parent->children) {
        if (child->move.x == x && child->move.y == y) {
            return child.get();
        }
    }
    
    return nullptr;
}

void MCTS::printMovesFromNode(MCTS::Node* node, int topN) const {
    if (!node || node->children.empty()) {
        std::cout << "No moves analyzed for this position.\n";
        return;
    }
    
    // Collect and sort children by visits
    std::vector<Node*> children;  // This is fine since we're inside MCTS scope
    for (const auto& child : node->children) {
        children.push_back(child.get());
    }
    
    std::sort(children.begin(), children.end(), 
        [](Node* a, Node* b) { return a->visits > b->visits; });
    
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves ===\n";
    std::cout << std::setw(8) << "Move"
              << std::setw(10) << "Visits"
              << std::setw(12) << "Win Rate"
              << std::setw(12) << "UCB1"
              << std::setw(12) << "Status\n";
    std::cout << std::string(54, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double winRate = child->visits > 0 ? child->wins / child->visits : 0.0;
        double ucb1 = child->getUCB1Value(config_.explorationConstant, node->visits);

        std::string moveStr = PenteGame::displayMove(child->move.x, child->move.y);

        std::string status;
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(8) << moveStr
                  << std::setw(10) << child->visits
                  << std::setw(12) << std::fixed << std::setprecision(3) << winRate
                  << std::setw(12) << std::fixed << std::setprecision(3) << ucb1
                  << std::setw(12) << status
                  << "\n";
    }

    std::cout << "===================\n\n";
}

void MCTS::printBranch(const char* moveStr, int topN) const {
    // use utility to convert moveStr to x,y
    // parseMove is a static method, so we can use it directly
    auto [x, y] = PenteGame::parseMove(moveStr);

    printBranch(x, y, topN);

}

void MCTS::printBranch(int x, int y, int topN) const {
    if (!root_) {
        std::cout << "No search tree exists yet.\n";
        return;
    }
    
    // Find the child node corresponding to move (x, y)
    Node* targetNode = findChildNode(root_.get(), x, y);  // Node* is fine here
    
    if (!targetNode) {
        std::string moveStr = PenteGame::displayMove(x, y);
        std::cout << "Move " << moveStr << " not found in search tree.\n";
        std::cout << "This move may not have been explored yet.\n";
        return;
    }
    
    // Print info about the move itself
    std::string moveStr = PenteGame::displayMove(x, y);
    double winRate = targetNode->visits > 0 ? targetNode->wins / targetNode->visits : 0.0;
    
    std::cout << "\n=== Analysis for move " << moveStr << " ===\n";
    std::cout << "Visits: " << targetNode->visits << "\n";
    std::cout << "Win Rate: " << std::fixed << std::setprecision(3) << winRate << "\n";
    std::cout << "Player: " << (targetNode->player == PenteGame::BLACK ? "Black" : "White") << "\n";
    
    // Print the best responses/continuations from this position
    std::cout << "\nBest responses:\n";
    printMovesFromNode(targetNode, topN);
}

// ============================================================================
// Configuration
// ============================================================================

void MCTS::setConfig(const Config& config) {
    config_ = config;
}

const MCTS::Config& MCTS::getConfig() const {
    return config_;
}
