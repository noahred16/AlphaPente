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

double MCTS::Node::getUCB1Value(double explorationFactor) const {
    // Prioritize solved nodes
    if (this->solvedStatus == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (this->solvedStatus == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    if (this->visits == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double exploitation = this->totalValue / this->visits;
    double exploration = explorationFactor / std::sqrt(static_cast<double>(this->visits));

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
    root_->unprovenCount = root_->untriedMoves.size();

    // Only local copy of game for simulations
    PenteGame localGame;
    
    // Run MCTS iterations
    for (int i = 0; i < config_.maxIterations; i++) {
        
        if (root_->solvedStatus != SolvedStatus::UNSOLVED) {
            std::cout << "Root node solved status: " 
                      << (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "WIN" : "LOSS") 
                      << " after " << i << " iterations.\n";
            break; // exit early if root is solved
        }

        // Reset localGame to the start state of THIS search
        localGame.syncFrom(game);
        
        // Selection: traverse tree to find node to expand
        Node* node = select(root_.get(), localGame);
        
        // Expansion: add a new child node if not terminal
        if (!localGame.isGameOver() && !node->untriedMoves.empty()) {
            node = expand(node, localGame);
        }
        
        // Simulation: play out the game randomly, only if not already solved
        double result = 0.0;
        if (node->solvedStatus == SolvedStatus::UNSOLVED) {
            result = simulate(localGame);
        } else {
            // If already solved, assign result
            result = (node->solvedStatus == SolvedStatus::SOLVED_WIN) ? 1.0 : -1.0;
        }
        
        // Backpropagation: update statistics
        backpropagate(node, result);

        totalSimulations_++;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    totalSearchTime_ = std::chrono::duration<double>(endTime - startTime).count();
    
    return getBestMove();
}

PenteGame::Move MCTS::getBestMove() const {
    if (!root_ || root_->children.empty()) {
        std::cout << "Exiting: Unexpected state in getBestMove(). No children found.\n";
        std::cerr << "Error: No moves available to select as best move.\n";
        exit(1);
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

    // Fallback: If all are losses, just pick the one with the most visits
    if (!bestChild && !root_->children.empty()) {
        for (const auto& child : root_->children) {
            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                bestChild = child.get();
            }
        }
    }

    return bestChild->move;
}

// ============================================================================
// MCTS Phases
// ============================================================================

MCTS::Node* MCTS::select(Node* node, PenteGame& game) {
    // Traverse tree using UCB1 until we find a node that's not fully expanded. And not solved
    // We stop if:
    // 1. We find a node that isn't fully expanded (need to add children).
    // 2. We find a node that is already solved (no need to simulate further).
    while (node->isFullyExpanded() && !node->children.empty() && node->solvedStatus == SolvedStatus::UNSOLVED) {
        node = selectBestChild(node);
        
        // Apply move to game state
        if (node) {
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

    // swap is faster than erase from middle
    std::swap(node->untriedMoves[moveIndex], node->untriedMoves.back());
    PenteGame::Move move = node->untriedMoves.back();
    node->untriedMoves.pop_back();

    // aka the opponent of the player who made the move to reach this node
    PenteGame::Player startPlayer = game.getCurrentPlayer();

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
        // Initially, all legal moves are 'unproven' paths. 
        // As they are proven to be LOSSES, this count will decrement.
        child->unprovenCount = child->untriedMoves.size(); // Initially all moves are unproven
    } else {
        // Terminal node - determine solved status - if game is over its either a win or draw
        // Note: node->player is the player who made the move (before this child)
        // child->player is the opponent (whose turn it would be, but game is over)
        PenteGame::Player winner = game.getWinner();
        if (winner == startPlayer) {
            child->solvedStatus = SolvedStatus::SOLVED_WIN;
            child->unprovenCount = 0; // May as well set to 0
        } else {
            // Draw or loss is treated as UNSOLVED for simplicity. Throw an error, I don't expect to ever hit this case for normal pente
            child->solvedStatus = SolvedStatus::UNSOLVED;
            std::cout << "Exiting: Unexpected game over state in expand(). Winner: " << static_cast<int>(winner) << std::endl;
            std::cerr << "Error: Unexpected game over state. Winner: " << static_cast<int>(winner) << std::endl;
            exit(1);
        }
    }

    Node* childPtr = child.get();
    node->children.push_back(std::move(child));

    return childPtr;
}

double MCTS::simulate(PenteGame& simGame) {
    // Track who is to move at start - result must be from their perspective
    PenteGame::Player startPlayer = simGame.getCurrentPlayer();
    int depth = 0;

    // Play out game randomly until terminal or max depth
    while (!simGame.isGameOver() && depth < config_.maxSimulationDepth) {
        PenteGame::Move move = selectSimulationMove(simGame);
        simGame.makeMove(move.x, move.y);
        depth++;
    }

    // Evaluate result from the perspective of the player who MADE THE MOVE to reach this node
    // (i.e., the opponent of startPlayer)
    double result = evaluateTerminalState(simGame, depth);
    PenteGame::Player winner = simGame.getWinner();

    if (winner != PenteGame::NONE) {
        // winner == startPlayer means the mover (opponent) won, which means the person who played lost
        // winner != startPlayer means the mover (opponent) had no good moves and lost
        result *= (winner == startPlayer) ? -1.0 : 1.0;
    } else {
        // Draw or max depth reached - neutral
        result = 0.0;
    }

    // Roll the board back to original state
    for (int i = 0; i < depth; i++) {
        simGame.undoMove();
    }

    return result;
}


/**
 * DESIGN NOTE: unprovenCount tracking
 * This counter is initialized to untriedMoves.size() and decremented whenever 
 * a child is proven to be a SOLVED_LOSS. 
 * * Safety Requirement: To maintain a correct count, each unique move/child 
 * should only trigger a decrement ONCE. Currently, this relies on the 
 * Selection/Expansion policy: once a branch is solved, it should not be 
 * re-selected in a way that triggers a duplicate decrement. 
 * * If future optimizations (like parallel search or RAVE) cause re-visits 
 * to solved nodes, a "wasAccountedFor" flag must be added to each node 
 * to ensure the parent's unprovenCount remains accurate.
 */
void MCTS::backpropagate(Node* node, double result) {
    // Standard update: propagate result up the tree
    Node* current = node;
    double currentResult = result;

    while (current != nullptr) {
        // Backpropagate stats
        current->visits++;
        current->totalValue += currentResult;
        current->wins += (currentResult > 0) ? 1 : 0;
        
        // --- SOLVER LOGIC (Minimax Propagation) ---
        // We use a "Push" model: current node updates its parent's status.

        // If we have a winning child, mark parent as LOSS. if the opp has a win, its an L
        if (current->solvedStatus == SolvedStatus::SOLVED_WIN && current->parent) {
            current->parent->solvedStatus = SolvedStatus::SOLVED_LOSS;
        }

        // Proven win if all children moves (our opp) are proven losses (no unproven moves left)
        if (current->solvedStatus == SolvedStatus::SOLVED_LOSS && current->parent) {
            current->parent->unprovenCount--;

            // For sanity, heres a check to ensure we never drop below 0
            if (current->parent->unprovenCount < 0) {
                std::cerr << "FATAL ERROR: unprovenCount dropped below 0!" << std::endl;
                std::cerr << "Node: " << current << " Parent: " << current->parent << std::endl;
                // throw std::runtime_error("MCTS Solver logic error: Multiple decrements for same node.");
                exit(1);
            }

            if (current->parent->unprovenCount == 0) {
                current->parent->solvedStatus = SolvedStatus::SOLVED_WIN;
            }
        }

        // Flip result for parent (opponent's perspective)
        currentResult = currentResult * -1.0;
        current = current->parent;
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

MCTS::Node* MCTS::selectBestChild(Node* node) const {
    if (!node || node->children.empty()) {
        return nullptr;
    }
    
    Node* bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();

    // Pre-calculate the numerator of the exploration term
    // Formula: C * sqrt(ln(N))
    double explorationFactor = config_.explorationConstant * std::sqrt(std::log(static_cast<double>(node->visits)));
    
    for (const auto& child : node->children) {
        double value = child->getUCB1Value(explorationFactor);
        if (value > bestValue) {
            bestValue = value;
            bestChild = child.get();
        }
    }

    if (!bestChild) {
        // std::cerr << "Warning: selectBestChild found no best child.\n";
        bestChild = node->children.front().get();
    }
    
    return bestChild;
}

// TODO make better depth weight. scale between 0.1 and 1
double MCTS::evaluateTerminalState(const PenteGame& game, int depth) const {
    // PenteGame::Player winner = game.getWinner();
    // PenteGame::Player rootPlayer = root_ ? root_->player : PenteGame::NONE;
    
    // if (winner == rootPlayer) {
    //     // Scale exponential to range [0.5, 1.0]
    //     // Win is always better than draw, but faster is much better
    //     double rawScore = std::exp(-depth / 20.0);
    //     return 0.5 + (rawScore * 0.5);  // Maps [0,1] -> [0.5,1.0]
    // } else if (winner == PenteGame::NONE) {
    //     return 0.5;
    // } else {
    //     // Loss: scale to [0.0, 0.5]
    //     double rawScore = std::exp(-depth / 20.0);
    //     return 0.5 - (rawScore * 0.5);  // Maps [0,1] -> [0.5,0.0]
    // }
    return 1.0; // Placeholder: always return win for testing
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
    // sims
    std::cout << "Total simulations: " << totalSimulations_ << ". Tree size: " << getTreeSize() << ". Root visits: " << getTotalVisits() << "\n";

    // timing x mins y secs (ints)
    std::cout << "Total search time: " << static_cast<int>(totalSearchTime_ / 60) << " min " << static_cast<int>(totalSearchTime_) % 60 << " sec\n";
    std::cout << "Simulations/second: " << std::fixed << std::setprecision(0)
              << (totalSearchTime_ > 0 ? totalSimulations_ / totalSearchTime_ : 0) << "\n";

    // win? loss? unsolved? - the status is from the prospective of the parent. so if the root is a "win" that means, the person who last played would get the win.
    std::cout << "Solved status: " << 
        (root_ ? 
            (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "SOLVED_WIN - All moves lead to a loss" :
             root_->solvedStatus == SolvedStatus::SOLVED_LOSS ? "SOLVED_LOSS - At least one move leads to a win" :
             "Unsolved")
            : "N/A") << " And Root avg value: " << std::fixed << std::setprecision(3)
                  << (root_ && root_->visits > 0 ? root_->totalValue / root_->visits : 0.0) << "\n";
    // best move and its avg value
    std::cout << "Best move: " << PenteGame::displayMove(getBestMove().x, getBestMove().y) << "\n";
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
        [](Node* a, Node* b) { 
            // prioritize solved wins first, then by visits
            if (a->solvedStatus == SolvedStatus::SOLVED_WIN && b->solvedStatus != SolvedStatus::SOLVED_WIN) {
                return true;
            }
            if (a->solvedStatus != SolvedStatus::SOLVED_WIN && b->solvedStatus == SolvedStatus::SOLVED_WIN) {
                return false;
            }
            return a->visits > b->visits;
        });
    
    int movesConsidered = root_->children.size();
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves of "
              << movesConsidered << " Considered ===\n";
    std::cout << std::setw(8) << "Move"
              << std::setw(12) << "Visits"
              << std::setw(12) << "Wins"
              << std::setw(12) << "Avg Value"
              << std::setw(12) << "UCB1"
              << std::setw(12) << "Status\n";
    std::cout << std::string(68, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double avgScore = child->visits > 0 ? child->totalValue / child->visits : 0.0;

        double explorationFactor = config_.explorationConstant * std::sqrt(std::log(static_cast<double>(root_->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);

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
                  << std::setw(12) << child->visits
                  << std::setw(12) << child->wins  // Reset to default formatting
                  << std::setw(12) << std::fixed << std::setprecision(3) << avgScore
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
        [](Node* a, Node* b) { 
            // prioritize solved wins first, then by visits
            if (a->solvedStatus == SolvedStatus::SOLVED_WIN && b->solvedStatus != SolvedStatus::SOLVED_WIN) {
                return true;
            }
            if (a->solvedStatus != SolvedStatus::SOLVED_WIN && b->solvedStatus == SolvedStatus::SOLVED_WIN) {
                return false;
            }
            return a->visits > b->visits;
        });
    
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves ===\n";
    std::cout << std::setw(8) << "Move"
              << std::setw(10) << "Visits"
              << std::setw(12) << "Avg Value"
              << std::setw(12) << "UCB1"
              << std::setw(12) << "Status\n";
    std::cout << std::string(54, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double avgValue = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        double explorationFactor = config_.explorationConstant * std::sqrt(std::log(static_cast<double>(node->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);

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
                  << std::setw(12) << std::fixed << std::setprecision(3) << avgValue
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
    double avgValue = targetNode->visits > 0 ? targetNode->totalValue / targetNode->visits : 0.0;
    
    std::cout << "\n=== Analysis for move " << moveStr << " ===\n";
    std::cout << "Visits: " << targetNode->visits << "\n";
    std::cout << "Avg Value: " << std::fixed << std::setprecision(3) << avgValue << "\n";
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
