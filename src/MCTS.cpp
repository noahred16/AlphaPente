#include "MCTS.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <iostream>
#include <iomanip>

// ============================================================================
// Node Implementation
// ============================================================================

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

double MCTS::Node::getPUCTValue(double explorationFactor, int parentVisits) const {
    if (this->solvedStatus == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (this->solvedStatus == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    double exploitation = (this->visits == 0) ? 0.0 : this->totalValue / this->visits;
    double exploration = explorationFactor * this->prior * 
               std::sqrt(static_cast<double>(parentVisits)) / (1.0 + this->visits);
    
    return exploitation + exploration;
}

// ============================================================================
// MCTS Constructor/Destructor
// ============================================================================

MCTS::MCTS(const Config& config)
    : config_(config)
    , arena_(config.arenaSize)
    , root_(nullptr)
    , rng_(std::random_device{}())
    , totalSimulations_(0)
    , totalSearchTime_(0.0) {
}

MCTS::~MCTS() {
    // Arena destructor handles all memory - no need to traverse tree
}

// ============================================================================
// Arena Allocation Helpers
// ============================================================================

MCTS::Node* MCTS::allocateNode() {
    Node* node = arena_.allocate<Node>();
    if (!node) {
        std::cerr << "FATAL: Arena out of memory! Used: " << arena_.bytesUsed()
                  << " / " << arena_.totalSize() << " bytes\n";
        throw std::bad_alloc();
    }
    // Zero-initialize the node (placement new with default constructor)
    new (node) Node();
    return node;
}

void MCTS::initNodeChildren(Node* node, int capacity) {
    if (capacity <= 0) {
        node->children = nullptr;
        node->childCapacity = 0;
        return;
    }

    node->children = arena_.allocate<Node*>(capacity);
    if (!node->children) {
        std::cerr << "FATAL: Arena out of memory for children array!\n";
        throw std::bad_alloc();
    }
    node->childCapacity = static_cast<uint16_t>(capacity);
    node->childCount = 0;
}

void MCTS::initNodeUntriedMoves(Node* node, const std::vector<PenteGame::Move>& moves) {
    size_t count = moves.size();
    if (count == 0) {
        node->untriedMoves = nullptr;
        node->untriedMoveCount = 0;
        return;
    }

    node->untriedMoves = arena_.allocate<PenteGame::Move>(count);
    if (!node->untriedMoves) {
        std::cerr << "FATAL: Arena out of memory for untried moves!\n";
        throw std::bad_alloc();
    }

    for (size_t i = 0; i < count; i++) {
        node->untriedMoves[i] = moves[i];
    }
    node->untriedMoveCount = static_cast<uint16_t>(count);
}

// ============================================================================
// Main Search Interface
// ============================================================================

PenteGame::Move MCTS::search(const PenteGame& game) {
    auto startTime = std::chrono::high_resolution_clock::now();

    // Reset statistics and arena
    reset();
    clearTree();

    // Initialize root node
    root_ = allocateNode();
    root_->player = game.getCurrentPlayer();

    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();
    initNodeUntriedMoves(root_, legalMoves);
    initNodeChildren(root_, static_cast<int>(legalMoves.size()));
    root_->unprovenCount = static_cast<int16_t>(legalMoves.size());

    // Local copy of game for simulations
    PenteGame localGame;

    // Run MCTS iterations
    for (int i = 0; i < config_.maxIterations; i++) {

        if (root_->solvedStatus != SolvedStatus::UNSOLVED) {
            std::cout << "Root node solved status: "
                      << (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "WIN" : "LOSS")
                      << " after " << i << " iterations.\n";
            break;
        }

        // Reset localGame to the start state of THIS search
        localGame.syncFrom(game);

        // Selection: traverse tree to find node to expand
        Node* node = select(root_, localGame);

        // Expansion: add a new child node if not terminal
        if (!localGame.isGameOver() && node->untriedMoveCount > 0) {
            node = expand(node, localGame);
        }

        // Simulation: play out the game randomly, only if not already solved
        double result = 0.0;
        if (node->solvedStatus == SolvedStatus::UNSOLVED) {
            result = simulate(localGame);
        } else {
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
    PROFILE_SCOPE("MCTS::getBestMove");
    if (!root_ || root_->childCount == 0) {
        std::cout << "Exiting: Unexpected state in getBestMove(). No children found.\n";
        std::cerr << "Error: No moves available to select as best move.\n";
        // print the game board
        // get a copy of the game
        GameUtils::printBoard(PenteGame());
        exit(1);
    }

    // First, check if any child is a proven win - always choose that
    for (uint16_t i = 0; i < root_->childCount; i++) {
        Node* child = root_->children[i];
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            return child->move;
        }
    }

    // Select child with most visits (most robust), excluding proven losses
    Node* bestChild = nullptr;
    int maxVisits = -1;

    for (uint16_t i = 0; i < root_->childCount; i++) {
        Node* child = root_->children[i];
        if (child->solvedStatus != SolvedStatus::SOLVED_LOSS && child->visits > maxVisits) {
            maxVisits = child->visits;
            bestChild = child;
        }
    }

    // Fallback: If all are losses, just pick the one with the most visits
    if (!bestChild && root_->childCount > 0) {
        for (uint16_t i = 0; i < root_->childCount; i++) {
            Node* child = root_->children[i];
            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                bestChild = child;
            }
        }
    }

    return bestChild->move;
}

// ============================================================================
// MCTS Phases
// ============================================================================

MCTS::Node* MCTS::select(Node* node, PenteGame& game) {
    PROFILE_SCOPE("MCTS::select");
    // Traverse tree using UCB1 until we find a node that's not fully expanded and not solved
    while (node->isFullyExpanded() && node->childCount > 0 && node->solvedStatus == SolvedStatus::UNSOLVED) {

        if (config_.searchMode == SearchMode::PUCT) {
            updateChildrenPriors(node, game);
        }

        node = selectBestChild(node);
        game.makeMove(node->move.x, node->move.y);
    }

    return node;
}

MCTS::Node* MCTS::expand(Node* node, PenteGame& game) {
    PROFILE_SCOPE("MCTS::expand");
    if (node->untriedMoveCount == 0) {
        return node;
    }

    // Pick a random untried move
    std::uniform_int_distribution<uint16_t> dist(0, node->untriedMoveCount - 1);
    uint16_t moveIndex = dist(rng_);

    // Swap with last and "pop" (just decrement count)
    std::swap(node->untriedMoves[moveIndex], node->untriedMoves[node->untriedMoveCount - 1]);
    PenteGame::Move move = node->untriedMoves[node->untriedMoveCount - 1];
    node->untriedMoveCount--;

    // Track the player who is about to move
    PenteGame::Player startPlayer = game.getCurrentPlayer();

    // Apply move to game
    game.makeMove(move.x, move.y);

    // Create new child node from arena
    Node* child = allocateNode();
    child->move = move;
    child->player = (node->player == PenteGame::BLACK)
                    ? PenteGame::WHITE
                    : PenteGame::BLACK;
    child->parent = node;
    child->prior = -1.0f; // will be updated in the selection phase

    // Get legal moves for this new state
    if (!game.isGameOver()) {
        std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();
        initNodeUntriedMoves(child, legalMoves);
        initNodeChildren(child, static_cast<int>(legalMoves.size()));
        child->unprovenCount = static_cast<int16_t>(legalMoves.size());
    } else {
        // Terminal node - determine solved status
        PenteGame::Player winner = game.getWinner();
        if (winner == startPlayer) {
            child->solvedStatus = SolvedStatus::SOLVED_WIN;
            child->unprovenCount = 0;
            // NEW: Immediately propagate win to parent
            if (node->solvedStatus == SolvedStatus::UNSOLVED) {
                node->solvedStatus = SolvedStatus::SOLVED_LOSS;
            }
        } else {
            child->solvedStatus = SolvedStatus::UNSOLVED;
            std::cout << "Exiting: Unexpected game over state in expand(). Winner: "
                      << static_cast<int>(winner) << std::endl;
            std::cerr << "Error: Unexpected game over state. Winner: "
                      << static_cast<int>(winner) << std::endl;
            exit(1);
        }
    }

    // Add child to parent's children array
    node->children[node->childCount] = child;
    node->childCount++;

    return child;
}

double MCTS::simulate(const PenteGame& gameState) {
    PROFILE_SCOPE("MCTS::simulate");

    // Work on a copy - discarded at end of function
    PenteGame simGame = gameState;

    PenteGame::Player startPlayer = simGame.getCurrentPlayer();
    PenteGame::Player winner = PenteGame::NONE;
    int depth = 0;

    // Play out game randomly until terminal or max depth
    while ((winner = simGame.getWinner()) == PenteGame::NONE && depth < config_.maxSimulationDepth) {
        PenteGame::Move move = simGame.getRandomLegalMove();
        simGame.makeMove(move.x, move.y);
        depth++;
    }

    // Evaluate result
    double result = evaluateTerminalState(simGame, depth);

    if (winner != PenteGame::NONE) {
        result *= (winner == startPlayer) ? -1.0 : 1.0;
    } else {
        result = 0.0;
    }

    // Copy is discarded when function returns - no undo needed
    return result;
}

void MCTS::backpropagate(Node* node, double result) {
    PROFILE_SCOPE("MCTS::backpropagate");
    Node* current = node;
    double currentResult = result;

    while (current != nullptr) {
        // Backpropagate stats
        current->visits++;
        current->totalValue += currentResult;
        current->wins += (currentResult > 0) ? 1 : 0;

        // --- SOLVER LOGIC (Minimax Propagation) ---
        if (current->solvedStatus == SolvedStatus::SOLVED_WIN && current->parent) {
            current->parent->solvedStatus = SolvedStatus::SOLVED_LOSS;
        }

        if (current->solvedStatus == SolvedStatus::SOLVED_LOSS && current->parent) {
            current->parent->unprovenCount--;

            if (current->parent->unprovenCount < 0) {
                std::cerr << "FATAL ERROR: unprovenCount dropped below 0!" << std::endl;
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

void MCTS::updateChildrenPriors(Node* node, const PenteGame& game) {
    PROFILE_SCOPE("MCTS::updateChildrenPriors");
    // No children? Nothing to do
    if (node->childCount == 0) {
        return;
    }
    
    // Check if already evaluated (first child's prior will be >= 0)
    if (node->children[0]->prior >= 0.0f) {
        return; // Already evaluated
    }
    
    // No evaluator? throw error
    if (!config_.evaluator) {
        std::cerr << "FATAL ERROR: PUCT search mode selected but no evaluator provided!" << std::endl;
        exit(1);
    }
    
    // Evaluate policy for this position
    auto policy = config_.evaluator->evaluatePolicy(game);
    
    // check that childCount matches policy size
    if (node->childCount != policy.size()) {
        std::cerr << "FATAL ERROR: Mismatch between child count and policy size!" << std::endl;
        exit(1);
    }

    // Assign priors to children
    // Policy is returned in same order as childMoves, so direct assignment
    for (uint16_t i = 0; i < node->childCount && i < policy.size(); i++) {
        node->children[i]->prior = policy[i];
    }
}

MCTS::Node* MCTS::selectBestChild(Node* node) const {
    if (!node || node->childCount == 0) {
        return nullptr;
    }

    Node* bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();


    if (config_.searchMode == SearchMode::UCB1) {
        double explorationFactor = config_.explorationConstant *
                                std::sqrt(std::log(static_cast<double>(node->visits)));
        for (uint16_t i = 0; i < node->childCount; i++) {
            Node* child = node->children[i];
            double value = child->getUCB1Value(explorationFactor);
            if (value > bestValue) {
                bestValue = value;
                bestChild = child;
            }
        }
    } else if (config_.searchMode == SearchMode::PUCT) {
        double explorationFactor = config_.explorationConstant;
        for (uint16_t i = 0; i < node->childCount; i++) {
            Node* child = node->children[i];
            double value = child->getPUCTValue(explorationFactor, node->visits);
            if (value > bestValue) {
                bestValue = value;
                bestChild = child;
            }
        }
    }

    if (!bestChild) {
        bestChild = node->children[0];
    }

    return bestChild;
}

double MCTS::evaluateTerminalState(const PenteGame& game, int depth) const {
    PROFILE_SCOPE("MCTS::evaluateTerminalState");
    return 1.0; // Placeholder: always return win for testing
}


// ============================================================================
// Tree Management
// ============================================================================

void MCTS::reset() {
    totalSimulations_ = 0;
    totalSearchTime_ = 0.0;
}

void MCTS::clearTree() {
    // O(1) tree destruction - just reset the arena offset
    arena_.reset();
    root_ = nullptr;
}

MCTS::Node* MCTS::copySubtree(Node* source, MCTSArena& destArena) {
    if (!source) return nullptr;

    // Allocate new node in destination arena
    Node* dest = destArena.allocate<Node>();
    if (!dest) {
        std::cerr << "FATAL: Destination arena out of memory during subtree copy!\n";
        throw std::bad_alloc();
    }

    // Copy scalar fields
    dest->move = source->move;
    dest->player = source->player;
    dest->solvedStatus = source->solvedStatus;
    dest->childCount = source->childCount;
    dest->childCapacity = source->childCapacity;
    dest->untriedMoveCount = source->untriedMoveCount;
    dest->unprovenCount = source->unprovenCount;
    dest->visits = source->visits;
    dest->wins = source->wins;
    dest->totalValue = source->totalValue;
    dest->parent = nullptr;  // Will be set by parent during recursion

    // Copy untried moves array
    if (source->untriedMoveCount > 0 && source->untriedMoves) {
        dest->untriedMoves = destArena.allocate<PenteGame::Move>(source->untriedMoveCount);
        if (!dest->untriedMoves) {
            std::cerr << "FATAL: Destination arena out of memory for untried moves!\n";
            throw std::bad_alloc();
        }
        for (uint16_t i = 0; i < source->untriedMoveCount; i++) {
            dest->untriedMoves[i] = source->untriedMoves[i];
        }
    } else {
        dest->untriedMoves = nullptr;
    }

    // Copy children array and recursively copy child nodes
    if (source->childCount > 0 && source->children) {
        dest->children = destArena.allocate<Node*>(source->childCapacity);
        if (!dest->children) {
            std::cerr << "FATAL: Destination arena out of memory for children!\n";
            throw std::bad_alloc();
        }
        for (uint16_t i = 0; i < source->childCount; i++) {
            dest->children[i] = copySubtree(source->children[i], destArena);
            if (dest->children[i]) {
                dest->children[i]->parent = dest;
            }
        }
    } else {
        dest->children = nullptr;
    }

    return dest;
}

void MCTS::reuseSubtree(const PenteGame::Move& move) {
    if (!root_) {
        return;
    }

    // Find child matching the move
    Node* matchingChild = nullptr;
    for (uint16_t i = 0; i < root_->childCount; i++) {
        Node* child = root_->children[i];
        if (child->move.x == move.x && child->move.y == move.y) {
            matchingChild = child;
            break;
        }
    }

    if (!matchingChild) {
        // Move not in tree, clear everything
        clearTree();
        return;
    }

    // Create fresh arena and copy subtree into it
    MCTSArena freshArena(arena_.totalSize());
    Node* newRoot = copySubtree(matchingChild, freshArena);
    newRoot->parent = nullptr;

    // Swap arenas - old arena gets freed when freshArena destructor runs
    arena_.swap(freshArena);
    root_ = newRoot;
}

void MCTS::pruneTree(Node* keepNode) {
    // With arena allocation, we can't selectively free nodes
    // Just clear everything for now
    clearTree();
}

// ============================================================================
// Statistics and Debugging
// ============================================================================

int MCTS::getTotalVisits() const {
    return root_ ? root_->visits : 0;
}

int MCTS::countNodes(Node* node) const {
    if (!node) return 0;

    int count = 1;
    for (uint16_t i = 0; i < node->childCount; i++) {
        count += countNodes(node->children[i]);
    }
    return count;
}

int MCTS::getTreeSize() const {
    return countNodes(root_);
}

void MCTS::printStats() const {
    std::cout << "\n=== MCTS Statistics ===\n";
    std::cout << "Total simulations: " << totalSimulations_
              << ". Tree size: " << getTreeSize()
              << ". Root visits: " << getTotalVisits() << "\n";

    std::cout << "Total search time: " << static_cast<int>(totalSearchTime_ / 60)
              << " min " << static_cast<int>(totalSearchTime_) % 60 << " sec\n";
    std::cout << "Simulations/second: " << std::fixed << std::setprecision(0)
              << (totalSearchTime_ > 0 ? totalSimulations_ / totalSearchTime_ : 0) << "\n";

    // Arena memory stats
    std::cout << "Arena memory: " << std::fixed << std::setprecision(1)
              << (arena_.bytesUsed() / (1024.0 * 1024.0)) << " MB / "
              << (arena_.totalSize() / (1024.0 * 1024.0)) << " MB ("
              << std::setprecision(1) << arena_.utilizationPercent() << "%)\n";

    std::cout << "Solved status: " <<
        (root_ ?
            (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "SOLVED_WIN - All moves lead to a loss" :
             root_->solvedStatus == SolvedStatus::SOLVED_LOSS ? "SOLVED_LOSS - At least one move leads to a win" :
             "Unsolved")
            : "N/A") << " And Root avg value: " << std::fixed << std::setprecision(3)
                  << (root_ && root_->visits > 0 ? root_->totalValue / root_->visits : 0.0) << "\n";

    if (root_ && root_->childCount > 0) {
        std::cout << "Best move: " << GameUtils::displayMove(getBestMove().x, getBestMove().y) << "\n";
    }
    std::cout << "=======================\n\n";
}

void MCTS::printBestMoves(int topN) const {
    if (!root_ || root_->childCount == 0) {
        std::cout << "No moves analyzed yet.\n";
        return;
    }

    // Collect children pointers into a vector for sorting
    std::vector<Node*> children;
    children.reserve(root_->childCount);
    for (uint16_t i = 0; i < root_->childCount; i++) {
        children.push_back(root_->children[i]);
    }

    std::sort(children.begin(), children.end(),
        [](Node* a, Node* b) {
            if (a->solvedStatus == SolvedStatus::SOLVED_WIN && b->solvedStatus != SolvedStatus::SOLVED_WIN) {
                return true;
            }
            if (a->solvedStatus != SolvedStatus::SOLVED_WIN && b->solvedStatus == SolvedStatus::SOLVED_WIN) {
                return false;
            }
            return a->visits > b->visits;
        });

    int movesConsidered = root_->childCount;
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves of "
              << movesConsidered << " Considered ===\n";
    std::cout << std::setw(8) << "Move"
              << std::setw(12) << "Visits"
              << std::setw(12) << "Wins"
              << std::setw(12) << "Avg Value"
              << std::setw(12) << "UCB1"
              << std::setw(12) << "PUCT"
              << std::setw(12) << "Status\n";
    std::cout << std::string(80, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double avgScore = child->visits > 0 ? child->totalValue / child->visits : 0.0;

        double explorationFactor = config_.explorationConstant *
                                   std::sqrt(std::log(static_cast<double>(root_->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, root_->visits);

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

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
                  << std::setw(12) << child->wins
                  << std::setw(12) << std::fixed << std::setprecision(3) << avgScore
                  << std::setw(12) << std::fixed << std::setprecision(3) << ucb1
                  << std::setw(12) << std::fixed << std::setprecision(3) << puct
                  << std::setw(12) << status
                  << "\n";
    }

    std::cout << "===================\n\n";
}

MCTS::Node* MCTS::findChildNode(MCTS::Node* parent, int x, int y) const {
    if (!parent) {
        return nullptr;
    }

    for (uint16_t i = 0; i < parent->childCount; i++) {
        Node* child = parent->children[i];
        if (child->move.x == x && child->move.y == y) {
            return child;
        }
    }

    return nullptr;
}

void MCTS::printMovesFromNode(MCTS::Node* node, int topN) const {
    if (!node || node->childCount == 0) {
        std::cout << "No moves analyzed for this position.\n";
        return;
    }

    std::vector<Node*> children;
    children.reserve(node->childCount);
    for (uint16_t i = 0; i < node->childCount; i++) {
        children.push_back(node->children[i]);
    }

    std::sort(children.begin(), children.end(),
        [](Node* a, Node* b) {
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
              << std::setw(12) << "PUCT"
              << std::setw(12) << "Status\n";
    std::cout << std::string(66, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        double avgValue = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        double explorationFactor = config_.explorationConstant *
                                   std::sqrt(std::log(static_cast<double>(node->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, node->visits);

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

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
                  << std::setw(12) << std::fixed << std::setprecision(3) << puct
                  << std::setw(12) << status
                  << "\n";
    }

    std::cout << "===================\n\n";
}

void MCTS::printBranch(const char* moveStr, int topN) const {
    auto [x, y] = GameUtils::parseMove(moveStr);
    printBranch(x, y, topN);
}

void MCTS::printBranch(int x, int y, int topN) const {
    if (!root_) {
        std::cout << "No search tree exists yet.\n";
        return;
    }

    Node* targetNode = findChildNode(root_, x, y);

    if (!targetNode) {
        std::string moveStr = GameUtils::displayMove(x, y);
        std::cout << "Move " << moveStr << " not found in search tree.\n";
        std::cout << "This move may not have been explored yet.\n";
        return;
    }

    std::string moveStr = GameUtils::displayMove(x, y);
    double avgValue = targetNode->visits > 0 ? targetNode->totalValue / targetNode->visits : 0.0;

    std::cout << "\n=== Analysis for move " << moveStr << " ===\n";
    std::cout << "Visits: " << targetNode->visits << "\n";
    std::cout << "Avg Value: " << std::fixed << std::setprecision(3) << avgValue << "\n";
    std::cout << "Player: " << (targetNode->player == PenteGame::BLACK ? "Black" : "White") << "\n";

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
