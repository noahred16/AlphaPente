#include "MCTS.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cassert>

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

double MCTS::Node::getPUCTValue(double explorationFactor, int parentVisits, float prior) const {
    if (this->solvedStatus == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (this->solvedStatus == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    double exploitation = (this->visits == 0) ? 0.0 : this->totalValue / this->visits;
    double exploration =
        explorationFactor * prior * std::sqrt(static_cast<double>(parentVisits)) / (1.0 + this->visits);

    return exploitation + exploration;
}

// ============================================================================
// MCTS Constructor/Destructor
// ============================================================================

MCTS::MCTS(const Config &config)
    : config_(config), arena_(config.arenaSize), root_(nullptr), rng_(std::random_device{}()), totalSimulations_(0),
      totalSearchTime_(0.0) {}

MCTS::~MCTS() {
    // Arena destructor handles all memory - no need to traverse tree
}

// ============================================================================
// Arena Allocation Helpers
// ============================================================================

MCTS::Node *MCTS::allocateNode() {
    Node *node = arena_.allocate<Node>();
    assert(reinterpret_cast<std::uintptr_t>(node) % alignof(Node) == 0);

    if (!node) {
        std::cerr << "FATAL: Arena out of memory! Used: " << arena_.bytesUsed() << " / " << arena_.totalSize()
                  << " bytes\n";
        throw std::bad_alloc();
    }
    // Zero-initialize the node (placement new with default constructor)
    new (node) Node();
    return node;
}

void MCTS::initNodeChildren(Node *node, int capacity) {
    if (capacity <= 0) {
        node->children = nullptr;
        node->childCapacity = 0;
        return;
    }

    node->children = arena_.allocate<Node *>(capacity);
    if (!node->children) {
        std::cerr << "FATAL: Arena out of memory for children array!\n";
        throw std::bad_alloc();
    }
    std::memset(node->children, 0, sizeof(Node *) * capacity);
    node->childCapacity = static_cast<uint16_t>(capacity);
    node->childCount = 0;
}

// ============================================================================
// Main Search Interface
// ============================================================================

PenteGame::Move MCTS::search(const PenteGame &game) {
    this->startSimulations_ = totalSimulations_; // For tracking how many sims were done in this search
    this->game = game;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Reset statistics and arena
    // reset();
    // clearTree();

    // Initialize root node if a fresh sim
    if (!root_) {
        root_ = allocateNode();
    }
    root_->player = game.getCurrentPlayer();

    // Local copy of game for simulations
    PenteGame localGame;

    // Run MCTS iterations
    for (int i = 0; i < config_.maxIterations; i++) {

        if (root_->solvedStatus != SolvedStatus::UNSOLVED) {
            std::cout << "Root node solved status: "
                      << (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "WIN" : "LOSS") << " after " << i
                      << " iterations.\n";
            break;
        }

        // Reset localGame to the start state of THIS search
        localGame.syncFrom(game);

        // Selection: traverse tree to find node to expand
        Node *node = select(root_, localGame);

        // Expansion: add a new child node if not terminal
        PenteGame::Player winner = localGame.getWinner();
        if (winner != PenteGame::NONE) {
            PenteGame::Player opponent = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
            // if the game is over. that means the last move made was a winning move.
            // which means the current player lost. which means this move was a "solve win" for the opponent.
            if (winner == opponent) {
                node->solvedStatus = SolvedStatus::SOLVED_WIN;
                node->unprovenCount = 0;
                backpropagate(node, 1.0);
            } else {
                node->solvedStatus = SolvedStatus::SOLVED_LOSS;
                node->unprovenCount = 0;
                backpropagate(node, -1.0);
            }
            totalSimulations_++;
            continue;
        }
        node = expand(node, localGame);

        // Simulation: play out the game randomly, only if not already solved
        double result = simulate(node, localGame);

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
    if (!root_ || root_->childCapacity == 0) {
        std::cout << "Exiting: Unexpected state in getBestMove(). No children found.\n";
        std::cerr << "Error: No moves available to select as best move.\n";
        GameUtils::printBoard(PenteGame());
        exit(1);
    }

    // First, check if any child is a proven win - always choose that
    for (int i = 0; i < root_->childCapacity; i++) {
        Node *child = root_->children[i];
        if (child && child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            return child->move;
        }
    }

    // Select child with most visits (most robust), excluding proven losses
    Node *bestChild = nullptr;
    int maxVisits = -1;

    for (int i = 0; i < root_->childCapacity; i++) {
        Node *child = root_->children[i];
        if (!child)
            continue;
        if (child->solvedStatus != SolvedStatus::SOLVED_LOSS && child->visits > maxVisits) {
            maxVisits = child->visits;
            bestChild = child;
        }
    }

    // Fallback: If all are losses, just pick the one with the most visits
    if (!bestChild) {
        for (int i = 0; i < root_->childCapacity; i++) {
            Node *child = root_->children[i];
            if (!child)
                continue;
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

MCTS::Node *MCTS::select(Node *node, PenteGame &game) {
    PROFILE_SCOPE("MCTS::select");

    while (!node->isTerminal() && node->evaluated) {
        int best = selectBestMoveIndex(node, game);
        PenteGame::Move move = node->moves[best];
        game.makeMove(move.x, move.y);

        Node *child = node->children[best];

        // if child is null, we need to create it (lazy expansion). Otherwise we continue down the tree.
        if (!child) {
            // Check transposition table 
            uint64_t hash = game.getHash();
            auto it = nodeTranspositionTable.find(hash);
            bool foundInTable = (it != nodeTranspositionTable.end());

            if (!foundInTable) {
                // lazy expansion
                child = allocateNode();
                child->move = move;
                child->player = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
                nodeTranspositionTable[hash] = child;
            } else {
                child = it->second;
            }

            node->children[best] = child;
            node->childCount++;
        }

        child->parent = node; // TODO TEMP Ensure parent pointer is set (in case we found an existing node)
        node = child;
    }
    return node;
}

MCTS::Node *MCTS::expand(Node *node, PenteGame &game) {
    PROFILE_SCOPE("MCTS::expand");

    if (node->solvedStatus != SolvedStatus::UNSOLVED) {
        return node;
    }

    if (node->evaluated) {
        std::cout << "This should never happen: expanding a node that has already been evaluated.\n";
        std::cerr << "Error: Attempting to expand a node that has already been evaluated. This indicates a logic error "
                     "in the MCTS implementation.\n";
        // game util print
        std::cerr << "Solved Status of node: " << static_cast<int>(node->solvedStatus) << "\n";
        GameUtils::printGameState(game);
        exit(1);
    }

    // if this is the first node being expanded, we need to allocate all children nodes and set priors
    if (node->childCount == 0) {

        int childCapacity = static_cast<int>(game.getLegalMoves().size());
        // policy - vector of pairs of moves and priors, ordered by prior, filtered to legal moves.
        // std::vector<std::pair<Move, float>> policy value - static evaluation of the position
        auto [policy, value] = config_.evaluator->evaluate(game);
        initNodeChildren(node, childCapacity);

        // Arena-allocate moves and priors arrays, then populate from policy
        node->moves = arena_.allocate<PenteGame::Move>(childCapacity);
        node->priors = arena_.allocate<float>(childCapacity);
        // Mark all priors as unset (-1.0f sentinel). Policy loop overwrites the ones we have data for.
        std::fill(node->priors, node->priors + childCapacity, -1.0f);
        // We allocate the full size, even if the policy returns empty. In selection we can do a lazy policy calc.
        int policyCount = static_cast<int>(policy.size());
        for (int i = 0; i < policyCount; i++) {
            node->moves[i] = policy[i].first;
            node->priors[i] = policy[i].second;
        }

        node->value = value;
        node->expanded = true;
        node->evaluated = true;
        node->unprovenCount = childCapacity;

    } else {
        // this should never happen right?
        std::cout << "This should never happen: expanding a node that already has children allocated.\n";
        std::cerr << "Error: Attempting to expand a node that already has children allocated. This indicates a logic "
                     "error in the MCTS implementation.\n";
        exit(1);
    }

    return node;
}

double MCTS::simulate(Node *node, PenteGame &game) {
    PROFILE_SCOPE("MCTS::simulate");
    return node->value;
}

void MCTS::backpropagate(Node *node, double result) {
    PROFILE_SCOPE("MCTS::backpropagate");
    Node *current = node;
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

int MCTS::selectBestMoveIndex(Node *node, const PenteGame &game) const {
    if (!node || node->childCapacity == 0) {
        std::cerr << "FATAL ERROR: selectBestMoveIndex called with null node or node with no moves.\n";
        exit(1);
    }

    // only supporting PUCT mode
    if (config_.searchMode != SearchMode::PUCT) {
        std::cerr << "FATAL ERROR: selectBestMoveIndex only supports PUCT mode in this implementation.\n";
        exit(1);
    }

    int bestIndex = -1;
    double bestValue = -std::numeric_limits<double>::infinity();

    // Lazy policy load: -1.0f sentinel means priors weren't populated during expand
    if (node->priors[0] < 0.0f) {
        std::vector<std::pair<PenteGame::Move, float>> movePriors = config_.evaluator->evaluatePolicy(game);
        for (int i = 0; i < node->childCapacity; i++) {
            node->moves[i] = movePriors[i].first;
            node->priors[i] = movePriors[i].second;
        }
    }

    double explorationFactor = config_.explorationConstant;
    // use moves and priors arrays to compute PUCT values without dereferencing child pointers
    for (uint16_t i = 0; i < node->childCapacity; i++) {
        // if the child is null, it means it hasn't been expanded yet. we can still compute its PUCT value using the
        // prior and 0 visits.
        double value;
        if (node->children[i]) {
            Node *child = node->children[i];
            value = child->getPUCTValue(explorationFactor, node->visits, node->priors[i]);
        } else {
            // unexpanded child - use prior and 0 visits for PUCT calculation
            double exploitation = 0.0; // no visits yet, so exploitation is 0
            double exploration =
                explorationFactor * node->priors[i] * std::sqrt(static_cast<double>(node->visits)) / (1.0);
            value = exploitation + exploration;
        }
        if (value > bestValue) {
            bestValue = value;
            bestIndex = i;
        }
    }

    // if -1 still then all are proven losses, 
    // this could happen if there is symmetry 
    // and that one node represents more than 1 move but still only decrements the unproven count by 1
    if (bestIndex == -1) {
        assert(node->unprovenCount >= 1);
        assert(node->childCount == node->childCapacity);
        assert(node->children[0] != nullptr);
        node->unprovenCount = 1; // last node to get sent through. 
        // slightly redundant since we already know this node is a proven loss...    
        return 0;
    }

    return bestIndex;
}

double MCTS::evaluateTerminalState(const PenteGame &game, int depth) const {
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

MCTS::Node *MCTS::copySubtree(Node *source, MCTSArena &destArena) {
    if (!source)
        return nullptr;

    // Allocate new node in destination arena
    Node *dest = destArena.allocate<Node>();
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
    dest->unprovenCount = source->unprovenCount;
    dest->visits = source->visits;
    dest->wins = source->wins;
    dest->totalValue = source->totalValue;
    dest->parent = nullptr; // Will be set by parent during recursion
    dest->children = nullptr;
    dest->moves = nullptr;
    dest->priors = nullptr;
    dest->expanded = source->expanded;
    dest->evaluated = source->evaluated;
    dest->value = source->value;

    // Copy moves/priors arrays and children (all sized by childCapacity)
    if (source->childCapacity > 0 && source->children) {
        dest->children = destArena.allocate<Node *>(source->childCapacity);
        dest->moves = destArena.allocate<PenteGame::Move>(source->childCapacity);
        dest->priors = destArena.allocate<float>(source->childCapacity);
        if (!dest->children || !dest->moves || !dest->priors) {
            std::cerr << "FATAL: Destination arena out of memory for children!\n";
            throw std::bad_alloc();
        }
        std::memcpy(dest->moves, source->moves, sizeof(PenteGame::Move) * source->childCapacity);
        std::memcpy(dest->priors, source->priors, sizeof(float) * source->childCapacity);
        for (int i = 0; i < source->childCapacity; i++) {
            dest->children[i] = copySubtree(source->children[i], destArena);
            if (dest->children[i]) {
                dest->children[i]->parent = dest;
            }
        }
    }

    return dest;
}

void MCTS::reuseSubtree(const PenteGame::Move &move) {
    if (!root_) {
        return;
    }

    // Find child matching the move (sparse array - scan all slots)
    Node *matchingChild = nullptr;
    for (int i = 0; i < root_->childCapacity; i++) {
        Node *child = root_->children[i];
        if (child && child->move.x == move.x && child->move.y == move.y) {
            matchingChild = child;
            break;
        }
    }

    if (!matchingChild) {
        // Move not in tree, clear everything
        clearTree();
        return;
    }

    // Just move root down - parent pointer is preserved for undo
    root_ = matchingChild;
}

bool MCTS::undoSubtree() {
    if (!root_ || !root_->parent) {
        return false;
    }
    root_ = root_->parent;
    return true;
}

void MCTS::pruneTree(Node *keepNode) {
    // With arena allocation, we can't selectively free nodes
    // Just clear everything for now
    clearTree();
}

// ============================================================================
// Statistics and Debugging
// ============================================================================

int MCTS::getTotalVisits() const { return root_ ? root_->visits : 0; }

int MCTS::countNodes(Node *node, std::unordered_set<Node *> &visited) const {
    if (!node || !visited.insert(node).second)
        return 0;

    int count = 1;
    for (int i = 0; i < node->childCapacity; i++) {
        if (node->children[i]) {
            count += countNodes(node->children[i], visited);
        }
    }
    return count;
}

int MCTS::getTreeSize() const {
    std::unordered_set<Node *> visited;
    return countNodes(root_, visited);
}

void MCTS::printStats() const {
    std::cout << "\n=== MCTS Statistics ===\n";
    int treeSize = getTreeSize();
    int totalVisits = getTotalVisits();
    int transpositionTableSize = nodeTranspositionTable.size();
    std::cout << "Total simulations: " << GameUtils::formatWithCommas(totalSimulations_)
              << ". Tree size: " << GameUtils::formatWithCommas(treeSize)
              << ". Root visits: " << GameUtils::formatWithCommas(totalVisits)
              << ". Transposition table size: " << GameUtils::formatWithCommas(transpositionTableSize) << "\n";

    std::cout << "Search time: " << static_cast<int>(totalSearchTime_ / 60) << " min "
              << static_cast<int>(totalSearchTime_) % 60 << " sec\n";

    // total - startSimulations_ to get sims for just this search
    int simsThisSearch = totalSimulations_ - startSimulations_;
    std::cout << "Simulations/second: " << std::fixed << std::setprecision(0)
              << (totalSearchTime_ > 0 ? simsThisSearch / totalSearchTime_ : 0) << "\n";

    // Arena memory stats
    std::cout << "Arena memory: " << std::fixed << std::setprecision(1) << (arena_.bytesUsed() / (1024.0 * 1024.0))
              << " MB / " << (arena_.totalSize() / (1024.0 * 1024.0)) << " MB (" << std::setprecision(1)
              << arena_.utilizationPercent() << "%)\n";

    std::cout << "Solved status: "
              << (root_ ? (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "SOLVED_WIN - All moves lead to a loss"
                           : root_->solvedStatus == SolvedStatus::SOLVED_LOSS
                               ? "SOLVED_LOSS - At least one move leads to a win"
                               : "Unsolved")
                        : "N/A")
              << " And Root avg value: " << std::fixed << std::setprecision(3)
              << (root_ && root_->visits > 0 ? root_->totalValue / root_->visits : 0.0) << "\n";

    if (root_ && root_->childCapacity > 0) {
        std::cout << "Best move: " << GameUtils::displayMove(getBestMove().x, getBestMove().y) << "\n";
    }
    std::cout << "=======================\n\n";
}

void MCTS::printBestMoves(int topN) const {
    if (!root_ || root_->childCapacity == 0) {
        std::cout << "No moves analyzed yet.\n";
        return;
    }

    // Collect non-null children pointers into a vector for sorting
    std::vector<Node *> children;
    children.reserve(root_->childCapacity);
    for (int i = 0; i < root_->childCapacity; i++) {
        if (root_->children[i]) {
            children.push_back(root_->children[i]);
        }
    }

    std::sort(children.begin(), children.end(), [](Node *a, Node *b) {
        if (a->solvedStatus == SolvedStatus::SOLVED_WIN && b->solvedStatus != SolvedStatus::SOLVED_WIN) {
            return true;
        }
        if (a->solvedStatus != SolvedStatus::SOLVED_WIN && b->solvedStatus == SolvedStatus::SOLVED_WIN) {
            return false;
        }

        return a->visits > b->visits;
    });

    int movesConsidered = static_cast<int>(children.size());
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves of " << movesConsidered
              << " Considered ===\n";
    std::cout << std::setw(6) << "Move" << std::setw(10) << "Visits" << std::setw(10) << "Wins" << std::setw(10)
              << "MoveEval" << std::setw(10) << "Prior" << std::setw(10) << "Avg Val" << std::setw(10) << "UCB1"
              << std::setw(10) << "PUCT" << std::setw(10) << "Status\n";
    std::cout << std::string(86, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node *child = children[i];
        double avgScore = child->visits > 0 ? child->totalValue / child->visits : 0.0;

        double explorationFactor =
            config_.explorationConstant * std::sqrt(std::log(static_cast<double>(root_->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, root_->visits, root_->priors[i]);

        double moveEval = this->game.evaluateMove(child->move);

        // if moveEval = 1, skip and inc i so it doesnt count towards topN
        // if (moveEval == 1.0) {
        //     continue;
        // }

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

        std::string status;
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(6) << moveStr << std::setw(10) << child->visits << std::setw(10) << child->wins
                  << std::setw(10) << static_cast<int>(moveEval) << std::setw(10) << std::fixed << std::setprecision(3)
                  << root_->priors[i] << std::setw(10) << std::fixed << std::setprecision(3) << avgScore << std::setw(10)
                  << std::fixed << std::setprecision(3) << ucb1 << std::setw(10) << std::fixed << std::setprecision(3)
                  << puct << std::setw(10) << status << "\n";
    }

    std::cout << std::string(74, '=') << "\n\n";
}

MCTS::Node *MCTS::findChildNode(MCTS::Node *parent, int x, int y) const {
    if (!parent) {
        return nullptr;
    }

    for (int i = 0; i < parent->childCapacity; i++) {
        Node *child = parent->children[i];
        if (child && child->move.x == x && child->move.y == y) {
            return child;
        }
    }

    return nullptr;
}

void MCTS::printMovesFromNode(MCTS::Node *node, int topN) const {
    if (!node || node->childCapacity == 0) {
        std::cout << "No moves analyzed for this position.\n";
        return;
    }

    std::vector<Node *> children;
    children.reserve(node->childCapacity);
    for (int i = 0; i < node->childCapacity; i++) {
        if (node->children[i]) {
            children.push_back(node->children[i]);
        }
    }

    std::sort(children.begin(), children.end(), [](Node *a, Node *b) {
        if (a->solvedStatus == SolvedStatus::SOLVED_WIN && b->solvedStatus != SolvedStatus::SOLVED_WIN) {
            return true;
        }
        if (a->solvedStatus != SolvedStatus::SOLVED_WIN && b->solvedStatus == SolvedStatus::SOLVED_WIN) {
            return false;
        }
        return a->visits > b->visits;
    });

    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves ===\n";
    std::cout << std::setw(6) << "Move" << std::setw(10) << "Visits" << std::setw(10) << "Avg Val" << std::setw(10)
              << "Prior" << std::setw(10) << "UCB1" << std::setw(10) << "PUCT" << std::setw(10) << "Status\n";
    std::cout << std::string(66, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node *child = children[i];
        double avgValue = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        double explorationFactor = config_.explorationConstant * std::sqrt(std::log(static_cast<double>(node->visits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, node->visits, node->priors[i]);

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

        std::string status;
        if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(6) << moveStr << std::setw(10) << child->visits << std::setw(10) << std::fixed
                  << std::setprecision(3) << avgValue << std::setw(10) << std::fixed << std::setprecision(3)
                  << root_->priors[i] << std::setw(10) << std::fixed << std::setprecision(3) << ucb1 << std::setw(10)
                  << std::fixed << std::setprecision(3) << puct << std::setw(10) << status << "\n";
    }

    std::cout << "===================\n\n";
}

void MCTS::printBranch(const char *moveStr, int topN) const {
    auto [x, y] = GameUtils::parseMove(moveStr);
    printBranch(x, y, topN);
}

void MCTS::printBranch(int x, int y, int topN) const {
    if (!root_) {
        std::cout << "No search tree exists yet.\n";
        return;
    }

    Node *targetNode = findChildNode(root_, x, y);

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

void MCTS::setConfig(const Config &config) { config_ = config; }

const MCTS::Config &MCTS::getConfig() const { return config_; }
