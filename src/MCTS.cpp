#include "MCTS.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <iostream>
#include <iomanip>

// TEMP NOTES: Multi-threaded implementation:
// - Worker and Inference threads. 
// - Policy/Value queue for the inference thread to process new nodes (policy value)
// - Backprop queue for worker threads to handle backpropagation.
// Thread-Safe Queue
// BU-UCT (Balance the Unobserved UCT) - Aggregate rollouts before backpropagation
// 2 Performance Cores (P-cores) - each counts as 2 logical processors
// 8 Efficient Cores (E-cores)
// 12 threads total
// Ensure the inference thread uses a dedicated P-core for low latency
// for each node, you typically add a small lock just for expansion / structural mutations
// oh gosh, the value inference happens separately. 
// we may need another queue for value updates


// ============================================================================
// Node Implementation
// ============================================================================

double MCTS::Node::getUCB1Value(double explorationFactor) const {
    // Prioritize solved nodes
    auto status = this->solvedStatus.load(std::memory_order_relaxed);
    if (status == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (status == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    int32_t v = this->visits.load(std::memory_order_relaxed);
    if (v == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double exploitation = this->totalValue.load(std::memory_order_relaxed) / v;
    double exploration = explorationFactor / std::sqrt(static_cast<double>(v));

    return exploitation + exploration;
}

double MCTS::Node::getPUCTValue(double explorationFactor, int parentVisits) const {
    auto status = this->solvedStatus.load(std::memory_order_relaxed);
    if (status == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (status == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    int32_t v = this->visits.load(std::memory_order_relaxed);
    double exploitation = (v == 0) ? 0.0 : this->totalValue.load(std::memory_order_relaxed) / v;
    double exploration = explorationFactor * this->prior *
               std::sqrt(static_cast<double>(parentVisits)) / (1.0 + v);

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


// ============================================================================
// Main Search Interface
// ============================================================================

PenteGame::Move MCTS::search(const PenteGame& game) {

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

        auto rootStatus = root_->solvedStatus.load(std::memory_order_relaxed);
        if (rootStatus != SolvedStatus::UNSOLVED) {
            std::cout << "Root node solved status: "
                      << (rootStatus == SolvedStatus::SOLVED_WIN ? "WIN" : "LOSS")
                      << " after " << i << " iterations.\n";
            break;
        }

        // Reset localGame to the start state of THIS search
        localGame.syncFrom(game);

        // Selection: traverse tree to find node to expand
        Node* node = select(root_, localGame);

        // Expansion: add a new child node if not terminal
        PenteGame::Player winner = localGame.getWinner();
        if (winner != PenteGame::NONE) {
            PenteGame::Player opponent = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
            // if the game is over. that means the last move made was a winning move. 
            // which means the current player lost. which means this move was a "solve win" for the opponent.
            if (winner == opponent) {
                node->solvedStatus.store(SolvedStatus::SOLVED_WIN, std::memory_order_relaxed);
                node->unprovenCount.store(0, std::memory_order_relaxed);
                backpropagate(node, 1.0);
            } else {
                node->solvedStatus.store(SolvedStatus::SOLVED_LOSS, std::memory_order_relaxed);
                node->unprovenCount.store(0, std::memory_order_relaxed);
                backpropagate(node, -1.0);
            }
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
        if (child->solvedStatus.load(std::memory_order_relaxed) == SolvedStatus::SOLVED_WIN) {
            return child->move;
        }
    }

    // Select child with most visits (most robust), excluding proven losses
    Node* bestChild = nullptr;
    int32_t maxVisits = -1;

    for (uint16_t i = 0; i < root_->childCount; i++) {
        Node* child = root_->children[i];
        int32_t v = child->visits.load(std::memory_order_relaxed);
        if (child->solvedStatus.load(std::memory_order_relaxed) != SolvedStatus::SOLVED_LOSS && v > maxVisits) {
            maxVisits = v;
            bestChild = child;
        }
    }

    // Fallback: If all are losses, just pick the one with the most visits
    if (!bestChild && root_->childCount > 0) {
        for (uint16_t i = 0; i < root_->childCount; i++) {
            Node* child = root_->children[i];
            int32_t v = child->visits.load(std::memory_order_relaxed);
            if (v > maxVisits) {
                maxVisits = v;
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
    while (node->isFullyExpanded() && node->solvedStatus.load(std::memory_order_relaxed) == SolvedStatus::UNSOLVED) {
        if (node->childCount == 0 && config_.heuristicMode == HeuristicMode::HEURISTIC) {
            // lazy policy load
            std::vector<std::pair<PenteGame::Move, float>> movePriors = config_.evaluator->evaluatePolicy(game);
            initializeNodePriors(node, movePriors);
            continue;
        }
        node = selectBestChild(node);
        game.makeMove(node->move.x, node->move.y);
    }

    return node;
}

MCTS::Node* MCTS::expand(Node* node, PenteGame& game) {
    PROFILE_SCOPE("MCTS::expand");
    
    // if this is the first node being expanded, we need to allocate all children nodes and set priors
    if (node->childCount == 0) {
        int legalMoveCount = static_cast<int>(game.getLegalMoves().size());
        initNodeChildren(node, legalMoveCount);
        // movePriors - vector of pairs of moves and priors, ordered by prior, filtered to legal moves
        // value - static evaluation of the position
        auto [movePriors, value] = config_.evaluator->evaluate(game);

        initializeNodePriors(node, movePriors);

        node->value = value;
        node->state.store(NodeState::EXPANDED, std::memory_order_release);
        node->unprovenCount.store(static_cast<int16_t>(legalMoveCount), std::memory_order_relaxed);

    } else {
        // this should never happen right? 
        std::cout << "This should never happen: expanding a node that already has children allocated.\n";
        std::cerr << "Error: Attempting to expand a node that already has children allocated. This indicates a logic error in the MCTS implementation.\n";
        exit(1);
    }

    return node;
}

double MCTS::simulate(Node* node, PenteGame& game) {
    PROFILE_SCOPE("MCTS::simulate");

    if (node->value != 0.0f) {
        return node->value;
    }


    // Fallback: random rollout (when no evaluator, or evaluation returned 0)
    PenteGame simGame = game;

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
        // Backpropagate stats (relaxed: no ordering needed for counters)
        current->visits.fetch_add(1, std::memory_order_relaxed);
        current->wins.fetch_add((currentResult > 0) ? 1 : 0, std::memory_order_relaxed);

        // CAS loop for atomic double add (no fetch_add for double in C++17)
        double expected = current->totalValue.load(std::memory_order_relaxed);
        while (!current->totalValue.compare_exchange_weak(
            expected, expected + currentResult, std::memory_order_relaxed));

        // --- SOLVER LOGIC (Minimax Propagation) ---
        auto status = current->solvedStatus.load(std::memory_order_relaxed);
        if (status == SolvedStatus::SOLVED_WIN && current->parent) {
            current->parent->solvedStatus.store(SolvedStatus::SOLVED_LOSS, std::memory_order_relaxed);
        }

        if (status == SolvedStatus::SOLVED_LOSS && current->parent) {
            int16_t oldCount = current->parent->unprovenCount.fetch_sub(1, std::memory_order_relaxed);

            if (oldCount - 1 < 0) {
                std::cerr << "FATAL ERROR: unprovenCount dropped below 0!" << std::endl;
                exit(1);
            }

            if (oldCount == 1) { // was 1, now 0 — all children are losses
                current->parent->solvedStatus.store(SolvedStatus::SOLVED_WIN, std::memory_order_relaxed);
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


// allocate children priors. takes the movePriors from the evaluator and a node reference. void. used for lazy policy loading
void MCTS::initializeNodePriors(Node* node, const std::vector<std::pair<PenteGame::Move, float>>& movePriors) {
    // allocate children
    for (const auto& [move, prior] : movePriors) {
        Node* child = allocateNode();
        child->move = move;
        child->player = (node->player == PenteGame::BLACK)
            ? PenteGame::WHITE
            : PenteGame::BLACK;
        child->parent = node;
        child->prior = prior;

        node->children[node->childCount] = child;
        node->childCount++;
    }
}

MCTS::Node* MCTS::selectBestChild(Node* node) const {
    if (!node || node->childCount == 0) {
        return nullptr;
    }

    Node* bestChild = nullptr;
    double bestValue = -std::numeric_limits<double>::infinity();


    int32_t parentVisits = node->visits.load(std::memory_order_relaxed);

    if (config_.searchMode == SearchMode::UCB1) {
        double explorationFactor = config_.explorationConstant *
                                std::sqrt(std::log(static_cast<double>(parentVisits)));
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
            double value = child->getPUCTValue(explorationFactor, parentVisits);
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
    dest->solvedStatus.store(source->solvedStatus.load());
    dest->childCount = source->childCount;
    dest->childCapacity = source->childCapacity;
    dest->unprovenCount.store(source->unprovenCount.load());
    dest->visits.store(source->visits.load());
    dest->wins.store(source->wins.load());
    dest->totalValue.store(source->totalValue.load());
    dest->state.store(source->state.load());
    dest->parent = nullptr;  // Will be set by parent during recursion
    dest->children = nullptr; // Will be set below if there are children
    dest->prior = source->prior;
    dest->value = source->value;

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
    return root_ ? root_->visits.load() : 0;
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
    std::cout << "Total simulations: " << GameUtils::formatWithCommas(totalSimulations_)
              << ". Tree size: " << GameUtils::formatWithCommas(getTreeSize())
              << ". Root visits: " << GameUtils::formatWithCommas(getTotalVisits()) << "\n";

    std::cout << "Total search time: " << static_cast<int>(totalSearchTime_ / 60)
              << " min " << static_cast<int>(totalSearchTime_) % 60 << " sec\n";
    std::cout << "Simulations/second: " << std::fixed << std::setprecision(0)
              << (totalSearchTime_ > 0 ? totalSimulations_ / totalSearchTime_ : 0) << "\n";

    // Arena memory stats
    std::cout << "Arena memory: " << std::fixed << std::setprecision(1)
              << (arena_.bytesUsed() / (1024.0 * 1024.0)) << " MB / "
              << (arena_.totalSize() / (1024.0 * 1024.0)) << " MB ("
              << std::setprecision(1) << arena_.utilizationPercent() << "%)\n";

    double rootAvgValue = 0.0;
    if (root_) {
        int32_t rv = root_->visits.load(std::memory_order_relaxed);
        if (rv > 0) rootAvgValue = root_->totalValue.load(std::memory_order_relaxed) / rv;
    }
    auto rootSolvedStatus = root_ ? root_->solvedStatus.load(std::memory_order_relaxed) : SolvedStatus::UNSOLVED;
    std::cout << "Solved status: " <<
        (root_ ?
            (rootSolvedStatus == SolvedStatus::SOLVED_WIN ? "SOLVED_WIN - All moves lead to a loss" :
             rootSolvedStatus == SolvedStatus::SOLVED_LOSS ? "SOLVED_LOSS - At least one move leads to a win" :
             "Unsolved")
            : "N/A") << " And Root avg value: " << std::fixed << std::setprecision(3)
                  << rootAvgValue << "\n";

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
            auto sa = a->solvedStatus.load(std::memory_order_relaxed);
            auto sb = b->solvedStatus.load(std::memory_order_relaxed);
            if (sa == SolvedStatus::SOLVED_WIN && sb != SolvedStatus::SOLVED_WIN) {
                return true;
            }
            if (sa != SolvedStatus::SOLVED_WIN && sb == SolvedStatus::SOLVED_WIN) {
                return false;
            }

            return a->visits.load(std::memory_order_relaxed) > b->visits.load(std::memory_order_relaxed);
        });

    int movesConsidered = root_->childCount;
    int32_t rootVisits = root_->visits.load(std::memory_order_relaxed);
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves of "
              << movesConsidered << " Considered ===\n";
    std::cout << std::setw(6) << "Move"
              << std::setw(10) << "Visits"
              << std::setw(10) << "Wins"
              << std::setw(10) << "MoveEval"
              << std::setw(10) << "Prior"
              << std::setw(10) << "Avg Val"
              << std::setw(10) << "UCB1"
              << std::setw(10) << "PUCT"
              << std::setw(8) << "Status\n";
    std::cout << std::string(74, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        int32_t cv = child->visits.load(std::memory_order_relaxed);
        int32_t cw = child->wins.load(std::memory_order_relaxed);
        double ctv = child->totalValue.load(std::memory_order_relaxed);
        double avgScore = cv > 0 ? ctv / cv : 0.0;

        double explorationFactor = config_.explorationConstant *
                                   std::sqrt(std::log(static_cast<double>(rootVisits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, rootVisits);

        double moveEval = this->game.evaluateMove(child->move);

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

        auto childStatus = child->solvedStatus.load(std::memory_order_relaxed);
        std::string status;
        if (childStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (childStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(6) << moveStr
                  << std::setw(10) << cv
                  << std::setw(10) << cw
                  << std::setw(10) << static_cast<int>(moveEval)
                  << std::setw(10) << std::fixed << std::setprecision(3) << child->prior
                  << std::setw(10) << std::fixed << std::setprecision(3) << avgScore
                  << std::setw(10) << std::fixed << std::setprecision(3) << ucb1
                  << std::setw(10) << std::fixed << std::setprecision(3) << puct
                  << std::setw(8) << status
                  << "\n";
    }

    std::cout << std::string(74, '=') << "\n\n";
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
            auto sa = a->solvedStatus.load(std::memory_order_relaxed);
            auto sb = b->solvedStatus.load(std::memory_order_relaxed);
            if (sa == SolvedStatus::SOLVED_WIN && sb != SolvedStatus::SOLVED_WIN) {
                return true;
            }
            if (sa != SolvedStatus::SOLVED_WIN && sb == SolvedStatus::SOLVED_WIN) {
                return false;
            }
            return a->visits.load(std::memory_order_relaxed) > b->visits.load(std::memory_order_relaxed);
        });

    int32_t nodeVisits = node->visits.load(std::memory_order_relaxed);
    std::cout << "\n=== Top " << std::min(topN, (int)children.size()) << " Moves ===\n";
    std::cout << std::setw(6) << "Move"
              << std::setw(10) << "Visits"
              << std::setw(10) << "Avg Val"
              << std::setw(10) << "Prior"
              << std::setw(10) << "UCB1"
              << std::setw(10) << "PUCT"
              << std::setw(8) << "Status\n";
    std::cout << std::string(64, '-') << "\n";

    for (int i = 0; i < std::min(topN, (int)children.size()); i++) {
        Node* child = children[i];
        int32_t cv = child->visits.load(std::memory_order_relaxed);
        double avgValue = cv > 0 ? child->totalValue.load(std::memory_order_relaxed) / cv : 0.0;
        double explorationFactor = config_.explorationConstant *
                                   std::sqrt(std::log(static_cast<double>(nodeVisits)));
        double ucb1 = child->getUCB1Value(explorationFactor);
        double puct = child->getPUCTValue(config_.explorationConstant, nodeVisits);

        std::string moveStr = GameUtils::displayMove(child->move.x, child->move.y);

        auto childStatus = child->solvedStatus.load(std::memory_order_relaxed);
        std::string status;
        if (childStatus == SolvedStatus::SOLVED_WIN) {
            status = "WIN";
        } else if (childStatus == SolvedStatus::SOLVED_LOSS) {
            status = "LOSS";
        } else {
            status = "-";
        }

        std::cout << std::setw(6) << moveStr
                  << std::setw(10) << cv
                  << std::setw(10) << std::fixed << std::setprecision(3) << avgValue
                  << std::setw(10) << std::fixed << std::setprecision(3) << child->prior
                  << std::setw(10) << std::fixed << std::setprecision(3) << ucb1
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
    int32_t tv = targetNode->visits.load(std::memory_order_relaxed);
    double avgValue = tv > 0 ? targetNode->totalValue.load(std::memory_order_relaxed) / tv : 0.0;

    std::cout << "\n=== Analysis for move " << moveStr << " ===\n";
    std::cout << "Visits: " << tv << "\n";
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

// ============================================================================
// Parallel Search
// ============================================================================

PenteGame::Move MCTS::parallelSearch(const PenteGame& game,
                                     const ParallelConfig& pconfig) {
    this->game = game;
    auto startTime = std::chrono::high_resolution_clock::now();

    virtualLossWeight_ = pconfig.virtualLossWeight;

    clearTree();
    root_ = allocateNode();
    root_->player = game.getCurrentPlayer();

    // Shared queues
    ThreadSafeQueue<InferenceRequest> inferenceQueue;
    ThreadSafeQueue<InferenceResult> backpropQueue;

    // Control flags
    std::atomic<bool> stopFlag{false};
    std::atomic<int> iterationCount{0};

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < pconfig.numWorkers; ++i) {
        workers.emplace_back([this, i, &stopFlag, &iterationCount,
                             &inferenceQueue, &backpropQueue, &game, &pconfig]() {
            this->workerThread(i, stopFlag, iterationCount,
                             inferenceQueue, backpropQueue, game, pconfig);
        });
    }

    // Launch inference thread (only for batched NN mode)
    std::thread infThread;
    if (pconfig.useInferenceThread && config_.evaluator) {
        infThread = std::thread([this, &stopFlag, &inferenceQueue,
                                &backpropQueue, &pconfig]() {
            this->inferenceThread(stopFlag, inferenceQueue, backpropQueue,
                                pconfig.batchSize, pconfig.batchTimeoutMs);
        });
    }

    // Wait for iterations to complete or root to be solved
    int lastCount = 0;
    int stallTicks = 0;
    while (iterationCount.load(std::memory_order_relaxed) < config_.maxIterations
           && root_->solvedStatus.load(std::memory_order_relaxed) == SolvedStatus::UNSOLVED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int currentCount = iterationCount.load(std::memory_order_relaxed);
        if (currentCount == lastCount) {
            stallTicks++;
            if (stallTicks >= 200) { // 2 seconds with no progress
                std::cerr << "WATCHDOG: Search stalled at " << currentCount
                          << "/" << config_.maxIterations << " iterations\n";
                break;
            }
        } else {
            stallTicks = 0;
            lastCount = currentCount;
        }
    }

    // Signal threads to stop and join
    stopFlag.store(true, std::memory_order_relaxed);
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    if (infThread.joinable()) infThread.join();

    auto endTime = std::chrono::high_resolution_clock::now();
    totalSearchTime_ = std::chrono::duration<double>(endTime - startTime).count();
    totalSimulations_ = iterationCount.load();

    return getBestMove();
}

// ============================================================================
// Worker Thread
// ============================================================================

void MCTS::workerThread(
    int threadId,
    std::atomic<bool>& stopFlag,
    std::atomic<int>& iterationCount,
    ThreadSafeQueue<InferenceRequest>& inferenceQueue,
    ThreadSafeQueue<InferenceResult>& backpropQueue,
    const PenteGame& rootGame,
    const ParallelConfig& pconfig)
{
    // print startup message with thread ID
    std::string threadMsg = "Worker thread " + std::to_string(threadId) + " started. \n";
    std::cout << threadMsg;

    PenteGame localGame;

    while (!stopFlag.load(std::memory_order_relaxed)
           && iterationCount.load(std::memory_order_relaxed) < config_.maxIterations) {
        // PHASE 1: Process pending inference results (priority)
        InferenceResult result;
        if (backpropQueue.tryPop(result)) {
            iterationCount.fetch_add(1, std::memory_order_relaxed);
            { PROFILE_SCOPE("worker::expand");
              expandParallel(result.node, result.movePriors, result.value); }
            { PROFILE_SCOPE("worker::backprop");
              backpropWithVirtualLoss(result.node, result.value, virtualLossWeight_); }
            continue;
        }

        // PHASE 2: Selection - traverse tree to find leaf
        { PROFILE_SCOPE("worker::syncFrom");
          localGame.syncFrom(rootGame); }

        Node* leaf;
        { PROFILE_SCOPE("worker::select");
          leaf = selectParallel(root_, localGame); }
        if (!leaf) continue;

        // Check for terminal game state or solver-proved node
        PenteGame::Player winner = localGame.getWinner();
        if (winner != PenteGame::NONE || leaf->isTerminal()) {
            PROFILE_SCOPE("worker::terminal");
            iterationCount.fetch_add(1, std::memory_order_relaxed);
            double terminalResult;
            if (winner != PenteGame::NONE) {
                // Game is actually over
                PenteGame::Player opponent = (leaf->player == PenteGame::BLACK)
                                            ? PenteGame::WHITE : PenteGame::BLACK;
                SolvedStatus terminalStatus = (winner == opponent)
                                             ? SolvedStatus::SOLVED_WIN : SolvedStatus::SOLVED_LOSS;
                solveNode(leaf, terminalStatus);
                terminalResult = (winner == opponent) ? 1.0 : -1.0;
            } else {
                // Proved via solver propagation (minimax)
                auto status = leaf->solvedStatus.load(std::memory_order_relaxed);
                terminalResult = (status == SolvedStatus::SOLVED_WIN) ? 1.0 : -1.0;
            }
            backpropWithVirtualLoss(leaf, terminalResult, virtualLossWeight_);
            continue;
        }

        // Try to claim this leaf for expansion (CAS: UNEXPANDED → EXPANDING)
        // acq_rel on success (publish ownership), relaxed on failure (just retry)
        NodeState expected = NodeState::UNEXPANDED;
        if (!leaf->state.compare_exchange_strong(expected, NodeState::EXPANDING,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            // Another thread owns this leaf — cancel VL and retry
            PROFILE_SCOPE("worker::casFail");
            cancelVirtualLoss(leaf, virtualLossWeight_);
            continue;
        }

        // Committed to productive work — count the iteration
        iterationCount.fetch_add(1, std::memory_order_relaxed);

        // PHASE 3: Evaluate position (we own this leaf)
        if (pconfig.useInferenceThread) {
            // Queue for batched inference - virtual losses stay in place
            InferenceRequest req;
            req.node = leaf;
            req.gameState = localGame;
            req.threadId = threadId;
            inferenceQueue.push(std::move(req));
        } else {
            // Inline evaluation (CPU-only mode)
            std::vector<std::pair<PenteGame::Move, float>> movePriors;
            float value;
            { PROFILE_SCOPE("worker::evaluate");
              auto result = config_.evaluator->evaluate(localGame);
              movePriors = std::move(result.first);
              value = result.second; }
            { PROFILE_SCOPE("worker::expand");
              expandParallel(leaf, movePriors, value); }
            double simResult = static_cast<double>(value);
            if (value == 0.0f) {
                PROFILE_SCOPE("worker::rollout");
                simResult = rollout(localGame);
            }
            { PROFILE_SCOPE("worker::backprop");
              backpropWithVirtualLoss(leaf, simResult, virtualLossWeight_); }
        }
    }
}

// ============================================================================
// Inference Thread
// ============================================================================

void MCTS::inferenceThread(
    std::atomic<bool>& stopFlag,
    ThreadSafeQueue<InferenceRequest>& inferenceQueue,
    ThreadSafeQueue<InferenceResult>& backpropQueue,
    int batchSize,
    int batchTimeoutMs)
{
    std::vector<InferenceRequest> batch;
    batch.reserve(batchSize);

    while (!stopFlag.load(std::memory_order_relaxed)) {
        batch.clear();

        // Collect a batch (wait for first item, then drain what's available)
        InferenceRequest first;
        if (!inferenceQueue.waitPop(first, batchTimeoutMs)) {
            continue; // Timeout, check stopFlag
        }
        batch.push_back(std::move(first));
        inferenceQueue.drainTo(batch, batchSize - 1);

        // Evaluate each position and push results
        // TODO: Replace with batched NN inference when available
        for (auto& req : batch) {
            auto [movePriors, value] = config_.evaluator->evaluate(req.gameState);
            float finalValue = value;
            if (value == 0.0f) {
                finalValue = static_cast<float>(rollout(req.gameState));
            }

            InferenceResult result;
            result.node = req.node;
            result.movePriors = std::move(movePriors);
            result.value = finalValue;
            backpropQueue.push(std::move(result));
        }
    }
}

// ============================================================================
// Parallel MCTS Phases
// ============================================================================

MCTS::Node* MCTS::selectParallel(Node* node, PenteGame& game) {
    addVirtualLoss(node, virtualLossWeight_);

    while (true) {
        if (node->isTerminal()) {
            return node;
        }

        if (!node->isFullyExpanded()) {
            return node;
        }

        // Lazy policy load (heuristic mode)
        if (node->childCount == 0 && config_.heuristicMode == HeuristicMode::HEURISTIC) {
            // Evaluate policy OUTSIDE the lock (expensive ~12us computation)
            std::vector<std::pair<PenteGame::Move, float>> movePriors =
                config_.evaluator->evaluatePolicy(game);
            // Lightweight lock only for the install step (pointer writes)
            std::lock_guard<std::mutex> lock(lazyPolicyMutex_);
            if (node->childCount == 0 && !movePriors.empty()) {
                initNodeChildren(node, static_cast<int>(movePriors.size()));
                initializeNodePriors(node, movePriors);
            }
        }

        Node* bestChild = selectBestChild(node);
        if (!bestChild) {
            return node;
        }

        addVirtualLoss(bestChild, virtualLossWeight_);
        game.makeMove(bestChild->move.x, bestChild->move.y);
        node = bestChild;
    }
}

void MCTS::expandParallel(
    Node* node,
    const std::vector<std::pair<PenteGame::Move, float>>& movePriors,
    float value)
{
    // No lock needed: CAS (UNEXPANDED→EXPANDING) ensures single-writer,
    // and arena allocator is now lock-free.

    node->value = value;

    if (!movePriors.empty()) {
        int count = static_cast<int>(movePriors.size());
        node->children = arena_.allocate<Node*>(count);
        if (!node->children) {
            std::cerr << "FATAL: Arena out of memory during parallel expand!\n";
            throw std::bad_alloc();
        }
        node->childCapacity = static_cast<uint16_t>(count);

        for (int i = 0; i < count; ++i) {
            Node* child = arena_.allocate<Node>();
            if (!child) {
                std::cerr << "FATAL: Arena out of memory for child node!\n";
                throw std::bad_alloc();
            }
            new (child) Node();
            child->move = movePriors[i].first;
            child->prior = movePriors[i].second;
            child->parent = node;
            child->player = (node->player == PenteGame::BLACK)
                           ? PenteGame::WHITE : PenteGame::BLACK;
            node->children[i] = child;
        }

        node->childCount = static_cast<uint16_t>(count);
        node->unprovenCount.store(static_cast<int16_t>(count), std::memory_order_relaxed);
    }

    // Release: ensures all child writes are visible before state=EXPANDED
    node->state.store(NodeState::EXPANDED, std::memory_order_release);
}

void MCTS::backpropWithVirtualLoss(Node* node, double result, int vlWeight) {
    Node* current = node;
    double value = result;

    while (current != nullptr) {
        // Remove virtual loss (was added during selection)
        removeVirtualLoss(current, vlWeight);

        // Record real visit and stats
        current->visits.fetch_add(1, std::memory_order_relaxed);
        current->wins.fetch_add((value > 0) ? 1 : 0, std::memory_order_relaxed);

        // CAS loop for atomic double add (C++17)
        double expected = current->totalValue.load(std::memory_order_relaxed);
        while (!current->totalValue.compare_exchange_weak(
            expected, expected + value, std::memory_order_relaxed));

        // Flip value for parent (opponent's perspective)
        value = -value;
        current = current->parent;
    }
}

void MCTS::addVirtualLoss(Node* node, int weight) {
    node->visits.fetch_add(weight, std::memory_order_relaxed);
    node->virtualLoss.fetch_add(weight, std::memory_order_relaxed);
}

void MCTS::removeVirtualLoss(Node* node, int weight) {
    node->visits.fetch_sub(weight, std::memory_order_relaxed);
    node->virtualLoss.fetch_sub(weight, std::memory_order_relaxed);
}

void MCTS::cancelVirtualLoss(Node* node, int weight) {
    // Remove virtual loss from entire selection path without adding real visits.
    // Used when a wasted selection finds an already-expanded leaf.
    Node* current = node;
    while (current != nullptr) {
        removeVirtualLoss(current, weight);
        current = current->parent;
    }
}

// ============================================================================
// Parallel Solver Propagation
// ============================================================================

void MCTS::solveNode(Node* node, SolvedStatus status) {
    // CAS ensures each node is solved at most once — no double-counting
    SolvedStatus expected = SolvedStatus::UNSOLVED;
    if (!node->solvedStatus.compare_exchange_strong(
            expected, status, std::memory_order_relaxed)) {
        return; // Already solved by another thread
    }

    if (!node->parent) return;

    if (status == SolvedStatus::SOLVED_WIN) {
        // This move is a proven win → parent position is a loss
        solveNode(node->parent, SolvedStatus::SOLVED_LOSS);
    } else if (status == SolvedStatus::SOLVED_LOSS) {
        // This move is a proven loss → one fewer unproven child for parent
        int16_t oldCount = node->parent->unprovenCount.fetch_sub(1, std::memory_order_relaxed);
        if (oldCount == 1) {
            // All children are now losses → parent position is a win
            solveNode(node->parent, SolvedStatus::SOLVED_WIN);
        }
    }
}

// ============================================================================
// Rollout (Thread-Safe)
// ============================================================================

double MCTS::rollout(PenteGame& game) const {
    PenteGame::Player startPlayer = game.getCurrentPlayer();
    PenteGame::Player winner = PenteGame::NONE;
    int depth = 0;

    while ((winner = game.getWinner()) == PenteGame::NONE
           && depth < config_.maxSimulationDepth) {
        PenteGame::Move move = game.getRandomLegalMove();
        game.makeMove(move.x, move.y);
        depth++;
    }

    if (winner != PenteGame::NONE) {
        double result = evaluateTerminalState(game, depth);
        return result * ((winner == startPlayer) ? -1.0 : 1.0);
    }
    return 0.0;
}
