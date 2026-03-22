#include "MCTS.hpp"
#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include "Profiler.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

// ============================================================================
// Node Implementation
// ============================================================================


double MCTS::Node::getPUCTValue(double explorationFactor, double sqrtParentVisits, float prior) const {
    PROFILE_SCOPE("MCTS::Node::getPUCTValue");
    if (this->solvedStatus == SolvedStatus::SOLVED_WIN) {
        return std::numeric_limits<double>::infinity();
    }
    if (this->solvedStatus == SolvedStatus::SOLVED_LOSS) {
        return -std::numeric_limits<double>::infinity();
    }

    double exploitation = (this->visits == 0) ? 0.0 : this->totalValue / this->visits;
    double exploration = explorationFactor * prior * sqrtParentVisits / (1.0 + this->visits);

    return exploitation + exploration;
}

// ============================================================================
// MCTS Constructor/Destructor
// ============================================================================

MCTS::MCTS(const Config &config)
    : config_(config), arena_(config.arenaSize), root_(nullptr),
      rng_(config.seed ? config.seed : std::random_device{}()), totalSimulations_(0), totalSearchTime_(0.0) {}

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
    PROFILE_SCOPE("MCTS::initNodeChildren");
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

    std::vector<Node *> searchPath;
    searchPath.reserve(400); // 19x19=361 max moves in a game + a few for caps. 400 to be safe

    // Reset statistics and arena
    // reset();
    // clearTree();

    // Initialize root node if a fresh sim
    if (!root_) {
        root_ = allocateNode();
    }
    root_->player = game.getCurrentPlayer();

    // Local copy of game for simulations - inherit seed from MCTS config for reproducibility
    PenteGame::Config localGameConfig = game.getConfig();
    localGameConfig.seed = config_.seed;
    PenteGame localGame(localGameConfig);

    // Run MCTS iterations
    for (int i = 0; i < config_.maxIterations; i++) {
        searchPath.clear();
        searchPath.push_back(root_);

        if (root_->solvedStatus != SolvedStatus::UNSOLVED) {
            // std::cout << "Root node solved status: "
            //           << (root_->solvedStatus == SolvedStatus::SOLVED_WIN ? "WIN" : "LOSS") << " after " << i
            //           << " iterations.\n";
            break;
        }

        // Reset localGame to the start state of THIS search
        localGame.syncFrom(game);

        // Selection: traverse tree to find node to expand
        Node *node = select(root_, localGame, searchPath);

        // Expansion: add a new child node if not terminal
        PenteGame::Player winner = localGame.getWinner();
        if (winner != PenteGame::NONE) {
            PenteGame::Player opponent = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
            // if the game is over. that means the last move made was a winning move.
            // which means the current player lost. which means this move was a "solve win" for the opponent.
            if (winner == opponent) {
                node->solvedStatus = SolvedStatus::SOLVED_WIN;
                node->unprovenCount = 0;
                backpropagate(node, 1.0, searchPath);
            } else {
                node->solvedStatus = SolvedStatus::SOLVED_LOSS;
                node->unprovenCount = 0;
                backpropagate(node, -1.0, searchPath);
            }
            totalSimulations_++;
            continue;
        }
        node = expand(node, localGame);

        // Simulation: play out the game randomly, only if not already solved
        double result = simulate(node, localGame);

        // Backpropagation: update statistics
        backpropagate(node, result, searchPath);

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
            return root_->moves[i];
        }
    }

    // Select child with most visits (most robust), excluding proven losses
    int bestIndex = -1;
    int maxVisits = -1;

    for (int i = 0; i < root_->childCapacity; i++) {
        Node *child = root_->children[i];
        if (!child)
            continue;
        if (child->solvedStatus != SolvedStatus::SOLVED_LOSS && child->visits > maxVisits) {
            maxVisits = child->visits;
            bestIndex = i;
        }
    }

    // Fallback: If all are losses, just pick the one with the most visits
    if (bestIndex == -1) {
        for (int i = 0; i < root_->childCapacity; i++) {
            Node *child = root_->children[i];
            if (!child)
                continue;
            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                bestIndex = i;
            }
        }
    }

    PenteGame::Move bestMove = root_->moves[bestIndex];

    // Un-rotate from canonical to physical coords if root was expanded in canonical mode
    if (root_->canonicalSym >= 0) {
        int rootSym = 0;
        this->game.getCanonicalHash(rootSym);
        int px, py;
        Zobrist::instance().applyInverseSym(rootSym, bestMove.x, bestMove.y, px, py);
        return PenteGame::Move(px, py);
    }

    return bestMove;
}

// ============================================================================
// MCTS Phases
// ============================================================================

MCTS::Node *MCTS::select(Node *node, PenteGame &game, std::vector<Node *> &searchPath) {
    PROFILE_SCOPE("MCTS::select");

    while (node->evaluated && !node->isTerminal()) {
        // Always recompute currentSym from the live game state — never use node->canonicalSym,
        // since transposition nodes may be reached from different orientations.
        // Compute once here and pass to selectBestMoveIndex to avoid a second getCanonicalHash call.
        int currentSym = -1;
        if (node->canonicalSym >= 0) {
            game.getCanonicalHash(currentSym);
        }

        int best = selectBestMoveIndex(node, game, currentSym);
        PenteGame::Move canonMove = node->moves[best];
        PenteGame::Move physMove = canonMove;

        assert(node->moves != nullptr);

        // Un-rotate canonical moves to physical coords before making the move.
        if (currentSym >= 0) {
            const auto &zob = Zobrist::instance();
            int px, py;
            zob.applyInverseSym(currentSym, canonMove.x, canonMove.y, px, py);
            physMove = PenteGame::Move(px, py);
        }

        assert(physMove.x >= 0 && physMove.x < PenteGame::BOARD_SIZE);
        assert(physMove.y >= 0 && physMove.y < PenteGame::BOARD_SIZE);

        game.makeMove(physMove.x, physMove.y);

        Node *child = node->children[best];

        // Use canonical hash as TT key when within the depth limit
        bool useCanonical = (config_.canonicalHashDepth > 0 &&
                             game.getMoveCount() <= config_.canonicalHashDepth);

        // When a canonical parent (canonicalSym >= 0) has a non-canonical child
        // (useCanonical == false), the same canonical move slot [best] maps to different
        // physical states depending on which physical orientation reached this parent.
        // children[best] may point to a node from a different orientation — always TT-lookup.
        bool needsTTLookup = (!child) || (node->canonicalSym >= 0 && !useCanonical);

        if (needsTTLookup) {
            int childSym = -1;
            uint64_t hash = useCanonical ? game.getCanonicalHash(childSym) : game.getHash();

            auto it = nodeTranspositionTable.find(hash);
            bool foundInTable = (it != nodeTranspositionTable.end());

            if (!foundInTable) {
                // lazy expansion
                child = allocateNode();
                child->move = canonMove;
                child->player = (node->player == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
                nodeTranspositionTable[hash] = child;
            } else {
                child = it->second;
            }

            child->positionHash = hash;
            bool wasNull = (node->children[best] == nullptr);
            node->children[best] = child;
            if (wasNull) node->childCount++;
        }

        assert(child != nullptr);

        node = child;
        searchPath.push_back(node);
    }
    return node;
}

MCTS::Node *MCTS::expand(Node *node, PenteGame &game) {
    PROFILE_SCOPE("MCTS::expand");

    if (node->solvedStatus != SolvedStatus::UNSOLVED) {
        return node;
    }

    assert(node->expanded == false);
    assert(node->evaluated == false);
    assert(node->childCount == 0);

    // this being the first node being expanded, we need to allocate all children nodes and set priors

    // Cap branching factor to the number of promising moves (cells near existing stones).
    // All legal moves are still valid candidates for the evaluator, but PUCT only considers this many.
    int childCapacity = static_cast<int>(game.getLegalMoves().size());

    // Determine whether to store moves in canonical coordinates
    bool useCanonical = (config_.canonicalHashDepth > 0 &&
                            game.getMoveCount() <= config_.canonicalHashDepth);
    int canonSym = -1;
    if (useCanonical) {
        game.getCanonicalHash(canonSym);
    }

    // policy - vector of pairs of moves and priors, ordered by prior, filtered to legal moves.
    // std::vector<std::pair<Move, float>> policy value - static evaluation of the position
    auto [policy, value] = config_.evaluator->evaluate(game);
    initNodeChildren(node, childCapacity);

    // Arena-allocate moves and priors arrays, then populate from policy
    node->moves = arena_.allocate<PenteGame::Move>(childCapacity);
    assert(node->moves != nullptr);
    node->priors = arena_.allocate<float>(childCapacity);
    // Mark all priors as unset (-1.0f sentinel). Policy loop overwrites the ones we have data for.
    std::fill(node->priors, node->priors + childCapacity, -1.0f);
    // We allocate the full size, even if the policy returns empty. In selection we can do a lazy policy calc.
    int policyCount = static_cast<int>(policy.size());

    // assert(policyCount == childCapacity); // lazy loading sets policy to 0

    for (int i = 0; i < policyCount; i++) {
        node->moves[i] = policy[i].first;

        // assert moves are legal, between 0 and board size
        assert(node->moves[i].x >= 0 && node->moves[i].x < PenteGame::BOARD_SIZE);
        assert(node->moves[i].y >= 0 && node->moves[i].y < PenteGame::BOARD_SIZE);

        node->priors[i] = policy[i].second;
    }

    // Rotate physical moves to canonical coordinates so transpositions share the same move list
    if (useCanonical) {
        const auto &zob = Zobrist::instance();
        for (int i = 0; i < policyCount; i++) {
            int cx, cy;
            zob.applySymToMove(canonSym, node->moves[i].x, node->moves[i].y, cx, cy);
            node->moves[i] = PenteGame::Move(cx, cy);
        }
        node->canonicalSym = static_cast<int8_t>(canonSym);
    }

    node->value = value;
    node->expanded = true;
    node->evaluated = true;
    node->unprovenCount = childCapacity;


    return node;
}

double MCTS::simulate(Node *node, PenteGame &game) {
    PROFILE_SCOPE("MCTS::simulate");
    return node->value;
}

void MCTS::backpropagate(Node *node, double result, std::vector<Node *> &searchPath) {
    PROFILE_SCOPE("MCTS::backpropagate");
    double currentResult = result;

    // assert currnet node is on top of the "stack" search path
    assert(!searchPath.empty());

    Node *current = searchPath.back();
    searchPath.pop_back();

    assert(current == node);

    Node *parent;
    // parent pointer no longer available

    while (true) {

        // Backpropagate stats
        current->visits++;
        current->totalValue += currentResult;
        current->wins += (currentResult > 0) ? 1 : 0;

        if (!searchPath.empty()) {
            parent = searchPath.back();
            searchPath.pop_back();
        } else {
            break;
        }

        // --- SOLVER LOGIC (Minimax Propagation) ---
        if (current->solvedStatus == SolvedStatus::SOLVED_WIN) {
            parent->solvedStatus = SolvedStatus::SOLVED_LOSS;
        }

        if (current->solvedStatus == SolvedStatus::SOLVED_LOSS) {
            parent->unprovenCount--;

            assert(parent->unprovenCount >= 0);

            if (parent->unprovenCount == 0) {
                parent->solvedStatus = SolvedStatus::SOLVED_WIN;
            }
        }

        // Flip result for parent (opponent's perspective)
        currentResult = currentResult * -1.0;
        current = parent;
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

int MCTS::selectBestMoveIndex(Node *node, const PenteGame &game, int currentSym) const {
    PROFILE_SCOPE("MCTS::selectBestMoveIndex");
    assert(node != nullptr && node->childCapacity > 0);
    assert(config_.searchMode == SearchMode::PUCT);
    // Note: positionHash may be canonical when canonicalHashDepth > 0, so we skip the hash assert

    // Lazy policy load: -1.0f sentinel means priors weren't populated during expand
    if (node->priors[0] < 0.0f) {
        PROFILE_SCOPE("MCTS::selectBestMoveIndex::LazyPolicyLoad");
        std::vector<std::pair<PenteGame::Move, float>> movePriors = config_.evaluator->evaluatePolicy(game);

        int priorsSize = static_cast<int>(movePriors.size());
        int childCapSize = static_cast<int>(node->childCapacity);
        // Policy may return more moves than childCapacity (e.g. all legal vs capped promising).
        // Take only the first childCapacity moves; evaluatePolicy sorts best-first so these are top-K.
        int count = std::min(priorsSize, childCapSize);

        for (int i = 0; i < count; i++) {
            node->moves[i] = movePriors[i].first;
            node->priors[i] = movePriors[i].second;
        }

        // evaluatePolicy returns physical moves; rotate to canonical if node uses canonical coords
        // currentSym was already computed by the caller (select), so no need to call getCanonicalHash again.
        if (currentSym >= 0) {
            // assert(currentSym == node->canonicalSym); not true since transposition nodes can be reached from different orientations
            const auto &zob = Zobrist::instance();
            for (int i = 0; i < node->childCapacity; i++) {
                int cx, cy;
                zob.applySymToMove(currentSym, node->moves[i].x, node->moves[i].y, cx, cy);
                node->moves[i] = PenteGame::Move(cx, cy);
            }
        }
    } else {
        PROFILE_SCOPE("MCTS::selectBestMoveIndex::PolicyAlreadySet");
    }

    // base case
    if (node->nextPriorIdx == 0 && node->children[0] == nullptr) {
        return 0;
    }

    const int cap = (int)node->childCapacity;
    const double explorationFactor = config_.explorationConstant;
    const double sqrtN = std::sqrt((double)node->visits);

    // nextPriorIdx is the last expanded child index and priors are in order from highest to lowest
    int lastExpanded = std::min((int)node->nextPriorIdx, cap - 1);
    assert(lastExpanded >= 0);

    // --- 1) Best among expanded children ---
    int bestIndex = -1;
    double bestValue = -std::numeric_limits<double>::infinity();
    for (uint16_t i = 0; i <= lastExpanded; i++) {
        assert(node->children[i] != nullptr && "Expanded index must have a child node");

        Node *child = node->children[i];
        // Full PUCT score uses child's stats (N,W/Q) + this action's prior
        double score = child->getPUCTValue(explorationFactor, sqrtN, node->priors[i]);

        if (score > bestValue) {
            bestValue = score;
            bestIndex = i;
        }
    }

    // --- 2) Compare against frontier candidate (unexpanded) ---
    const int frontier = lastExpanded + 1;
    if (frontier < cap) {
        // Must be unexpanded per invariant
        assert(node->children[frontier] == nullptr);
        assert(node->visits > 0);

        // For an unexpanded action: N_a=0, Q=0 (no child stats yet)
        // exploitation = 0.0 since Q=0
        const double prior = node->priors[frontier];
        if (prior <= 0.0f && bestIndex != -1) {
            return bestIndex;
        }
        const double frontierScore = explorationFactor * node->priors[frontier] * sqrtN; // /(1+0)

        if (frontierScore > bestValue) {
            // Select the frontier; caller should expand it right after.
            bestIndex = frontier;
            bestValue = frontierScore;
            node->nextPriorIdx = frontier; // Move the frontier forward
        }
    }

    // if -1 still then all are proven losses,
    // this could happen if there is symmetry
    // and that one node represents more than 1 move but still only decrements the unproven count by 1
    if (bestIndex == -1) {
        // std::cerr << "WARNING: All child nodes are proven losses. Selecting the first unproven move as a last
        // resort.\n";
        assert(node->unprovenCount >= 1);
        assert(node->childCount == node->childCapacity);
        assert(node->children[0] != nullptr);
        node->unprovenCount = 1; // last node to get sent through.
        // slightly redundant since we already know this node is a proven loss...
        return 0;
    }

    return bestIndex;
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
    reusePath.clear();
    nodeTranspositionTable.clear();
}

void MCTS::reuseSubtree(const PenteGame::Move &move) {
    if (!root_) {
        return;
    }

    // Find child matching the move by comparing the parent's moves[] array,
    // not child->move, since transposition nodes may carry a move from a different parent context.
    // root_->moves[] may be in canonical coords, so convert the physical move first.
    PenteGame::Move searchMove = move;
    if (root_->canonicalSym >= 0) {
        int rootSym = 0;
        this->game.getCanonicalHash(rootSym);
        int cx, cy;
        Zobrist::instance().applySymToMove(rootSym, move.x, move.y, cx, cy);
        searchMove = PenteGame::Move(cx, cy);
    }

    Node *matchingChild = nullptr;
    for (int i = 0; i < root_->childCapacity; i++) {
        if (root_->moves[i].x == searchMove.x && root_->moves[i].y == searchMove.y) {
            matchingChild = root_->children[i]; // may be null if not yet visited
            break;
        }
    }

    if (!matchingChild) {
        // Move not in tree, clear everything
        clearTree();
        return;
    }

    reusePath.push_back(root_);
    root_ = matchingChild;
}

bool MCTS::undoSubtree() {
    if (reusePath.empty()) {
        return false;
    }
    root_ = reusePath.back();
    reusePath.pop_back();
    return true;
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

    int simsThisSearch = totalSimulations_ - startSimulations_;
    std::cout << "Search time: " << static_cast<int>(totalSearchTime_ / 60) << " min "
              << static_cast<int>(totalSearchTime_) % 60 << " sec (" << std::fixed << std::setprecision(0)
              << GameUtils::formatWithCommas(totalSearchTime_ > 0 ? simsThisSearch / totalSearchTime_ : 0) << " sims/sec)\n";

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

    struct MoveInfo {
        std::string moveStr;
        int visits;
        int wins;
        int moveEval;
        float prior;
        double avgVal;
        double puct;
        SolvedStatus solvedStatus;
    };

    std::vector<MoveInfo> moves;
    moves.reserve(root_->childCapacity);

    // Precompute physical-coord translation for canonical root moves
    int rootSym = -1;
    if (root_->canonicalSym >= 0) {
        this->game.getCanonicalHash(rootSym);
    }

    for (int i = 0; i < root_->childCapacity; i++) {
        Node *child = root_->children[i];
        if (!child)
            continue;

        // Convert canonical move back to physical coords for display/eval
        PenteGame::Move physMove = root_->moves[i];
        if (rootSym >= 0) {
            int px, py;
            Zobrist::instance().applyInverseSym(rootSym, physMove.x, physMove.y, px, py);
            physMove = PenteGame::Move(px, py);
        }

        MoveInfo info;
        info.moveStr = GameUtils::displayMove(physMove.x, physMove.y);
        info.visits = child->visits;
        info.wins = child->wins;
        info.moveEval = static_cast<int>(this->game.evaluateMove(physMove));
        info.prior = root_->priors[i];
        info.avgVal = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        info.puct = child->getPUCTValue(config_.explorationConstant, std::sqrt((double)root_->visits), root_->priors[i]);
        info.solvedStatus = child->solvedStatus;
        moves.push_back(info);
    }

    std::sort(moves.begin(), moves.end(), [](const MoveInfo &a, const MoveInfo &b) {
        if (a.solvedStatus == SolvedStatus::SOLVED_WIN && b.solvedStatus != SolvedStatus::SOLVED_WIN)
            return true;
        if (a.solvedStatus != SolvedStatus::SOLVED_WIN && b.solvedStatus == SolvedStatus::SOLVED_WIN)
            return false;
        return a.visits > b.visits;
    });

    int movesConsidered = static_cast<int>(moves.size());
    std::cout << "\n=== Top " << std::min(topN, movesConsidered) << " Moves of " << movesConsidered
              << " Considered ===\n";
    std::cout << std::setw(6) << "Move" << std::setw(10) << "Visits" << std::setw(10) << "Wins" << std::setw(10)
              << "MoveEval" << std::setw(10) << "Prior" << std::setw(10) << "Avg Val" 
              << std::setw(10) << "PUCT" << std::setw(10) << "Status\n";
    std::cout << std::string(86, '-') << "\n";

    for (int i = 0; i < std::min(topN, movesConsidered); i++) {
        const auto &m = moves[i];
        const char *status = m.solvedStatus == SolvedStatus::SOLVED_WIN    ? "WIN"
                             : m.solvedStatus == SolvedStatus::SOLVED_LOSS ? "LOSS"
                                                                           : "-";

        std::cout << std::setw(6) << m.moveStr << std::setw(10) << m.visits << std::setw(10) << m.wins << std::setw(10)
                  << m.moveEval << std::setw(10) << std::fixed << std::setprecision(3) << m.prior << std::setw(10)
                  << std::fixed << std::setprecision(3) << m.avgVal << std::setw(10) << std::fixed
                  << std::setw(10) << std::fixed << std::setprecision(3) << m.puct
                  << std::setw(10) << status << "\n";
    }

    std::cout << std::string(86, '=') << "\n\n";
}

MCTS::Node *MCTS::findChildNode(MCTS::Node *parent, int x, int y) const {
    if (!parent) {
        return nullptr;
    }

    // parent->moves[] may be in canonical coords; convert input physical coords to match.
    int sx = x, sy = y;
    if (parent->canonicalSym >= 0) {
        int rootSym = 0;
        this->game.getCanonicalHash(rootSym);
        Zobrist::instance().applySymToMove(rootSym, x, y, sx, sy);
    }

    for (int i = 0; i < parent->childCapacity; i++) {
        if (parent->moves[i].x == sx && parent->moves[i].y == sy) {
            return parent->children[i];
        }
    }

    return nullptr;
}

void MCTS::printMovesFromNode(MCTS::Node *node, int topN) const {
    if (!node || node->childCapacity == 0) {
        std::cout << "No moves analyzed for this position.\n";
        return;
    }

    struct MoveInfo {
        std::string moveStr;
        int visits;
        double avgVal;
        float prior;
        double puct;
        SolvedStatus solvedStatus;
    };

    std::vector<MoveInfo> moves;
    moves.reserve(node->childCapacity);

    for (int i = 0; i < node->childCapacity; i++) {
        Node *child = node->children[i];
        if (!child)
            continue;

        // Convert canonical move back to physical coords for display
        PenteGame::Move physMove = node->moves[i];
        if (node->canonicalSym >= 0) {
            int px, py;
            Zobrist::instance().applyInverseSym(node->canonicalSym, physMove.x, physMove.y, px, py);
            physMove = PenteGame::Move(px, py);
        }

        MoveInfo info;
        info.moveStr = GameUtils::displayMove(physMove.x, physMove.y);
        info.visits = child->visits;
        info.avgVal = child->visits > 0 ? child->totalValue / child->visits : 0.0;
        info.prior = node->priors[i];
        info.puct = child->getPUCTValue(config_.explorationConstant, std::sqrt((double)node->visits), node->priors[i]);
        info.solvedStatus = child->solvedStatus;
        moves.push_back(info);
    }

    std::sort(moves.begin(), moves.end(), [](const MoveInfo &a, const MoveInfo &b) {
        if (a.solvedStatus == SolvedStatus::SOLVED_WIN && b.solvedStatus != SolvedStatus::SOLVED_WIN)
            return true;
        if (a.solvedStatus != SolvedStatus::SOLVED_WIN && b.solvedStatus == SolvedStatus::SOLVED_WIN)
            return false;
        return a.visits > b.visits;
    });

    int movesConsidered = static_cast<int>(moves.size());
    std::cout << "\n=== Top " << std::min(topN, movesConsidered) << " Moves ===\n";
    std::cout << std::setw(6) << "Move" << std::setw(10) << "Visits" << std::setw(10) << "Avg Val" << std::setw(10)
              << "Prior" << std::setw(10) << "PUCT" << std::setw(10) << "Status\n";
    std::cout << std::string(66, '-') << "\n";

    for (int i = 0; i < std::min(topN, movesConsidered); i++) {
        const auto &m = moves[i];
        const char *status = m.solvedStatus == SolvedStatus::SOLVED_WIN    ? "WIN"
                             : m.solvedStatus == SolvedStatus::SOLVED_LOSS ? "LOSS"
                                                                           : "-";

        std::cout << std::setw(6) << m.moveStr << std::setw(10) << m.visits << std::setw(10) << std::fixed
                  << std::setprecision(3) << m.avgVal << std::setw(10) << std::fixed << std::setprecision(3) << m.prior
                  << std::setw(10) << std::fixed << std::setw(10) << std::fixed
                  << std::setprecision(3) << m.puct << std::setw(10) << status << "\n";
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
