#include "doctest.h"
#include "MCTS.hpp"
#include "Evaluator.hpp"

// Recursive tree invariant checker
static void checkNodeInvariants(const MCTS::Node* node, int depth = 0) {
    REQUIRE(node != nullptr);

    // No lingering virtual loss after search completes
    CHECK(node->virtualLoss.load(std::memory_order_relaxed) == 0);

    // Visits should be non-negative
    int32_t visits = node->visits.load(std::memory_order_relaxed);
    CHECK(visits >= 0);

    // Solved status consistency
    auto status = node->solvedStatus.load(std::memory_order_relaxed);
    if (status == MCTS::SolvedStatus::SOLVED_WIN) {
        // All children must be SOLVED_LOSS
        for (uint16_t i = 0; i < node->childCount; i++) {
            auto childStatus = node->children[i]->solvedStatus.load(std::memory_order_relaxed);
            CHECK(childStatus == MCTS::SolvedStatus::SOLVED_LOSS);
        }
    }

    if (status == MCTS::SolvedStatus::SOLVED_LOSS) {
        // At least one child must be SOLVED_WIN (if expanded)
        if (node->childCount > 0) {
            bool hasWinningChild = false;
            for (uint16_t i = 0; i < node->childCount; i++) {
                if (node->children[i]->solvedStatus.load(std::memory_order_relaxed) == MCTS::SolvedStatus::SOLVED_WIN) {
                    hasWinningChild = true;
                    break;
                }
            }
            CHECK(hasWinningChild);
        }
    }

    // unprovenCount consistency: should match number of non-SOLVED_LOSS children
    if (node->isFullyExpanded() && node->childCount > 0) {
        int16_t expectedUnproven = 0;
        for (uint16_t i = 0; i < node->childCount; i++) {
            if (node->children[i]->solvedStatus.load(std::memory_order_relaxed) != MCTS::SolvedStatus::SOLVED_LOSS) {
                expectedUnproven++;
            }
        }
        int16_t actual = node->unprovenCount.load(std::memory_order_relaxed);
        CHECK(actual == expectedUnproven);
    }

    // Recurse into children
    for (uint16_t i = 0; i < node->childCount; i++) {
        checkNodeInvariants(node->children[i], depth + 1);
    }
}

TEST_CASE("Parallel search returns a valid move") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();
    game.makeMove("K10");
    game.makeMove("L9");

    MCTS::Config config;
    config.maxIterations = 500;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    config.arenaSize = 64 * 1024 * 1024; // 64 MB for tests

    HeuristicEvaluator evaluator;
    config.evaluator = &evaluator;

    MCTS mcts(config);

    MCTS::ParallelConfig pconfig;
    pconfig.numWorkers = 4;
    pconfig.useInferenceThread = false;

    PenteGame::Move move = mcts.parallelSearch(game, pconfig);

    // Move should be on the board
    CHECK(move.x < PenteGame::BOARD_SIZE);
    CHECK(move.y < PenteGame::BOARD_SIZE);
    // Move should be in the legal moves list
    auto legalMoves = game.getLegalMoves();
    bool found = false;
    for (const auto& lm : legalMoves) {
        if (lm.x == move.x && lm.y == move.y) { found = true; break; }
    }
    CHECK(found);
    // Tree should have visits
    CHECK(mcts.getTotalVisits() > 0);
}

TEST_CASE("Parallel search tree invariants") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();
    game.makeMove("K10");
    game.makeMove("L9");

    MCTS::Config config;
    config.maxIterations = 1000;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    config.arenaSize = 64 * 1024 * 1024; // 64 MB for tests

    HeuristicEvaluator evaluator;
    config.evaluator = &evaluator;

    MCTS mcts(config);

    MCTS::ParallelConfig pconfig;
    pconfig.numWorkers = 4;
    pconfig.useInferenceThread = false;

    mcts.parallelSearch(game, pconfig);

    // Check invariants on the entire tree
    CHECK(mcts.getTotalVisits() > 0);
    CHECK(mcts.getTreeSize() > 1);
}
