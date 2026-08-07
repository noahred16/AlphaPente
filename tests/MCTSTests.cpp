#include "Evaluator.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"

// doctest must come AFTER the torch-dependent headers: the c10 logging header
// (pulled in via Evaluator.hpp when WITH_TORCH is set) defines its own CHECK
// macro, which silently shadows doctest's CHECK and turns assertions into no-ops.
#ifdef CHECK
#undef CHECK
#endif
#include "doctest.h"

// A 3x3 gomoku board can never produce a winner (five-in-a-row is physically
// impossible on a 3-wide board), so every complete game is a draw. This makes
// the WIN/LOSS branches of MCTS::search's main loop unreachable and isolates
// the SOLVED_DRAW bubbling path added in backpropagate(): if it works, the
// root fully solves as a proven draw well within a generous iteration budget.
TEST_CASE("MCTS proves an unwinnable board as a solved draw at the root") {
    PenteGame::Config config = PenteGame::Config::gomoku(); // no captures, no tournament rule
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9); // forced opening move (center); leaves 8 real branches to test aggregation over

    HeuristicEvaluator evaluator;
    MCTS::Config mctsConfig;
    mctsConfig.evaluator = &evaluator;
    mctsConfig.maxIterations = 50000; // generous for an 8-cell board
    mctsConfig.seed = 42;
    mctsConfig.arenaSize = 64ull * 1024 * 1024;

    MCTS mcts(mctsConfig);
    mcts.search(game);

    // Every root move must lead to a proven draw.
    auto topMoves = mcts.getTopMoves(8);
    REQUIRE(topMoves.size() == 8);
    for (const auto &m : topMoves) {
        CHECK(m.solvedStatus == MCTS::SolvedStatus::SOLVED_DRAW);
    }

    int visitsAfterFirstSearch = mcts.getTotalVisits();
    REQUIRE(visitsAfterFirstSearch < mctsConfig.maxIterations); // proved before exhausting the budget

    // Root itself is now terminal: a second search should do no further work at
    // all (search()'s main loop breaks on iteration 0 once root is solved).
    mcts.search(game);
    CHECK(mcts.getTotalVisits() == visitsAfterFirstSearch);
}

// Same scenario, against the separate multi-threaded ParallelMCTS implementation,
// which has its own atomic/CAS-based SOLVED_DRAW bubbling (see backpropagate(),
// select()'s bestIndex<0 fallback, and selectBestMoveIndex() in ParallelMCTS.cpp).
TEST_CASE("ParallelMCTS proves an unwinnable board as a solved draw at the root") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9); // forced opening move (center); leaves 8 real branches to test aggregation over

    HeuristicEvaluator evaluator;
    ParallelMCTS::Config mctsConfig;
    mctsConfig.evaluator = &evaluator;
    // ParallelMCTS has no progressive widening (unlike single-threaded MCTS) and 4
    // workers explore somewhat redundantly, so it empirically needs ~127K iterations
    // to fully resolve this same 8-branch position (vs. 662 single-threaded) -- budget
    // well above that.
    mctsConfig.maxIterations = 500000;
    mctsConfig.seed = 42;
    mctsConfig.numWorkerThreads = 4;
    mctsConfig.numEvalThreads = 0; // inline mode: cheap heuristic evaluator, no queue round-trip
    mctsConfig.arenaSize = 64ull * 1024 * 1024;

    ParallelMCTS mcts(mctsConfig);
    mcts.search(game);

    const auto *root = mcts.getRoot();
    REQUIRE(root != nullptr);
    REQUIRE(root->childCapacity == 8);
    for (int i = 0; i < root->childCapacity; ++i) {
        auto *child = root->children[i].load();
        REQUIRE(child != nullptr);
        CHECK(child->solvedStatus.load() == ParallelMCTS::SolvedStatus::SOLVED_DRAW);
    }
    CHECK(root->solvedStatus.load() == ParallelMCTS::SolvedStatus::SOLVED_DRAW);

    // Proved well before exhausting the iteration budget.
    CHECK(mcts.getTotalVisits() < mctsConfig.maxIterations);
}

// Canonical-hash transposition sharing (ParallelMCTS::Config::canonicalHashDepth):
// symmetric-equivalent root replies should transpose to the *same* shared child
// node instead of each getting their own independent subtree. After a single
// center opening move, a 5x5 board is fully D4-symmetric, so several of White's
// legal replies (the "promising" cells near the existing stone) are rotations/
// reflections of each other and must collapse together.
TEST_CASE("ParallelMCTS shares symmetric children via canonical-hash transposition") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9); // forced center opening; fully symmetric position for White to reply to

    HeuristicEvaluator evaluator;
    ParallelMCTS::Config mctsConfig;
    mctsConfig.evaluator = &evaluator;
    mctsConfig.maxIterations = 20000; // enough for PUCT to touch every legal reply
    mctsConfig.seed = 42;
    mctsConfig.numWorkerThreads = 4;
    mctsConfig.numEvalThreads = 0;
    mctsConfig.arenaSize = 64ull * 1024 * 1024;
    mctsConfig.canonicalHashDepth = 10; // default; explicit for clarity

    ParallelMCTS mcts(mctsConfig);
    mcts.search(game);

    const auto *root = mcts.getRoot();
    REQUIRE(root != nullptr);
    REQUIRE(root->childCapacity > 0);

    int sharedPairs = 0;
    for (int i = 0; i < root->childCapacity; ++i) {
        auto *ci = root->children[i].load();
        if (!ci) continue;
        for (int j = i + 1; j < root->childCapacity; ++j) {
            auto *cj = root->children[j].load();
            if (ci == cj) sharedPairs++;
        }
    }
    CHECK(sharedPairs > 0);
}

// Same position, canonicalHashDepth=0 (disabled): every reply must get its own
// independent node -- proves the depth gate actually gates the sharing rather
// than it happening unconditionally.
TEST_CASE("ParallelMCTS does not share children when canonicalHashDepth is disabled") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);

    HeuristicEvaluator evaluator;
    ParallelMCTS::Config mctsConfig;
    mctsConfig.evaluator = &evaluator;
    mctsConfig.maxIterations = 20000;
    mctsConfig.seed = 42;
    mctsConfig.numWorkerThreads = 4;
    mctsConfig.numEvalThreads = 0;
    mctsConfig.arenaSize = 64ull * 1024 * 1024;
    mctsConfig.canonicalHashDepth = 0; // disabled

    ParallelMCTS mcts(mctsConfig);
    mcts.search(game);

    const auto *root = mcts.getRoot();
    REQUIRE(root != nullptr);
    REQUIRE(root->canonicalSym < 0);

    for (int i = 0; i < root->childCapacity; ++i) {
        auto *ci = root->children[i].load();
        if (!ci) continue;
        for (int j = i + 1; j < root->childCapacity; ++j) {
            auto *cj = root->children[j].load();
            CHECK(ci != cj);
        }
    }
}
