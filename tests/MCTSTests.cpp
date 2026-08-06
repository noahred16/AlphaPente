#include "Evaluator.hpp"
#include "MCTS.hpp"
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
