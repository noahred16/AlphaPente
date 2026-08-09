#include "PNS.hpp"
#include "PenteGame.hpp"
#include "doctest.h"

// Mirrors tests/MCTSTests.cpp's "MCTS proves an unwinnable board as a solved
// draw at the root" case: a 3x3 gomoku board can never produce a five-in-a-row
// winner, so every complete game is a draw. This isolates PNS's terminal
// detection (board-full draw), OR/AND aggregation, threshold-based df-pn
// descent, and canonical-hash DAG sharing (3x3 has plenty of symmetric
// positions) all at once, and cross-checks against the already-verified MCTS
// result on the identical scenario.
TEST_CASE("PNS proves an unwinnable 3x3 board as a draw at the root") {
    PenteGame::Config config = PenteGame::Config::gomoku(); // no captures, no tournament rule
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9); // forced opening move (center)

    PNS pns;
    bool solved = pns.solve(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNS::Outcome::DRAW);
}

// Directed test isolating the OR-node WIN short-circuit and real five-in-a-row
// terminal detection (impossible to exercise on a 3x3 board, whose longest
// line is only 3 cells). Black has four in a row (8,9)-(11,9) on row y=9 with
// only one empty cell left in that row, (7,9); it's Black's move.
TEST_CASE("PNS finds an immediate five-in-a-row win") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();

    game.makeMove(9, 9);  // B forced center
    game.makeMove(7, 7);  // W (elsewhere)
    game.makeMove(8, 9);  // B
    game.makeMove(7, 8);  // W (elsewhere)
    game.makeMove(10, 9); // B
    game.makeMove(11, 7); // W (elsewhere)
    game.makeMove(11, 9); // B - now B has 8,9,10,11 in row y=9; only (7,9) is empty in that row
    game.makeMove(11, 8); // W (elsewhere) - doesn't block

    REQUIRE(game.getCurrentPlayer() == PenteGame::BLACK);
    REQUIRE(game.getWinner() == PenteGame::NONE); // not already over

    PNS pns;
    bool solved = pns.solve(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNS::Outcome::WIN);
}

// Directed test isolating capture-based terminal detection (a distinct code
// path from five-in-a-row, exercised through PNS's own move enumeration and
// expandNode - not shared with the five-in-a-row test above). capturesToWin
// is lowered to 2 so a single bracket capture ends the game immediately:
// B(7,8) W(8,8) W(9,8) B(10,8) is a standard 2-stone bracket - Black's move at
// (10,8) captures both White stones and reaches capturesToWin.
TEST_CASE("PNS finds an immediate capture win") {
    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = 5;
    config.tournamentRule = false; // matches apps/Pente.cpp's auto-disable below boardSize 7
    config.capturesToWin = 2;
    PenteGame game(config);
    game.reset();

    game.makeMove(9, 9); // B forced center
    game.makeMove(8, 8); // W
    game.makeMove(7, 8); // B
    game.makeMove(9, 8); // W

    REQUIRE(game.getCurrentPlayer() == PenteGame::BLACK);
    REQUIRE(game.getWinner() == PenteGame::NONE); // not already over

    PNS pns;
    bool solved = pns.solve(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNS::Outcome::WIN);
}
