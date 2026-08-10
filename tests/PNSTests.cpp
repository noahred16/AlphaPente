#include "PNS.hpp"
#include "PenteGame.hpp"
#include "doctest.h"
#include <cstdio>

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
    // At most the 8 remaining empty cells can be played before the board is
    // full; exact optimal-line length isn't asserted (not hand-verified),
    // just that depth tracking produced something in the plausible range.
    CHECK(pns.getRootDepth() > 0);
    CHECK(pns.getRootDepth() <= 8);
}

// solveExhaustive() exists specifically because solve() leaves most reachable
// positions UNKNOWN (df-pn stops exploring once a branch is no longer needed
// to prove the root). This checks the actual distinguishing property: same
// root result as solve() on the identical scenario, but resolves the full
// reachable DAG rather than a minimal subset - every node solveExhaustive()
// ever touches (getNodeCount()) ends up resolved (exportResolved().size()
// equal to it), unlike solve()'s partial coverage.
TEST_CASE("PNS::solveExhaustive resolves every reachable position, not just enough to prove the root") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);

    PNS pns;
    bool solved = pns.solveExhaustive(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNS::Outcome::DRAW); // matches the plain solve() result above
    CHECK(pns.exportResolved().size() == pns.getNodeCount());
}

// Regression test for the recursion-depth safety valve (Config::maxRecursionDepth):
// a real long single-threaded 4x4 run once SIGSEGV'd (stack overflow - mid()
// recurses per-ply, passing a full PenteGame >8KB by value each level) once
// df-pn committed deep into one narrow line. Forcing a tiny depth budget here
// makes that condition trivially reachable without needing an actually-deep
// search, and checks it stops gracefully (like maxNodes/maxSeconds) instead
// of crashing or hanging.
TEST_CASE("PNS stops gracefully when maxRecursionDepth is hit, rather than crashing") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);

    PNS::Config pnsConfig;
    pnsConfig.maxRecursionDepth = 2; // far too shallow to solve 3x3
    PNS pns(pnsConfig);
    bool solved = pns.solve(game);

    CHECK_FALSE(solved);
    CHECK(pns.getRootOutcome() == PNS::Outcome::UNKNOWN);
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
    CHECK(pns.getRootDepth() == 1); // wins on the very next move
}

// Full checkpoint/resume round trip: cap maxNodes low enough that solve()
// can't finish on the first pass, load that checkpoint into a fresh PNS
// instance with a real budget, and confirm resuming reaches the exact same
// result an uninterrupted solve does. This is the scenario checkpointing
// exists for - a long run split across two separate processes/sessions.
TEST_CASE("PNS checkpoint save/load resumes an interrupted solve to the same result as an uninterrupted one") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);

    PNS baseline;
    REQUIRE(baseline.solve(game));
    REQUIRE(baseline.getRootOutcome() == PNS::Outcome::DRAW);

    const std::string path = "/tmp/pns_checkpoint_test_resume.bin";
    std::remove(path.c_str());

    // First pass: maxNodes=20 is far short of the ~264 nodes a full 3x3
    // solve touches (see the first TEST_CASE's comment above), so this
    // cannot finish - but solve() still writes a final checkpoint of
    // whatever partial DAG it built, since checkpointPath is set.
    PNS::Config cfg1;
    cfg1.maxNodes = 20;
    cfg1.checkpointPath = path;
    PNS pns1(cfg1);
    bool solved1 = pns1.solve(game);
    CHECK_FALSE(solved1);
    CHECK(pns1.getRootOutcome() == PNS::Outcome::UNKNOWN);
    CHECK(pns1.getNodeCount() > 0);
    CHECK(pns1.getNodeCount() <= cfg1.maxNodes);

    // Resume in a completely fresh PNS instance with a real budget.
    PNS pns2;
    REQUIRE(pns2.loadCheckpoint(path, game));
    CHECK(pns2.getNodeCount() == pns1.getNodeCount()); // picked up exactly where it left off
    bool solved2 = pns2.solve(game);

    CHECK(solved2);
    CHECK(pns2.getRootOutcome() == PNS::Outcome::DRAW);
    CHECK(pns2.getRootDepth() == baseline.getRootDepth());

    std::remove(path.c_str());
}

// loadCheckpoint() must refuse a checkpoint written for a different root
// player rather than silently resuming against a mismatched position - a
// checkpoint's proof numbers are only meaningful relative to whoever was
// proving when it was written (see the doc comment on loadCheckpoint()).
TEST_CASE("PNS::loadCheckpoint rejects a checkpoint whose root player doesn't match rootGame") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame blackToMove(config);
    blackToMove.reset();

    PenteGame whiteToMove(config);
    whiteToMove.reset();
    whiteToMove.makeMove(9, 9);

    const std::string path = "/tmp/pns_checkpoint_test_mismatch.bin";
    std::remove(path.c_str());

    PNS::Config cfg;
    cfg.checkpointPath = path;
    PNS pns(cfg);
    REQUIRE(pns.solve(whiteToMove)); // writes a final checkpoint keyed to White-to-move

    PNS pns2;
    CHECK_FALSE(pns2.loadCheckpoint(path, blackToMove)); // Black-to-move root: mismatched

    std::remove(path.c_str());
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
    CHECK(pns.getRootDepth() == 1); // wins on the very next move
}
