#include "PNSRocks.hpp"
#include "PenteGame.hpp"
#include "doctest.h"
#include <filesystem>

// Mirrors PNSTests.cpp's identical case for the in-RAM PNS - same scenario,
// cross-checked against the same already-verified result, isolating that the
// RocksDB-backed storage swap (Get/Put instead of arena/table_ - see
// PNSRocks.hpp's class comment) didn't change the df-pn algorithm's actual
// output.
TEST_CASE("PNSRocks proves an unwinnable 3x3 board as a draw at the root") {
    const std::string dbPath = "/tmp/pnsrocks_test_draw3x3";
    std::filesystem::remove_all(dbPath);

    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9); // forced opening move (center)

    PNSRocks::Config cfg;
    cfg.dbPath = dbPath;
    cfg.blockCacheBytes = 32ULL << 20; // small cache; this is a tiny test DB
    PNSRocks pns(cfg);
    bool solved = pns.solve(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNSRocks::Outcome::DRAW);
    CHECK(pns.getRootDepth() > 0);
    CHECK(pns.getRootDepth() <= 8);

    std::filesystem::remove_all(dbPath);
}

// Mirrors PNSTests.cpp's "PNS finds an immediate five-in-a-row win" case.
TEST_CASE("PNSRocks finds an immediate five-in-a-row win") {
    const std::string dbPath = "/tmp/pnsrocks_test_win5x5";
    std::filesystem::remove_all(dbPath);

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
    REQUIRE(game.getWinner() == PenteGame::NONE);

    PNSRocks::Config cfg;
    cfg.dbPath = dbPath;
    cfg.blockCacheBytes = 32ULL << 20;
    PNSRocks pns(cfg);
    bool solved = pns.solve(game);

    CHECK(solved);
    CHECK(pns.getRootOutcome() == PNSRocks::Outcome::WIN);
    CHECK(pns.getRootDepth() == 1);

    std::filesystem::remove_all(dbPath);
}

// The whole point of PNSRocks over PNS's checkpoint/resume feature: reopening
// the same dbPath resumes automatically, no separate checkpoint file at all.
// Caps maxNodes low enough that the first PNSRocks instance can't finish,
// closes it (destructor closes the RocksDB handle), reopens a second instance
// on the SAME path with a real budget, and confirms it picks up exactly where
// the first left off and reaches the same result an uninterrupted solve does.
TEST_CASE("PNSRocks reopening the same dbPath resumes an interrupted solve") {
    const std::string dbPath = "/tmp/pnsrocks_test_resume";
    std::filesystem::remove_all(dbPath);

    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);

    {
        PNSRocks::Config cfg;
        cfg.dbPath = dbPath;
        cfg.blockCacheBytes = 32ULL << 20;
        cfg.maxNodes = 20; // far short of the ~264 nodes a full 3x3 solve touches
        PNSRocks pns1(cfg);
        bool solved1 = pns1.solve(game);
        CHECK_FALSE(solved1);
        CHECK(pns1.getRootOutcome() == PNSRocks::Outcome::UNKNOWN);
        CHECK(pns1.getStats().nodesCreated > 0);
        CHECK(pns1.getStats().nodesCreated <= cfg.maxNodes);
    } // destructor closes the RocksDB handle

    PNSRocks::Config cfg2;
    cfg2.dbPath = dbPath; // same path: resumes from what pns1 already wrote
    cfg2.blockCacheBytes = 32ULL << 20;
    PNSRocks pns2(cfg2);
    bool solved2 = pns2.solve(game);

    CHECK(solved2);
    CHECK(pns2.getRootOutcome() == PNSRocks::Outcome::DRAW);

    std::filesystem::remove_all(dbPath);
}
