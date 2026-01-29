#include "doctest.h"
#include "PenteGame.hpp"

TEST_CASE("PenteGame initial state") {
    PenteGame game;
    game.reset();

    CHECK(game.getCurrentPlayer() == PenteGame::BLACK);
    CHECK(game.getMoveCount() == 0);
    CHECK(game.getBlackCaptures() == 0);
    CHECK(game.getWhiteCaptures() == 0);
    CHECK(game.isGameOver() == false);
}

TEST_CASE("PenteGame make move") {
    PenteGame game;
    game.reset();

    CHECK(game.makeMove("K10") == true);
    CHECK(game.getMoveCount() == 1);
    CHECK(game.getCurrentPlayer() == PenteGame::WHITE);
    CHECK(game.getStoneAt(9, 9) == PenteGame::BLACK);
}

TEST_CASE("PenteGame undo move") {
    PenteGame game;
    game.reset();

    game.makeMove("K10");
    game.makeMove("L10");

    CHECK(game.getMoveCount() == 2);
    CHECK(game.canUndo() == true);

    game.undoMove();

    CHECK(game.getMoveCount() == 1);
    CHECK(game.getCurrentPlayer() == PenteGame::WHITE);
    CHECK(game.getStoneAt(10, 9) == PenteGame::NONE);
}

TEST_CASE("PenteGame config presets") {
    PenteGame pente(PenteGame::Config::pente());
    CHECK(pente.getConfig().capturesToWin == 10);
    CHECK(pente.getConfig().capturesEnabled == true);

    PenteGame gomoku(PenteGame::Config::gomoku());
    CHECK(gomoku.getConfig().capturesEnabled == false);

    PenteGame keryo(PenteGame::Config::keryoPente());
    CHECK(keryo.getConfig().capturesToWin == 15);
    CHECK(keryo.getConfig().keryoRules == true);
}

TEST_CASE("PenteGame getPromisingMoves distance 2") {
    PenteGame game;
    game.reset();

    // Place a stone at center K10 (9,9)
    game.makeMove("K10");

    auto moves = game.getPromisingMoves(2);

    // With distance 2, should get a 5x5 area around the stone minus the stone itself
    // 5x5 = 25, minus 1 occupied = 24 promising moves
    CHECK(moves.size() == 24);

    // Verify all moves are within distance 2 of (9,9)
    for (const auto& move : moves) {
        int dx = std::abs(move.x - 9);
        int dy = std::abs(move.y - 9);
        CHECK(dx <= 2);
        CHECK(dy <= 2);
        // Should not include the occupied square
        bool isOccupied = (move.x == 9 && move.y == 9);
        CHECK_FALSE(isOccupied);
    }
}
