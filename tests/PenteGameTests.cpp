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

TEST_CASE("PenteGame evaluateMove no captures") {
    PenteGame game;
    game.reset();

    // Place center stone
    game.makeMove("K10");  // Black at (9,9)

    // White's move with no capture potential should return 1
    PenteGame::Move move(10, 9);  // L10
    CHECK(game.evaluateMove(move) == 1.0f);
}

TEST_CASE("PenteGame evaluateMove single capture") {
    PenteGame game;
    game.reset();

    // Setup: X O O _ pattern where X will complete capture
    // Black at K10 (9,9), White at L10 (10,9), White at M10 (11,9)
    // Black move at N10 (12,9) should capture
    game.makeMove("K10");  // Black (9,9)
    game.makeMove("L10");  // White (10,9)
    game.makeMove("J10");  // Black (8,9) - need another black move
    game.makeMove("M10");  // White (11,9)

    // Now it's Black's turn - N10 should capture the two white stones
    PenteGame::Move captureMove(12, 9);  // N10
    CHECK(game.evaluateMove(captureMove) == 7.0f);  // 1 + 1 capture * 6 = 7
}

TEST_CASE("PenteGame evaluateMove double capture") {
    PenteGame game;
    game.reset();

    // Setup: Two B W W _ patterns meeting at (12,9) = O10
    // Horizontal: B(9,9) W(10,9) W(11,9) _(12,9)
    // Vertical: B(12,6) W(12,7) W(12,8) _(12,9)

    game.makeMove("K10");  // Black (9,9) center - required first move
    game.makeMove("L10");  // White (10,9)
    game.makeMove("N7");   // Black (12,6)
    game.makeMove("M10");  // White (11,9) - horizontal: B(9,9) W(10,9) W(11,9) _(12,9)
    game.makeMove("K9");   // Black (9,8) - dummy
    game.makeMove("N8");   // White (12,7)
    game.makeMove("K8");   // Black (9,7) - dummy
    game.makeMove("N9");   // White (12,8) - vertical: B(12,6) W(12,7) W(12,8) _(12,9)

    // Now Black at N10 (12,9) captures in both directions
    PenteGame::Move doubleCapture(12, 9);  // N10
    CHECK(game.evaluateMove(doubleCapture) == 13.0f);  // 1 + 2 captures * 6 = 13
}

TEST_CASE("PenteGame evaluateMove detects block") {
    PenteGame game;
    game.reset();

    // Setup so it's Black's turn and K6 (9,5) blocks a capture threat
    // Block pattern from position: myStone - myStone - oppStone
    // So we need: K7(B) K8(B) K9(W), then K6 blocks
    game.makeMove("K10");  // Black (9,9) center
    game.makeMove("L10");  // White (10,9) dummy
    game.makeMove("K7");   // Black (9,6)
    game.makeMove("L9");   // White (10,8) dummy
    game.makeMove("K8");   // Black (9,7)
    game.makeMove("K9");   // White (9,8) - the threat anchor

    // Column K: K7(B,9,6) K8(B,9,7) K9(W,9,8)
    // It's Black's turn
    // Check K6 (9,5) - direction (0,1):
    //   (9,6)=B=my, (9,7)=B=my, (9,8)=W=opp
    // Pattern: my my opp = BLOCK!

    PenteGame::Move blockMove(9, 5);  // K6
    CHECK(game.evaluateMove(blockMove) == 5.0f);  // 1 + 1 block * 4 = 5
}

TEST_CASE("PenteGame evaluateMove creates solid open three") {
    PenteGame game;
    game.reset();

    // Setup: _ X X _ where placing at either end creates open three _ X X X _
    game.makeMove("K10");  // Black (9,9) center
    game.makeMove("L7");   // White dummy
    game.makeMove("L10");  // Black (10,9)
    game.makeMove("L8");   // White dummy

    // Now placing at M10 (11,9) creates _ K10 L10 M10 _ = _ X X X _
    // Pattern: J10 empty, K10 Black, L10 Black, M10 new, N10 empty
    PenteGame::Move openThreeMove(11, 9);  // M10
    // Score: 1 (default) + 15 (open three) = 16
    CHECK(game.evaluateMove(openThreeMove) == 16.0f);
}

TEST_CASE("PenteGame evaluateMove creates gap open three") {
    PenteGame game;
    game.reset();

    // Setup: X _ X pattern, placing to make X _ X X (gap three)
    game.makeMove("K10");  // Black (9,9) center
    game.makeMove("L7");   // White dummy
    game.makeMove("M10");  // Black (11,9)
    game.makeMove("L8");   // White dummy

    // Now placing at N10 (12,9) creates K10 _ M10 N10 = X _ X X
    // Need both ends open: J10 empty, O10 empty
    PenteGame::Move gapThreeMove(12, 9);  // N10
    // This should be pattern 1: P _ X X
    // Score: 1 (default) + 15 (gap open three) = 16
    CHECK(game.evaluateMove(gapThreeMove) == 16.0f);
}

TEST_CASE("PenteGame evaluateMove detects block of open three") {
    PenteGame game;
    game.reset();

    // Create White's open three in a column
    game.makeMove("K10");  // Black (9,9) center
    game.makeMove("L11");  // White (10,10)
    game.makeMove("M10");  // Black far
    game.makeMove("L12");  // White (10,11)
    game.makeMove("M9");   // Black far
    game.makeMove("L13");  // White (10,12)

    // White has L11(10,10), L12(10,11), L13(10,12) - three in column L
    // L10(10,9) empty, L14(10,13) empty - open three!
    // Black at L10 or L14 blocks

    PenteGame::Move blockMove(10, 9);  // L10
    // Score: 1 (default) + 20 (block open three) + 15 (creates open three K10-L10-M10) = 36
    CHECK(game.evaluateMove(blockMove) == 36.0f);
}

TEST_CASE("PenteGame evaluateMove verifies capture pattern") {
    PenteGame game;
    game.reset();

    // Simple verification: X _ O O X pattern
    // Black K10, White L10, White M10, Black needs to be at N10 to capture
    game.makeMove("K10");  // Black (9,9)
    game.makeMove("L10");  // White (10,9)
    game.makeMove("N10");  // Black (12,9)
    game.makeMove("M10");  // White (11,9)

    // Now it's Black's turn, but the pattern is K10(B) L10(W) M10(W) N10(B)
    // A move between them won't capture - we need X O O _ pattern
    // Let's check a non-capturing move
    PenteGame::Move nonCapture(9, 8);  // K9
    CHECK(game.evaluateMove(nonCapture) == 1.0f);
}
