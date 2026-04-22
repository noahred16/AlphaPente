#include "doctest.h"
#include "NNInputEncoder.hpp"
#include "PenteGame.hpp"

static constexpr int PS = NNInputEncoder::PLANE_SIZE;

TEST_CASE("NNInputEncoder - tensor dimensions") {
    PenteGame game;
    auto tensor = NNInputEncoder::encode(game);
    CHECK(int(tensor.size()) == NNInputEncoder::TOTAL_SIZE);
}

TEST_CASE("NNInputEncoder - empty board has no stones") {
    PenteGame game;
    auto tensor = NNInputEncoder::encode(game);

    for (int i = 0; i < PS; ++i) {
        CHECK(tensor[0 * PS + i] == 0.0f); // current player stones
        CHECK(tensor[1 * PS + i] == 0.0f); // opponent stones
    }
}

TEST_CASE("NNInputEncoder - color plane") {
    PenteGame game;
    CHECK(game.getCurrentPlayer() == PenteGame::BLACK);

    // Black to move → color plane all 1.0
    auto tensor = NNInputEncoder::encode(game);
    for (int i = 0; i < PS; ++i)
        CHECK(tensor[2 * PS + i] == 1.0f);

    // After one move, White to move → color plane all 0.0
    game.makeMove("K10");
    tensor = NNInputEncoder::encode(game);
    for (int i = 0; i < PS; ++i)
        CHECK(tensor[2 * PS + i] == 0.0f);
}

TEST_CASE("NNInputEncoder - stone planes are perspective-relative") {
    PenteGame game;
    game.makeMove("K10"); // Black plays center (9,9)

    // Now White to move. Black's stone should be in plane 1 (opponent).
    auto tensor = NNInputEncoder::encode(game);
    int centerIdx = 9 * NNInputEncoder::BOARD_SIZE + 9;

    CHECK(tensor[0 * PS + centerIdx] == 0.0f); // not White's stone
    CHECK(tensor[1 * PS + centerIdx] == 1.0f); // is Black's (opponent's) stone

    // Every other cell should be 0 on both stone planes
    for (int i = 0; i < PS; ++i) {
        if (i != centerIdx) {
            CHECK(tensor[0 * PS + i] == 0.0f);
            CHECK(tensor[1 * PS + i] == 0.0f);
        }
    }
}

TEST_CASE("NNInputEncoder - stone planes switch after second move") {
    PenteGame game;
    game.makeMove("K10"); // Black (9,9)
    game.makeMove("J10"); // White (8,9)

    // Black to move again. Black stone at (9,9) → plane 0. White stone at (8,9) → plane 1.
    auto tensor = NNInputEncoder::encode(game);
    int blackIdx = 9 * NNInputEncoder::BOARD_SIZE + 9;
    int whiteIdx = 9 * NNInputEncoder::BOARD_SIZE + 8;

    CHECK(tensor[0 * PS + blackIdx] == 1.0f); // Black's (current) stone
    CHECK(tensor[1 * PS + whiteIdx] == 1.0f); // White's (opponent) stone
    CHECK(tensor[0 * PS + whiteIdx] == 0.0f);
    CHECK(tensor[1 * PS + blackIdx] == 0.0f);
}

TEST_CASE("NNInputEncoder - move count plane") {
    PenteGame game;

    auto tensor = NNInputEncoder::encode(game);
    float expected = 0.0f / float(PS);
    for (int i = 0; i < PS; ++i)
        CHECK(tensor[5 * PS + i] == doctest::Approx(expected));

    game.makeMove("K10");
    game.makeMove("J10");
    tensor = NNInputEncoder::encode(game);
    expected = 2.0f / float(PS);
    for (int i = 0; i < PS; ++i)
        CHECK(tensor[5 * PS + i] == doctest::Approx(expected));
}

TEST_CASE("NNInputEncoder - capture planes reflect game state") {
    // Use a position with known captures from the existing PenteGame tests:
    // Black K10(9,9), White L10(10,9), Black J10(8,9), White M10(11,9),
    // then Black N10(12,9) captures the two white stones.
    PenteGame game;

    // Verify no captures initially
    auto tensor = NNInputEncoder::encode(game);
    for (int i = 0; i < PS; ++i) {
        CHECK(tensor[3 * PS + i] == 0.0f);
        CHECK(tensor[4 * PS + i] == 0.0f);
    }

    game.makeMove("K10"); // Black (9,9)
    game.makeMove("L10"); // White (10,9)
    game.makeMove("J10"); // Black (8,9)
    game.makeMove("M10"); // White (11,9)
    game.makeMove("N10"); // Black (12,9) — captures W at (10,9) and (11,9)

    CHECK(game.getBlackCaptures() > 0);

    // Now White to move. Black (opponent) has captures → plane 4.
    tensor = NNInputEncoder::encode(game);
    float expectedOppCaps = float(game.getBlackCaptures()) / float(game.getConfig().capturesToWin);
    float expectedMyCaps  = float(game.getWhiteCaptures()) / float(game.getConfig().capturesToWin);

    for (int i = 0; i < PS; ++i) {
        CHECK(tensor[3 * PS + i] == doctest::Approx(expectedMyCaps));
        CHECK(tensor[4 * PS + i] == doctest::Approx(expectedOppCaps));
    }
}

TEST_CASE("NNInputEncoder - all values in [0, 1]") {
    PenteGame game;
    game.makeMove("K10");
    game.makeMove("L10");
    game.makeMove("J10");
    game.makeMove("M10");
    game.makeMove("N10");

    auto tensor = NNInputEncoder::encode(game);
    for (float v : tensor) {
        CHECK(v >= 0.0f);
        CHECK(v <= 1.0f);
    }
}
