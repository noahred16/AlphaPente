#include "doctest.h"
#include "BitBoard.hpp"
#include "RenjuRules.hpp"
#include "PenteGame.hpp"
#include <algorithm>

TEST_CASE("RenjuRules isOverline: exact five is not overline") {
    BitBoard black(19);
    // X X X X _   -- placing at (4,9) makes exactly five in a row
    black.setBit(0, 9);
    black.setBit(1, 9);
    black.setBit(2, 9);
    black.setBit(3, 9);

    CHECK(RenjuRules::isOverline(black, 4, 9) == false);
}

TEST_CASE("RenjuRules isOverline: six in a row is overline") {
    BitBoard black(19);
    // X X X X _ X   -- placing at (4,9) makes six in a row
    black.setBit(0, 9);
    black.setBit(1, 9);
    black.setBit(2, 9);
    black.setBit(3, 9);
    black.setBit(5, 9);

    CHECK(RenjuRules::isOverline(black, 4, 9) == true);
}

TEST_CASE("RenjuRules isOverline: seven in a row is overline") {
    BitBoard black(19);
    black.setBit(0, 9);
    black.setBit(1, 9);
    black.setBit(2, 9);
    black.setBit(3, 9);
    black.setBit(5, 9);
    black.setBit(6, 9);

    CHECK(RenjuRules::isOverline(black, 4, 9) == true);
}

TEST_CASE("RenjuRules isOverline: vertical six in a row is overline") {
    BitBoard black(19);
    black.setBit(9, 0);
    black.setBit(9, 1);
    black.setBit(9, 2);
    black.setBit(9, 4);
    black.setBit(9, 5);

    CHECK(RenjuRules::isOverline(black, 9, 3) == true);
}

TEST_CASE("RenjuRules isOverline: diagonal six in a row is overline") {
    BitBoard black(19);
    black.setBit(0, 0);
    black.setBit(1, 1);
    black.setBit(2, 2);
    black.setBit(4, 4);
    black.setBit(5, 5);

    CHECK(RenjuRules::isOverline(black, 3, 3) == true);
}

TEST_CASE("RenjuRules isOverline: unrelated stones elsewhere don't trigger it") {
    BitBoard black(19);
    black.setBit(0, 0);
    black.setBit(18, 18);

    CHECK(RenjuRules::isOverline(black, 9, 9) == false);
}

TEST_CASE("RenjuRules isOverline: four in a row only, no overline") {
    BitBoard black(19);
    black.setBit(0, 9);
    black.setBit(1, 9);
    black.setBit(2, 9);

    CHECK(RenjuRules::isOverline(black, 3, 9) == false);
}

TEST_CASE("RenjuRules isOverline: near board edge does not crash or false-trigger") {
    BitBoard black(19);
    black.setBit(0, 0);
    black.setBit(1, 0);
    black.setBit(2, 0);
    black.setBit(3, 0);

    CHECK(RenjuRules::isOverline(black, 4, 0) == false); // exactly five, board edge cuts it off
}

// ============================================================================
// PenteGame integration: Config::renju() must actually enforce the forbidden-move rules
// through getLegalMoves()/isLegalMove(), not just expose an unused rule engine.
// ============================================================================

namespace {
bool inLegalMoves(const PenteGame &game, int x, int y) {
    const auto &legal = game.getLegalMoves();
    return std::any_of(legal.begin(), legal.end(),
                        [x, y](const PenteGame::Move &m) { return m.x == x && m.y == y; });
}
} // namespace

TEST_CASE("PenteGame Renju: a black double-three is excluded from legal moves") {
    PenteGame game(PenteGame::Config::renju());
    // Cross of two open threes around (9,9), with harmless White filler moves in between.
    game.makeMove(8, 9);  // Black
    game.makeMove(2, 2);  // White filler
    game.makeMove(10, 9); // Black
    game.makeMove(2, 3);  // White filler
    game.makeMove(9, 8);  // Black
    game.makeMove(2, 4);  // White filler
    game.makeMove(9, 10); // Black
    game.makeMove(2, 5);  // White filler

    REQUIRE(game.getCurrentPlayer() == PenteGame::BLACK);
    CHECK(game.isLegalMove(9, 9) == false);
    CHECK(inLegalMoves(game, 9, 9) == false);
}

TEST_CASE("PenteGame Renju: the same double-three shape is legal in standard Pente") {
    PenteGame game(PenteGame::Config::pente());
    game.makeMove(8, 9);
    game.makeMove(2, 2);
    game.makeMove(10, 9);
    game.makeMove(2, 3);
    game.makeMove(9, 8);
    game.makeMove(2, 4);
    game.makeMove(9, 10);
    game.makeMove(2, 5);

    REQUIRE(game.getCurrentPlayer() == PenteGame::BLACK);
    CHECK(game.isLegalMove(9, 9) == true);
    CHECK(inLegalMoves(game, 9, 9) == true);
}

TEST_CASE("PenteGame Renju: White has no forbidden-move restriction") {
    PenteGame game(PenteGame::Config::renju());
    // Same cross shape, but built from White stones, with Black filler moves in between.
    game.makeMove(2, 2);  // Black filler
    game.makeMove(8, 9);  // White
    game.makeMove(2, 3);  // Black filler
    game.makeMove(10, 9); // White
    game.makeMove(2, 4);  // Black filler
    game.makeMove(9, 8);  // White
    game.makeMove(2, 5);  // Black filler
    game.makeMove(9, 10); // White
    game.makeMove(2, 6);  // Black filler

    REQUIRE(game.getCurrentPlayer() == PenteGame::WHITE);
    CHECK(game.isLegalMove(9, 9) == true);
    CHECK(inLegalMoves(game, 9, 9) == true);
}

TEST_CASE("PenteGame Renju: exact five in a row wins for Black") {
    PenteGame game(PenteGame::Config::renju());
    game.makeMove(5, 9);  // Black
    game.makeMove(2, 2);  // White filler
    game.makeMove(6, 9);  // Black
    game.makeMove(2, 3);  // White filler
    game.makeMove(7, 9);  // Black
    game.makeMove(2, 4);  // White filler
    game.makeMove(8, 9);  // Black
    game.makeMove(2, 5);  // White filler
    game.makeMove(9, 9);  // Black: completes 5..9, exactly five

    CHECK(game.getWinner() == PenteGame::BLACK);
}

TEST_CASE("PenteGame Renju: a Black overline does not win") {
    PenteGame game(PenteGame::Config::renju());
    game.makeMove(2, 9);  // Black
    game.makeMove(2, 2);  // White filler
    game.makeMove(3, 9);  // Black
    game.makeMove(2, 3);  // White filler
    game.makeMove(4, 9);  // Black
    game.makeMove(2, 4);  // White filler
    game.makeMove(5, 9);  // Black
    game.makeMove(2, 5);  // White filler
    game.makeMove(7, 9);  // Black
    game.makeMove(2, 6);  // White filler
    // makeMove(x,y) is unconditional (no legality gate) - placed directly to exercise the win
    // check itself: filling the gap at x=6 makes six in a row (2..7), an overline.
    game.makeMove(6, 9);  // Black

    CHECK(game.getWinner() != PenteGame::BLACK);
}

TEST_CASE("PenteGame Renju: the same overline still wins in standard Pente") {
    PenteGame game(PenteGame::Config::pente());
    game.makeMove(2, 9);
    game.makeMove(2, 2);
    game.makeMove(3, 9);
    game.makeMove(2, 3);
    game.makeMove(4, 9);
    game.makeMove(2, 4);
    game.makeMove(5, 9);
    game.makeMove(2, 5);
    game.makeMove(7, 9);
    game.makeMove(2, 6);
    game.makeMove(6, 9);

    CHECK(game.getWinner() == PenteGame::BLACK);
}

TEST_CASE("PenteGame Renju regression: H5 from the reported game is excluded from legal moves") {
    // 1.H8 H7 2.H6 J7 3.G7 J9 4.J5 K4 5.F8 E9 6.G8 J8 7.J10 H9 8.G9 G6 9.F5 G10
    // 10.K7 F6 11.K9 E10 12.E8 D8 13.L8 H11 14.M7 N6
    // The engine reported H5 as a winning move; it is in fact a Black double-three.
    PenteGame game(PenteGame::Config::renju());
    const char *moves[] = {"H8", "H7", "H6", "J7", "G7", "J9", "J5", "K4", "F8", "E9",
                            "G8", "J8", "J10", "H9", "G9", "G6", "F5", "G10", "K7", "F6",
                            "K9", "E10", "E8", "D8", "L8", "H11", "M7", "N6"};
    for (const char *m : moves) {
        REQUIRE(game.makeMove(m));
    }

    REQUIRE(game.getCurrentPlayer() == PenteGame::BLACK);
    CHECK(game.isLegalMove(7, 4) == false);   // H5
    CHECK(inLegalMoves(game, 7, 4) == false); // H5
}
