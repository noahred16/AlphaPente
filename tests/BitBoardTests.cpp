#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "BitBoard.hpp"
#include "PenteGame.hpp"

TEST_CASE("BitBoard basic set and get") {
    BitBoard board(19);

    CHECK(board.getBit(5, 5) == false);
    board.setBit(5, 5);
    CHECK(board.getBit(5, 5) == true);
}

TEST_CASE("BitBoard clear bit") {
    BitBoard board(19);

    board.setBit(10, 10);
    CHECK(board.getBit(10, 10) == true);

    board.clearBit(10, 10);
    CHECK(board.getBit(10, 10) == false);
}

TEST_CASE("BitBoard clear all") {
    BitBoard board(19);

    board.setBit(0, 0);
    board.setBit(18, 18);
    board.setBit(9, 9);

    board.clear();

    CHECK(board.getBit(0, 0) == false);
    CHECK(board.getBit(18, 18) == false);
    CHECK(board.getBit(9, 9) == false);
}

TEST_CASE("BitBoard OR operator") {
    BitBoard a(19);
    BitBoard b(19);

    a.setBit(1, 1);
    b.setBit(2, 2);

    BitBoard result = a | b;

    CHECK(result.getBit(1, 1) == true);
    CHECK(result.getBit(2, 2) == true);
    CHECK(result.getBit(3, 3) == false);
}

TEST_CASE("BitBoard getSetPositions with pair") {
    BitBoard board(19);

    board.setBit(3, 4);
    board.setBit(7, 8);

    auto positions = board.getSetPositions<std::pair<int, int>>();

    REQUIRE(positions.size() == 2);

    // Check that both positions are present (order may vary)
    bool found_3_4 = false, found_7_8 = false;
    for (const auto& pos : positions) {
        if (pos.first == 3 && pos.second == 4) found_3_4 = true;
        if (pos.first == 7 && pos.second == 8) found_7_8 = true;
    }
    CHECK(found_3_4);
    CHECK(found_7_8);
}

TEST_CASE("BitBoard getSetPositions with PenteGame::Move") {
    BitBoard board(19);

    board.setBit(9, 9);   // Center (K10 in Pente notation)
    board.setBit(0, 0);   // Corner

    auto moves = board.getSetPositions<PenteGame::Move>();

    REQUIRE(moves.size() == 2);

    bool found_center = false, found_corner = false;
    for (const auto& move : moves) {
        if (move.x == 9 && move.y == 9) found_center = true;
        if (move.x == 0 && move.y == 0) found_corner = true;
    }
    CHECK(found_center);
    CHECK(found_corner);
}

TEST_CASE("BitBoard dilate single stone") {
    BitBoard board(19);

    board.setBit(5, 5);
    BitBoard dilated = board.dilate();

    // Original position should still be set
    CHECK(dilated.getBit(5, 5) == true);

    // All 8 neighbors should be set
    CHECK(dilated.getBit(4, 4) == true);  // top-left
    CHECK(dilated.getBit(5, 4) == true);  // top
    CHECK(dilated.getBit(6, 4) == true);  // top-right
    CHECK(dilated.getBit(4, 5) == true);  // left
    CHECK(dilated.getBit(6, 5) == true);  // right
    CHECK(dilated.getBit(4, 6) == true);  // bottom-left
    CHECK(dilated.getBit(5, 6) == true);  // bottom
    CHECK(dilated.getBit(6, 6) == true);  // bottom-right

    // Positions further away should not be set
    CHECK(dilated.getBit(3, 3) == false);
    CHECK(dilated.getBit(7, 7) == false);
}

TEST_CASE("BitBoard dilate corner respects boundaries") {
    BitBoard board(19);

    board.setBit(0, 0);  // Top-left corner
    BitBoard dilated = board.dilate();

    // Should set valid neighbors only
    CHECK(dilated.getBit(0, 0) == true);
    CHECK(dilated.getBit(1, 0) == true);
    CHECK(dilated.getBit(0, 1) == true);
    CHECK(dilated.getBit(1, 1) == true);

    // Should have exactly 4 positions set (corner + 3 neighbors)
    auto positions = dilated.getSetPositions();
    CHECK(positions.size() == 4);
}

TEST_CASE("BitBoard dilate merges adjacent stones") {
    BitBoard board(19);

    // Two adjacent stones
    board.setBit(5, 5);
    board.setBit(6, 5);

    BitBoard dilated = board.dilate();

    // Both originals
    CHECK(dilated.getBit(5, 5) == true);
    CHECK(dilated.getBit(6, 5) == true);

    // Shared neighbors should be set (merged)
    CHECK(dilated.getBit(5, 4) == true);
    CHECK(dilated.getBit(6, 4) == true);
    CHECK(dilated.getBit(5, 6) == true);
    CHECK(dilated.getBit(6, 6) == true);

    // Count total - should be 12 (4x3 rectangle minus corners? let's just check)
    // Actually: 2 stones each expand to 3x3 = 9, but they overlap
    // Stone at (5,5): covers (4,4) to (6,6)
    // Stone at (6,5): covers (5,4) to (7,6)
    // Combined: (4,4) to (7,6) = 4 wide x 3 tall = 12
    auto positions = dilated.getSetPositions();
    CHECK(positions.size() == 12);
}

TEST_CASE("BitBoard dilate2 single stone") {
    BitBoard board(19);
    board.setBit(5, 5);
    BitBoard dilated = board.dilate2();

    // Expected 5x5 pattern centered at (5,5) -> top-left at (3,3)
    int pattern[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
    };

    int originX = 3, originY = 3;
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            bool expected = pattern[dy][dx] == 1;
            CHECK(dilated.getBit(originX + dx, originY + dy) == expected);
        }
    }

    // Verify nothing set outside the 5x5
    CHECK(dilated.getBit(2, 2) == false);
    CHECK(dilated.getBit(8, 8) == false);

    CHECK(dilated.getSetPositions().size() == 25);
}

TEST_CASE("BitBoard dilate2 corner respects boundaries") {
    BitBoard board(19);
    board.setBit(0, 0);
    BitBoard dilated = board.dilate2();

    // Only the valid 3x3 portion should be set
    int pattern[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
    };

    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            CHECK(dilated.getBit(dx, dy) == (pattern[dy][dx] == 1));
        }
    }

    CHECK(dilated.getSetPositions().size() == 9);
}

TEST_CASE("BitBoard dilate1_5 single stone") {
    BitBoard board(19);
    board.setBit(5, 5);
    BitBoard dilated = board.dilate1_5();

    // Cross/diamond pattern: 3x3 core + cardinal/diagonal distance-2
    // Knight's move positions are NOT included
    int pattern[5][5] = {
        {1, 0, 1, 0, 1},  // corners + center column
        {0, 1, 1, 1, 0},  // inner row
        {1, 1, 1, 1, 1},  // center row (full)
        {0, 1, 1, 1, 0},  // inner row
        {1, 0, 1, 0, 1},  // corners + center column
    };

    int originX = 3, originY = 3;
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            bool expected = pattern[dy][dx] == 1;
            CHECK(dilated.getBit(originX + dx, originY + dy) == expected);
        }
    }

    // 9 (3x3 core) + 4 (cardinals) + 4 (diagonal corners) = 17
    CHECK(dilated.getSetPositions().size() == 17);
}

TEST_CASE("BitBoard dilate1_5 corner respects boundaries") {
    BitBoard board(19);
    board.setBit(0, 0);
    BitBoard dilated = board.dilate1_5();

    // From dilate(): (0,0), (1,0), (0,1), (1,1)
    // From distance-2: (2,0), (0,2), (2,2)
    int pattern[3][3] = {
        {1, 1, 1},  // (0,0), (1,0), (2,0)
        {1, 1, 0},  // (0,1), (1,1), not (2,1)
        {1, 0, 1},  // (0,2), not (1,2), (2,2)
    };

    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            CHECK(dilated.getBit(dx, dy) == (pattern[dy][dx] == 1));
        }
    }

    // 4 from dilate + 3 distance-2 extensions = 7
    CHECK(dilated.getSetPositions().size() == 7);
}
