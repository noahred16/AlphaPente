#include "doctest.h"
#include "BitBoard.hpp"
#include "RenjuRules.hpp"

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
