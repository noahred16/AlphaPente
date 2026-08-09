#include "doctest.h"
#include "RenjuForbiddenPointFinder.hpp"

using Finder = RenjuForbiddenPointFinder;

// ============================================================================
// Basic board mechanics
// ============================================================================

TEST_CASE("RenjuForbiddenPointFinder: fresh board is all empty") {
    Finder f(15);
    CHECK(f.getStone(0, 0) == Finder::EMPTY);
    CHECK(f.getStone(7, 7) == Finder::EMPTY);
    CHECK(f.getStone(14, 14) == Finder::EMPTY);
}

TEST_CASE("RenjuForbiddenPointFinder: setStone/getStone round-trip") {
    Finder f(15);
    f.setStone(3, 4, Finder::BLACK);
    f.setStone(5, 6, Finder::WHITE);
    CHECK(f.getStone(3, 4) == Finder::BLACK);
    CHECK(f.getStone(5, 6) == Finder::WHITE);
    CHECK(f.getStone(4, 4) == Finder::EMPTY);
}

TEST_CASE("RenjuForbiddenPointFinder: clear resets to empty") {
    Finder f(15);
    f.setStone(3, 4, Finder::BLACK);
    f.clear();
    CHECK(f.getStone(3, 4) == Finder::EMPTY);
}

TEST_CASE("RenjuForbiddenPointFinder: out-of-board coordinates read as BORDER") {
    Finder f(15);
    CHECK(f.getStone(-1, 0) == Finder::BORDER);
    CHECK(f.getStone(0, -1) == Finder::BORDER);
    CHECK(f.getStone(15, 0) == Finder::BORDER);
    CHECK(f.getStone(0, 15) == Finder::BORDER);
}

// ============================================================================
// isFive - black needs exactly 5, white needs >= 5
// ============================================================================

TEST_CASE("isFive: black exact five in a row (horizontal)") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    f.setStone(3, 5, Finder::BLACK);
    CHECK(f.isFive(4, 5, 0, 0) == true);
    CHECK(f.isFive(4, 5, 0) == true);
}

TEST_CASE("isFive: black six in a row is NOT a five (overline excluded)") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    f.setStone(3, 5, Finder::BLACK);
    f.setStone(5, 5, Finder::BLACK);
    CHECK(f.isFive(4, 5, 0, 0) == false);
    CHECK(f.isFive(4, 5, 0) == false);
}

TEST_CASE("isFive: white six in a row still counts as a win") {
    Finder f(15);
    f.setStone(0, 5, Finder::WHITE);
    f.setStone(1, 5, Finder::WHITE);
    f.setStone(2, 5, Finder::WHITE);
    f.setStone(3, 5, Finder::WHITE);
    f.setStone(5, 5, Finder::WHITE);
    CHECK(f.isFive(4, 5, 1, 0) == true);
    CHECK(f.isFive(4, 5, 1) == true);
}

TEST_CASE("isFive: only four in a row is not a five") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    CHECK(f.isFive(3, 5, 0, 0) == false);
    CHECK(f.isFive(3, 5, 0) == false);
}

TEST_CASE("isFive: vertical and diagonal directions") {
    Finder f(15);
    f.setStone(7, 0, Finder::BLACK);
    f.setStone(7, 1, Finder::BLACK);
    f.setStone(7, 2, Finder::BLACK);
    f.setStone(7, 3, Finder::BLACK);
    CHECK(f.isFive(7, 4, 0, 1) == true); // {0,1} vertical

    Finder f2(15);
    f2.setStone(0, 0, Finder::BLACK);
    f2.setStone(1, 1, Finder::BLACK);
    f2.setStone(2, 2, Finder::BLACK);
    f2.setStone(3, 3, Finder::BLACK);
    CHECK(f2.isFive(4, 4, 0, 2) == true); // {1,1} diagonal

    Finder f3(15);
    f3.setStone(4, 0, Finder::BLACK);
    f3.setStone(3, 1, Finder::BLACK);
    f3.setStone(2, 2, Finder::BLACK);
    f3.setStone(1, 3, Finder::BLACK);
    CHECK(f3.isFive(0, 4, 0, 3) == true); // {1,-1} anti-diagonal
}

// ============================================================================
// isOverline - black only
// ============================================================================

TEST_CASE("isOverline: exact five is not an overline") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    f.setStone(3, 5, Finder::BLACK);
    CHECK(f.isOverline(4, 5) == false);
}

TEST_CASE("isOverline: six in a row is an overline") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    f.setStone(3, 5, Finder::BLACK);
    f.setStone(5, 5, Finder::BLACK);
    CHECK(f.isOverline(4, 5) == true);
}

TEST_CASE("isOverline: seven in a row is an overline") {
    Finder f(15);
    for (int x : {0, 1, 2, 3, 5, 6}) f.setStone(x, 5, Finder::BLACK);
    CHECK(f.isOverline(4, 5) == true);
}

// ============================================================================
// isFour
// ============================================================================

TEST_CASE("isFour: single-sided four (blocked by the board edge) is still a four") {
    Finder f(15);
    f.setStone(0, 0, Finder::BLACK);
    f.setStone(1, 0, Finder::BLACK);
    f.setStone(2, 0, Finder::BLACK);
    CHECK(f.isFour(3, 0, 0, 0) == true);
}

TEST_CASE("isFour: four boxed in by opponent stones on both ends is not a real four") {
    Finder f(15);
    f.setStone(4, 0, Finder::WHITE);
    f.setStone(5, 0, Finder::BLACK);
    f.setStone(6, 0, Finder::BLACK);
    f.setStone(7, 0, Finder::BLACK);
    f.setStone(9, 0, Finder::WHITE);
    CHECK(f.isFour(8, 0, 0, 0) == false);
}

TEST_CASE("isFour: a four whose only completions would overline is not a real four") {
    // B _ B B B B _ B  (candidate fills the gap at x=3, making the run {2,3,4,5});
    // both flanking gaps (x=1 and x=6) would extend it to six, not five.
    Finder f(15);
    f.setStone(0, 1, Finder::BLACK);
    f.setStone(2, 1, Finder::BLACK);
    f.setStone(4, 1, Finder::BLACK);
    f.setStone(5, 1, Finder::BLACK);
    f.setStone(7, 1, Finder::BLACK);
    CHECK(f.isOverline(3, 1) == false); // the candidate itself only makes a four, not an overline
    CHECK(f.isFour(3, 1, 0, 0) == false);
}

TEST_CASE("isFour: white four is unaffected by the black overline exception") {
    Finder f(15);
    f.setStone(0, 1, Finder::WHITE);
    f.setStone(2, 1, Finder::WHITE);
    f.setStone(4, 1, Finder::WHITE);
    f.setStone(5, 1, Finder::WHITE);
    f.setStone(7, 1, Finder::WHITE);
    CHECK(f.isFour(3, 1, 1, 0) == true);
}

// ============================================================================
// isOpenFour - 0, 1, or 2 live completions
// ============================================================================

TEST_CASE("isOpenFour: open four with both ends live returns 2") {
    Finder f(15);
    f.setStone(5, 3, Finder::BLACK);
    f.setStone(6, 3, Finder::BLACK);
    f.setStone(7, 3, Finder::BLACK);
    CHECK(f.isOpenFour(8, 3, 0, 0) == 2);
}

TEST_CASE("isOpenFour: four blocked on one end returns 1") {
    Finder f(15);
    f.setStone(0, 0, Finder::BLACK);
    f.setStone(1, 0, Finder::BLACK);
    f.setStone(2, 0, Finder::BLACK);
    CHECK(f.isOpenFour(3, 0, 0, 0) == 1);
}

TEST_CASE("isOpenFour: four blocked on both ends returns 0") {
    Finder f(15);
    f.setStone(4, 0, Finder::WHITE);
    f.setStone(5, 0, Finder::BLACK);
    f.setStone(6, 0, Finder::BLACK);
    f.setStone(7, 0, Finder::BLACK);
    f.setStone(9, 0, Finder::WHITE);
    CHECK(f.isOpenFour(8, 0, 0, 0) == 0);
}

// ============================================================================
// isOpenThree
// ============================================================================

TEST_CASE("isOpenThree: classic open three with room on both sides") {
    Finder f(15);
    f.setStone(5, 4, Finder::BLACK);
    f.setStone(6, 4, Finder::BLACK);
    CHECK(f.isOpenThree(7, 4, 0, 0) == true);
}

TEST_CASE("isOpenThree: three pinned against the board edge is not a real (open) three") {
    Finder f(15);
    f.setStone(0, 2, Finder::BLACK);
    f.setStone(1, 2, Finder::BLACK);
    CHECK(f.isOpenThree(2, 2, 0, 0) == false);
}

TEST_CASE("isOpenThree: three immediately boxed in on one side is not open") {
    // White sits right against the run, so every way of growing this three towards a four can
    // only ever pick up one live completion (the far side), never two.
    Finder f(15);
    f.setStone(1, 6, Finder::WHITE);
    f.setStone(2, 6, Finder::BLACK);
    f.setStone(3, 6, Finder::BLACK);
    CHECK(f.isOpenThree(4, 6, 0, 0) == false);
}

// ============================================================================
// isDoubleFour
// ============================================================================

TEST_CASE("isDoubleFour: two real fours crossing at the candidate") {
    Finder f(15);
    // Horizontal four through (7,7): 5,6,_,8
    f.setStone(5, 7, Finder::BLACK);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    // Vertical four through (7,7): 4,5,6,_
    f.setStone(7, 4, Finder::BLACK);
    f.setStone(7, 5, Finder::BLACK);
    f.setStone(7, 6, Finder::BLACK);
    CHECK(f.isDoubleFour(7, 7) == true);
}

TEST_CASE("isDoubleFour: only one four present is not a double-four") {
    Finder f(15);
    f.setStone(5, 7, Finder::BLACK);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    CHECK(f.isDoubleFour(7, 7) == false);
}

TEST_CASE("isDoubleFour: a real four plus a fake (overline-only) four is not a double-four") {
    Finder f(15);
    // Real horizontal four through (7,7).
    f.setStone(5, 7, Finder::BLACK);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    // Fake vertical "four" through (7,7): both completions would overline (mirrors the isFour test;
    // real stones at offsets -3,-1,+1,+2 from the candidate, run = candidate's offsets -1..+2).
    f.setStone(7, 4, Finder::BLACK);
    f.setStone(7, 6, Finder::BLACK);
    f.setStone(7, 8, Finder::BLACK);
    f.setStone(7, 9, Finder::BLACK);
    f.setStone(7, 11, Finder::BLACK);
    CHECK(f.isFour(7, 7, 0, 1) == false);  // sanity: the vertical line is not a real four
    CHECK(f.isDoubleFour(7, 7) == false);
}

// ============================================================================
// isDoubleThree
// ============================================================================

TEST_CASE("isDoubleThree: two open threes crossing at the candidate") {
    Finder f(15);
    // Horizontal three through (7,7): 6,_,8
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    // Vertical three through (7,7): 6,_,8
    f.setStone(7, 6, Finder::BLACK);
    f.setStone(7, 8, Finder::BLACK);
    CHECK(f.isDoubleThree(7, 7) == true);
}

TEST_CASE("isDoubleThree: only one open three is not a double-three") {
    Finder f(15);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    CHECK(f.isDoubleThree(7, 7) == false);
}

// ============================================================================
// isForbidden - combines the rules, with five-in-a-row taking priority
// ============================================================================

TEST_CASE("isForbidden: overline is forbidden") {
    Finder f(15);
    f.setStone(0, 5, Finder::BLACK);
    f.setStone(1, 5, Finder::BLACK);
    f.setStone(2, 5, Finder::BLACK);
    f.setStone(3, 5, Finder::BLACK);
    f.setStone(5, 5, Finder::BLACK);
    CHECK(f.isForbidden(4, 5) == true);
}

TEST_CASE("isForbidden: double-four is forbidden") {
    Finder f(15);
    f.setStone(5, 7, Finder::BLACK);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    f.setStone(7, 4, Finder::BLACK);
    f.setStone(7, 5, Finder::BLACK);
    f.setStone(7, 6, Finder::BLACK);
    CHECK(f.isForbidden(7, 7) == true);
}

TEST_CASE("isForbidden: double-three is forbidden") {
    Finder f(15);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    f.setStone(7, 6, Finder::BLACK);
    f.setStone(7, 8, Finder::BLACK);
    CHECK(f.isForbidden(7, 7) == true);
}

TEST_CASE("isForbidden: a plain single three or single four is legal") {
    Finder f(15);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    CHECK(f.isForbidden(7, 7) == false);
}

TEST_CASE("isForbidden: completing an exact five always wins, even alongside a double-three") {
    Finder f(15);
    // Horizontal exact five through (7,7): 3,4,5,6,_ ; kept off the board edge at both (2,7) and
    // (8,7) so it can't accidentally extend into a six.
    f.setStone(3, 7, Finder::BLACK);
    f.setStone(4, 7, Finder::BLACK);
    f.setStone(5, 7, Finder::BLACK);
    f.setStone(6, 7, Finder::BLACK);
    // Two more open threes crossing the same point, which would be a forbidden double-three
    // on their own.
    f.setStone(7, 6, Finder::BLACK);
    f.setStone(7, 8, Finder::BLACK);
    f.setStone(6, 6, Finder::BLACK);
    f.setStone(8, 8, Finder::BLACK);

    CHECK(f.isFive(7, 7, 0) == true);
    CHECK(f.isDoubleThree(7, 7) == true); // would be forbidden in isolation...
    CHECK(f.isForbidden(7, 7) == false);  // ...but the five takes priority
}

TEST_CASE("isForbidden: an occupied point is never forbidden") {
    Finder f(15);
    f.setStone(7, 7, Finder::BLACK);
    CHECK(f.isForbidden(7, 7) == false);
}

// ============================================================================
// findForbiddenPoints
// ============================================================================

TEST_CASE("findForbiddenPoints: finds the double-three point and nothing unrelated") {
    Finder f(15);
    f.setStone(6, 7, Finder::BLACK);
    f.setStone(8, 7, Finder::BLACK);
    f.setStone(7, 6, Finder::BLACK);
    f.setStone(7, 8, Finder::BLACK);

    auto forbidden = f.findForbiddenPoints();
    bool found = false;
    for (auto &c : forbidden) {
        if (c.x == 7 && c.y == 7) found = true;
        CHECK(f.getStone(c.x, c.y) == Finder::EMPTY);
    }
    CHECK(found == true);
}

TEST_CASE("findForbiddenPoints: empty board has no forbidden points") {
    Finder f(15);
    CHECK(f.findForbiddenPoints().empty());
}

// ============================================================================
// Regression: the exact game reported as a bug (see H5 in the game log below).
// Moves: 1.H8 H7 2.H6 J7 3.G7 J9 4.J5 K4 5.F8 E9 6.G8 J8 7.J10 H9 8.G9 G6 9.F5 G10
//        10.K7 F6 11.K9 E10 12.E8 D8 13.L8 H11 14.M7 N6
// The engine reported H5 as a winning move; H5 is in fact a Black double-three and must be
// forbidden. Coordinates below are 0-based, local to the 15x15 Renju board (physical - minIdx).
// ============================================================================

TEST_CASE("Regression: H5 in the reported game is a forbidden black double-three") {
    Finder f(15);
    // Black
    for (auto [x, y] : {std::pair{5, 5}, {5, 3}, {4, 4}, {6, 2}, {3, 5}, {4, 5},
                         {6, 7}, {4, 6}, {3, 2}, {7, 4}, {7, 6}, {2, 5}, {8, 5}, {9, 4}}) {
        f.setStone(x, y, Finder::BLACK);
    }
    // White
    for (auto [x, y] : {std::pair{5, 4}, {6, 4}, {6, 6}, {7, 1}, {2, 6}, {6, 5},
                         {5, 6}, {4, 3}, {4, 7}, {3, 3}, {2, 7}, {1, 5}, {5, 8}, {10, 3}}) {
        f.setStone(x, y, Finder::WHITE);
    }

    // H5 -> (5, 2) local.
    CHECK(f.getStone(5, 2) == Finder::EMPTY);
    CHECK(f.isOverline(5, 2) == false);
    CHECK(f.isDoubleFour(5, 2) == false);
    CHECK(f.isDoubleThree(5, 2) == true);
    CHECK(f.isForbidden(5, 2) == true);
}
