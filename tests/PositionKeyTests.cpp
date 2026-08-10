#include "PositionKey.hpp"
#include "PenteGame.hpp"
#include "doctest.h"

// PositionKey is an exact (bijective) packing, not a statistical hash - these
// tests check pack/unpack round-tripping and D4-symmetry canonicalization,
// the two properties the PNS transposition DAG (Phase 2) will rely on.

TEST_CASE("PositionKey round-trips an empty board") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();

    PositionKey key = PositionKey::pack(game);
    auto unpacked = PositionKey::unpack(key, 5);

    for (int i = 0; i < 25; ++i) {
        CHECK(unpacked.cell[static_cast<size_t>(i)] == PenteGame::NONE);
    }
    CHECK(unpacked.sideToMove == PenteGame::BLACK);
    CHECK(unpacked.blackCaptures == 0);
    CHECK(unpacked.whiteCaptures == 0);
}

TEST_CASE("PositionKey round-trips stones and side-to-move without captures") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();

    game.makeMove(9, 9); // black, forced center
    game.makeMove(8, 8); // white
    game.makeMove(10, 11); // black

    PositionKey key = PositionKey::pack(game);
    auto unpacked = PositionKey::unpack(key, 5);

    const int lo = 7; // minIdx() for a 5-wide window centered in a 19-wide grid
    auto cellAt = [&](int x, int y) { return unpacked.cell[static_cast<size_t>((y - lo) * 5 + (x - lo))]; };

    CHECK(cellAt(9, 9) == PenteGame::BLACK);
    CHECK(cellAt(8, 8) == PenteGame::WHITE);
    CHECK(cellAt(10, 11) == PenteGame::BLACK);
    CHECK(unpacked.sideToMove == PenteGame::WHITE); // black just moved
    CHECK(unpacked.blackCaptures == 0);
    CHECK(unpacked.whiteCaptures == 0);
}

TEST_CASE("PositionKey round-trips capture counts") {
    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = 5;
    config.tournamentRule = false; // matches apps/Pente.cpp's auto-disable below boardSize 7
    PenteGame game(config);
    game.reset();

    // Bracket capture: B(7,8) W(8,8) W(9,8) B(10,8) - Black's move at (10,8)
    // captures the two flanked White stones.
    game.makeMove(9, 9);  // black, forced center
    game.makeMove(8, 8);  // white
    game.makeMove(7, 8);  // black
    game.makeMove(9, 8);  // white
    game.makeMove(10, 8); // black completes the bracket, captures (8,8) and (9,8)

    REQUIRE(game.getBlackCaptures() == 2); // captures count stones removed (2 per bracket), not events
    REQUIRE(game.getWhiteCaptures() == 0);
    REQUIRE(game.getStoneAt(8, 8) == PenteGame::NONE); // captured
    REQUIRE(game.getStoneAt(9, 8) == PenteGame::NONE); // captured

    PositionKey key = PositionKey::pack(game);
    auto unpacked = PositionKey::unpack(key, 5);

    const int lo = 7;
    auto cellAt = [&](int x, int y) { return unpacked.cell[static_cast<size_t>((y - lo) * 5 + (x - lo))]; };

    CHECK(cellAt(9, 9) == PenteGame::BLACK);
    CHECK(cellAt(7, 8) == PenteGame::BLACK);
    CHECK(cellAt(10, 8) == PenteGame::BLACK);
    CHECK(cellAt(8, 8) == PenteGame::NONE);
    CHECK(cellAt(9, 8) == PenteGame::NONE);
    CHECK(unpacked.sideToMove == PenteGame::WHITE);
    CHECK(unpacked.blackCaptures == 2);
    CHECK(unpacked.whiteCaptures == 0);
}

TEST_CASE("PositionKey canonicalization is identical for a physically rotated board") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;

    // Window-relative coords (0..4), physical = 7 + relative.
    // An asymmetric shape so all 8 D4 images are distinct (no accidental symmetry).
    const int lo = 7;
    struct WMove {
        int wx, wy;
    };
    const WMove shapeA[] = {{2, 2}, {0, 0}, {1, 0}, {3, 4}};

    PenteGame gameA(config);
    gameA.reset();
    for (const auto &m : shapeA) gameA.makeMove(lo + m.wx, lo + m.wy);

    // rot90CW under the same convention as PositionKey::packSym / Zobrist: ox=y, oy=n-x, n=4.
    PenteGame gameB(config);
    gameB.reset();
    for (const auto &m : shapeA) {
        int ox = m.wy;
        int oy = 4 - m.wx;
        gameB.makeMove(lo + ox, lo + oy);
    }

    int symA = -1, symB = -1;
    PositionKey canonA = PositionKey::canonical(gameA, symA);
    PositionKey canonB = PositionKey::canonical(gameB, symB);

    CHECK(canonA == canonB);
}

TEST_CASE("PositionKey canonicalization picks the same key regardless of which symmetry is physically played") {
    // All 8 physical rotations/reflections of one shape must canonicalize to one key.
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    const int lo = 7;
    const int n = 4;

    struct WMove {
        int wx, wy;
    };
    const WMove shapeA[] = {{2, 2}, {0, 1}, {1, 0}, {3, 4}};

    auto applySym = [&](int sym, int x, int y, int &ox, int &oy) {
        switch (sym) {
        case 0: ox = x; oy = y; break;
        case 1: ox = y; oy = n - x; break;
        case 2: ox = n - x; oy = n - y; break;
        case 3: ox = n - y; oy = x; break;
        case 4: ox = n - x; oy = y; break;
        case 5: ox = y; oy = x; break;
        case 6: ox = x; oy = n - y; break;
        case 7: ox = n - y; oy = n - x; break;
        }
    };

    PositionKey reference;
    reference.bits = 0;
    bool first = true;
    for (int sym = 0; sym < 8; ++sym) {
        PenteGame game(config);
        game.reset();
        for (const auto &m : shapeA) {
            int ox, oy;
            applySym(sym, m.wx, m.wy, ox, oy);
            game.makeMove(lo + ox, lo + oy);
        }
        int outSym = -1;
        PositionKey canon = PositionKey::canonical(game, outSym);
        if (first) {
            reference = canon;
            first = false;
        } else {
            CHECK(canon == reference);
        }
    }
}

// Regression test for a real bug: PenteGame::minIdx()/maxIdx() truncate
// asymmetrically for even boardSize (BOARD_SIZE=19 is odd), so boardSize=4
// actually yields a 5-wide window (physical [7,12), not [7,11)). Packing
// against the requested boardSize=4 instead of that true width silently
// dropped the 5th row/column from the key, aliasing distinct positions
// together - this was the actual root cause of an "impossible" recursion
// depth (>50000, far beyond any legally reachable 4x4 game length) seen in a
// real long solve5x5 run. The real 5x5/3x3 targets are odd-sized and
// unaffected (no truncation), but PositionKey must still be correct for any
// supported boardSize.
TEST_CASE("PositionKey packs the true (possibly wider-than-requested) window for even boardSize") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 4;
    PenteGame gameA(config);
    gameA.reset();
    gameA.makeMove(9, 9); // forced center

    PenteGame gameB = gameA.clone();

    // (11, 9) is inside the true 5-wide window (physical [7,12)) but outside
    // the naive 4-wide interpretation [7,11) - exactly the cell the bug
    // dropped. A stone there must still affect the packed key. This is
    // White's move (move index 1, right after Black's forced-center index 0).
    gameA.makeMove(11, 9);
    gameB.makeMove(10, 9); // a different, in-bounds-either-way move instead

    PositionKey keyA = PositionKey::pack(gameA);
    PositionKey keyB = PositionKey::pack(gameB);
    CHECK(keyA != keyB);

    const int trueWindowSize = 5; // maxIdx() - minIdx() for boardSize=4, not config.boardSize
    auto unpacked = PositionKey::unpack(keyA, trueWindowSize);
    const int lo = 7;
    CHECK(unpacked.cell[static_cast<size_t>((9 - lo) * trueWindowSize + (11 - lo))] == PenteGame::WHITE);
}
