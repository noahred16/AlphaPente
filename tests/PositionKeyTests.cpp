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
