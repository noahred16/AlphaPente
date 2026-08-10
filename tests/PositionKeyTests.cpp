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

// Regression test for a real bug (now fixed at the source): PenteGame's
// minIdx()/maxIdx() used to truncate asymmetrically for even boardSize
// (BOARD_SIZE=19 is odd), so boardSize=4 used to actually yield a 5-wide
// window (physical [7,12), not [7,11)) - PositionKey packing against the
// requested boardSize=4 instead of that (wrong) true width silently dropped
// the 5th row/column from the key, aliasing distinct positions together.
// PenteGame::maxIdx() is now minIdx()+boardSize, giving a genuinely 4-wide
// window - this checks PositionKey packs all 16 cells of it correctly (no
// off-by-one gaps at the far edge) and that a cell outside the true window
// (physical x/y=11) is correctly out of bounds.
TEST_CASE("PositionKey packs a genuinely 4-wide window correctly") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 4;
    PenteGame game(config);
    game.reset();

    REQUIRE(game.minIdx() == 7);
    REQUIRE(game.maxIdx() == 11); // exclusive: window is {7,8,9,10}, genuinely 4 wide

    game.makeMove(9, 9);   // forced center - still in-bounds for the 4-wide window
    game.makeMove(10, 10); // farthest legal corner of the true 4-wide window (White)

    PositionKey key = PositionKey::pack(game);
    auto unpacked = PositionKey::unpack(key, 4);

    const int lo = 7;
    CHECK(unpacked.cell[static_cast<size_t>((9 - lo) * 4 + (9 - lo))] == PenteGame::BLACK);
    CHECK(unpacked.cell[static_cast<size_t>((10 - lo) * 4 + (10 - lo))] == PenteGame::WHITE);
}

// Regression test for a second bug that followed directly from fixing the
// first one: a genuinely 4-wide window sits at physical [7,11), which is
// NOT centered on the full 19x19 grid's own center (9) - true 4-wide window
// center would be 8.5, unlike odd boardSize where minIdx()/maxIdx() divide
// evenly and the window is grid-centered. PNS used to bridge canonical<->
// physical coordinates via Zobrist::applySymToMove/applyInverseSym, which
// is defined relative to the *grid's* center - correct only when the window
// happens to be grid-centered (odd boardSize), silently wrong for this
// asymmetric even-boardSize window. That produced wrong physical
// coordinates fed into PenteGame::makeMove() (which has no occupancy check
// of its own), corrupting real solve5x5 runs in a way that looked like an
// "impossible" recursion depth (>50000 on a board that can't legally exceed
// ~34 plies) rather than an obvious crash.
//
// PositionKey::applySymToPhysical/applyInverseSymToPhysical replace that
// bridge with a window-relative transform, guaranteed self-consistent with
// pack()/canonical() by construction regardless of grid-centering. This
// checks exactly the property that broke: round-tripping through apply then
// inverse-apply recovers the original physical coordinate, and the image
// never leaves the window, for every symmetry and every cell of a
// genuinely-asymmetric (even boardSize) window.
TEST_CASE("PositionKey's physical symmetry transform round-trips correctly on an asymmetric even window") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 4;
    PenteGame game(config);
    game.reset();

    const int lo = game.minIdx();
    const int hi = game.maxIdx();
    REQUIRE(hi - lo == 4);

    for (int sym = 0; sym < 8; ++sym) {
        for (int py = lo; py < hi; ++py) {
            for (int px = lo; px < hi; ++px) {
                int fx, fy;
                PositionKey::applySymToPhysical(game, sym, px, py, fx, fy);
                CHECK(fx >= lo);
                CHECK(fx < hi);
                CHECK(fy >= lo);
                CHECK(fy < hi);

                int rx, ry;
                PositionKey::applyInverseSymToPhysical(game, sym, fx, fy, rx, ry);
                CHECK(rx == px);
                CHECK(ry == py);
            }
        }
    }
}

// PenteGame::loadRawState() exists for PNS checkpoint/resume (src/PNS.cpp):
// reconstructing a position directly from its packed key, rather than
// needing to replay a move history that was never actually recorded (a DAG
// node may be reachable via many different move orders). This checks the
// full round trip - pack a real (capture-involving) game, unpack it,
// reconstruct via loadRawState, and verify the result is byte-for-byte
// equivalent for everything that matters: stones, captures, side-to-move,
// moveCount, and critically the canonical key itself (PNS's resume logic
// looks nodes up by canonical key, so the reconstruction MUST reproduce it
// exactly) - plus that the reconstructed game still behaves correctly for
// further play (a second capture triggers normally).
TEST_CASE("PenteGame::loadRawState reconstructs a position exactly from its packed key") {
    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = 5;
    config.tournamentRule = false;
    PenteGame original(config);
    original.reset();

    // Same bracket-capture scenario used elsewhere: B(9,9) forced center,
    // then B(7,8)-W(8,8)-W(9,8)-B(10,8) captures the two whites.
    original.makeMove(9, 9);
    original.makeMove(8, 8);
    original.makeMove(7, 8);
    original.makeMove(9, 8);
    original.makeMove(10, 8);
    REQUIRE(original.getBlackCaptures() == 2);

    int origSym = -1;
    PositionKey origCanon = PositionKey::canonical(original, origSym);
    auto unpacked = PositionKey::unpack(PositionKey::pack(original), 5);

    PenteGame reconstructed(config);
    reconstructed.loadRawState(unpacked.cell.data(), unpacked.sideToMove, unpacked.blackCaptures, unpacked.whiteCaptures);

    for (int y = 7; y < 12; ++y) {
        for (int x = 7; x < 12; ++x) {
            CHECK(reconstructed.getStoneAt(x, y) == original.getStoneAt(x, y));
        }
    }
    CHECK(reconstructed.getCurrentPlayer() == original.getCurrentPlayer());
    CHECK(reconstructed.getBlackCaptures() == original.getBlackCaptures());
    CHECK(reconstructed.getWhiteCaptures() == original.getWhiteCaptures());
    CHECK(reconstructed.getMoveCount() == original.getMoveCount());

    int reconSym = -1;
    PositionKey reconCanon = PositionKey::canonical(reconstructed, reconSym);
    CHECK(reconCanon == origCanon);

    // Further play still works correctly on the reconstructed game - a
    // simple move updates state exactly as it would on a normally-played
    // game, confirming loadRawState() left it in a genuinely valid state
    // (correct hash/bitboards/moveCount), not just a snapshot that happens
    // to read back the same values.
    REQUIRE(reconstructed.getCurrentPlayer() == PenteGame::WHITE);
    reconstructed.makeMove(11, 11);
    CHECK(reconstructed.getStoneAt(11, 11) == PenteGame::WHITE);
    CHECK(reconstructed.getMoveCount() == original.getMoveCount() + 1);
    CHECK(reconstructed.getCurrentPlayer() == PenteGame::BLACK);

    // And a genuine second capture still works, confirming captures (which
    // depend on correctly-set bitboards/hash, not just cosmetics) function
    // normally on the reconstructed game - a fresh bracket on row y=10
    // (still within the 5-wide window [7,12)): B(7,10) W(8,10) ... W(9,10)
    // B(10,10) captures.
    reconstructed.makeMove(7, 10);  // Black
    reconstructed.makeMove(8, 10);  // White
    reconstructed.makeMove(7, 11);  // Black (elsewhere)
    reconstructed.makeMove(9, 10);  // White
    reconstructed.makeMove(10, 10); // Black completes bracket B(7,10) W(8,10) W(9,10) B(10,10)
    CHECK(reconstructed.getBlackCaptures() == original.getBlackCaptures() + 2);
    CHECK(reconstructed.getStoneAt(8, 10) == PenteGame::NONE);
    CHECK(reconstructed.getStoneAt(9, 10) == PenteGame::NONE);
}
