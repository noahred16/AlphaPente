#include "PositionKey.hpp"
#include <cassert>

namespace {

// D4 symmetry transform applied to window-relative coordinates (0..n), same
// convention/ordering as Zobrist::precomputeSymmetry.
void applySym(int sym, int n, int x, int y, int &ox, int &oy) {
    switch (sym) {
    case 0:
        ox = x;
        oy = y;
        break; // identity
    case 1:
        ox = y;
        oy = n - x;
        break; // rot90 CW
    case 2:
        ox = n - x;
        oy = n - y;
        break; // rot180
    case 3:
        ox = n - y;
        oy = x;
        break; // rot270 CW
    case 4:
        ox = n - x;
        oy = y;
        break; // mirror L<->R
    case 5:
        ox = y;
        oy = x;
        break; // mirror + rot90
    case 6:
        ox = x;
        oy = n - y;
        break; // mirror + rot180
    case 7:
        ox = n - y;
        oy = n - x;
        break; // mirror + rot270
    default:
        ox = x;
        oy = y;
        break;
    }
}

} // namespace

PositionKey PositionKey::packSym(const PenteGame &game, int sym) {
    const int lo = game.minIdx();
    // The TRUE logical window is [minIdx(), maxIdx()), not necessarily
    // config().boardSize cells wide: minIdx()=(BOARD_SIZE-boardSize)/2
    // truncates for even boardSize (BOARD_SIZE=19 is odd), so
    // maxIdx()-minIdx() can be one cell wider than the requested boardSize
    // - e.g. boardSize=4 actually yields a 5-wide window. Packing against
    // the requested boardSize instead of this actual width would silently
    // drop a whole row/column from the key, aliasing distinct positions
    // together. Odd boardSize (the actual 5x5 target, and 3x3) has no such
    // truncation, so this only bites even sizes - but must be handled
    // generally rather than assumed away.
    const int boardSize = game.maxIdx() - lo;
    assert(boardSize >= 1 && boardSize <= kMaxBoardSize && "PositionKey packing only supports boardSize <= 5");
    const int n = boardSize - 1;

    uint64_t bits = 0;
    for (int y = 0; y < boardSize; ++y) {
        for (int x = 0; x < boardSize; ++x) {
            PenteGame::Player p = game.getStoneAt(lo + x, lo + y);
            if (p == PenteGame::NONE) continue;
            int ox, oy;
            applySym(sym, n, x, y, ox, oy);
            int cell = oy * boardSize + ox;
            uint64_t code = (p == PenteGame::BLACK) ? 1ULL : 2ULL;
            bits |= code << (2 * cell);
        }
    }

    const int blackCap = game.getBlackCaptures();
    const int whiteCap = game.getWhiteCaptures();
    // blackCaptures/whiteCaptures count captured stones (not events), so this holds as
    // long as capturesToWin <= 16, true for every Config preset (Pente=10, Keryo=15).
    assert(blackCap >= 0 && blackCap < 16 && "black capture count exceeds 4-bit packing");
    assert(whiteCap >= 0 && whiteCap < 16 && "white capture count exceeds 4-bit packing");

    const int base = 2 * boardSize * boardSize;
    bits |= static_cast<uint64_t>(game.getCurrentPlayer() == PenteGame::WHITE ? 1 : 0) << base;
    bits |= static_cast<uint64_t>(blackCap) << (base + 1);
    bits |= static_cast<uint64_t>(whiteCap) << (base + 5);

    return PositionKey{bits};
}

PositionKey PositionKey::pack(const PenteGame &game) { return packSym(game, 0); }

PositionKey PositionKey::canonical(const PenteGame &game, int &outSym) {
    PositionKey best = packSym(game, 0);
    outSym = 0;
    for (int s = 1; s < 8; ++s) {
        PositionKey cand = packSym(game, s);
        if (cand.bits < best.bits) {
            best = cand;
            outSym = s;
        }
    }
    return best;
}

PositionKey::Unpacked PositionKey::unpack(const PositionKey &key, int windowSize) {
    assert(windowSize >= 1 && windowSize <= kMaxBoardSize && "PositionKey unpacking only supports windowSize <= 5");
    Unpacked out;
    const uint64_t bits = key.bits;

    for (int y = 0; y < windowSize; ++y) {
        for (int x = 0; x < windowSize; ++x) {
            int cell = y * windowSize + x;
            uint64_t code = (bits >> (2 * cell)) & 0x3ULL;
            PenteGame::Player p = (code == 1) ? PenteGame::BLACK : (code == 2) ? PenteGame::WHITE : PenteGame::NONE;
            out.cell[static_cast<size_t>(cell)] = p;
        }
    }

    const int base = 2 * windowSize * windowSize;
    out.sideToMove = ((bits >> base) & 0x1ULL) ? PenteGame::WHITE : PenteGame::BLACK;
    out.blackCaptures = static_cast<int>((bits >> (base + 1)) & 0xFULL);
    out.whiteCaptures = static_cast<int>((bits >> (base + 5)) & 0xFULL);

    return out;
}
