#ifndef POSITIONKEY_HPP
#define POSITIONKEY_HPP

#include "PenteGame.hpp"
#include <array>
#include <cstdint>
#include <functional>

// Exact (bijective, not hashed) packing of a PenteGame position into a 64-bit
// integer, for use as a Proof-Number-Search transposition/DAG key.
//
// Unlike Zobrist::computeCanonicalHash (a *statistical* 64-bit hash - fine for
// MCTS's heuristic transposition table, where a wrong hit just wastes some
// search, but not safe as the sole key for a *definitive* proof database:
// birthday-bound collision probability becomes non-negligible once the number
// of distinct positions reaches ~10^8-10^9), PositionKey packs the logical
// board exactly: no two distinct positions can ever produce the same key, so
// there is nothing to argue statistically about collisions.
//
// Layout (LSB first), for a boardSize x boardSize logical window:
//   2 bits per cell (00=empty, 01=black, 10=white), row-major over the window
//                                              -> 2 * boardSize * boardSize bits
//   1 bit  side-to-move   (0=BLACK, 1=WHITE)
//   4 bits black stones captured (PenteGame::getBlackCaptures(), 0 until just
//                                  under capturesToWin, e.g. 0-9 for Pente's
//                                  default capturesToWin=10)
//   4 bits white stones captured (symmetric)
// For boardSize=5: 2*25 + 1 + 4 + 4 = 59 bits, fits a uint64_t. This scheme
// only fits boardSize <= 5 (boardSize=6 would need 2*36+9=81 bits); pack()
// asserts on that bound rather than silently overflowing.
class PositionKey {
  public:
    static constexpr int kMaxBoardSize = 5;

    uint64_t bits = 0;

    bool operator==(const PositionKey &o) const { return bits == o.bits; }
    bool operator!=(const PositionKey &o) const { return bits != o.bits; }
    bool operator<(const PositionKey &o) const { return bits < o.bits; }

    // Exact packing of the physical position, no symmetry applied.
    static PositionKey pack(const PenteGame &game);

    // Packing under one of the 8 D4 symmetries (same indexing convention as
    // Zobrist: 0=identity, 1=rot90CW, 2=rot180, 3=rot270CW, 4=mirror L<->R,
    // 5=mirror+rot90, 6=mirror+rot180, 7=mirror+rot270), applied to
    // window-relative coordinates.
    static PositionKey packSym(const PenteGame &game, int sym);

    // Canonical packing: the minimum PositionKey over all 8 D4 symmetries of
    // the logical window. outSym receives which symmetry produced it, so
    // callers can un-rotate stored move/child info back to physical
    // coordinates.
    static PositionKey canonical(const PenteGame &game, int &outSym);

    // Reconstructed contents of a packed key, for round-trip testing and
    // (later) book export. boardSize must match what the key was packed with.
    struct Unpacked {
        std::array<PenteGame::Player, kMaxBoardSize * kMaxBoardSize> cell{};
        PenteGame::Player sideToMove = PenteGame::BLACK;
        int blackCaptures = 0;
        int whiteCaptures = 0;
    };
    static Unpacked unpack(const PositionKey &key, int boardSize);
};

namespace std {
template <> struct hash<PositionKey> {
    size_t operator()(const PositionKey &k) const noexcept { return std::hash<uint64_t>{}(k.bits); }
};
} // namespace std

#endif // POSITIONKEY_HPP
