#include "Zobrist.hpp"

uint64_t Zobrist::computeFullHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
                                  int whiteCap) const {
    uint64_t h = 0;

    // XOR in black stones
    auto blackPos = blackStones.getSetPositions<>();
    for (auto [x, y] : blackPos) {
        h ^= stoneKeys[0][y * 19 + x];
    }

    // XOR in white stones
    auto whitePos = whiteStones.getSetPositions<>();
    for (auto [x, y] : whitePos) {
        h ^= stoneKeys[1][y * 19 + x];
    }

    // XOR in capture counts
    h ^= captureKeys[0][blackCap];
    h ^= captureKeys[1][whiteCap];

    return h;
}
