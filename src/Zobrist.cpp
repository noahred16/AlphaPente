#include "Zobrist.hpp"
#include "Profiler.hpp"

uint64_t Zobrist::computeFullHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
                                  int whiteCap) const {
    PROFILE_SCOPE("Zobrist::computeFullHash");
    uint64_t h = 0;

    blackStones.forEachSetBit([&](int cell) { h ^= stoneKeys[0][cell]; });
    whiteStones.forEachSetBit([&](int cell) { h ^= stoneKeys[1][cell]; });

    h ^= captureKeys[0][blackCap];
    h ^= captureKeys[1][whiteCap];

    return h;
}

// uint64_t computeCanonicalHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
uint64_t Zobrist::computeCanonicalHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
                                       int whiteCap) const {
    PROFILE_SCOPE("Zobrist::computeCanonicalHash");
    uint64_t h[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // XOR stones (using precomposed symmetry keys)
    blackStones.forEachSetBit([&](int cell) {
        for (int s = 0; s < 8; ++s) {
            h[s] ^= symStoneKeys[s][0][cell];
        }
    });

    whiteStones.forEachSetBit([&](int cell) {
        for (int s = 0; s < 8; ++s) {
            h[s] ^= symStoneKeys[s][1][cell];
        }
    });

    // Captures are symmetry-invariant; add once to each candidate
    const uint64_t capPart = captureKeys[0][blackCap] ^ captureKeys[1][whiteCap];
    for (int s = 0; s < 8; ++s) {
        h[s] ^= capPart;
    }

    // Return minimum hash as canonical representative
    uint64_t best = h[0];
    for (int s = 1; s < 8; ++s) {
        if (h[s] < best)
            best = h[s];
    }
    return best;
}
