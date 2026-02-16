#include "Zobrist.hpp"

void Zobrist::initSymmetryMap() {
    constexpr int N = BOARD_SIZE - 1; // 18
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            int cell = y * BOARD_SIZE + x;
            // D4 symmetries for NxN board centered at (N/2, N/2)
            symmetryMap[0][cell] = y * BOARD_SIZE + x;              // identity
            symmetryMap[1][cell] = x * BOARD_SIZE + (N - y);        // rotate 90 CW: (x,y)->(N-y,x)
            symmetryMap[2][cell] = (N - y) * BOARD_SIZE + (N - x);  // rotate 180: (x,y)->(N-x,N-y)
            symmetryMap[3][cell] = (N - x) * BOARD_SIZE + y;        // rotate 270 CW: (x,y)->(y,N-x)
            symmetryMap[4][cell] = y * BOARD_SIZE + (N - x);        // reflect horiz: (x,y)->(N-x,y)
            symmetryMap[5][cell] = (N - y) * BOARD_SIZE + x;        // reflect vert: (x,y)->(x,N-y)
            symmetryMap[6][cell] = x * BOARD_SIZE + y;              // reflect main diag: (x,y)->(y,x)
            symmetryMap[7][cell] = (N - x) * BOARD_SIZE + (N - y);  // reflect anti-diag: (x,y)->(N-y,N-x)
        }
    }
}

uint64_t Zobrist::computeFullHash(const BitBoard& blackStones, const BitBoard& whiteStones,
                                   int blackCap, int whiteCap, int currentPlayer) const {
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

    // XOR in side to move (convention: XOR when it's WHITE's turn)
    if (currentPlayer == 2) { // WHITE
        h ^= sideToMoveKey;
    }

    return h;
}

void Zobrist::computeAllHashes(uint64_t out[NUM_SYMMETRIES],
                                const BitBoard& blackStones, const BitBoard& whiteStones,
                                int blackCap, int whiteCap, int currentPlayer) const {
    for (int s = 0; s < NUM_SYMMETRIES; ++s)
        out[s] = 0;

    // XOR in black stones across all symmetries
    auto blackPos = blackStones.getSetPositions<>();
    for (auto [x, y] : blackPos) {
        int cell = y * BOARD_SIZE + x;
        for (int s = 0; s < NUM_SYMMETRIES; ++s)
            out[s] ^= stoneKeys[0][symmetryMap[s][cell]];
    }

    // XOR in white stones across all symmetries
    auto whitePos = whiteStones.getSetPositions<>();
    for (auto [x, y] : whitePos) {
        int cell = y * BOARD_SIZE + x;
        for (int s = 0; s < NUM_SYMMETRIES; ++s)
            out[s] ^= stoneKeys[1][symmetryMap[s][cell]];
    }

    // Capture counts and side-to-move are symmetric — same for all 8
    uint64_t shared = captureKeys[0][blackCap] ^ captureKeys[1][whiteCap];
    if (currentPlayer == 2) // WHITE
        shared ^= sideToMoveKey;

    for (int s = 0; s < NUM_SYMMETRIES; ++s)
        out[s] ^= shared;
}
