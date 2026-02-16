#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include "BitBoard.hpp"
#include <cstdint>
#include <random>

class Zobrist {
public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;  // 361
    static constexpr int MAX_CAPTURES = 16;       // 0..15 covers Keryo
    static constexpr int NUM_SYMMETRIES = 8;

    uint64_t stoneKeys[2][BOARD_CELLS];    // [playerIdx][y*19+x]
    uint64_t captureKeys[2][MAX_CAPTURES]; // [playerIdx][captureCount]
    uint64_t sideToMoveKey;

    // symmetryMap[s][cell] = transformed cell index for symmetry s
    int symmetryMap[NUM_SYMMETRIES][BOARD_CELLS];

    static const Zobrist& instance() {
        static const Zobrist z;
        return z;
    }

    // playerIndex: BLACK=1 -> 0, WHITE=2 -> 1
    static int playerIndex(int p) { return p - 1; }

    uint64_t computeFullHash(const BitBoard& blackStones, const BitBoard& whiteStones,
                             int blackCap, int whiteCap, int currentPlayer) const;

    void computeAllHashes(uint64_t out[NUM_SYMMETRIES],
                          const BitBoard& blackStones, const BitBoard& whiteStones,
                          int blackCap, int whiteCap, int currentPlayer) const;

private:
    Zobrist() {
        std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < BOARD_CELLS; ++i)
                stoneKeys[p][i] = rng();
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < MAX_CAPTURES; ++i)
                captureKeys[p][i] = rng();
        sideToMoveKey = rng();
        initSymmetryMap();
    }

    void initSymmetryMap();
};

#endif // ZOBRIST_HPP
