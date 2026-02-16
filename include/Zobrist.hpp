#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include "BitBoard.hpp"
#include <cstdint>
#include <random>

class Zobrist {
  public:
    static constexpr int BOARD_CELLS = 19 * 19; // 361
    static constexpr int MAX_CAPTURES = 16;     // 0..15 covers Keryo

    uint64_t stoneKeys[2][BOARD_CELLS];    // [playerIdx][y*19+x]
    uint64_t captureKeys[2][MAX_CAPTURES]; // [playerIdx][captureCount]
    uint64_t sideToMoveKey;

    static const Zobrist &instance() {
        static const Zobrist z;
        return z;
    }

    // playerIndex: BLACK=1 -> 0, WHITE=2 -> 1
    static int playerIndex(int p) { return p - 1; }

    uint64_t computeFullHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap, int whiteCap,
                             int currentPlayer) const;

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
    }
};

#endif // ZOBRIST_HPP
