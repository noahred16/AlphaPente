#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include "BitBoard.hpp"
#include <cstdint>
#include <random>

class Zobrist {
  public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int BOARD_CELLS = 19 * 19; // 361
    static constexpr int MAX_CAPTURES = 16;     // 0..15 covers Keryo

    uint64_t stoneKeys[2][BOARD_CELLS];    // [playerIdx][y*19+x]
    uint64_t captureKeys[2][MAX_CAPTURES]; // [playerIdx][captureCount]

    static const Zobrist &instance() {
        static const Zobrist z;
        return z;
    }

    // playerIndex: BLACK=1 -> 0, WHITE=2 -> 1
    static int playerIndex(int p) { return p - 1; }

    uint64_t computeFullHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
                             int whiteCap) const;
    uint64_t computeCanonicalHash(const BitBoard &blackStones, const BitBoard &whiteStones, int blackCap,
                                  int whiteCap) const;

  private:
    int symMap[8][BOARD_CELLS];

    // Optional micro-opt: symStoneKeys[sym][player][cell] = stoneKeys[player][ symMap[sym][cell] ]
    uint64_t symStoneKeys[8][2][BOARD_CELLS];

    Zobrist() {
        std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < BOARD_CELLS; ++i) {
                stoneKeys[p][i] = rng();
            }
        }
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < MAX_CAPTURES; ++i) {
                captureKeys[p][i] = rng();
            }
        }

        precomputeSymmetry();
    }

    void precomputeSymmetry() {
        const int N = BOARD_SIZE;
        const int n = N - 1;

        auto applySym = [&](int sym, int x, int y, int &ox, int &oy) {
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
        };

        for (int sym = 0; sym < 8; ++sym) {
            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    int cell = y * N + x;
                    int tx, ty;
                    applySym(sym, x, y, tx, ty);
                    int tcell = ty * N + tx;
                    symMap[sym][cell] = tcell;
                }
            }
        }

        // Precompose keys for speed: symStoneKeys[sym][p][cell] is the key to XOR for a stone at 'cell'
        // under symmetry 'sym'.
        for (int sym = 0; sym < 8; ++sym) {
            for (int p = 0; p < 2; ++p) {
                for (int cell = 0; cell < BOARD_CELLS; ++cell) {
                    symStoneKeys[sym][p][cell] = stoneKeys[p][symMap[sym][cell]];
                }
            }
        }
    }
};

#endif // ZOBRIST_HPP
