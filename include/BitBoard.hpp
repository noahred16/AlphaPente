#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include <cstdint>
#include <utility>
#include <vector>

class BitBoard {
  private:
    static constexpr int MAX_BOARD_SIZE = 19;
    static constexpr int BITS_PER_UINT64 = 64;
    static constexpr int NUM_SEGMENTS = (MAX_BOARD_SIZE * MAX_BOARD_SIZE + BITS_PER_UINT64 - 1) / BITS_PER_UINT64; // 6

    uint64_t board[NUM_SEGMENTS];
    int boardSize;

    // Masks to prevent wrapping around the edges
    static uint64_t MASK_NOT_COL_0[NUM_SEGMENTS];
    static uint64_t MASK_NOT_COL_18[NUM_SEGMENTS];
    static uint64_t MASK_NOT_COL_0_1[NUM_SEGMENTS];
    static uint64_t MASK_NOT_COL_17_18[NUM_SEGMENTS];
    static bool masksInitialized;

    int toIndex(int x, int y) const { return y * boardSize + x; }

  public:
    BitBoard(int size = 19);

    static void initMasks();
    void applyMask(const uint64_t maskArray[NUM_SEGMENTS]);

    // Core operations (with bounds checking)
    void setBit(int x, int y);
    void clearBit(int x, int y);
    bool getBit(int x, int y) const;
    void clear();

    // Unchecked operations (caller must guarantee bounds)
    inline bool getBitUnchecked(int x, int y) const {
        int index = y * boardSize + x;
        return (board[index >> 6] >> (index & 63)) & 1;
    }

    inline void setBitUnchecked(int x, int y) {
        int index = y * boardSize + x;
        board[index >> 6] |= (1ULL << (index & 63));
    }

    inline void clearBitUnchecked(int x, int y) {
        int index = y * boardSize + x;
        board[index >> 6] &= ~(1ULL << (index & 63));
    }

    // Just keep OR for finding occupied squares
    BitBoard &operator|=(const BitBoard &other);
    BitBoard operator|(const BitBoard &other) const;
    BitBoard operator&(const BitBoard &other) const;
    BitBoard operator~() const;

    // Dilation (expand by 1 in all 8 directions)
    BitBoard dilate() const;
    BitBoard dilate1_5() const; // Dilate by 1.5 steps
    BitBoard dilate2() const;   // Dilate by 2 steps
    BitBoard shiftFixed(int count) const;
    void orShifted(int count, const BitBoard &source);

    // Extract all set positions
    template <typename T = std::pair<int, int>> std::vector<T> getSetPositions() const;
};

// In BitBoard.hpp after the class definition:
template <typename T> std::vector<T> BitBoard::getSetPositions() const {
    std::vector<T> positions;

    for (int seg = 0; seg < NUM_SEGMENTS; ++seg) {
        uint64_t word = board[seg];

        while (word) {
            int bit_pos = __builtin_ctzll(word);
            int global_pos = seg * BITS_PER_UINT64 + bit_pos;
            int y = global_pos / boardSize;
            int x = global_pos % boardSize;

            if (x < boardSize && y < boardSize) {
                positions.push_back({x, y});
            }

            word &= word - 1;
        }
    }

    return positions;
}

#endif // BITBOARD_HPP
