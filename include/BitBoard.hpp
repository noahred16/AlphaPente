#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include <cstdint>

class BitBoard {
private:
    static constexpr int MAX_BOARD_SIZE = 19;
    static constexpr int BITS_PER_UINT64 = 64;
    static constexpr int NUM_SEGMENTS = (MAX_BOARD_SIZE * MAX_BOARD_SIZE + BITS_PER_UINT64 - 1) / BITS_PER_UINT64;
    
    uint64_t board[NUM_SEGMENTS];
    int boardSize;
    
    int toIndex(int x, int y) const {
        return y * boardSize + x;
    }

public:
    BitBoard(int size = 19);
    
    // Core operations
    void setBit(int x, int y);
    void clearBit(int x, int y);
    bool getBit(int x, int y) const;
    void clear();
    
    // Just keep OR for finding occupied squares
    BitBoard operator|(const BitBoard& other) const;
};

#endif // BITBOARD_HPP
