#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include <cstdint>
#include <vector>
#include <utility>

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
    BitBoard operator&(const BitBoard& other) const;
    BitBoard operator~() const;
    
    // Dilation (expand by 1 in all 8 directions)
    BitBoard dilate() const;
    
    // Extract all set positions
    template<typename T = std::pair<int, int>>
    std::vector<T> getSetPositions() const;
};

// In BitBoard.hpp after the class definition:
template<typename T>
std::vector<T> BitBoard::getSetPositions() const {
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
