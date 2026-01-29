#include "BitBoard.hpp"
#include <cstring>  // for memset and memcpy

BitBoard::BitBoard(int size) : boardSize(size) {
    // Initialize all bits to 0
    std::memset(board, 0, sizeof(board));
}

void BitBoard::setBit(int x, int y) {
    // Check bounds
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
        return;
    }
    
    int index = toIndex(x, y);
    int segment = index / BITS_PER_UINT64;
    int bit = index % BITS_PER_UINT64;
    
    board[segment] |= (1ULL << bit);
}

void BitBoard::clearBit(int x, int y) {
    // Check bounds
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
        return;
    }
    
    int index = toIndex(x, y);
    int segment = index / BITS_PER_UINT64;
    int bit = index % BITS_PER_UINT64;
    
    board[segment] &= ~(1ULL << bit);
}

bool BitBoard::getBit(int x, int y) const {
    // Return false for out of bounds
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
        return false;
    }
    
    int index = toIndex(x, y);
    int segment = index / BITS_PER_UINT64;
    int bit = index % BITS_PER_UINT64;
    
    return (board[segment] >> bit) & 1;
}

void BitBoard::clear() {
    std::memset(board, 0, sizeof(board));
}

BitBoard BitBoard::operator|(const BitBoard& other) const {
    BitBoard result(boardSize);
    
    // OR each segment
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        result.board[i] = board[i] | other.board[i];
    }
    
    return result;
}

BitBoard BitBoard::operator&(const BitBoard& other) const {
    BitBoard result(boardSize);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        result.board[i] = board[i] & other.board[i];
    }
    return result;
}

BitBoard BitBoard::operator~() const {
    BitBoard result(boardSize);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        result.board[i] = ~board[i];
    }
    // Mask off bits beyond boardSize * boardSize
    int totalBits = boardSize * boardSize;
    int lastSegment = totalBits / BITS_PER_UINT64;
    int lastBit = totalBits % BITS_PER_UINT64;
    if (lastBit > 0) {
        result.board[lastSegment] &= (1ULL << lastBit) - 1;
    }
    return result;
}

BitBoard BitBoard::dilate() const {
    BitBoard result = *this;
    
    // Process each segment
    for (int seg = 0; seg < NUM_SEGMENTS; ++seg) {
        uint64_t word = board[seg];
        
        while (word) {
            int bit_pos = __builtin_ctzll(word);
            int global_pos = seg * BITS_PER_UINT64 + bit_pos;
            int y = global_pos / boardSize;
            int x = global_pos % boardSize;
            
            // Set 8 neighbors directly
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize) {
                        int nIndex = ny * boardSize + nx;
                        result.board[nIndex / BITS_PER_UINT64] |= (1ULL << (nIndex % BITS_PER_UINT64));
                    }
                }
            }
            
            word &= word - 1;
        }
    }
    
    return result;
}