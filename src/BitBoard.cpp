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