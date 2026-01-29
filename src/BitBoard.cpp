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

uint64_t BitBoard::MASK_NOT_COL_0[NUM_SEGMENTS] = {0};
uint64_t BitBoard::MASK_NOT_COL_18[NUM_SEGMENTS] = {0};
uint64_t BitBoard::MASK_NOT_COL_0_1[NUM_SEGMENTS] = {0};
uint64_t BitBoard::MASK_NOT_COL_17_18[NUM_SEGMENTS] = {0};
bool BitBoard::masksInitialized = false;

void BitBoard::initMasks() {
    if (masksInitialized) return;

    // 1. Initialize ALL masks to all-ones (0xFFFFFFFFFFFFFFFF)
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        MASK_NOT_COL_0[i] = ~0ULL;
        MASK_NOT_COL_18[i] = ~0ULL;
        MASK_NOT_COL_0_1[i] = ~0ULL;      // Added this
        MASK_NOT_COL_17_18[i] = ~0ULL;    // Added this
    }

    // 2. Punch out the prohibited columns
    for (int y = 0; y < 19; y++) {
        // Distance 1 columns
        int i0 = y * 19 + 0;
        MASK_NOT_COL_0[i0 / 64] &= ~(1ULL << (i0 % 64));
        
        int i18 = y * 19 + 18;
        MASK_NOT_COL_18[i18 / 64] &= ~(1ULL << (i18 % 64));

        // Distance 2 columns
        for (int x : {0, 1}) {
            int idx = y * 19 + x;
            MASK_NOT_COL_0_1[idx / 64] &= ~(1ULL << (idx % 64));
        }
        for (int x : {17, 18}) {
            int idx = y * 19 + x;
            MASK_NOT_COL_17_18[idx / 64] &= ~(1ULL << (idx % 64));
        }
    }

    // Handle the "Dead Bits" (Bits 361-383 in the 6th segment)
    // This prevents ~occupied from suggesting moves outside the 19x19 grid
    int totalBits = 19 * 19;
    int lastSegment = totalBits / 64;
    int remainingBits = totalBits % 64;
    uint64_t validBitsMask = (1ULL << remainingBits) - 1;
    
    // Apply this to all masks to keep the "junk" area clean
    MASK_NOT_COL_0[lastSegment] &= validBitsMask;
    MASK_NOT_COL_18[lastSegment] &= validBitsMask;
    MASK_NOT_COL_0_1[lastSegment] &= validBitsMask;
    MASK_NOT_COL_17_18[lastSegment] &= validBitsMask;

    masksInitialized = true;
}

// In BitBoard.cpp
void BitBoard::applyMask(const uint64_t maskArray[NUM_SEGMENTS]) {
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        board[i] &= maskArray[i];
    }
}

// Logic for Dilate (Distance 1)
BitBoard BitBoard::dilate() const {
    if (!masksInitialized) initMasks();

    BitBoard res = *this;

    // Vertical (No masking needed because there is no 'wrap' at top/bottom)
    res |= shiftFixed(19);  // Down
    res |= shiftFixed(-19); // Up

    // Prepare masked versions for Horizontal/Diagonal movement
    BitBoard maskL = *this; 
    maskL.applyMask(MASK_NOT_COL_0); // Safe to move Left
    
    BitBoard maskR = *this; 
    maskR.applyMask(MASK_NOT_COL_18); // Safe to move Right

    // Horizontal
    res |= maskL.shiftFixed(-1);
    res |= maskR.shiftFixed(1);

    // Diagonals
    res |= maskL.shiftFixed(-20); // Up-Left  (-19 - 1)
    res |= maskL.shiftFixed(18);  // Down-Left (+19 - 1)
    res |= maskR.shiftFixed(-18); // Up-Right (-19 + 1)
    res |= maskR.shiftFixed(20);  // Down-Right (+19 + 1)

    return res;
}

BitBoard& BitBoard::operator|=(const BitBoard& other) {
    // Standard Pente board is 6 segments (361 bits)
    // Unrolling this loop or using SIMD is possible, but 6 iterations is already blazing fast.
    for (int i = 0; i < NUM_SEGMENTS; ++i) {
        this->board[i] |= other.board[i];
    }
    return *this;
}

// In BitBoard.cpp
void BitBoard::orShifted(int count, const BitBoard& source) {
    if (count == 0) {
        *this |= source;
        return;
    }

    if (count > 0) { // Shift Forward (Right/Down)
        int wordShift = count >> 6;
        int bitShift = count & 63;
        for (int i = 0; i < NUM_SEGMENTS; ++i) {
            int target = i + wordShift;
            if (target < NUM_SEGMENTS) {
                this->board[target] |= (source.board[i] << bitShift);
                if (target + 1 < NUM_SEGMENTS && bitShift > 0)
                    this->board[target + 1] |= (source.board[i] >> (64 - bitShift));
            }
        }
    } else { // Shift Backward (Left/Up)
        int c = -count;
        int wordShift = c >> 6;
        int bitShift = c & 63;
        for (int i = NUM_SEGMENTS - 1; i >= 0; --i) {
            int target = i - wordShift;
            if (target >= 0) {
                this->board[target] |= (source.board[i] >> bitShift);
                if (target - 1 >= 0 && bitShift > 0)
                    this->board[target - 1] |= (source.board[i] << (64 - bitShift));
            }
        }
    }
}

BitBoard BitBoard::dilate1_5() const {
    if (!masksInitialized) initMasks();

    // Start with Distance 1
    BitBoard res = this->dilate(); 

    // Reuse these buffers to avoid calling the constructor 20 times
    BitBoard maskL2 = *this; maskL2.applyMask(MASK_NOT_COL_0_1);
    BitBoard maskR2 = *this; maskR2.applyMask(MASK_NOT_COL_17_18);

    // Use our new orShifted to modify 'res' in place
    res.orShifted(38, *this);  // Down 2
    res.orShifted(-38, *this); // Up 2

    res.orShifted(-2, maskL2);  // Left 2
    res.orShifted(-40, maskL2); // Up 2, Left 2
    res.orShifted(36, maskL2);  // Down 2, Left 2

    res.orShifted(2, maskR2);   // Right 2
    res.orShifted(-36, maskR2); // Up 2, Right 2
    res.orShifted(40, maskR2);  // Down 2, Right 2

    return res;
}

// BitBoard BitBoard::dilate1_5() const {
//     if (!masksInitialized) initMasks();

//     // Start with the standard 3x3 expansion
//     BitBoard res = this->dilate(); 

//     // Add only the straight-line extensions (Distance 2)
//     // Vertical
//     res |= shiftFixed(38);  // Down 2
//     res |= shiftFixed(-38); // Up 2

//     // Horizontal
//     BitBoard maskL2 = *this;
//     maskL2.applyMask(MASK_NOT_COL_0_1);
//     res |= maskL2.shiftFixed(-2); // Left 2

//     BitBoard maskR2 = *this;
//     maskR2.applyMask(MASK_NOT_COL_17_18);
//     res |= maskR2.shiftFixed(2);  // Right 2

//     // Diagonals (Straight lines only)
//     res |= maskL2.shiftFixed(-40); // Up 2, Left 2
//     res |= maskL2.shiftFixed(36);  // Down 2, Left 2
//     res |= maskR2.shiftFixed(-36); // Up 2, Right 2
//     res |= maskR2.shiftFixed(40);  // Down 2, Right 2

//     return res;
// }

// Optimized Fixed Shifter for the 6-segment array
BitBoard BitBoard::dilate2() const {
    // Start with distance 1 expansion
    BitBoard res = this->dilate();

    // Add distance 2 expansion
    // Vertical (no masking needed)
    res |= shiftFixed(38);  // Down 2
    res |= shiftFixed(-38); // Up 2

    // Prepare masks for distance-2 horizontal movement
    BitBoard maskL2 = *this;
    maskL2.applyMask(MASK_NOT_COL_0_1);

    BitBoard maskR2 = *this;
    maskR2.applyMask(MASK_NOT_COL_17_18);

    // Horizontal distance 2
    res |= maskL2.shiftFixed(-2);  // Left 2
    res |= maskR2.shiftFixed(2);   // Right 2

    // Diagonal distance 2 (corners of 5x5)
    res |= maskL2.shiftFixed(-40); // Up 2, Left 2
    res |= maskR2.shiftFixed(-36); // Up 2, Right 2
    res |= maskL2.shiftFixed(36);  // Down 2, Left 2
    res |= maskR2.shiftFixed(40);  // Down 2, Right 2

    // Prepare masks for distance-1 horizontal movement
    BitBoard maskL1 = *this;
    maskL1.applyMask(MASK_NOT_COL_0);

    BitBoard maskR1 = *this;
    maskR1.applyMask(MASK_NOT_COL_18);

    // Knight's move positions (Up/Down 2, Left/Right 1)
    res |= maskL1.shiftFixed(-39); // Up 2, Left 1
    res |= maskR1.shiftFixed(-37); // Up 2, Right 1
    res |= maskL1.shiftFixed(37);  // Down 2, Left 1
    res |= maskR1.shiftFixed(39);  // Down 2, Right 1

    // Knight's move positions (Up/Down 1, Left/Right 2)
    res |= maskL2.shiftFixed(-21); // Up 1, Left 2
    res |= maskR2.shiftFixed(-17); // Up 1, Right 2
    res |= maskL2.shiftFixed(17);  // Down 1, Left 2
    res |= maskR2.shiftFixed(21);  // Down 1, Right 2

    return res;
}

BitBoard BitBoard::shiftFixed(int count) const {
    BitBoard res(boardSize);
    if (count > 0) { // Shift "Forward" (Right/Down)
        int wordShift = count >> 6; // count / 64
        int bitShift = count & 63;  // count % 64
        // uint64_t carry = 0; TODO sus
        for (int i = 0; i < NUM_SEGMENTS; ++i) {
            int target = i + wordShift;
            if (target < NUM_SEGMENTS) {
                res.board[target] |= (board[i] << bitShift);
                if (target + 1 < NUM_SEGMENTS && bitShift > 0)
                    res.board[target + 1] |= (board[i] >> (64 - bitShift));
            }
        }
    } else { // Shift "Backward" (Left/Up)
        int c = -count;
        int wordShift = c >> 6;
        int bitShift = c & 63;
        for (int i = NUM_SEGMENTS - 1; i >= 0; --i) {
            int target = i - wordShift;
            if (target >= 0) {
                res.board[target] |= (board[i] >> bitShift);
                if (target - 1 >= 0 && bitShift > 0)
                    res.board[target - 1] |= (board[i] << (64 - bitShift));
            }
        }
    }
    return res;
}