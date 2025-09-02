#include <gtest/gtest.h>
#include "core/bitboard.hpp"

using namespace core;

class BitBoardTest : public ::testing::Test {
protected:
    BitBoard board;
};

TEST_F(BitBoardTest, EmptyBoardTest) {
    // Test that a new board is empty
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            EXPECT_TRUE(board.is_empty(row, col));
            EXPECT_EQ(board.get_stone(row, col), 0);
        }
    }
}

TEST_F(BitBoardTest, SetAndGetStone) {
    // Test setting and getting player 1 stone
    board.set_stone(9, 9, 1);
    EXPECT_EQ(board.get_stone(9, 9), 1);
    EXPECT_FALSE(board.is_empty(9, 9));
    
    // Test setting and getting player -1 stone
    board.set_stone(5, 5, -1);
    EXPECT_EQ(board.get_stone(5, 5), -1);
    EXPECT_FALSE(board.is_empty(5, 5));
    
    // Verify other positions remain empty
    EXPECT_TRUE(board.is_empty(0, 0));
    EXPECT_TRUE(board.is_empty(18, 18));
}

TEST_F(BitBoardTest, OverwriteStone) {
    // Place player 1 stone
    board.set_stone(10, 10, 1);
    EXPECT_EQ(board.get_stone(10, 10), 1);
    
    // Overwrite with player -1 stone
    board.set_stone(10, 10, -1);
    EXPECT_EQ(board.get_stone(10, 10), -1);
    
    // Overwrite with player 1 stone again
    board.set_stone(10, 10, 1);
    EXPECT_EQ(board.get_stone(10, 10), 1);
}

TEST_F(BitBoardTest, RemoveStone) {
    // Place stones
    board.set_stone(7, 8, 1);
    board.set_stone(8, 7, -1);
    
    // Verify they exist
    EXPECT_EQ(board.get_stone(7, 8), 1);
    EXPECT_EQ(board.get_stone(8, 7), -1);
    
    // Remove them
    board.remove_stone(7, 8);
    board.remove_stone(8, 7);
    
    // Verify they're gone
    EXPECT_TRUE(board.is_empty(7, 8));
    EXPECT_TRUE(board.is_empty(8, 7));
    EXPECT_EQ(board.get_stone(7, 8), 0);
    EXPECT_EQ(board.get_stone(8, 7), 0);
}

TEST_F(BitBoardTest, CornerPositions) {
    // Test all four corners
    board.set_stone(0, 0, 1);      // Top-left
    board.set_stone(0, 18, -1);    // Top-right  
    board.set_stone(18, 0, 1);     // Bottom-left
    board.set_stone(18, 18, -1);   // Bottom-right
    
    EXPECT_EQ(board.get_stone(0, 0), 1);
    EXPECT_EQ(board.get_stone(0, 18), -1);
    EXPECT_EQ(board.get_stone(18, 0), 1);
    EXPECT_EQ(board.get_stone(18, 18), -1);
}

TEST_F(BitBoardTest, EdgePositions) {
    // Test edge positions
    board.set_stone(0, 9, 1);      // Top edge
    board.set_stone(18, 9, -1);    // Bottom edge
    board.set_stone(9, 0, 1);      // Left edge
    board.set_stone(9, 18, -1);    // Right edge
    
    EXPECT_EQ(board.get_stone(0, 9), 1);
    EXPECT_EQ(board.get_stone(18, 9), -1);
    EXPECT_EQ(board.get_stone(9, 0), 1);
    EXPECT_EQ(board.get_stone(9, 18), -1);
}

TEST_F(BitBoardTest, BitIndexCalculation) {
    // Test that our bit index calculation works correctly
    // Position (0,0) should map to bit 0
    board.set_stone(0, 0, 1);
    EXPECT_EQ(board.get_stone(0, 0), 1);
    
    // Position (0,18) should map to bit 18
    board.set_stone(0, 18, -1);
    EXPECT_EQ(board.get_stone(0, 18), -1);
    
    // Position (1,0) should map to bit 19
    board.set_stone(1, 0, 1);
    EXPECT_EQ(board.get_stone(1, 0), 1);
    
    // Position (18,18) should map to bit 360 (last position)
    board.set_stone(18, 18, -1);
    EXPECT_EQ(board.get_stone(18, 18), -1);
}

TEST_F(BitBoardTest, All361Positions) {
    // Test that we can set and retrieve stones at all 361 board positions
    int stone_value = 1;
    
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            board.set_stone(row, col, stone_value);
            EXPECT_EQ(board.get_stone(row, col), stone_value);
            
            // Alternate between player 1 and -1
            stone_value = -stone_value;
        }
    }
    
    // Verify all positions are still correct
    stone_value = 1;
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            EXPECT_EQ(board.get_stone(row, col), stone_value);
            stone_value = -stone_value;
        }
    }
}