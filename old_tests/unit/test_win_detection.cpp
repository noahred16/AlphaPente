#include <gtest/gtest.h>
#include "core/bitboard.hpp"

using namespace core;

class WinDetectionTest : public ::testing::Test {
protected:
    BitBoard board;
    
    void SetUp() override {
        board = BitBoard();
    }
    
    void place_horizontal_line(int row, int start_col, int player, int count) {
        for (int i = 0; i < count; ++i) {
            board.set_stone(row, start_col + i, player);
        }
    }
    
    void place_vertical_line(int start_row, int col, int player, int count) {
        for (int i = 0; i < count; ++i) {
            board.set_stone(start_row + i, col, player);
        }
    }
    
    void place_diagonal_line(int start_row, int start_col, int dr, int dc, int player, int count) {
        for (int i = 0; i < count; ++i) {
            board.set_stone(start_row + i * dr, start_col + i * dc, player);
        }
    }
};

TEST_F(WinDetectionTest, HorizontalWin) {
    // Test horizontal win in middle of board
    place_horizontal_line(9, 7, 1, 5);
    
    // Check all positions in the line detect the win
    for (int col = 7; col < 12; ++col) {
        EXPECT_TRUE(board.check_five_in_a_row(9, col, 1))
            << "Failed at position (" << 9 << "," << col << ")";
    }
    
    // Check positions outside the line don't detect win
    EXPECT_FALSE(board.check_five_in_a_row(9, 6, 1));
    EXPECT_FALSE(board.check_five_in_a_row(9, 12, 1));
    EXPECT_FALSE(board.check_five_in_a_row(8, 9, 1));
}

TEST_F(WinDetectionTest, VerticalWin) {
    // Test vertical win
    place_vertical_line(5, 10, -1, 5);
    
    // Check all positions in the line detect the win
    for (int row = 5; row < 10; ++row) {
        EXPECT_TRUE(board.check_five_in_a_row(row, 10, -1))
            << "Failed at position (" << row << "," << 10 << ")";
    }
    
    // Check positions outside don't detect win
    EXPECT_FALSE(board.check_five_in_a_row(4, 10, -1));
    EXPECT_FALSE(board.check_five_in_a_row(10, 10, -1));
}

TEST_F(WinDetectionTest, DiagonalWinUpRight) {
    // Test diagonal win (/)
    place_diagonal_line(10, 5, -1, 1, 1, 5);
    
    // Check all positions in the diagonal detect the win
    for (int i = 0; i < 5; ++i) {
        int row = 10 - i;
        int col = 5 + i;
        EXPECT_TRUE(board.check_five_in_a_row(row, col, 1))
            << "Failed at position (" << row << "," << col << ")";
    }
}

TEST_F(WinDetectionTest, DiagonalWinDownRight) {
    // Test diagonal win (\)
    place_diagonal_line(3, 8, 1, 1, -1, 5);
    
    // Check all positions in the diagonal detect the win
    for (int i = 0; i < 5; ++i) {
        int row = 3 + i;
        int col = 8 + i;
        EXPECT_TRUE(board.check_five_in_a_row(row, col, -1))
            << "Failed at position (" << row << "," << col << ")";
    }
}

TEST_F(WinDetectionTest, EdgeWinConditions) {
    // Test horizontal win at left edge
    place_horizontal_line(5, 0, 1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(5, 2, 1));
    
    // Test horizontal win at right edge
    place_horizontal_line(10, 14, -1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(10, 16, -1));
    
    // Test vertical win at top edge
    place_vertical_line(0, 9, 1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(2, 9, 1));
    
    // Test vertical win at bottom edge
    place_vertical_line(14, 7, -1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(16, 7, -1));
}

TEST_F(WinDetectionTest, CornerWinConditions) {
    // Test diagonal win from top-left corner
    place_diagonal_line(0, 0, 1, 1, 1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(2, 2, 1));
    
    // Test diagonal win to bottom-right corner
    place_diagonal_line(14, 14, 1, 1, -1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(16, 16, -1));
    
    // Test diagonal win from top-right
    place_diagonal_line(0, 18, 1, -1, 1, 5);
    EXPECT_TRUE(board.check_five_in_a_row(2, 16, 1));
}

TEST_F(WinDetectionTest, NearMissConditions) {
    // Test 4 in a row (should not win)
    place_horizontal_line(9, 7, 1, 4);
    EXPECT_FALSE(board.check_five_in_a_row(9, 9, 1));
    
    // Test gap in 5 stones
    place_horizontal_line(5, 3, -1, 2);
    place_horizontal_line(5, 6, -1, 2);
    EXPECT_FALSE(board.check_five_in_a_row(5, 4, -1));
    EXPECT_FALSE(board.check_five_in_a_row(5, 6, -1));
    
    // Test mixed players
    board.set_stone(12, 5, 1);
    board.set_stone(12, 6, -1);
    board.set_stone(12, 7, 1);
    board.set_stone(12, 8, 1);
    board.set_stone(12, 9, 1);
    EXPECT_FALSE(board.check_five_in_a_row(12, 7, 1));
}

TEST_F(WinDetectionTest, LongerThanFiveWin) {
    // Test 6 in a row (should still win)
    place_horizontal_line(8, 4, 1, 6);
    EXPECT_TRUE(board.check_five_in_a_row(8, 6, 1));
    
    // Test 7 in a row
    place_vertical_line(2, 15, -1, 7);
    EXPECT_TRUE(board.check_five_in_a_row(5, 15, -1));
}

TEST_F(WinDetectionTest, NoWinOnEmptyBoard) {
    // Test that empty positions don't show wins
    EXPECT_FALSE(board.check_five_in_a_row(9, 9, 1));
    EXPECT_FALSE(board.check_five_in_a_row(9, 9, -1));
}

TEST_F(WinDetectionTest, SingleStoneNoWin) {
    // Test single stone doesn't create win
    board.set_stone(10, 10, 1);
    EXPECT_FALSE(board.check_five_in_a_row(10, 10, 1));
}

TEST_F(WinDetectionTest, AllBoardPositions) {
    // Test that we can place stones and check wins at all 361 positions
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            BitBoard test_board;
            if (col <= 14) {  // Can place horizontal line
                for (int i = 0; i < 5; ++i) {
                    test_board.set_stone(row, col + i, 1);
                }
                EXPECT_TRUE(test_board.check_five_in_a_row(row, col + 2, 1))
                    << "Failed horizontal win at position (" << row << "," << col << ")";
            }
        }
    }
}