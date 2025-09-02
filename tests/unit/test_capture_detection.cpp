#include <gtest/gtest.h>
#include "core/game_state.hpp"
#include <algorithm>

using namespace core;

class CaptureDetectionTest : public ::testing::Test {
protected:
    GameState game;
    
    void SetUp() override {
        // GameState is already default-constructed as a member
        // No need to assign since copy assignment is deleted
    }
    
    void setup_basic_capture_pattern() {
        // setup 
        game.reset();
        // Set up: X - O - O - ? pattern for player 1 to capture
        game.make_move(5, 5);  // Player 1
        game.make_move(5, 6);  // Player -1  
        game.make_move(10, 10); // Player 1 (dummy move)
        game.make_move(5, 7);  // Player -1
        game.make_move(11, 11); // Player 1 (dummy move)
        // Now it's player -1's turn, but we want to test player 1 capturing
        game.switch_player(); // Make it player 1's turn
    }
};

TEST_F(CaptureDetectionTest, BasicHorizontalCapture) {
    setup_basic_capture_pattern();
    
    // Player 1 places stone at (5,8) to complete capture pattern
    MoveDelta delta = game.make_move(5, 8);
    
    // Should have captured 2 stones
    EXPECT_EQ(delta.capture_count, 2);
    EXPECT_EQ(game.get_captures(1), 2);
    EXPECT_EQ(game.get_captures(-1), 0);
    
    // Captured positions should be (5,6) and (5,7) in some order
    EXPECT_EQ(delta.captured_stones[0].row, 5);
    EXPECT_EQ(delta.captured_stones[1].row, 5);
    // Check that we captured the right columns (order may vary)
    std::array<int, 2> captured_cols = {delta.captured_stones[0].col, delta.captured_stones[1].col};
    std::sort(captured_cols.begin(), captured_cols.end());
    EXPECT_EQ(captured_cols[0], 6);
    EXPECT_EQ(captured_cols[1], 7);
    
    // Board should show empty positions where stones were captured
    EXPECT_TRUE(game.is_empty(5, 6));
    EXPECT_TRUE(game.is_empty(5, 7));
    EXPECT_EQ(game.get_stone(5, 5), 1);
    EXPECT_EQ(game.get_stone(5, 8), 1);
}

TEST_F(CaptureDetectionTest, VerticalCapture) {
    // Set up vertical capture pattern
    game.make_move(5, 10);  // Player 1
    game.make_move(6, 10);  // Player -1
    game.make_move(10, 5);  // Player 1 (dummy)
    game.make_move(7, 10);  // Player -1
    game.make_move(11, 5);  // Player 1 (dummy)
    game.switch_player();   // Make it player 1's turn
    
    MoveDelta delta = game.make_move(8, 10);
    
    EXPECT_EQ(delta.capture_count, 2);
    EXPECT_EQ(game.get_captures(1), 2);
    EXPECT_TRUE(game.is_empty(6, 10));
    EXPECT_TRUE(game.is_empty(7, 10));
}

TEST_F(CaptureDetectionTest, DiagonalCapture) {
    // Set up diagonal capture pattern
    game.make_move(5, 5);   // Player 1
    game.make_move(6, 6);   // Player -1
    game.make_move(10, 10); // Player 1 (dummy)
    game.make_move(7, 7);   // Player -1
    game.make_move(11, 11); // Player 1 (dummy)
    game.switch_player();   // Make it player 1's turn
    
    MoveDelta delta = game.make_move(8, 8);
    
    EXPECT_EQ(delta.capture_count, 2);
    EXPECT_EQ(game.get_captures(1), 2);
    EXPECT_TRUE(game.is_empty(6, 6));
    EXPECT_TRUE(game.is_empty(7, 7));
}

TEST_F(CaptureDetectionTest, MultipleCaptures) {
    // Set up multiple capture opportunities
    // Horizontal capture
    game.make_move(5, 5);   // Player 1
    game.make_move(5, 6);   // Player -1
    game.make_move(10, 10); // Player 1 (dummy)
    game.make_move(5, 7);   // Player -1
    game.make_move(11, 11); // Player 1 (dummy)
    
    // Vertical capture setup
    game.make_move(6, 8);   // Player -1
    game.make_move(8, 8);   // Player 1
    game.make_move(7, 8);   // Player -1

    // Player 1 can capture both horizontally and vertically
    MoveDelta delta1 = game.make_move(5, 8);   // Player 1
    EXPECT_EQ(delta1.capture_count, 4);
    EXPECT_EQ(game.get_captures(1), 4);
    EXPECT_TRUE(game.is_empty(5, 6));
    EXPECT_TRUE(game.is_empty(5, 7));
    EXPECT_TRUE(game.is_empty(6, 8));
    EXPECT_TRUE(game.is_empty(7, 8));
}

TEST_F(CaptureDetectionTest, NoFalsePositives) {
    // Test patterns that should NOT result in captures
    
    // Only one opponent stone
    game.make_move(5, 5);   // Player 1
    game.make_move(5, 6);   // Player -1
    game.make_move(5, 7);   // Player 1
    MoveDelta delta = game.make_move(5, 8);   // Player -1
    EXPECT_EQ(delta.capture_count, 0);
    
    // Reset game state for next test
    game.reset();
    
    // Three opponent stones (too many)
    game.make_move(10, 10); // Player 1
    game.make_move(10, 11); // Player -1
    game.make_move(10, 12); // Player 1
    game.make_move(10, 13); // Player -1
    game.make_move(10, 14); // Player 1
    delta = game.make_move(10, 15); // Player -1
    EXPECT_EQ(delta.capture_count, 0);
}

TEST_F(CaptureDetectionTest, EdgeCaptures) {
    // Test captures near board edges
    
    // Capture at left edge
    game.make_move(9, 0);   // Player 1
    game.make_move(9, 1);   // Player -1
    game.make_move(10, 10); // Player 1 (dummy)
    game.make_move(9, 2);   // Player -1
    game.make_move(11, 11); // Player 1 (dummy)
    game.switch_player();   // Make it player 1's turn
    
    MoveDelta delta = game.make_move(9, 3);
    EXPECT_EQ(delta.capture_count, 2);
    
    // Test capture at top edge
    game.reset();
    game.make_move(0, 9);   // Player 1
    game.make_move(1, 9);   // Player -1
    game.make_move(10, 10); // Player 1 (dummy)
    game.make_move(2, 9);   // Player -1
    game.make_move(11, 11); // Player 1 (dummy)
    game.switch_player();   // Make it player 1's turn
    
    delta = game.make_move(3, 9);
    EXPECT_EQ(delta.capture_count, 2);
}

TEST_F(CaptureDetectionTest, UndoCapture) {
    setup_basic_capture_pattern();
    
    // Make capturing move
    MoveDelta delta = game.make_move(5, 8);
    EXPECT_EQ(game.get_captures(1), 2);
    EXPECT_TRUE(game.is_empty(5, 6));
    EXPECT_TRUE(game.is_empty(5, 7));
    
    // Undo the move
    game.undo_move();
    
    // Captures should be restored
    EXPECT_EQ(game.get_captures(1), 0);
    EXPECT_EQ(game.get_stone(5, 6), -1);
    EXPECT_EQ(game.get_stone(5, 7), -1);
    EXPECT_TRUE(game.is_empty(5, 8));
}

TEST_F(CaptureDetectionTest, CaptureWinCondition) {
    // Test win by capturing 10 stones
    // Need to set up multiple capture opportunities
    
    // First capture
    game.make_move(0, 0);   // Player 1
    game.make_move(0, 1);   // Player -1
    game.make_move(0, 2);   // Player 1
    game.make_move(0, 3);   // Player -1
    game.make_move(0, 4);   // Player 1 (dummy)
    game.switch_player();   // Back to player 1
    game.make_move(0, 5);   // Player 1 captures (2 total)
    
    // Second capture  
    game.make_move(1, 0);   // Player -1
    game.make_move(1, 1);   // Player 1
    game.make_move(1, 2);   // Player -1
    game.make_move(1, 3);   // Player 1
    game.make_move(1, 4);   // Player -1 captures (2 total for player -1)
    
    // Continue pattern to get player 1 to 10 captures
    for (int row = 2; row < 6; ++row) {
        game.make_move(row, 0);   // Player 1
        game.make_move(row, 1);   // Player -1
        game.make_move(row, 2);   // Player 1  
        game.make_move(row, 3);   // Player -1
        game.make_move(18, row);  // Player 1 (dummy)
        game.switch_player();     // Back to player 1
        MoveDelta delta = game.make_move(row, 4);  // Player 1 captures
        
        if (game.get_captures(1) >= 10) {
            EXPECT_TRUE(game.check_win(row, 4, 1));
            EXPECT_TRUE(game.is_terminal());
            EXPECT_EQ(game.get_winner(), 1);
            break;
        }
    }
}

TEST_F(CaptureDetectionTest, NoCaptureOnWrongPlayer) {
    // Test that stones of same color don't get "captured"
    game.make_move(5, 5);   // Player 1
    game.make_move(5, 6);   // Player -1
    game.make_move(5, 7);   // Player 1
    game.make_move(5, 8);   // Player -1
    game.make_move(5, 9);   // Player 1
    
    MoveDelta delta = game.make_move(10, 10);  // Player -1 (dummy move)
    EXPECT_EQ(delta.capture_count, 0);
    EXPECT_EQ(game.get_captures(-1), 0);
}