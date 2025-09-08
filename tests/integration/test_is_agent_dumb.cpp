#include <gtest/gtest.h>
#include "mcts/mcts_engine.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include <iostream>
#include <iomanip>

using namespace mcts;
using namespace core;

class IsAgentDumbTest : public ::testing::Test {
protected:
    void SetUp() override {
        distance_rings_ = std::make_unique<DistanceRings>();
        move_generator_ = std::make_unique<MoveGenerator>(*distance_rings_);
    }
    
    std::unique_ptr<DistanceRings> distance_rings_;
    std::unique_ptr<MoveGenerator> move_generator_;
};

TEST_F(IsAgentDumbTest, AgentShouldPlayNearExistingStones) {
    // Set up a game state with stones in the center area
    auto state = std::make_unique<GameState>();
    
    // Make some moves to create a focused area of play
    state->make_move(9, 10);  // K10 (Human)  
    state->make_move(12, 9);  // J7 (Engine)
    state->make_move(8, 10);  // K9 (Human)
    state->make_move(9, 14);  // O10 (Engine) 
    state->make_move(7, 10);  // K8 (Human)
    
    // Now it's the engine's turn
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    
    // Run a search
    Position engine_move = engine->search(1000, 2000.0);
    
    // Print the board state for debugging
    std::cout << "\nBoard state after moves:\n";
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    for (int row = 0; row < 19; row++) {
        printf("%2d  ", 19-row);
        for (int col = 0; col < 19; col++) {
            int stone = state->get_stone(row, col);
            if (stone == 1) std::cout << "X ";
            else if (stone == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        printf("  %d\n", 19-row);
    }
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    
    // Print the engine's move
    if (engine_move.row != -1 && engine_move.col != -1) {
        char col_char = 'A' + engine_move.col;
        int display_row = 19 - engine_move.row;
        std::cout << "\nEngine played: " << col_char << display_row << "\n";
    } else {
        std::cout << "\nEngine returned invalid move!\n";
    }
    
    // Check that the move is valid
    ASSERT_NE(engine_move.row, -1);
    ASSERT_NE(engine_move.col, -1);
    ASSERT_TRUE(state->is_empty(engine_move.row, engine_move.col));
    
    // Check that the move is reasonably close to existing stones
    // Find the closest existing stone to the engine's move
    int min_distance = 999;
    for (int row = 0; row < 19; row++) {
        for (int col = 0; col < 19; col++) {
            if (!state->is_empty(row, col)) {
                int dist = std::max(std::abs(row - engine_move.row), 
                                   std::abs(col - engine_move.col));
                min_distance = std::min(min_distance, dist);
            }
        }
    }
    
    std::cout << "Closest distance to existing stone: " << min_distance << "\n";
    
    // The engine should not play moves that are more than 3 squares away from any existing stone
    EXPECT_LE(min_distance, 3) << "Engine played " << char('A' + engine_move.col) << (19 - engine_move.row) 
                               << " which is " << min_distance << " squares from the nearest stone. This is too far!";
    
    // Even better - it should usually be within 2 squares
    if (min_distance > 2) {
        std::cout << "WARNING: Engine played " << min_distance << " squares away. Not ideal but within tolerance.\n";
    }
}

TEST_F(IsAgentDumbTest, AgentShouldNotPlayInCorners) {
    // Set up a game state with stones in the center
    auto state = std::make_unique<GameState>();
    
    // Make a few center moves
    state->make_move(9, 9);   // J10
    state->make_move(10, 10); // K11
    state->make_move(8, 8);   // I9
    
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    
    // Run multiple searches to see if it ever picks corner moves
    for (int i = 0; i < 3; i++) {
        Position move = engine->search(500, 1000.0);
        
        ASSERT_NE(move.row, -1);
        ASSERT_NE(move.col, -1);
        
        // Check it's not in the corners
        bool is_corner = (move.row <= 2 && move.col <= 2) ||      // Top-left
                        (move.row <= 2 && move.col >= 16) ||      // Top-right  
                        (move.row >= 16 && move.col <= 2) ||      // Bottom-left
                        (move.row >= 16 && move.col >= 16);       // Bottom-right
        
        char col_char = 'A' + move.col;
        int display_row = 19 - move.row;
        
        EXPECT_FALSE(is_corner) << "Engine played in corner: " << col_char << display_row 
                               << " on attempt " << (i+1);
        
        if (!is_corner) {
            std::cout << "Attempt " << (i+1) << ": Engine played " << col_char << display_row 
                      << " (good - not a corner)\n";
        }
        
        // Reset engine for next attempt
        engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    }
}

TEST_F(IsAgentDumbTest, MoveGeneratorProducesGoodMoves) {
    // Test that the move generator itself is working correctly
    auto state = std::make_unique<GameState>();
    
    // Set up the same problematic position
    state->make_move(9, 10);  // K10 (Human)  
    state->make_move(12, 9);  // J7 (Engine)
    state->make_move(8, 10);  // K9 (Human)
    state->make_move(9, 14);  // O10 (Engine) 
    state->make_move(7, 10);  // K8 (Human)
    
    // Test move generation at different visit levels
    std::vector<int> test_visits = {0, 10, 100, 500};
    
    for (int visits : test_visits) {
        auto moves = move_generator_->generate_ordered_moves(*state, visits);
        
        std::cout << "\nWith " << visits << " visits, top 10 moves:\n";
        for (size_t i = 0; i < std::min(size_t(10), moves.size()); i++) {
            char col_char = 'A' + moves[i].col;
            int display_row = 19 - moves[i].row;
            std::cout << "  " << (i+1) << ". " << col_char << display_row << "\n";
        }
        
        // Check that the first few moves are close to existing stones
        ASSERT_GT(moves.size(), 0);
        
        for (size_t i = 0; i < std::min(size_t(5), moves.size()); i++) {
            Position move = moves[i];
            
            // Find closest stone distance
            int min_distance = 999;
            for (int row = 0; row < 19; row++) {
                for (int col = 0; col < 19; col++) {
                    if (!state->is_empty(row, col)) {
                        int dist = std::max(std::abs(row - move.row), 
                                           std::abs(col - move.col));
                        min_distance = std::min(min_distance, dist);
                    }
                }
            }
            
            char col_char = 'A' + move.col;
            int display_row = 19 - move.row;
            
            EXPECT_LE(min_distance, 3) << "Move generator's #" << (i+1) << " move " 
                                       << col_char << display_row 
                                       << " is " << min_distance << " squares away (too far)";
        }
    }
}

TEST_F(IsAgentDumbTest, EngineConsistencyTest) {
    // Test that the engine consistently makes reasonable moves
    auto state = std::make_unique<GameState>();
    
    // Simple center game
    state->make_move(9, 9);   // J10
    state->make_move(10, 10); // K11  
    
    std::vector<Position> engine_moves;
    
    // Run 5 independent searches
    for (int i = 0; i < 5; i++) {
        auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
        Position move = engine->search(200, 500.0);
        
        ASSERT_NE(move.row, -1);
        ASSERT_NE(move.col, -1);
        ASSERT_TRUE(state->is_empty(move.row, move.col));
        
        engine_moves.push_back(move);
        
        char col_char = 'A' + move.col;
        int display_row = 19 - move.row;
        std::cout << "Search " << (i+1) << ": " << col_char << display_row << "\n";
    }
    
    // Check that all moves are reasonable (within distance 3 of existing stones)
    int reasonable_moves = 0;
    for (const auto& move : engine_moves) {
        int min_distance = 999;
        for (int row = 0; row < 19; row++) {
            for (int col = 0; col < 19; col++) {
                if (!state->is_empty(row, col)) {
                    int dist = std::max(std::abs(row - move.row), 
                                       std::abs(col - move.col));
                    min_distance = std::min(min_distance, dist);
                }
            }
        }
        
        if (min_distance <= 3) {
            reasonable_moves++;
        }
    }
    
    std::cout << "Reasonable moves: " << reasonable_moves << " out of " << engine_moves.size() << "\n";
    
    // At least 4 out of 5 moves should be reasonable
    EXPECT_GE(reasonable_moves, 4) << "Only " << reasonable_moves << " out of 5 moves were reasonable";
}


// my fav test.. NS
TEST_F(IsAgentDumbTest, AgentShouldBlockOpenThrees) {
    // Test that the engine blocks open threes (fundamental Pente defense)
    
    // Helper function to create the test board state
    auto create_test_state = []() {
        auto state = std::make_unique<GameState>();
        // Set up an open three for the human player (engine must block)
        // Human stones: J10, K10, L10 (horizontal open three)
        state->make_move(9, 9);   // J10 (Human)
        state->make_move(5, 5);   // F6 (Engine - irrelevant move)
        state->make_move(9, 10);  // K10 (Human) 
        state->make_move(6, 6);   // G7 (Engine - irrelevant move)
        state->make_move(9, 11);  // L10 (Human) - creates open three
        return state;
    };
    
    auto state = create_test_state();
    
    // Print the board state for debugging
    std::cout << "\nBoard state with open three (J10-K10-L10):\n";
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    for (int row = 0; row < 19; row++) {
        printf("%2d  ", 19-row);
        for (int col = 0; col < 19; col++) {
            int stone = state->get_stone(row, col);
            if (stone == 1) std::cout << "X ";
            else if (stone == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        printf("  %d\n", 19-row);
    }
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    
    // Run multiple searches to test consistency
    std::vector<Position> blocking_moves;
    Position expected_blocks[2] = {{9, 8}, {9, 12}}; // I10 or M10
    
    for (int attempt = 0; attempt < 3; attempt++) {
        // Create fresh state and engine for each attempt
        state = create_test_state();
        auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
        
        Position engine_move = engine->search(1000, 2000.0);
        
        ASSERT_NE(engine_move.row, -1);
        ASSERT_NE(engine_move.col, -1);
        ASSERT_TRUE(state->is_empty(engine_move.row, engine_move.col));
        
        char col_char = 'A' + engine_move.col;
        int display_row = 19 - engine_move.row;
        std::cout << "Attempt " << (attempt + 1) << ": Engine played " << col_char << display_row << "\n";
        
        // Print out the top 10 most visited moves from the root node
        auto top_children = engine->get_root()->get_top_children(10);
        std::cout << "DEBUG: Top " << top_children.size() << " moves by visits:\n";
        for (size_t i = 0; i < top_children.size(); i++) {
            const auto* child = top_children[i];
            const auto& move = child->get_move();
            char move_col = 'A' + move.col;
            int move_row = 19 - move.row;
            double win_rate = child->get_win_rate();
            std::cout << "  " << (i+1) << ". " << move_col << move_row 
                      << " (visits: " << child->get_visits() 
                      << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
        }

        // Check if it's a blocking move
        bool is_blocking = (engine_move.row == expected_blocks[0].row && engine_move.col == expected_blocks[0].col) ||
                          (engine_move.row == expected_blocks[1].row && engine_move.col == expected_blocks[1].col);
        
        if (is_blocking) {
            blocking_moves.push_back(engine_move);
        }
    }
    
    std::cout << "Blocking moves: " << blocking_moves.size() << " out of 3 attempts\n";
    
    // The engine should block the open three at least 2 out of 3 times
    EXPECT_GE(blocking_moves.size(), 2) << "Engine only blocked the open three " << blocking_moves.size() << " out of 3 times. Expected at least 2.";
}

TEST_F(IsAgentDumbTest, AgentShouldBlockVerticalOpenThrees) {
    // Test vertical open three blocking
    auto state = std::make_unique<GameState>();
    
    // Set up a vertical open three for the human player
    // Human stones: J9, J10, J11 (vertical open three)
    state->make_move(10, 9);  // J8 (Human)
    state->make_move(5, 5);   // F6 (Engine - irrelevant)
    state->make_move(9, 9);   // J10 (Human) 
    state->make_move(6, 6);   // G7 (Engine - irrelevant)
    state->make_move(8, 9);   // J11 (Human) - creates vertical open three
    
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    
    std::cout << "\nBoard state with vertical open three (J8-J10-J11):\n";
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    for (int row = 0; row < 19; row++) {
        printf("%2d  ", 19-row);
        for (int col = 0; col < 19; col++) {
            int stone = state->get_stone(row, col);
            if (stone == 1) std::cout << "X ";
            else if (stone == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        printf("  %d\n", 19-row);
    }
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    
    Position expected_blocks[2] = {{11, 9}, {7, 9}}; // J7 or J12
    int blocking_count = 0;
    
    for (int attempt = 0; attempt < 3; attempt++) {
        Position engine_move = engine->search(1000, 2000.0);
        
        ASSERT_NE(engine_move.row, -1);
        ASSERT_NE(engine_move.col, -1);
        
        char col_char = 'A' + engine_move.col;
        int display_row = 19 - engine_move.row;
        std::cout << "Attempt " << (attempt + 1) << ": Engine played " << col_char << display_row << "\n";
        
        bool is_blocking = (engine_move.row == expected_blocks[0].row && engine_move.col == expected_blocks[0].col) ||
                          (engine_move.row == expected_blocks[1].row && engine_move.col == expected_blocks[1].col);
        
        if (is_blocking) {
            blocking_count++;
        }
        
        engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    }
    
    std::cout << "Vertical blocking moves: " << blocking_count << " out of 3 attempts\n";
    EXPECT_GE(blocking_count, 2) << "Engine only blocked the vertical open three " << blocking_count << " out of 3 times.";
}

TEST_F(IsAgentDumbTest, AgentShouldBlockDiagonalOpenThrees) {
    // Test diagonal open three blocking
    auto state = std::make_unique<GameState>();
    
    // Set up a diagonal open three for the human player
    // Human stones: I9, J10, K11 (diagonal open three)
    state->make_move(10, 8);  // I9 (Human)
    state->make_move(5, 5);   // F6 (Engine - irrelevant)
    state->make_move(9, 9);   // J10 (Human) 
    state->make_move(6, 6);   // G7 (Engine - irrelevant)
    state->make_move(8, 10);  // K11 (Human) - creates diagonal open three
    
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    
    std::cout << "\nBoard state with diagonal open three (I9-J10-K11):\n";
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    for (int row = 0; row < 19; row++) {
        printf("%2d  ", 19-row);
        for (int col = 0; col < 19; col++) {
            int stone = state->get_stone(row, col);
            if (stone == 1) std::cout << "X ";
            else if (stone == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        printf("  %d\n", 19-row);
    }
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    
    Position expected_blocks[2] = {{11, 7}, {7, 11}}; // H8 or L12
    int blocking_count = 0;
    
    for (int attempt = 0; attempt < 3; attempt++) {
        Position engine_move = engine->search(1000, 2000.0);
        
        ASSERT_NE(engine_move.row, -1);
        ASSERT_NE(engine_move.col, -1);
        
        char col_char = 'A' + engine_move.col;
        int display_row = 19 - engine_move.row;
        std::cout << "Attempt " << (attempt + 1) << ": Engine played " << col_char << display_row << "\n";
        
        bool is_blocking = (engine_move.row == expected_blocks[0].row && engine_move.col == expected_blocks[0].col) ||
                          (engine_move.row == expected_blocks[1].row && engine_move.col == expected_blocks[1].col);
        
        if (is_blocking) {
            blocking_count++;
        }
        
        engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    }
    
    std::cout << "Diagonal blocking moves: " << blocking_count << " out of 3 attempts\n";
    EXPECT_GE(blocking_count, 2) << "Engine only blocked the diagonal open three " << blocking_count << " out of 3 times.";
}