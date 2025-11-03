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

        // With stronger move-weighted scoring (0.05/move), need sufficient sampling
        Position engine_move = engine->search(15000, 30000.0);
        
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



TEST_F(IsAgentDumbTest, AgentShouldBlockFourThreat) {
    // Test that the engine blocks four-in-a-row threats (critical Pente defense)
    
    // Helper function to create the test board state
    auto create_test_state = []() {
        auto state = std::make_unique<GameState>();
        // Set up a four-in-a-row threat for the human player (engine must block)
        // Human stones: J10, K10, L10, M10 (horizontal four threat)
        // Cover one side with engine stones (I10)
        // expect engine to block the other side
        state->make_move(9, 9);   // J10 (Human)
        state->make_move(9, 8);   // I10 (Engine - covers one side)
        state->make_move(9, 10);  // K10 (Human) 
        state->make_move(6, 6);   // G7 (Engine - irrelevant move)
        state->make_move(9, 11);  // L10 (Human)
        state->make_move(7, 7);   // H8 (Engine - irrelevant move) 
        state->make_move(9, 12);  // M10 (Human) - creates four-in-a-row threat
        return state;
    };
    
    auto state = create_test_state();

    // DEBUG: Check what moves are being generated at different distances
    auto moves_dist1 = move_generator_->generate_ordered_moves(*state, 0, 100, 1);
    auto moves_dist2 = move_generator_->generate_ordered_moves(*state, 0, 100, 2);
    auto moves_dist3 = move_generator_->generate_ordered_moves(*state, 0, 100, 3);

    std::cout << "\nDEBUG Move counts:\n";
    std::cout << "  Distance 1: " << moves_dist1.size() << " moves\n";
    std::cout << "  Distance 2: " << moves_dist2.size() << " moves\n";
    std::cout << "  Distance 3: " << moves_dist3.size() << " moves\n";

    std::cout << "\nDistance-1 moves (first 20):\n";
    for (size_t i = 0; i < std::min(size_t(20), moves_dist1.size()); i++) {
        std::cout << "  " << (i+1) << ". " << moves_dist1[i].to_string();
        if (moves_dist1[i].row == 9 && (moves_dist1[i].col == 8 || moves_dist1[i].col == 12)) {
            std::cout << " <- BLOCKING MOVE!";
        }
        std::cout << "\n";
    }

    auto candidate_moves = moves_dist3;
    std::cout << "\nDEBUG: Using distance-3, Generated " << candidate_moves.size() << " candidate moves\n";
    std::cout << "Looking for N10 which is row=9, col=13\n";
    bool found_n10 = false;
    for (size_t i = 0; i < candidate_moves.size(); i++) {
        if (candidate_moves[i].row == 9 && candidate_moves[i].col == 13) {
            std::cout << "Found N10 at position " << i << " in candidate list!\n";
            found_n10 = true;
            break;
        }
    }
    if (!found_n10) {
        std::cout << "N10 NOT in candidate list! First 20 candidates:\n";
        for (size_t i = 0; i < std::min(size_t(20), candidate_moves.size()); i++) {
            std::cout << "  " << (i+1) << ". " << candidate_moves[i].to_string() << "\n";
        }
    }

    // Print the board state for debugging
    std::cout << "\nBoard state with four threat (J10-K10-L10-M10):\n";
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
    Position expected_blocks[1] = {{9, 13}}; // N10 (only remaining blocking move since I10 is occupied)
    
    // Create fresh state and engine for each attempt
    state = create_test_state();
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator_);
    
    // With stronger move-weighted scoring, four-threat should be detectable
    Position engine_move = engine->search(15000, 30000.0);
    
    ASSERT_NE(engine_move.row, -1);
    ASSERT_NE(engine_move.col, -1);
    ASSERT_TRUE(state->is_empty(engine_move.row, engine_move.col));
    
    char col_char = 'A' + engine_move.col;
    int display_row = 19 - engine_move.row;
    std::cout << "Attempt 1: Engine played " << col_char << display_row << "\n";
    
    // Print out all most visited moves from the root node
    auto top_children = engine->get_root()->get_top_children(-1);
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

    // anaylze the top choice move
    if (!top_children.empty()) {
        const auto* best_child = top_children[0];
        const auto& move = best_child->get_move();
        char move_col = 'A' + move.col;
        int move_row = 19 - move.row;
        double win_rate = best_child->get_win_rate();
        std::cout << "DEBUG: Top choice move is " << move_col << move_row 
                    << " (visits: " << best_child->get_visits() 
                    << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
        // lets look at its children too
        std::cout << "   DEBUG: Top choice move's top children:\n";
        auto best_child_children = best_child->get_top_children(-1);
        std::cout << "   DEBUG: It has " << best_child_children.size() << " children:\n";
        for (size_t i = 0; i < best_child_children.size(); i++)
        {
            const auto* child = best_child_children[i];
            const auto& move = child->get_move();
            char move_col = 'A' + move.col;
            int move_row = 19 - move.row;
            double win_rate = child->get_win_rate();
            std::cout << "     " << (i+1) << ". " << move_col << move_row 
                        << " (visits: " << child->get_visits() 
                        << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
        }
        // it has zero for some reason? 
        // print the class info
        if (best_child_children.size() == 0) {
            std::cout << "   DEBUG: Top choice move has no children! This is unexpected.\n";
            std::cout << "   DEBUG: Best child move details - visits: " << best_child->get_visits() 
                      << ", wins: " << best_child->get_wins() 
                      << ", win rate: " << std::fixed << std::setprecision(2) << best_child->get_win_rate() << "\n";
        }
    }

    // Print out the visit amounts for the correct answers
    auto all_children = engine->get_root()->get_top_children(-1);
    std::cout << "DEBUG: Blocking move statistics (total children: " << all_children.size() << "):\n";
    
    bool found_blocking_move = false;
    for (const auto* child : all_children) {
        const auto& move = child->get_move();
        if (move.row == expected_blocks[0].row && move.col == expected_blocks[0].col) {
            char move_col = 'A' + move.col;
            int move_row = 19 - move.row;
            double win_rate = child->get_win_rate();
            std::cout << "  Blocking move " << move_col << move_row 
                        << " (visits: " << child->get_visits() 
                        << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
            found_blocking_move = true;
        }
    }
    
    if (!found_blocking_move) {
        char expected_col = 'A' + expected_blocks[0].col;
        int expected_row = 19 - expected_blocks[0].row;
        std::cout << "  Expected blocking move " << expected_col << expected_row << " was NOT explored!\n";
    }

    // Check if it's a blocking move
    bool is_blocking = (engine_move.row == expected_blocks[0].row && engine_move.col == expected_blocks[0].col);
    
    if (is_blocking) {
        blocking_moves.push_back(engine_move);
    }
    
    std::cout << "Four threat blocking moves: " << blocking_moves.size() << " out of 1 attempts\n";
    
    // The engine should block the four threat
    EXPECT_GE(blocking_moves.size(), 1) << "Engine failed to block the four-in-a-row threat.";
}
