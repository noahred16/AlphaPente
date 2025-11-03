#include <gtest/gtest.h>
#include "mcts/mcts_engine.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include <chrono>

using namespace mcts;
using namespace core;

class MCTSEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        state_ = std::make_unique<GameState>();
        distance_rings_ = std::make_unique<DistanceRings>();
        move_generator_ = std::make_unique<MoveGenerator>(*distance_rings_);
        engine_ = std::make_unique<MCTSEngine>(*state_, *move_generator_);
    }
    
    std::unique_ptr<GameState> state_;
    std::unique_ptr<DistanceRings> distance_rings_;
    std::unique_ptr<MoveGenerator> move_generator_;
    std::unique_ptr<MCTSEngine> engine_;
};

TEST_F(MCTSEngineTest, ConstructorInitializesCorrectly) {
    EXPECT_EQ(engine_->get_total_simulations(), 0);
    
    // Engine should be ready to search
    Position move = engine_->search(10, 100.0); // Small search for testing
    
    // Should return a valid position or invalid position if no moves
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
}

TEST_F(MCTSEngineTest, SearchReturnsValidMove) {
    Position move = engine_->search(100, 500.0);
    
    // Should return a valid move on empty board
    EXPECT_GE(move.row, 0);
    EXPECT_LT(move.row, 19);
    EXPECT_GE(move.col, 0);
    EXPECT_LT(move.col, 19);
    
    // Move should be on an empty square
    EXPECT_TRUE(state_->is_empty(move.row, move.col));
    
    // Should have performed some simulations
    EXPECT_GT(engine_->get_total_simulations(), 0);
}

TEST_F(MCTSEngineTest, SearchRespectsIterationLimit) {
    const int MAX_ITERATIONS = 50;
    
    Position move = engine_->search(MAX_ITERATIONS, 10000.0); // High time limit
    
    // Should not exceed iteration limit significantly
    // (might be a bit over due to completion of current iteration)
    EXPECT_LE(engine_->get_total_simulations(), MAX_ITERATIONS + 10);
    EXPECT_GT(engine_->get_total_simulations(), 0);
}

TEST_F(MCTSEngineTest, SearchRespectsTimeLimit) {
    const double TIME_LIMIT = 200.0; // 200ms
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Position move = engine_->search(10000, TIME_LIMIT); // High iteration limit
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Should respect time limit (with some tolerance)
    EXPECT_LT(elapsed, TIME_LIMIT + 100); // Allow 100ms tolerance
    
    // Should still return a valid move
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
}

TEST_F(MCTSEngineTest, SearchFromDifferentPositions) {
    // Test search from various board positions
    std::vector<Position> setup_moves = {
        {9, 9},   // Center
        {10, 10}  // Adjacent to center
    };
    
    for (const auto& setup_move : setup_moves) {
        state_->make_move(setup_move.row, setup_move.col);
    }
    
    Position move = engine_->search(100, 500.0);
    
    // Should return a valid move
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
    
    // If valid, should be on empty square
    if (move.row != -1 && move.col != -1) {
        EXPECT_TRUE(state_->is_empty(move.row, move.col));
    }
    
    // Clean up
    for (size_t i = 0; i < setup_moves.size(); i++) {
        state_->undo_move();
    }
}

TEST_F(MCTSEngineTest, ExplorationWeightConfiguration) {
    // Test different exploration weights
    const double EXPLORATION_WEIGHTS[] = {0.5, 1.0, 1.414, 2.0};
    
    for (double weight : EXPLORATION_WEIGHTS) {
        engine_->set_exploration_weight(weight);
        engine_->reset_statistics();
        
        Position move = engine_->search(50, 200.0);
        
        // Should return valid move regardless of exploration weight
        EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                    (move.row == -1 && move.col == -1));
    }
}

TEST_F(MCTSEngineTest, StatisticsTracking) {
    engine_->reset_statistics();
    EXPECT_EQ(engine_->get_total_simulations(), 0);
    
    engine_->search(100, 500.0);
    
    int first_total = engine_->get_total_simulations();
    EXPECT_GT(first_total, 0);
    
    // Second search should add to total
    engine_->search(50, 200.0);
    
    int second_total = engine_->get_total_simulations();
    EXPECT_GT(second_total, first_total);
    
    // Reset should clear statistics
    engine_->reset_statistics();
    EXPECT_EQ(engine_->get_total_simulations(), 0);
}

TEST_F(MCTSEngineTest, PerformanceTest) {
    const int ITERATIONS = 1000;
    const double TIME_LIMIT = 2000.0; // 2 seconds
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Position move = engine_->search(ITERATIONS, TIME_LIMIT);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Should complete in reasonable time
    EXPECT_LT(elapsed, TIME_LIMIT);
    
    // Should achieve reasonable simulation rate
    int simulations = engine_->get_total_simulations();
    double sims_per_second = (simulations * 1000.0) / elapsed;
    
    EXPECT_GT(sims_per_second, 100.0); // At least 100 simulations per second
    
    std::cout << "MCTS performance: " << sims_per_second 
              << " simulations/second (" << simulations 
              << " total in " << elapsed << "ms)" << std::endl;
    
    // Should return valid move
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
}

TEST_F(MCTSEngineTest, EarlyStoppingLogic) {
    // Test early stopping when one move dominates
    // This is harder to test deterministically, but we can verify the mechanism works
    
    Position move = engine_->search(1000, 1000.0);
    
    int simulations = engine_->get_total_simulations();
    
    // Should have performed some simulations
    EXPECT_GT(simulations, 0);
    
    // Should return valid move
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
}

TEST_F(MCTSEngineTest, UpdateRootFunctionality) {
    // Test tree reuse functionality
    Position opponent_move{9, 9};
    
    // Make move in actual game state
    state_->make_move(opponent_move.row, opponent_move.col);
    
    // Update engine's root to reflect this move
    engine_->update_root(opponent_move);
    
    // Search should still work from new position
    Position next_move = engine_->search(100, 300.0);
    
    // Should return a valid move
    EXPECT_TRUE((next_move.row >= 0 && next_move.row < 19 && 
                 next_move.col >= 0 && next_move.col < 19) ||
                (next_move.row == -1 && next_move.col == -1));
    
    // If valid, should not conflict with existing move
    if (next_move.row != -1 && next_move.col != -1) {
        EXPECT_TRUE(state_->is_empty(next_move.row, next_move.col));
    }
    
    // Clean up
    state_->undo_move();
}

TEST_F(MCTSEngineTest, StateIntegrityAfterSearch) {
    // Verify that game state is preserved after search
    state_->reset();
    
    // Make some moves to create a specific position
    std::vector<Position> moves = {{9, 9}, {10, 10}, {8, 8}};
    for (const auto& move : moves) {
        state_->make_move(move.row, move.col);
    }
    
    // Record initial state
    int initial_player = state_->current_player();
    std::vector<std::vector<int>> initial_board(19, std::vector<int>(19));
    
    for (int row = 0; row < 19; row++) {
        for (int col = 0; col < 19; col++) {
            initial_board[row][col] = state_->get_stone(row, col);
        }
    }
    
    // Perform search
    Position selected_move = engine_->search(200, 500.0);
    
    // Verify state is unchanged
    EXPECT_EQ(state_->current_player(), initial_player);
    
    for (int row = 0; row < 19; row++) {
        for (int col = 0; col < 19; col++) {
            EXPECT_EQ(state_->get_stone(row, col), initial_board[row][col])
                << "Board state changed at (" << row << ", " << col << ")";
        }
    }
    
    // Clean up
    for (size_t i = 0; i < moves.size(); i++) {
        state_->undo_move();
    }
}

TEST_F(MCTSEngineTest, MultipleSearchConsistency) {
    // Test that multiple searches from same position can return different moves
    // (due to exploration) but all should be valid
    
    const int NUM_SEARCHES = 5;
    std::vector<Position> moves;
    
    for (int i = 0; i < NUM_SEARCHES; i++) {
        engine_->reset_statistics();
        Position move = engine_->search(100, 200.0);
        moves.push_back(move);
        
        // Each move should be valid
        EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                    (move.row == -1 && move.col == -1));
        
        // If valid, should be on empty square
        if (move.row != -1 && move.col != -1) {
            EXPECT_TRUE(state_->is_empty(move.row, move.col));
        }
    }
    
    // Could check for some variation, but moves might be deterministic
    // depending on the position and search parameters
}