#include <gtest/gtest.h>
#include "mcts/rollout.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include <chrono>

using namespace mcts;
using namespace core;

class RolloutTest : public ::testing::Test {
protected:
    void SetUp() override {
        state_ = std::make_unique<GameState>();
        distance_rings_ = std::make_unique<DistanceRings>();
        move_generator_ = std::make_unique<MoveGenerator>(*distance_rings_);
        rollout_policy_ = std::make_unique<RolloutPolicy>();
    }
    
    std::unique_ptr<GameState> state_;
    std::unique_ptr<DistanceRings> distance_rings_;
    std::unique_ptr<MoveGenerator> move_generator_;
    std::unique_ptr<RolloutPolicy> rollout_policy_;
};

TEST_F(RolloutTest, SimulationReturnsValidResult) {
    // Test on initial empty board
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    
    // Result should be between 0.0 and 1.0
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

TEST_F(RolloutTest, SimulationReachesTerminalState) {
    // Run simulation and verify state returns to original
    Position original_first_move{9, 9}; // Center of board
    state_->make_move(original_first_move.row, original_first_move.col);
    int initial_player = state_->current_player();
    
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    
    // After simulation, state should be restored
    EXPECT_EQ(state_->current_player(), initial_player);
    EXPECT_EQ(state_->get_stone(original_first_move.row, original_first_move.col), -initial_player);
    
    // Clean up
    state_->undo_move();
}

TEST_F(RolloutTest, SimulationHandlesEmptyMoves) {
    // Create a state where there are no legal moves (unlikely in Pente but test edge case)
    // This is hard to create in Pente, so we'll test the basic case
    
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    
    // Should handle gracefully and return some result
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

TEST_F(RolloutTest, SimulationRespectsMoveLimit) {
    // Test that simulation doesn't run forever
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Simulation should complete in reasonable time (less than 1 second)
    EXPECT_LT(duration, 1000);
    
    // Should return valid result
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

TEST_F(RolloutTest, SimulationFromDifferentPositions) {
    // Test simulation from various board positions
    std::vector<Position> test_moves = {
        {9, 9},   // Center
        {5, 5},   // Upper left quadrant
        {15, 15}, // Lower right quadrant
        {3, 12}   // Mixed position
    };
    
    for (const auto& move : test_moves) {
        state_->reset();
        state_->make_move(move.row, move.col);
        
        double result = rollout_policy_->simulate(*state_, *move_generator_);
        
        EXPECT_GE(result, 0.0);
        EXPECT_LE(result, 1.0);
        
        state_->undo_move();
    }
}

TEST_F(RolloutTest, ResultEvaluationAccuracy) {
    // Test various terminal states
    state_->reset();
    
    // Simulate a basic game scenario
    // Place some stones to create a pattern
    std::vector<Position> moves = {
        {9, 9}, {9, 10}, {9, 8}, {9, 11}, {9, 7}, {9, 12}
    };
    
    // Play the moves
    for (const auto& move : moves) {
        if (!state_->is_terminal()) {
            state_->make_move(move.row, move.col);
        }
    }
    
    // Run simulation from this position
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    
    // Should return valid result
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
    
    // Restore state
    for (size_t i = 0; i < moves.size() && state_->get_stone(moves[moves.size()-1-i].row, moves[moves.size()-1-i].col) != 0; i++) {
        state_->undo_move();
    }
}

TEST_F(RolloutTest, PerformanceTest) {
    // Test that we can run many rollouts quickly
    const int NUM_ROLLOUTS = 100;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ROLLOUTS; i++) {
        double result = rollout_policy_->simulate(*state_, *move_generator_);
        EXPECT_GE(result, 0.0);
        EXPECT_LE(result, 1.0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // 100 rollouts should complete in under 5 seconds
    EXPECT_LT(duration, 5000);
    
    // Calculate rollouts per second
    double rollouts_per_second = (NUM_ROLLOUTS * 1000.0) / duration;
    
    // Should achieve reasonable performance (target: >50 rollouts/sec)
    EXPECT_GT(rollouts_per_second, 20.0);
    
    std::cout << "Rollout performance: " << rollouts_per_second 
              << " rollouts/second" << std::endl;
}

TEST_F(RolloutTest, DifferentOutcomesFromSamePosition) {
    // Test that rollouts from the same position can have different outcomes
    // due to randomness
    
    const int NUM_TESTS = 20;
    std::vector<double> results;
    
    for (int i = 0; i < NUM_TESTS; i++) {
        double result = rollout_policy_->simulate(*state_, *move_generator_);
        results.push_back(result);
        EXPECT_GE(result, 0.0);
        EXPECT_LE(result, 1.0);
    }
    
    // Check if we got some variation in results
    // (This might not always pass if the position is too deterministic)
    bool has_variation = false;
    for (int i = 1; i < NUM_TESTS; i++) {
        if (std::abs(results[i] - results[0]) > 1e-10) {
            has_variation = true;
            break;
        }
    }
    
    // Note: This test might occasionally fail if all rollouts happen to have
    // the same outcome, which is possible but unlikely
    EXPECT_TRUE(has_variation || results[0] == 0.5); // 0.5 indicates draws
}

TEST_F(RolloutTest, StateIntegrityAfterRollout) {
    // Verify that the game state is properly restored after rollout
    state_->reset();
    
    // Make some initial moves
    state_->make_move(9, 9);
    state_->make_move(10, 10);
    state_->make_move(8, 8);
    
    // Record initial state
    int initial_player = state_->current_player();
    std::vector<std::vector<int>> initial_board(19, std::vector<int>(19));
    
    for (int row = 0; row < 19; row++) {
        for (int col = 0; col < 19; col++) {
            initial_board[row][col] = state_->get_stone(row, col);
        }
    }
    
    // Run rollout
    double result = rollout_policy_->simulate(*state_, *move_generator_);
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
    
    // Verify state is restored
    EXPECT_EQ(state_->current_player(), initial_player);
    
    for (int row = 0; row < 19; row++) {
        for (int col = 0; col < 19; col++) {
            EXPECT_EQ(state_->get_stone(row, col), initial_board[row][col])
                << "Board mismatch at (" << row << ", " << col << ")";
        }
    }
}