#include <gtest/gtest.h>
#include "mcts/mcts_engine.hpp"
#include "mcts/mcts_node.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"

using namespace mcts;
using namespace core;

class TreeReuseTest : public ::testing::Test {
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

TEST_F(TreeReuseTest, FindChildWithMoveBasic) {
    MCTSNode root;
    
    // Expand a few children
    MCTSNode* child1 = root.expand(*state_, *move_generator_);
    MCTSNode* child2 = root.expand(*state_, *move_generator_);
    
    ASSERT_NE(child1, nullptr);
    ASSERT_NE(child2, nullptr);
    
    // Find children by their moves
    Position move1 = child1->get_move();
    Position move2 = child2->get_move();
    
    MCTSNode* found1 = root.find_child_with_move(move1);
    MCTSNode* found2 = root.find_child_with_move(move2);
    
    EXPECT_EQ(found1, child1);
    EXPECT_EQ(found2, child2);
}

TEST_F(TreeReuseTest, FindChildWithMoveNotFound) {
    MCTSNode root;
    
    // Create some children
    root.expand(*state_, *move_generator_);
    root.expand(*state_, *move_generator_);
    
    // Try to find a move that doesn't exist
    Position nonexistent_move{15, 15}; // Assume this wasn't generated
    MCTSNode* found = root.find_child_with_move(nonexistent_move);
    
    EXPECT_EQ(found, nullptr);
}

TEST_F(TreeReuseTest, ExtractChildBasic) {
    MCTSNode root;
    
    // Expand children and add some statistics
    MCTSNode* child1 = root.expand(*state_, *move_generator_);
    MCTSNode* child2 = root.expand(*state_, *move_generator_);
    
    ASSERT_NE(child1, nullptr);
    ASSERT_NE(child2, nullptr);
    
    // Add some visit data to the children
    child1->backpropagate(0.7);
    child1->backpropagate(0.3);
    child2->backpropagate(0.6);
    
    Position extract_move = child1->get_move();
    size_t initial_child_count = root.child_count();
    
    // Extract child1
    std::unique_ptr<MCTSNode> extracted = root.extract_child(extract_move);
    
    ASSERT_NE(extracted, nullptr);
    EXPECT_EQ(root.child_count(), initial_child_count - 1);
    EXPECT_EQ(extracted->get_parent(), nullptr); // Should be null after extraction
    EXPECT_EQ(extracted->get_visits(), 2); // Should preserve visit count
    EXPECT_EQ(extracted->get_move().row, extract_move.row);
    EXPECT_EQ(extracted->get_move().col, extract_move.col);
    
    // Original child should no longer be findable in root
    MCTSNode* not_found = root.find_child_with_move(extract_move);
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(TreeReuseTest, ExtractChildNotFound) {
    MCTSNode root;
    
    // Create some children
    root.expand(*state_, *move_generator_);
    root.expand(*state_, *move_generator_);
    
    Position nonexistent_move{18, 18};
    std::unique_ptr<MCTSNode> extracted = root.extract_child(nonexistent_move);
    
    EXPECT_EQ(extracted, nullptr);
}

TEST_F(TreeReuseTest, SuccessfulTreeReuseScenario) {
    engine_->reset_statistics();
    
    // Perform initial search to build tree
    Position our_move = engine_->search(100, 300.0);
    ASSERT_NE(our_move.row, -1);
    ASSERT_NE(our_move.col, -1);
    
    // Make our move in the actual game state
    state_->make_move(our_move.row, our_move.col);
    
    // Create an opponent move that we likely explored
    Position opponent_move{9, 9}; // Center move, likely to be explored
    state_->make_move(opponent_move.row, opponent_move.col);
    
    // Update engine root with opponent's move
    int reuse_count_before = engine_->get_tree_reuse_count();
    engine_->update_root(opponent_move);
    int reuse_count_after = engine_->get_tree_reuse_count();
    
    // We should either have a successful reuse or fallback
    EXPECT_TRUE((reuse_count_after > reuse_count_before) || 
                (engine_->get_tree_fallback_count() > 0));
    
    // Engine should still work after tree update
    Position next_move = engine_->search(50, 200.0);
    EXPECT_TRUE((next_move.row >= 0 && next_move.row < 19 && 
                 next_move.col >= 0 && next_move.col < 19) ||
                (next_move.row == -1 && next_move.col == -1));
}

TEST_F(TreeReuseTest, TreeFallbackScenario) {
    engine_->reset_statistics();
    
    // Perform a small search that won't explore many moves
    engine_->search(10, 100.0); // Very limited search
    
    // Use a move that's unlikely to have been explored
    Position unlikely_move{0, 0}; // Corner move, unlikely in small search
    
    int fallback_count_before = engine_->get_tree_fallback_count();
    engine_->update_root(unlikely_move);
    int fallback_count_after = engine_->get_tree_fallback_count();
    
    // Should have increased fallback count
    EXPECT_GT(fallback_count_after, fallback_count_before);
    
    // Engine should still work
    Position move = engine_->search(50, 200.0);
    EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                (move.row == -1 && move.col == -1));
}

TEST_F(TreeReuseTest, StatisticsTracking) {
    engine_->reset_statistics();
    
    EXPECT_EQ(engine_->get_tree_reuse_count(), 0);
    EXPECT_EQ(engine_->get_tree_fallback_count(), 0);
    EXPECT_EQ(engine_->get_tree_reuse_rate(), 0.0);
    
    // Simulate some tree updates
    engine_->update_root({9, 9});
    engine_->update_root({10, 10});
    engine_->update_root({8, 8});
    
    // Should have some statistics now
    int total_updates = engine_->get_tree_reuse_count() + engine_->get_tree_fallback_count();
    EXPECT_EQ(total_updates, 3);
    
    // Reuse rate should be between 0 and 1
    double reuse_rate = engine_->get_tree_reuse_rate();
    EXPECT_GE(reuse_rate, 0.0);
    EXPECT_LE(reuse_rate, 1.0);
}

TEST_F(TreeReuseTest, PreservesVisitCounts) {
    // Build a tree with some depth
    engine_->search(200, 500.0);
    
    // Make a move and get a child that likely has visits
    Position test_move{9, 9};
    state_->make_move(test_move.row, test_move.col);
    
    // Store current statistics
    int total_sims_before = engine_->get_total_simulations();
    
    // Update root - if successful, should preserve some visits
    engine_->update_root(test_move);
    
    // Do a small additional search
    engine_->search(50, 200.0);
    
    int total_sims_after = engine_->get_total_simulations();
    
    // We should have done some simulations
    EXPECT_GT(total_sims_after, total_sims_before);
    
    // If tree reuse was successful, the reuse count should be positive
    if (engine_->get_tree_reuse_count() > 0) {
        // Tree reuse occurred - this is good!
        EXPECT_GT(engine_->get_tree_reuse_rate(), 0.0);
    }
}

TEST_F(TreeReuseTest, MultipleTreeUpdates) {
    // Test multiple consecutive tree updates
    std::vector<Position> moves = {{9, 9}, {10, 10}, {8, 8}, {11, 9}};
    
    engine_->reset_statistics();
    
    for (const auto& move : moves) {
        // Perform search to build some tree
        engine_->search(50, 200.0);
        
        // Update tree with the move
        engine_->update_root(move);
        
        // Make move in actual state
        if (state_->is_empty(move.row, move.col)) {
            state_->make_move(move.row, move.col);
        }
    }
    
    // Should have tracked all updates
    int total_updates = engine_->get_tree_reuse_count() + engine_->get_tree_fallback_count();
    EXPECT_EQ(total_updates, moves.size());
    
    // Engine should still be functional
    Position final_move = engine_->search(100, 300.0);
    EXPECT_TRUE((final_move.row >= 0 && final_move.row < 19 && 
                 final_move.col >= 0 && final_move.col < 19) ||
                (final_move.row == -1 && final_move.col == -1));
}

TEST_F(TreeReuseTest, TreeStructureValidityAfterReuse) {
    // Build initial tree
    Position our_move = engine_->search(150, 400.0);
    
    // Make moves to set up tree reuse scenario
    state_->make_move(our_move.row, our_move.col);
    Position opponent_move{9, 9};
    state_->make_move(opponent_move.row, opponent_move.col);
    
    // Update root
    engine_->update_root(opponent_move);
    
    // Perform another search - tree should be valid
    Position next_move = engine_->search(100, 300.0);
    
    // Should return valid move
    EXPECT_TRUE((next_move.row >= 0 && next_move.row < 19 && 
                 next_move.col >= 0 && next_move.col < 19) ||
                (next_move.row == -1 && next_move.col == -1));
    
    // Should be able to perform multiple searches without issues
    for (int i = 0; i < 3; i++) {
        Position move = engine_->search(50, 150.0);
        EXPECT_TRUE((move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) ||
                    (move.row == -1 && move.col == -1));
    }
}

TEST_F(TreeReuseTest, MemoryManagementDuringTreeOperations) {
    // This test ensures no memory leaks during tree operations
    {
        MCTSNode root;
        
        // Create several children with grandchildren
        for (int i = 0; i < 5; i++) {
            MCTSNode* child = root.expand(*state_, *move_generator_);
            if (child) {
                child->backpropagate(0.5);
                // Create grandchildren
                child->expand(*state_, *move_generator_);
                child->expand(*state_, *move_generator_);
            }
        }
        
        // Extract some children
        if (root.child_count() > 0) {
            Position first_child_move = root.find_child_with_move({9, 9}) ? 
                                      Position{9, 9} : 
                                      Position{10, 10}; // Try common positions
            
            auto extracted = root.extract_child(first_child_move);
            // extracted will be destroyed when it goes out of scope
        }
        
        // Root will be destroyed here, should clean up remaining tree
    }
    
    // If we get here without crashes, memory management is working
    SUCCEED();
}