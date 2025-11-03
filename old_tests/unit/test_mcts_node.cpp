#include <gtest/gtest.h>
#include "mcts/mcts_node.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include <cmath>

using namespace mcts;
using namespace core;

class MCTSNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        state_ = std::make_unique<GameState>();
        distance_rings_ = std::make_unique<DistanceRings>();
        move_generator_ = std::make_unique<MoveGenerator>(*distance_rings_);
    }
    
    std::unique_ptr<GameState> state_;
    std::unique_ptr<DistanceRings> distance_rings_;
    std::unique_ptr<MoveGenerator> move_generator_;
};

TEST_F(MCTSNodeTest, ConstructorInitializesCorrectly) {
    MCTSNode root;
    
    EXPECT_EQ(root.get_visits(), 0);
    EXPECT_EQ(root.get_wins(), 0.0);
    EXPECT_EQ(root.get_win_rate(), 0.0);
    EXPECT_EQ(root.get_parent(), nullptr);
    EXPECT_TRUE(root.is_leaf());
    EXPECT_EQ(root.child_count(), 0);
}

TEST_F(MCTSNodeTest, ConstructorWithParentAndMove) {
    MCTSNode root;
    Position move{5, 5};
    MCTSNode child(&root, move);
    
    EXPECT_EQ(child.get_parent(), &root);
    EXPECT_EQ(child.get_move().row, 5);
    EXPECT_EQ(child.get_move().col, 5);
    EXPECT_EQ(child.get_visits(), 0);
    EXPECT_EQ(child.get_wins(), 0.0);
}

TEST_F(MCTSNodeTest, UCB1CalculationUnvisitedNode) {
    MCTSNode root;
    MCTSNode child(&root);
    
    // Unvisited nodes should return infinity
    EXPECT_EQ(child.ucb1_value(), std::numeric_limits<double>::infinity());
}

TEST_F(MCTSNodeTest, UCB1CalculationRootNode) {
    MCTSNode root;
    
    // Simulate some visits to root
    root.backpropagate(0.6); // 1 visit, 0.6 wins
    root.backpropagate(0.4); // 2 visits, 1.0 wins total
    
    // Root node should return win rate
    EXPECT_DOUBLE_EQ(root.ucb1_value(), 0.5); // 1.0 wins / 2 visits
}

TEST_F(MCTSNodeTest, UCB1CalculationWithExploration) {
    MCTSNode root;
    MCTSNode child(&root);
    
    // Add visits to root
    root.backpropagate(0.5); // 1 visit
    root.backpropagate(0.5); // 2 visits
    root.backpropagate(0.5); // 3 visits
    
    // Add visits to child (backpropagate automatically updates parent)
    child.backpropagate(0.8); // 1 visit to child, adds 1 visit to root (total 4 visits)
    
    // Calculate expected UCB1 value using actual visit counts
    int actual_parent_visits = root.get_visits();
    int actual_child_visits = child.get_visits();
    
    double exploitation = 0.8; // 0.8 wins / 1 visit
    double exploration = 1.414 * std::sqrt(std::log(actual_parent_visits) / actual_child_visits);
    double expected_ucb1 = exploitation + exploration;
    
    EXPECT_NEAR(child.ucb1_value(), expected_ucb1, 1e-10);
}

TEST_F(MCTSNodeTest, BackpropagationUpdatesStatistics) {
    MCTSNode root;
    
    EXPECT_EQ(root.get_visits(), 0);
    EXPECT_EQ(root.get_wins(), 0.0);
    
    root.backpropagate(1.0);
    EXPECT_EQ(root.get_visits(), 1);
    EXPECT_EQ(root.get_wins(), 1.0);
    EXPECT_EQ(root.get_win_rate(), 1.0);
    
    root.backpropagate(0.0);
    EXPECT_EQ(root.get_visits(), 2);
    EXPECT_EQ(root.get_wins(), 1.0);
    EXPECT_EQ(root.get_win_rate(), 0.5);
}

TEST_F(MCTSNodeTest, BackpropagationFlipsResultForParent) {
    MCTSNode root;
    MCTSNode child(&root);
    
    // Child wins (result = 1.0)
    child.backpropagate(1.0);
    
    // Child should have 1 visit, 1 win
    EXPECT_EQ(child.get_visits(), 1);
    EXPECT_EQ(child.get_wins(), 1.0);
    
    // Root should have 1 visit, 0 wins (flipped result)
    EXPECT_EQ(root.get_visits(), 1);
    EXPECT_EQ(root.get_wins(), 0.0);
}

TEST_F(MCTSNodeTest, ExpansionCreatesChild) {
    MCTSNode root;
    
    // Expansion should create a child node
    MCTSNode* expanded = root.expand(*state_, *move_generator_);
    
    EXPECT_NE(expanded, nullptr);
    EXPECT_EQ(root.child_count(), 1);
    EXPECT_FALSE(root.is_leaf());
    EXPECT_EQ(expanded->get_parent(), &root);
}

TEST_F(MCTSNodeTest, SelectionChoosesBestUCB1Child) {
    MCTSNode root;
    
    // Create some children manually and give them different statistics
    MCTSNode* child1 = root.expand(*state_, *move_generator_);
    MCTSNode* child2 = root.expand(*state_, *move_generator_);
    
    if (child1 && child2) {
        // Give different win rates to children
        child1->backpropagate(0.3); // Lower win rate
        child2->backpropagate(0.7); // Higher win rate
        
        // Best child should be the one with higher UCB1 value
        MCTSNode* best = root.select_best_child();
        
        // With equal visits, higher win rate should have higher UCB1
        EXPECT_NE(best, nullptr);
        EXPECT_TRUE(best == child1 || best == child2);
        
        // The one with higher win rate should be selected
        if (child1->get_visits() == child2->get_visits()) {
            EXPECT_EQ(best, child2); // Higher win rate
        }
    }
}

TEST_F(MCTSNodeTest, MostVisitedChildSelection) {
    MCTSNode root;
    
    // Create children
    MCTSNode* child1 = root.expand(*state_, *move_generator_);
    MCTSNode* child2 = root.expand(*state_, *move_generator_);
    
    if (child1 && child2) {
        // Give child2 more visits
        child1->backpropagate(0.5);
        child2->backpropagate(0.5);
        child2->backpropagate(0.5);
        
        MCTSNode* most_visited = root.get_most_visited_child();
        EXPECT_EQ(most_visited, child2);
        EXPECT_EQ(most_visited->get_visits(), 2);
    }
}

TEST_F(MCTSNodeTest, FullyExpandedDetection) {
    MCTSNode root;
    
    // Initially has no children, so is a leaf
    EXPECT_TRUE(root.is_leaf());
    
    // After first expansion, should have one child and potentially more moves to try
    MCTSNode* child = root.expand(*state_, *move_generator_);
    EXPECT_NE(child, nullptr);
    EXPECT_FALSE(root.is_leaf());
    
    // Keep expanding until fully expanded or no more children
    int expansion_count = 1;
    while (!root.is_fully_expanded() && child != nullptr && expansion_count < 100) {
        child = root.expand(*state_, *move_generator_);
        expansion_count++;
    }
    
    // Should eventually become fully expanded (when all moves are tried)
    EXPECT_TRUE(root.is_fully_expanded());
    EXPECT_GT(expansion_count, 1); // Should have expanded multiple times
}

TEST_F(MCTSNodeTest, NoMemoryLeaks) {
    // Create a tree structure and let it go out of scope
    {
        MCTSNode root;
        
        // Create several children
        for (int i = 0; i < 5; i++) {
            root.expand(*state_, *move_generator_);
        }
        
        // Create grandchildren
        MCTSNode* child = root.select_best_child();
        if (child) {
            child->expand(*state_, *move_generator_);
            child->expand(*state_, *move_generator_);
        }
        
        // Tree should be automatically cleaned up when root goes out of scope
    }
    
    // No explicit test for memory leaks here, but valgrind or similar tools
    // would catch issues. The unique_ptr usage should handle cleanup automatically.
    SUCCEED();
}