#include <gtest/gtest.h>
#include "core/distance_rings.hpp"
#include <chrono>
#include <set>

using namespace core;

class DistanceRingsTest : public ::testing::Test {
protected:
    DistanceRings rings;
};

// Test Chebyshev distance calculations
TEST_F(DistanceRingsTest, ChebyshevDistanceCalculations) {
    // Test center position (9,9)
    Position center{9, 9};
    
    // Distance 1 should include immediate neighbors (8 positions)
    auto distance1 = rings.get_positions_at_distance(center, 1);
    EXPECT_EQ(distance1.size(), 8);
    
    // Check that all distance 1 positions are adjacent
    for (const auto& pos : distance1) {
        int row_diff = std::abs(pos.row - center.row);
        int col_diff = std::abs(pos.col - center.col);
        int chebyshev_dist = std::max(row_diff, col_diff);
        EXPECT_EQ(chebyshev_dist, 1);
    }
    
    // Distance 2 should have 16 positions (3x3 - 1x1 = 8 positions at distance 1)
    auto distance2 = rings.get_positions_at_distance(center, 2);
    EXPECT_EQ(distance2.size(), 16);
    
    // Check that all distance 2 positions are correct
    for (const auto& pos : distance2) {
        int row_diff = std::abs(pos.row - center.row);
        int col_diff = std::abs(pos.col - center.col);
        int chebyshev_dist = std::max(row_diff, col_diff);
        EXPECT_EQ(chebyshev_dist, 2);
    }
}

// Test edge position behavior
TEST_F(DistanceRingsTest, EdgePositions) {
    // Test corner (0,0)
    Position corner{0, 0};
    
    auto distance1 = rings.get_positions_at_distance(corner, 1);
    EXPECT_EQ(distance1.size(), 3); // Only 3 neighbors for corner
    
    // Test edge position (0,9)
    Position edge{0, 9};
    auto edge_distance1 = rings.get_positions_at_distance(edge, 1);
    EXPECT_EQ(edge_distance1.size(), 5); // 5 neighbors for edge
    
    // Test that all positions are valid and within bounds
    for (const auto& pos : edge_distance1) {
        EXPECT_GE(pos.row, 0);
        EXPECT_LT(pos.row, 19);
        EXPECT_GE(pos.col, 0);
        EXPECT_LT(pos.col, 19);
    }
}

// Test bounds checking
TEST_F(DistanceRingsTest, BoundsChecking) {
    // Invalid position should return empty vector
    Position invalid{-1, 5};
    auto result = rings.get_positions_at_distance(invalid, 1);
    EXPECT_TRUE(result.empty());
    
    // Invalid distance should return empty vector
    Position valid{9, 9};
    auto invalid_dist = rings.get_positions_at_distance(valid, -1);
    EXPECT_TRUE(invalid_dist.empty());
    
    auto too_large_dist = rings.get_positions_at_distance(valid, 25);
    EXPECT_TRUE(too_large_dist.empty());
}

// Test get_ordered_moves_around_stones
TEST_F(DistanceRingsTest, OrderedMovesAroundStones) {
    std::vector<Position> stones = {{9, 9}, {10, 10}};
    
    // Get moves within distance 2
    auto moves = rings.get_ordered_moves_around_stones(stones, 2);
    
    // Should not be empty
    EXPECT_FALSE(moves.empty());
    
    // Should not contain the stone positions themselves
    for (const auto& move : moves) {
        EXPECT_NE(move, stones[0]);
        EXPECT_NE(move, stones[1]);
    }
    
    // All positions should be valid
    for (const auto& move : moves) {
        EXPECT_GE(move.row, 0);
        EXPECT_LT(move.row, 19);
        EXPECT_GE(move.col, 0);
        EXPECT_LT(move.col, 19);
    }
    
    // No duplicates should exist
    std::set<std::pair<int, int>> unique_moves;
    for (const auto& move : moves) {
        unique_moves.insert({move.row, move.col});
    }
    EXPECT_EQ(unique_moves.size(), moves.size());
}

// Test ordering (closer positions should come first)
TEST_F(DistanceRingsTest, MoveOrdering) {
    std::vector<Position> stones = {{9, 9}};
    
    auto moves = rings.get_ordered_moves_around_stones(stones, 3);
    
    // Check that moves are ordered by distance (distance 1 positions come before distance 2, etc.)
    int prev_max_distance = 0;
    for (const auto& move : moves) {
        int row_diff = std::abs(move.row - stones[0].row);
        int col_diff = std::abs(move.col - stones[0].col);
        int distance = std::max(row_diff, col_diff);
        
        EXPECT_GE(distance, prev_max_distance);
        prev_max_distance = std::max(prev_max_distance, distance);
    }
}

// Performance test - should initialize quickly
TEST_F(DistanceRingsTest, InitializationPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create new instance to test initialization time
    DistanceRings test_rings;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should initialize in under 100ms (target is ~50ms)
    EXPECT_LT(duration.count(), 100);
}

// Test maximum distance
TEST_F(DistanceRingsTest, MaximumDistance) {
    // From corner (0,0) to opposite corner (18,18) should be distance 18
    Position corner1{0, 0};
    Position corner2{18, 18};
    
    auto max_dist_positions = rings.get_positions_at_distance(corner1, 18);
    EXPECT_FALSE(max_dist_positions.empty());
    
    // corner2 should be in the distance 18 ring from corner1
    bool found_opposite_corner = false;
    for (const auto& pos : max_dist_positions) {
        if (pos == corner2) {
            found_opposite_corner = true;
            break;
        }
    }
    EXPECT_TRUE(found_opposite_corner);
}