#pragma once

#include "game_state.hpp"
#include "distance_rings.hpp"
#include "move_delta.hpp"
#include <vector>

namespace core {

class MoveGenerator {
private:
    const DistanceRings& distance_rings_;
    
    // Get all stone positions from the board (now O(1) thanks to GameState optimization)
    const std::vector<Position>& get_all_stones(const GameState& state) const;
    
    // Filter out occupied positions
    std::vector<Position> filter_legal_moves(
        const GameState& state, 
        const std::vector<Position>& candidate_moves) const;
    
    // Get center area moves for empty board
    std::vector<Position> get_center_area_moves() const;

public:
    explicit MoveGenerator(const DistanceRings& rings);
    ~MoveGenerator() = default;
    
    std::vector<Position> generate_ordered_moves(
        const GameState& state,
        int node_visits = 0,      // For progressive widening
        int max_moves = 50,       // Move limit override
        int max_distance = 3      // Distance limit override
    ) const;
    
    // Progressive widening strategy based on visit counts
    struct ProgressiveWidening {
        int max_moves;
        int max_distance;
        
        static ProgressiveWidening get_limits(int visits) {
            if (visits < 10) return {15, 1};        // Very focused search
            if (visits < 100) return {30, 2};       // Moderate expansion  
            if (visits < 1000) return {50, 3};      // Standard search
            return {80, 5};                         // Full exploration
        }
    };
};

} // namespace core