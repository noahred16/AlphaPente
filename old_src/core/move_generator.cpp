#include "core/move_generator.hpp"
#include <algorithm>
#include <iostream>

namespace core {

MoveGenerator::MoveGenerator(const DistanceRings& rings) 
    : distance_rings_(rings) {
}

std::vector<Position> MoveGenerator::generate_ordered_moves(
    const GameState& state,
    int node_visits,
    int max_moves,
    int max_distance) const {

    // Apply progressive widening if no overrides provided
    if (node_visits > 0 && max_moves == 50 && max_distance == 3) {
        auto limits = ProgressiveWidening::get_limits(node_visits);
        max_moves = limits.max_moves;
        max_distance = limits.max_distance;
    }

    const std::vector<Position>& all_stones = get_all_stones(state);
    
    // Handle empty board case
    if (all_stones.empty()) {
        auto center_moves = get_center_area_moves();
        return filter_legal_moves(state, center_moves);
    }
    
    // Get candidate moves around existing stones
    auto candidate_moves = distance_rings_.get_ordered_moves_around_stones(
        all_stones, max_distance);
    
    // Filter out occupied positions
    auto legal_moves = filter_legal_moves(state, candidate_moves);
    
    // Limit number of moves
    if (static_cast<int>(legal_moves.size()) > max_moves) {
        legal_moves.resize(max_moves);
    }

    return legal_moves;
}

const std::vector<Position>& MoveGenerator::get_all_stones(const GameState& state) const {
    // O(1) access thanks to GameState optimization!
    return state.get_stone_positions();
}

std::vector<Position> MoveGenerator::filter_legal_moves(
    const GameState& state, 
    const std::vector<Position>& candidate_moves) const {
    
    std::vector<Position> legal_moves;
    legal_moves.reserve(candidate_moves.size());
    
    for (const Position& pos : candidate_moves) {
        // Check bounds and occupancy
        if (pos.row >= 0 && pos.row < 19 && 
            pos.col >= 0 && pos.col < 19 && 
            state.is_empty(pos.row, pos.col)) {
            legal_moves.push_back(pos);
        }
    }
    
    return legal_moves;
}

std::vector<Position> MoveGenerator::get_center_area_moves() const {
    std::vector<Position> center_moves;
    center_moves.reserve(25); // 5x5 center area
    
    // Start with center position
    center_moves.emplace_back(9, 9);
    
    // Add positions in expanding rings around center
    for (int distance = 1; distance <= 2; ++distance) {
        auto positions_at_distance = distance_rings_.get_positions_at_distance(
            Position{9, 9}, distance);
        
        for (const Position& pos : positions_at_distance) {
            center_moves.push_back(pos);
        }
    }
    
    return center_moves;
}

} // namespace core