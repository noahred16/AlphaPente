#include "core/distance_rings.hpp"
#include <unordered_set>
#include <algorithm>

namespace core {

DistanceRings::DistanceRings() {
    precompute_all_distances();
}

void DistanceRings::precompute_all_distances() {
    // For each position on the board
    for (int center_row = 0; center_row < BOARD_SIZE; ++center_row) {
        for (int center_col = 0; center_col < BOARD_SIZE; ++center_col) {
            Position center{static_cast<int8_t>(center_row), static_cast<int8_t>(center_col)};
            int center_id = position_to_id(center);
            
            // For each other position on the board
            for (int target_row = 0; target_row < BOARD_SIZE; ++target_row) {
                for (int target_col = 0; target_col < BOARD_SIZE; ++target_col) {
                    Position target{static_cast<int8_t>(target_row), static_cast<int8_t>(target_col)};
                    
                    // Skip the center position itself
                    if (center == target) continue;
                    
                    int distance = chebyshev_distance(center, target);
                    
                    // Only store distances within our maximum range
                    if (distance <= MAX_DISTANCE) {
                        rings_[center_id][distance].push_back(target);
                    }
                }
            }
            
            // Sort each distance ring for consistent ordering
            for (int distance = 1; distance <= MAX_DISTANCE; ++distance) {
                std::sort(rings_[center_id][distance].begin(), rings_[center_id][distance].end(),
                    [](const Position& a, const Position& b) {
                        if (a.row != b.row) return a.row < b.row;
                        return a.col < b.col;
                    });
            }
        }
    }
}

const std::vector<Position>& DistanceRings::get_positions_at_distance(
    Position center, int distance) const noexcept {
    
    // Bounds checking
    if (center.row < 0 || center.row >= BOARD_SIZE || 
        center.col < 0 || center.col >= BOARD_SIZE ||
        distance < 0 || distance > MAX_DISTANCE) {
        static const std::vector<Position> empty;
        return empty;
    }
    
    int center_id = position_to_id(center);
    return rings_[center_id][distance];
}

std::vector<Position> DistanceRings::get_ordered_moves_around_stones(
    const std::vector<Position>& stone_positions,
    int max_distance) const {
    
    std::unordered_set<int> unique_positions; // Use position_id for uniqueness
    std::vector<std::vector<Position>> distance_groups(max_distance + 1);
    
    // Add stone positions to unique_positions to exclude them from results
    for (const Position& stone : stone_positions) {
        unique_positions.insert(position_to_id(stone));
    }
    
    // Collect all positions at each distance from any stone
    for (const Position& stone : stone_positions) {
        for (int distance = 1; distance <= max_distance; ++distance) {
            const auto& positions_at_distance = get_positions_at_distance(stone, distance);
            
            for (const Position& pos : positions_at_distance) {
                int pos_id = position_to_id(pos);
                
                // Only add if we haven't seen this position before
                if (unique_positions.find(pos_id) == unique_positions.end()) {
                    unique_positions.insert(pos_id);
                    distance_groups[distance].push_back(pos);
                }
            }
        }
    }
    
    // Build result vector ordered by distance (closest first)
    std::vector<Position> result;
    for (int distance = 1; distance <= max_distance; ++distance) {
        // Sort positions at this distance for consistency
        std::sort(distance_groups[distance].begin(), distance_groups[distance].end(),
            [](const Position& a, const Position& b) {
                if (a.row != b.row) return a.row < b.row;
                return a.col < b.col;
            });
        
        // Add to result
        result.insert(result.end(), distance_groups[distance].begin(), distance_groups[distance].end());
    }
    
    return result;
}

} // namespace core