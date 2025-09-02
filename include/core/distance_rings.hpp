#pragma once

#include "move_delta.hpp"
#include <vector>
#include <array>

namespace core {

class DistanceRings {
private:
    // Pre-allocated 2D arrays for maximum performance
    static constexpr int BOARD_SIZE = 19;
    static constexpr int MAX_POSITIONS = BOARD_SIZE * BOARD_SIZE;
    static constexpr int MAX_DISTANCE = 18; // Maximum Chebyshev distance on 19x19 board
    
    // rings_[position_id][distance] = vector of positions at that distance
    std::array<std::array<std::vector<Position>, MAX_DISTANCE + 1>, MAX_POSITIONS> rings_;
    
    inline int position_to_id(Position pos) const noexcept {
        return pos.row * BOARD_SIZE + pos.col;
    }
    
    inline Position id_to_position(int id) const noexcept {
        return Position{static_cast<int8_t>(id / BOARD_SIZE), static_cast<int8_t>(id % BOARD_SIZE)};
    }
    
    inline int chebyshev_distance(Position a, Position b) const noexcept {
        return std::max(std::abs(a.row - b.row), std::abs(a.col - b.col));
    }
    
    void precompute_all_distances();

public:
    DistanceRings();  // Precompute all distances at startup (~50ms)
    ~DistanceRings() = default;
    
    const std::vector<Position>& get_positions_at_distance(
        Position center, int distance) const noexcept;
    
    std::vector<Position> get_ordered_moves_around_stones(
        const std::vector<Position>& stone_positions,
        int max_distance = 3) const;
};

} // namespace core