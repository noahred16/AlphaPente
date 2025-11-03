#pragma once

#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include <random>

namespace mcts {

class RolloutPolicy {
public:
    RolloutPolicy();
    ~RolloutPolicy() = default;
    
    // Perform rollout simulation from current state, return result from current player's perspective
    double simulate(core::GameState& state, const core::MoveGenerator& move_gen);

private:
    std::mt19937 rng_;
    static constexpr int MAX_ROLLOUT_MOVES = 100;  // Prevent infinite games
    static constexpr int MAX_DISTANCE = 3;         // Focus moves near existing stones
    
    // Select a random move prioritizing moves near stones
    core::Position select_rollout_move(const core::GameState& state, 
                                     const core::MoveGenerator& move_gen);
    
    // Evaluate final game result from perspective of given player
    // moves_played: number of moves in rollout (used to prefer quick wins / slow losses)
    double evaluate_result(const core::GameState& state, int player, int moves_played = 0) const noexcept;
};

} // namespace mcts