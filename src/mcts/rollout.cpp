#include "mcts/rollout.hpp"
#include <algorithm>

namespace mcts {

RolloutPolicy::RolloutPolicy() : rng_(std::random_device{}()) {
}

double RolloutPolicy::simulate(core::GameState& state, const core::MoveGenerator& move_gen) {
    int starting_player = state.current_player();
    int moves_played = 0;
    
    // Play random moves until terminal or max moves
    while (!state.is_terminal() && moves_played < MAX_ROLLOUT_MOVES) {
        core::Position move = select_rollout_move(state, move_gen);
        
        // No legal moves available
        if (move.row == -1 || move.col == -1) {
            break;
        }
        
        state.make_move(move.row, move.col);
        moves_played++;
    }
    
    // Evaluate result from starting player's perspective
    double result = evaluate_result(state, starting_player);
    
    // Undo all moves to restore state
    for (int i = 0; i < moves_played; i++) {
        state.undo_move();
    }
    
    return result;
}

core::Position RolloutPolicy::select_rollout_move(const core::GameState& state, 
                                                const core::MoveGenerator& move_gen) {
    // Get moves prioritized by distance from existing stones
    std::vector<core::Position> moves = move_gen.generate_ordered_moves(
        state, 30, MAX_DISTANCE  // Limit moves for rollout efficiency
    );
    
    if (moves.empty()) {
        return {-1, -1}; // No legal moves
    }
    
    // Simple random selection from distance-ordered moves
    // Could be enhanced with more sophisticated heuristics
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng_)];
}

double RolloutPolicy::evaluate_result(const core::GameState& state, int player) const noexcept {
    if (!state.is_terminal()) {
        // Non-terminal position - return draw value
        return 0.5;
    }
    
    int winner = state.get_winner();
    
    if (winner == 0) {
        // Draw
        return 0.5;
    } else if (winner == player) {
        // Player wins
        return 1.0;
    } else {
        // Player loses
        return 0.0;
    }
}

} // namespace mcts