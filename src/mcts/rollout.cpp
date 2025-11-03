#include "mcts/rollout.hpp"
#include <algorithm>
#include <cmath>

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
    // Pass moves_played to reward quick wins and slow losses
    double result = evaluate_result(state, starting_player, moves_played);

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
    
    // Prefer closer moves using weighted selection
    // Moves are already ordered by distance (closest first)
    size_t num_moves = moves.size();
    
    // Use weighted selection favoring earlier (closer) moves
    std::vector<double> weights(num_moves);
    for (size_t i = 0; i < num_moves; i++) {
        // Exponential decay: earlier moves get much higher weight
        weights[i] = std::exp(-0.5 * i); // e^(-0.5*i)
    }
    
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return moves[dist(rng_)];
}

double RolloutPolicy::evaluate_result(const core::GameState& state, int player, int moves_played) const noexcept {
    if (!state.is_terminal()) {
        // Non-terminal position - return draw value
        return 0.5;
    }

    int winner = state.get_winner();

    if (winner == 0) {
        // Draw
        return 0.5;
    } else if (winner == player) {
        // Player wins - prefer quick wins
        // Reduce value based on move count (0.1 per move for very strong signal)
        // Example: Win in 1 move = 0.90, win in 5 moves = 0.51
        return std::max(0.51, 1.0 - (moves_played * 0.1));
    } else {
        // Player loses - prefer slow losses
        // Increase value based on move count (0.1 per move for very strong signal)
        // Example: Lose in 1 move = 0.10, lose in 5 moves = 0.49
        return std::min(0.49, 0.0 + (moves_played * 0.1));
    }
}

} // namespace mcts