#pragma once

#include "mcts_node.hpp"
#include "rollout.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "utils/timer.hpp"
#include <memory>
#include <random>

namespace mcts {

class MCTSEngine {
public:
    MCTSEngine(core::GameState& shared_state, 
               const core::MoveGenerator& move_generator);
    ~MCTSEngine() = default;
    
    // Main search interface
    core::Position search(int max_iterations = 1000, double time_limit_ms = 1000.0);
    
    // Tree reuse - move root to opponent's move subtree 
    void update_root(core::Position opponent_move);
    
    // Configuration
    void set_exploration_weight(double weight) { exploration_weight_ = weight; }
    
    // Statistics
    int get_total_simulations() const { return total_simulations_; }
    void reset_statistics() { total_simulations_ = 0; }

private:
    // Single shared state for all simulations - no copying!
    core::GameState& shared_state_;
    const core::MoveGenerator& move_generator_;
    RolloutPolicy rollout_policy_;
    
    // Tree management
    std::unique_ptr<MCTSNode> root_;
    double exploration_weight_ = 1.414;  // UCB1 exploration parameter
    
    // Statistics
    int total_simulations_ = 0;
    int moves_applied_ = 0;  // Track moves applied to shared state for proper undo
    
    // MCTS phases
    MCTSNode* select_node(MCTSNode* node);
    MCTSNode* expand_node(MCTSNode* node);
    double simulate_from_node(MCTSNode* node);
    void backpropagate(MCTSNode* node, double result);
    
    // Utilities
    bool should_stop_early() const;
    void apply_moves_to_state(MCTSNode* target_node);
    void restore_state_to_root();
};

} // namespace mcts