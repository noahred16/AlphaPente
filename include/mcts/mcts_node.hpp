#pragma once

#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace mcts {

class MCTSNode {
public:
    // Constructor - only root node has no move/parent
    MCTSNode(MCTSNode* parent = nullptr, core::Position move = {-1, -1});
    ~MCTSNode() = default;

    // Statistics accessors
    double get_win_rate() const noexcept;
    int get_visits() const noexcept { return visits_; }
    double get_wins() const noexcept { return wins_; }
    MCTSNode* get_parent() const noexcept { return parent_; }
    const core::Position& get_move() const noexcept { return move_; }
    
    // UCB1 calculation for selection
    double ucb1_value(double exploration_weight = 1.414) const noexcept;
    
    // Tree operations
    MCTSNode* select_best_child() const noexcept;
    MCTSNode* expand(core::GameState& state, const core::MoveGenerator& move_gen);
    void backpropagate(double result) noexcept;
    
    // Child management
    bool is_fully_expanded() const noexcept { return untried_moves_.empty(); }
    bool is_leaf() const noexcept { return children_.empty(); }
    size_t child_count() const noexcept { return children_.size(); }
    
    // Get best move for final selection
    MCTSNode* get_most_visited_child() const noexcept;
    
    // Tree reuse support
    MCTSNode* find_child_with_move(const core::Position& move) const noexcept;
    std::unique_ptr<MCTSNode> extract_child(const core::Position& move) noexcept;
    
    // Debug support - get top N children by visit count
    std::vector<const MCTSNode*> get_top_children(int count = 10) const;

private:
    // Hot data (accessed every simulation) - keep together for cache efficiency
    int visits_ = 0;
    double wins_ = 0.0;
    MCTSNode* parent_ = nullptr;
    
    // Cold data (accessed less frequently)
    core::Position move_;  // Move that led to this state
    std::vector<std::unique_ptr<MCTSNode>> children_;
    std::vector<core::Position> untried_moves_;
    
    // Initialize untried moves for this node
    void initialize_untried_moves(core::GameState& state, const core::MoveGenerator& move_gen);
};

} // namespace mcts