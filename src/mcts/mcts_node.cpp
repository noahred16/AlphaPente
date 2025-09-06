#include "mcts/mcts_node.hpp"
#include <algorithm>
#include <limits>
#include <random>

namespace mcts {

MCTSNode::MCTSNode(MCTSNode* parent, core::Position move)
    : parent_(parent), move_(move) {
}

double MCTSNode::get_win_rate() const noexcept {
    return visits_ > 0 ? wins_ / visits_ : 0.0;
}

double MCTSNode::ucb1_value(double exploration_weight) const noexcept {
    if (visits_ == 0) {
        return std::numeric_limits<double>::infinity(); // Unvisited nodes get highest priority
    }
    
    if (parent_ == nullptr || parent_->visits_ == 0) {
        return get_win_rate(); // Root node or no parent visits
    }
    
    double exploitation = get_win_rate();
    double exploration = exploration_weight * std::sqrt(std::log(parent_->visits_) / visits_);
    
    return exploitation + exploration;
}

MCTSNode* MCTSNode::select_best_child() const noexcept {
    if (children_.empty()) {
        return nullptr;
    }
    
    MCTSNode* best_child = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();
    
    for (const auto& child : children_) {
        double value = child->ucb1_value();
        if (value > best_value) {
            best_value = value;
            best_child = child.get();
        }
    }
    
    return best_child;
}

MCTSNode* MCTSNode::expand(core::GameState& state, const core::MoveGenerator& move_gen) {
    // Initialize untried moves if not done yet
    if (untried_moves_.empty() && children_.empty()) {
        initialize_untried_moves(state, move_gen);
    }
    
    // No moves to expand
    if (untried_moves_.empty()) {
        return nullptr;
    }
    
    // Take the first untried move
    core::Position move = untried_moves_.back();
    untried_moves_.pop_back();
    
    // Create new child node
    auto child = std::make_unique<MCTSNode>(this, move);
    MCTSNode* child_ptr = child.get();
    
    children_.push_back(std::move(child));
    
    return child_ptr;
}

void MCTSNode::backpropagate(double result) noexcept {
    visits_++;
    wins_ += result;
    
    if (parent_ != nullptr) {
        // Flip result for parent (opponent's perspective)
        parent_->backpropagate(1.0 - result);
    }
}

MCTSNode* MCTSNode::get_most_visited_child() const noexcept {
    if (children_.empty()) {
        return nullptr;
    }
    
    MCTSNode* best_child = nullptr;
    int max_visits = -1;
    
    for (const auto& child : children_) {
        if (child->visits_ > max_visits) {
            max_visits = child->visits_;
            best_child = child.get();
        }
    }
    
    return best_child;
}

void MCTSNode::initialize_untried_moves(core::GameState& state, const core::MoveGenerator& move_gen) {
    // Get legal moves for current position
    untried_moves_ = move_gen.generate_ordered_moves(state);
    
    // Shuffle to add randomization
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(untried_moves_.begin(), untried_moves_.end(), g);
}

} // namespace mcts