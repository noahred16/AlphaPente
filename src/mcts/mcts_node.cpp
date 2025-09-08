#include "mcts/mcts_node.hpp"
#include <algorithm>
#include <limits>
#include <random>
#include <iostream>

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
    // Get legal moves for current position - keep distance ordering!
    // Pass visit count for progressive widening
    untried_moves_ = move_gen.generate_ordered_moves(state, visits_);
    
    // DEBUG: Print first few untried moves
    std::cout << "DEBUG: Initialized " << untried_moves_.size() << " untried moves, node has " << visits_ << " visits, first 10:\n";
    // print out first 10 untried moves
    for (size_t i = 0; i < std::min(size_t(10), untried_moves_.size()); i++) {
        char col_char = 'A' + untried_moves_[i].col;
        int display_row = 19 - untried_moves_[i].row;
        std::cout << "  " << (i+1) << ". " << col_char << display_row << "\n";
    }
    
    // Don't shuffle! Distance ring ordering is important for move quality
    // MCTS will naturally explore the tree, no need for artificial randomization
}

MCTSNode* MCTSNode::find_child_with_move(const core::Position& move) const noexcept {
    for (const auto& child : children_) {
        const core::Position& child_move = child->get_move();
        if (child_move.row == move.row && child_move.col == move.col) {
            return child.get();
        }
    }
    return nullptr;
}

std::unique_ptr<MCTSNode> MCTSNode::extract_child(const core::Position& move) noexcept {
    for (auto it = children_.begin(); it != children_.end(); ++it) {
        const core::Position& child_move = (*it)->get_move();
        if (child_move.row == move.row && child_move.col == move.col) {
            // Extract the child and remove from children vector
            std::unique_ptr<MCTSNode> extracted_child = std::move(*it);
            children_.erase(it);
            
            // Update parent pointer to null (will become new root)
            extracted_child->parent_ = nullptr;
            
            return extracted_child;
        }
    }
    return nullptr; // Child not found
}

std::vector<const MCTSNode*> MCTSNode::get_top_children(int count) const {
    std::vector<const MCTSNode*> children_ptrs;
    children_ptrs.reserve(children_.size());
    
    // Convert unique_ptr vector to raw pointer vector for sorting
    for (const auto& child : children_) {
        children_ptrs.push_back(child.get());
    }
    
    // Sort by visit count (descending)
    std::sort(children_ptrs.begin(), children_ptrs.end(), 
              [](const MCTSNode* a, const MCTSNode* b) {
                  return a->get_visits() > b->get_visits();
              });
    
    // Return top N children, or all children if count is -1
    if (count > 0 && static_cast<int>(children_ptrs.size()) > count) {
        children_ptrs.resize(count);
    }
    
    return children_ptrs;
}

} // namespace mcts