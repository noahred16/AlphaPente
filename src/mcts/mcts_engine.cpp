#include "mcts/mcts_engine.hpp"
#include <algorithm>
#include <chrono>

namespace mcts {

MCTSEngine::MCTSEngine(core::GameState& shared_state, 
                       const core::MoveGenerator& move_generator)
    : shared_state_(shared_state), move_generator_(move_generator) {
    root_ = std::make_unique<MCTSNode>();
}

core::Position MCTSEngine::search(int max_iterations, double time_limit_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize root with current game state's legal moves
    if (root_->is_leaf() && !shared_state_.is_terminal()) {
        root_->expand(shared_state_, move_generator_);
    }
    
    int iterations = 0;
    
    while (iterations < max_iterations) {
        // Check time limit
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
            
        if (elapsed >= time_limit_ms) {
            break;
        }
        
        // MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation
        MCTSNode* selected_node = select_node(root_.get());
        MCTSNode* expanded_node = expand_node(selected_node);
        double result = simulate_from_node(expanded_node ? expanded_node : selected_node);
        backpropagate(expanded_node ? expanded_node : selected_node, result);
        
        iterations++;
        total_simulations_++;
        
        // Early stopping if one move dominates significantly
        if (iterations >= 100 && should_stop_early()) {
            break;
        }
    }
    
    // Select best move based on visit count (most robust)
    MCTSNode* best_child = root_->get_most_visited_child();
    
    return best_child ? best_child->get_move() : core::Position{-1, -1};
}

void MCTSEngine::update_root(core::Position opponent_move) {
    // Try to find the child node corresponding to opponent's move
    std::unique_ptr<MCTSNode> new_root = root_->extract_child(opponent_move);
    
    if (new_root != nullptr) {
        // Successfully found and extracted the subtree
        root_ = std::move(new_root);
        tree_reuse_count_++;
        
        // Optional: Log successful tree reuse
        // std::cout << "Tree reuse: preserved " << root_->get_visits() 
        //           << " visits from subtree" << std::endl;
    } else {
        // Opponent's move was not explored in our tree, create fresh root
        root_ = std::make_unique<MCTSNode>();
        tree_fallback_count_++;
        
        // Optional: Log tree fallback
        // std::cout << "Tree fallback: opponent move not in explored tree" << std::endl;
    }
}

MCTSNode* MCTSEngine::select_node(MCTSNode* node) {
    while (!node->is_leaf() && node->is_fully_expanded()) {
        node = node->select_best_child();
        if (node == nullptr) break;
    }
    return node;
}

MCTSNode* MCTSEngine::expand_node(MCTSNode* node) {
    if (node->is_fully_expanded() || shared_state_.is_terminal()) {
        return nullptr; // Cannot expand
    }
    
    // Apply moves from root to current node
    apply_moves_to_state(node);
    
    // Expand the node
    MCTSNode* new_node = node->expand(shared_state_, move_generator_);
    
    // Restore state back to root
    restore_state_to_root();
    
    return new_node;
}

double MCTSEngine::simulate_from_node(MCTSNode* node) {
    // Apply moves from root to simulation start node
    apply_moves_to_state(node);
    
    // Run rollout simulation
    double result = rollout_policy_.simulate(shared_state_, move_generator_);
    
    // Restore state back to root
    restore_state_to_root();
    
    return result;
}

void MCTSEngine::backpropagate(MCTSNode* node, double result) {
    node->backpropagate(result);
}

bool MCTSEngine::should_stop_early() const {
    MCTSNode* best_child = root_->get_most_visited_child();
    if (best_child == nullptr || root_->get_visits() < 200) {
        return false;
    }
    
    // Stop if best move has >60% of all visits
    double visit_ratio = static_cast<double>(best_child->get_visits()) / root_->get_visits();
    return visit_ratio > 0.6;
}

void MCTSEngine::apply_moves_to_state(MCTSNode* target_node) {
    if (target_node == nullptr) return;
    
    // Collect path from root to target
    std::vector<MCTSNode*> path;
    MCTSNode* current = target_node;
    
    while (current != nullptr && current != root_.get()) {
        path.push_back(current);
        current = current->get_parent();
    }
    
    // Apply moves in reverse order (root to target)
    std::reverse(path.begin(), path.end());
    moves_applied_ = 0;
    
    for (MCTSNode* node : path) {
        core::Position move = node->get_move();
        if (move.row != -1 && move.col != -1) {
            shared_state_.make_move(move.row, move.col);
            moves_applied_++;
        }
    }
}

void MCTSEngine::restore_state_to_root() {
    // Undo all moves we applied
    for (int i = 0; i < moves_applied_; i++) {
        shared_state_.undo_move();
    }
    moves_applied_ = 0;
}

} // namespace mcts