#include "mcts/mcts_engine.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

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

        std::cout << "\n=== MCTS ITERATION " << iterations + 1 << " ===\n";

        // MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation
        MCTSNode* selected_node = select_node(root_.get());
        MCTSNode* expanded_node = expand_node(selected_node);

        std::cout << "DEBUG MAIN: expanded_node is " << (expanded_node ? "NOT NULL" : "NULL") << "\n";

        double result = simulate_from_node(expanded_node ? expanded_node : selected_node);
        backpropagate(expanded_node ? expanded_node : selected_node, result);

        iterations++;
        total_simulations_++;

        // Stop after a few iterations for debugging
        if (iterations >= 5) {
            std::cout << "DEBUG: Stopping after 5 iterations for analysis\n";
            break;
        }
        
        // Early stopping if one move dominates significantly
        if (iterations >= 100 && should_stop_early()) {
            break;
        }
    }
    
    // Select best move based on visit count (most robust)
    MCTSNode* best_child = root_->get_most_visited_child();
    
    // DEBUG: Print top children for debugging
    std::cout << "DEBUG: Top MCTS children:\n";
    for (size_t i = 0; i < root_->child_count() && i < 5; i++) {
        // We need a way to iterate children - for now just show we found best
        if (best_child) {
            core::Position move = best_child->get_move();
            char col_char = 'A' + move.col;
            int display_row = 19 - move.row;
            std::cout << "  Best child: " << col_char << display_row 
                      << " (visits: " << best_child->get_visits() << ")\n";
            break;
        }
    }
    
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
    std::cout << "DEBUG SELECT: Starting selection from " << (node == root_.get() ? "ROOT" : "non-root") << "\n";

    int depth = 0;
    while (!node->is_leaf()) {
        std::cout << "DEBUG SELECT: At depth " << depth << ", node has " << node->child_count() << " children\n";
        node = node->select_best_child();
        if (node == nullptr) {
            std::cout << "DEBUG SELECT: select_best_child returned nullptr!\n";
            break;
        }
        const auto& move = node->get_move();
        std::cout << "DEBUG SELECT: Selected child with move " << move.to_string()
                  << " (visits: " << node->get_visits() << ")\n";
        depth++;
        if (depth > 10) {
            std::cout << "DEBUG SELECT: Breaking due to excessive depth!\n";
            break;
        }
    }

    std::cout << "DEBUG SELECT: Final selected node - move: " << node->get_move().to_string()
              << ", visits: " << node->get_visits()
              << ", is_leaf: " << node->is_leaf()
              << ", is_fully_expanded: " << node->is_fully_expanded() << "\n";

    return node;
}

MCTSNode* MCTSEngine::expand_node(MCTSNode* node) {
    std::cout << "DEBUG EXPAND: Attempting to expand node with move " << node->get_move().to_string() << "\n";
    std::cout << "DEBUG EXPAND: Node state - visits: " << node->get_visits()
              << ", is_fully_expanded: " << node->is_fully_expanded()
              << ", child_count: " << node->child_count() << "\n";

    if (node->is_fully_expanded()) {
        std::cout << "DEBUG EXPAND: Cannot expand - node is fully expanded\n";
        return nullptr;
    }

    if (shared_state_.is_terminal()) {
        std::cout << "DEBUG EXPAND: Cannot expand - game state is terminal\n";
        return nullptr;
    }

    // Apply moves from root to current node
    std::cout << "DEBUG EXPAND: Applying moves to reach target node\n";
    apply_moves_to_state(node);

    // Check if state is terminal after applying moves
    if (shared_state_.is_terminal()) {
        std::cout << "DEBUG EXPAND: Game state became terminal after applying moves\n";
        restore_state_to_root();
        return nullptr;
    }

    // Expand the node
    std::cout << "DEBUG EXPAND: Calling node->expand()\n";
    MCTSNode* new_node = node->expand(shared_state_, move_generator_);

    if (new_node == nullptr) {
        std::cout << "DEBUG EXPAND: node->expand() returned nullptr\n";
    } else {
        std::cout << "DEBUG EXPAND: Successfully created new child with move "
                  << new_node->get_move().to_string() << "\n";
    }

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