#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include "mcts/mcts_engine.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cctype>

class PenteUI {
private:
    core::GameState game_state_;
    std::unique_ptr<core::DistanceRings> distance_rings_;
    std::unique_ptr<core::MoveGenerator> move_generator_;
    std::unique_ptr<mcts::MCTSEngine> engine_;
    
    // Engine settings
    int engine_iterations_ = 10000;
    double engine_time_limit_ = 10000.0; // 10 seconds

    // Move history for undo (pairs of human+engine moves)
    std::vector<std::pair<core::Position, core::Position>> move_pairs_;
    
public:
    PenteUI() {
        distance_rings_ = std::make_unique<core::DistanceRings>();
        move_generator_ = std::make_unique<core::MoveGenerator>(*distance_rings_);
        engine_ = std::make_unique<mcts::MCTSEngine>(game_state_, *move_generator_);
    }
    
    void display_board() const {
        std::cout << "\n    ";
        // Column headers A-S
        for (int col = 0; col < 19; col++) {
            std::cout << char('A' + col) << " ";
        }
        std::cout << "\n\n";
        
        // Rows 19 down to 1
        for (int row = 0; row < 19; row++) {
            // Row number (19, 18, 17, ..., 1)
            std::cout << std::setw(2) << (19 - row) << "  ";
            
            // Board cells
            for (int col = 0; col < 19; col++) {
                int stone = game_state_.get_stone(row, col);
                if (stone == 1) {
                    std::cout << "X ";
                } else if (stone == -1) {
                    std::cout << "O ";
                } else {
                    std::cout << ". ";
                }
            }
            
            // Row number on right side too
            std::cout << " " << (19 - row);
            std::cout << "\n";
        }
        
        std::cout << "\n    ";
        // Column headers again at bottom
        for (int col = 0; col < 19; col++) {
            std::cout << char('A' + col) << " ";
        }
        std::cout << "\n\n";
    }
    
    // Parse move in K10 format (column letter + row number)
    core::Position parse_move(const std::string& move_str) const {
        if (move_str.length() < 2) {
            return {-1, -1}; // Invalid
        }
        
        char col_char = std::toupper(move_str[0]);
        std::string row_str = move_str.substr(1);
        
        // Parse column (A-S = 0-18)
        if (col_char < 'A' || col_char > 'S') {
            return {-1, -1}; // Invalid column
        }
        int col = col_char - 'A';
        
        // Parse row (1-19, where 1 is row 18, 19 is row 0)
        try {
            int display_row = std::stoi(row_str);
            if (display_row < 1 || display_row > 19) {
                return {-1, -1}; // Invalid row
            }
            int row = 19 - display_row; // Convert to 0-based, flipped
            return {static_cast<int8_t>(row), static_cast<int8_t>(col)};
        } catch (const std::exception&) {
            return {-1, -1}; // Invalid number
        }
    }
    
    std::string position_to_move_string(const core::Position& pos) const {
        if (pos.row < 0 || pos.row >= 19 || pos.col < 0 || pos.col >= 19) {
            return "Invalid";
        }
        char col_char = 'A' + pos.col;
        int display_row = 19 - pos.row;
        return std::string(1, col_char) + std::to_string(display_row);
    }
    
    void display_game_info() const {
        std::cout << "=== AlphaPente ===\n";
        std::cout << "Current player: " << (game_state_.current_player() == 1 ? "Human (X)" : "Engine (O)") << "\n";
        std::cout << "Captures - Human: " << game_state_.get_captures(1) 
                  << ", Engine: " << game_state_.get_captures(-1) << "\n";
        
        if (game_state_.is_terminal()) {
            int winner = game_state_.get_winner();
            if (winner == 1) {
                std::cout << "*** HUMAN WINS! ***\n";
            } else if (winner == -1) {
                std::cout << "*** ENGINE WINS! ***\n";
            } else {
                std::cout << "*** DRAW ***\n";
            }
        }
        std::cout << "\n";
    }
    
    void display_help() const {
        std::cout << "\nCommands:\n";
        std::cout << "  <move>  - Make a move (e.g., K10, A1, S19)\n";
        std::cout << "  help    - Show this help\n";
        std::cout << "  quit    - Exit game\n";
        std::cout << "  undo    - Undo last move pair (human + engine)\n";
        std::cout << "  config  - Show/change engine settings\n\n";
    }
    
    bool undo_last_move_pair() {
        if (move_pairs_.empty()) {
            std::cout << "No moves to undo!\n";
            return false;
        }
        
        // Get the last move pair
        auto last_pair = move_pairs_.back();
        move_pairs_.pop_back();
        
        // Undo the moves (engine move first, then human move)
        game_state_.undo_move(); // Engine move
        game_state_.undo_move(); // Human move
        
        std::string human_move = position_to_move_string(last_pair.first);
        std::string engine_move = position_to_move_string(last_pair.second);
        
        std::cout << "Undid: Human " << human_move << " and Engine " << engine_move << "\n";
        
        // Reset engine tree (since tree state is now different)
        engine_ = std::make_unique<mcts::MCTSEngine>(game_state_, *move_generator_);
        
        return true;
    }
    
    void run() {
        std::cout << "=== Welcome to AlphaPente ===\n";
        std::cout << "You are X, Engine is O. You go first!\n";
        display_help();
        
        std::string input;
        core::Position current_human_move{-1, -1}; // Track human move for pairing
        
        while (!game_state_.is_terminal()) {
            display_game_info();
            display_board();
            
            if (game_state_.current_player() == 1) {
                // Human turn
                std::cout << "Your move (or 'help'): ";
                std::getline(std::cin, input);
                
                // Trim whitespace
                input.erase(0, input.find_first_not_of(" \t"));
                input.erase(input.find_last_not_of(" \t") + 1);
                
                if (input.empty()) continue;
                
                if (input == "help") {
                    display_help();
                    continue;
                } else if (input == "quit") {
                    std::cout << "Thanks for playing!\n";
                    return;
                } else if (input == "undo") {
                    undo_last_move_pair();
                    continue;
                } else if (input == "config") {
                    // TODO: Implement config
                    std::cout << "Engine settings: " << engine_iterations_ 
                              << " iterations, " << engine_time_limit_ << "ms time limit\n";
                    continue;
                }
                
                // Try to parse as move
                core::Position move = parse_move(input);
                if (move.row == -1 || move.col == -1) {
                    std::cout << "Invalid move format. Use format like K10, A1, S19.\n";
                    continue;
                }
                
                if (!game_state_.is_empty(move.row, move.col)) {
                    std::cout << "Position " << input << " is already occupied!\n";
                    continue;
                }
                
                // Make human move
                game_state_.make_move(move.row, move.col);
                std::cout << "You played " << input << "\n";
                
                // Store human move for pairing
                current_human_move = move;
                
                // Update engine with human move (disabled tree reuse for debugging)
                // engine_->update_root(move);
                // Create fresh engine to avoid tree reuse bugs
                engine_ = std::make_unique<mcts::MCTSEngine>(game_state_, *move_generator_);
                
            } else {
                // Engine turn
                std::cout << "Engine thinking...\n";
                
                auto start_time = std::chrono::high_resolution_clock::now();
                engine_->reset_statistics();
                
                core::Position engine_move = engine_->search(engine_iterations_, engine_time_limit_);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto thinking_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();
                
                if (engine_move.row == -1 || engine_move.col == -1) {
                    std::cout << "Engine couldn't find a move!\n";
                    break;
                }
                
                // Make engine move
                game_state_.make_move(engine_move.row, engine_move.col);
                
                // Display engine move with statistics
                std::string move_str = position_to_move_string(engine_move);
                int simulations = engine_->get_total_simulations();
                double sims_per_sec = (simulations * 1000.0) / thinking_time;
                
                std::cout << "Engine played " << move_str 
                          << " (" << thinking_time << "ms, " 
                          << simulations << " simulations, "
                          << std::fixed << std::setprecision(0) << sims_per_sec << " sims/sec)\n";
                
                // Store the human+engine move pair for undo
                if (current_human_move.row != -1) {
                    move_pairs_.emplace_back(current_human_move, engine_move);
                    current_human_move = {-1, -1}; // Reset
                }
                
                // Update engine with its own move for tree reuse (disabled for debugging)
                // engine_->update_root(engine_move);
                // Create fresh engine to avoid tree reuse bugs  
                engine_ = std::make_unique<mcts::MCTSEngine>(game_state_, *move_generator_);
            }
        }
        
        // Final game state
        display_game_info();
        display_board();
        std::cout << "Game Over! Thanks for playing.\n";
    }
};

int main() {
    try {
        PenteUI ui;
        ui.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}