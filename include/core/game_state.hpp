#pragma once

#include "bitboard.hpp"
#include "move_delta.hpp"
#include <vector>

namespace core {

class GameState {
private:
    BitBoard board_;
    std::vector<MoveDelta> move_history_;
    bool is_player1_turn_ = true;
    std::array<int, 2> captures_ = {0, 0};
    
    void detect_and_execute_captures(int row, int col, int player, MoveDelta& delta) noexcept;
    
public:
    GameState() = default;
    ~GameState() = default;
    
    // Prevent copying for performance
    GameState(const GameState&) = delete;
    GameState& operator=(const GameState&) = delete;
    
    // Board access
    inline int get_stone(int row, int col) const noexcept {
        return board_.get_stone(row, col);
    }
    
    inline bool is_empty(int row, int col) const noexcept {
        return board_.is_empty(row, col);
    }
    
    // Player management
    inline int current_player() const noexcept {
        return is_player1_turn_ ? 1 : -1;
    }
    
    inline void switch_player() noexcept {
        is_player1_turn_ = !is_player1_turn_;
    }
    
    // Move system
    MoveDelta make_move(int row, int col) noexcept;
    void undo_move() noexcept;
    
    // Game state queries
    inline int get_captures(int player) const noexcept {
        return player == 1 ? captures_[0] : captures_[1];
    }
    
    // Win conditions
    bool is_terminal() const noexcept;
    int get_winner() const noexcept;
    bool check_win(int row, int col, int player) const noexcept;
    
    // Reset to initial state
    void reset() noexcept;
};

} // namespace core