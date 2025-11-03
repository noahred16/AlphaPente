#include "core/game_state.hpp"
#include <algorithm>

namespace core {

MoveDelta GameState::make_move(int row, int col) noexcept {
    MoveDelta delta;
    delta.move_pos = Position{static_cast<int8_t>(row), static_cast<int8_t>(col)};
    delta.capture_count = 0;
    delta.captures_before = captures_;
    
    int player = current_player();
    
    // Place the stone
    board_.set_stone(row, col, player);
    stone_positions_.emplace_back(static_cast<int8_t>(row), static_cast<int8_t>(col));
    
    // Detect and execute captures
    detect_and_execute_captures(row, col, player, delta);
    
    // Add to move history
    move_history_.push_back(delta);
    
    // Switch player
    switch_player();
    
    return delta;
}

void GameState::undo_move() noexcept {
    if (move_history_.empty()) return;
    
    MoveDelta last_move = move_history_.back();
    move_history_.pop_back();
    
    // Switch player back
    switch_player();
    
    // Remove the placed stone
    board_.remove_stone(last_move.move_pos.row, last_move.move_pos.col);
    stone_positions_.pop_back();  // Placed stone is always the last one added
    
    // Restore captured stones
    for (uint8_t i = 0; i < last_move.capture_count; ++i) {
        Position captured = last_move.captured_stones[i];
        board_.set_stone(captured.row, captured.col, -current_player());
        stone_positions_.push_back(captured);  // Add back to position vector
    }
    
    // Restore capture counts
    captures_ = last_move.captures_before;
}

bool GameState::check_win(int row, int col, int player) const noexcept {
    // Check five in a row
    if (board_.check_five_in_a_row(row, col, player)) {
        return true;
    }
    
    // Check capture win (10 or more captures)
    return get_captures(player) >= 10;
}

bool GameState::is_terminal() const noexcept {
    if (move_history_.empty()) return false;
    
    MoveDelta last_move = move_history_.back();
    int last_player = -current_player(); // Previous player (before switch)
    
    return check_win(last_move.move_pos.row, last_move.move_pos.col, last_player);
}

int GameState::get_winner() const noexcept {
    if (!is_terminal()) return 0;
    
    MoveDelta last_move = move_history_.back();
    return -current_player(); // Previous player won
}

void GameState::detect_and_execute_captures(int row, int col, int player, MoveDelta& delta) noexcept {
    constexpr int dr[] = {0, 1, 1, 1};
    constexpr int dc[] = {1, 0, 1, -1};
    
    for (int dir = 0; dir < 4; ++dir) {
        // Check both directions from the new move
        for (int direction_sign = -1; direction_sign <= 1; direction_sign += 2) {
            // Check for capture pattern: new_move - opponent - opponent - existing_player_stone
            int r1 = row + direction_sign * dr[dir];
            int c1 = col + direction_sign * dc[dir];
            int r2 = row + direction_sign * 2 * dr[dir];
            int c2 = col + direction_sign * 2 * dc[dir];
            int r3 = row + direction_sign * 3 * dr[dir];
            int c3 = col + direction_sign * 3 * dc[dir];
            
            // Check bounds
            if (r3 < 0 || r3 >= 19 || c3 < 0 || c3 >= 19) continue;
            
            // Check capture pattern: new_move - opponent - opponent - existing_player_stone
            if (board_.get_stone(r1, c1) == -player &&
                board_.get_stone(r2, c2) == -player &&
                board_.get_stone(r3, c3) == player) {
                
                // Execute capture
                board_.remove_stone(r1, c1);
                board_.remove_stone(r2, c2);
                
                // Remove captured stones from position vector
                Position captured1{static_cast<int8_t>(r1), static_cast<int8_t>(c1)};
                Position captured2{static_cast<int8_t>(r2), static_cast<int8_t>(c2)};
                
                auto it1 = std::find(stone_positions_.begin(), stone_positions_.end(), captured1);
                if (it1 != stone_positions_.end()) {
                    stone_positions_.erase(it1);
                }
                auto it2 = std::find(stone_positions_.begin(), stone_positions_.end(), captured2);
                if (it2 != stone_positions_.end()) {
                    stone_positions_.erase(it2);
                }
                
                // Record captured positions for undo
                delta.captured_stones[delta.capture_count++] = captured1;
                delta.captured_stones[delta.capture_count++] = captured2;
                
                // Update capture count
                if (player == 1) {
                    captures_[0] += 2;
                } else {
                    captures_[1] += 2;
                }
            }
        }
    }
}

void GameState::reset() noexcept {
    board_ = BitBoard{};
    move_history_.clear();
    is_player1_turn_ = true;
    captures_ = {0, 0};
    stone_positions_.clear();
}

} // namespace core