#include "core/bitboard.hpp"

namespace core {

bool BitBoard::check_five_in_a_row(int row, int col, int player) const noexcept {
    // First check if the position actually has the player's stone
    if (get_stone(row, col) != player) {
        return false;
    }
    
    // Direction vectors: horizontal, vertical, diagonal, diagonal
    constexpr int dr[] = {0, 1, 1, 1};
    constexpr int dc[] = {1, 0, 1, -1};
    
    for (int dir = 0; dir < 4; ++dir) {
        int count = 1; // Count the stone at (row, col)
        
        // Check positive direction
        int r = row + dr[dir];
        int c = col + dc[dir];
        while (r >= 0 && r < 19 && c >= 0 && c < 19 && get_stone(r, c) == player) {
            ++count;
            r += dr[dir];
            c += dc[dir];
        }
        
        // Check negative direction
        r = row - dr[dir];
        c = col - dc[dir];
        while (r >= 0 && r < 19 && c >= 0 && c < 19 && get_stone(r, c) == player) {
            ++count;
            r -= dr[dir];
            c -= dc[dir];
        }
        
        if (count >= 5) {
            return true;
        }
    }
    
    return false;
}

} // namespace core