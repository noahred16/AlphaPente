#pragma once

#include <array>
#include <cstdint>

namespace core {

class BitBoard {
private:
    std::array<uint64_t, 6> player1_bits_ = {0};
    std::array<uint64_t, 6> player2_bits_ = {0};
    
    inline std::pair<int, int> pos_to_bit_index(int row, int col) const noexcept {
        int bit_index = (row << 4) + (row << 1) + (row) + col; // row * 19 + col
        return {bit_index >> 6, bit_index & 63}; // / 64, % 64
    }
    
public:
    BitBoard() = default;
    ~BitBoard() = default;
    
    inline void set_stone(int row, int col, int player) noexcept {
        auto [array_idx, bit_pos] = pos_to_bit_index(row, col);
        uint64_t mask = 1ULL << bit_pos;
        
        if (player == 1) {
            player1_bits_[array_idx] |= mask;
            player2_bits_[array_idx] &= ~mask;
        } else if (player == -1) {
            player2_bits_[array_idx] |= mask;
            player1_bits_[array_idx] &= ~mask;
        }
    }
    
    inline void remove_stone(int row, int col) noexcept {
        auto [array_idx, bit_pos] = pos_to_bit_index(row, col);
        uint64_t mask = ~(1ULL << bit_pos);
        
        player1_bits_[array_idx] &= mask;
        player2_bits_[array_idx] &= mask;
    }
    
    inline int get_stone(int row, int col) const noexcept {
        auto [array_idx, bit_pos] = pos_to_bit_index(row, col);
        uint64_t mask = 1ULL << bit_pos;
        
        if (player1_bits_[array_idx] & mask) {
            return 1;
        } else if (player2_bits_[array_idx] & mask) {
            return -1;
        }
        return 0;
    }
    
    inline bool is_empty(int row, int col) const noexcept {
        return get_stone(row, col) == 0;
    }
    
    bool check_five_in_a_row(int row, int col, int player) const noexcept;
    
    // For testing - get raw bit arrays
    // Might not need, can just use the get_stone method
    // const std::array<uint64_t, 6>& get_player1_bits() const noexcept { return player1_bits_; }
    // const std::array<uint64_t, 6>& get_player2_bits() const noexcept { return player2_bits_; }
};

} // namespace core