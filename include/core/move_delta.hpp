#pragma once

#include <array>
#include <cstdint>

namespace core {

struct Position {
    int8_t row, col;
    
    Position() = default;
    Position(int8_t r, int8_t c) : row(r), col(c) {}
    
    bool operator==(const Position& other) const noexcept {
        return row == other.row && col == other.col;
    }
    
    bool operator!=(const Position& other) const noexcept {
        return !(*this == other);
    }
};

struct MoveDelta {
    Position move_pos;
    std::array<Position, 8> captured_stones;
    uint8_t capture_count = 0;
    std::array<int, 2> captures_before; // Both players' counts BEFORE this move
    
    MoveDelta() = default;
    MoveDelta(Position pos) : move_pos(pos) {}
    
    void add_capture(Position pos) {
        if (capture_count < 8) {
            captured_stones[capture_count++] = pos;
        }
    }
};

} // namespace core