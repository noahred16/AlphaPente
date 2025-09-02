#pragma once

#include "game_state.hpp"
#include <vector>

namespace core {

struct Move {
    int x, y;
    Move(int x, int y) : x(x), y(y) {}
};

class MoveGenerator {
public:
    MoveGenerator();
    ~MoveGenerator() = default;
    
    std::vector<Move> generateMoves(const GameState& state);

private:
};

} // namespace core