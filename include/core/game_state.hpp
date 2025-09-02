#pragma once

#include "bitboard.hpp"

namespace core {

class GameState {
public:
    GameState();
    ~GameState() = default;

private:
    Bitboard board_;
    int current_player_;
};

} // namespace core