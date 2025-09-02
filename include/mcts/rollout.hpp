#pragma once

#include "core/game_state.hpp"

namespace mcts {

class Rollout {
public:
    Rollout();
    ~Rollout() = default;
    
    double simulate(const core::GameState& state);

private:
    double randomPlayout(core::GameState state);
};

} // namespace mcts