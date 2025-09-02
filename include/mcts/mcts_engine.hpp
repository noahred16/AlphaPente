#pragma once

#include "mcts_node.hpp"
#include "core/game_state.hpp"
#include "core/move_generator.hpp"

namespace mcts {

class MCTSEngine {
public:
    MCTSEngine(int iterations = 10000);
    ~MCTSEngine() = default;
    
    core::Move getBestMove(const core::GameState& state);
    void setIterations(int iterations);

private:
    int iterations_;
    core::MoveGenerator move_generator_;
    
    MCTSNode* runSimulation(MCTSNode* root);
};

} // namespace mcts