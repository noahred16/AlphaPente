#pragma once

#include "core/game_state.hpp"
#include <vector>
#include <memory>

namespace mcts {

class MCTSNode {
public:
    MCTSNode(const core::GameState& state, MCTSNode* parent = nullptr);
    ~MCTSNode() = default;

    double getWinRate() const;
    int getVisits() const;
    bool isFullyExpanded() const;
    
    MCTSNode* select();
    MCTSNode* expand();
    double simulate();
    void backpropagate(double result);

private:
    core::GameState state_;
    MCTSNode* parent_;
    std::vector<std::unique_ptr<MCTSNode>> children_;
    
    int visits_;
    double wins_;
    bool fully_expanded_;
};

} // namespace mcts