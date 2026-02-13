#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "PenteGame.hpp"
#include <vector>
#include <utility>
#include <unordered_map>

// Abstract interface for position evaluation
class Evaluator {
public:
    virtual ~Evaluator() = default;

    // returns policy and value together
    virtual std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame& game) = 0;


    // returns orderd pairs of moves and their probabilities, sorted by probability. filtered to only legal moves.
    virtual std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame& game) = 0;

    // Returns expected value from current player's perspective
    // Returns: value in range [-1.0, 1.0] where 1.0 = current player winning
    virtual float evaluateValue(const PenteGame& game) = 0;

    void setMaxRolloutDepth(int depth) { maxRolloutDepth_ = depth; }

protected:
    int maxRolloutDepth_ = 200;

    // Random rollout from current position, returns value in same convention as evaluateValue
    float rollout(const PenteGame& game);
};

// Baseline - uniform policy, rollout for value
class UniformEvaluator : public Evaluator {
public:
    UniformEvaluator() = default;
    ~UniformEvaluator() override = default;
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame& game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame& game) override;
    float evaluateValue(const PenteGame& game) override;
};

// Heuristic Evaluator
class HeuristicEvaluator : public Evaluator {
public:
    HeuristicEvaluator() = default;
    ~HeuristicEvaluator() override = default;
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame& game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame& game) override;
    float evaluateValue(const PenteGame& game) override;
};




#endif // EVALUATOR_HPP