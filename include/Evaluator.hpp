#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "PenteGame.hpp"
#include <vector>
#include <utility>

// Abstract interface for position evaluation
class Evaluator {
public:
    virtual ~Evaluator() = default;
    
    // Returns prior probabilities for all legal moves
    // Returns: vector of (move, probability) pairs that sum to 1.0
    virtual std::vector<float> evaluatePolicy(const PenteGame& game) = 0;
    
    // Returns expected value from current player's perspective
    // Returns: value in range [-1.0, 1.0] where 1.0 = current player winning
    virtual float evaluateValue(const PenteGame& game) = 0;
};

// Baseline 
class UniformEvaluator : public Evaluator {
public:
    UniformEvaluator() = default;
    ~UniformEvaluator() override = default;
    std::vector<float> evaluatePolicy(const PenteGame& game) override;
    float evaluateValue(const PenteGame& game) override;
};

// Heuristic Evaluator
class HeuristicEvaluator : public Evaluator {
public:
    HeuristicEvaluator() = default;
    ~HeuristicEvaluator() override = default;
    std::vector<float> evaluatePolicy(const PenteGame& game) override;
    float evaluateValue(const PenteGame& game) override;
};




#endif // EVALUATOR_HPP