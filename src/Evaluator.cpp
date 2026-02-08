#include "Evaluator.hpp"
#include "Profiler.hpp"
#include "GameUtils.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
// UniformEvaluator Implementation
// ============================================================================
std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> UniformEvaluator::evaluate(const PenteGame& game) {
    auto policy = evaluatePolicy(game);
    float value = evaluateValue(game);
    return {policy, value};
}

std::vector<std::pair<PenteGame::Move, float>> UniformEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("UniformEvaluator::evaluatePolicy");
    // get moves from legalMoves
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();
    
    if (legalMoves.empty()) {
        return {};
    }
    
    float uniformProb = 1.0f / legalMoves.size();
    
    std::vector<std::pair<PenteGame::Move, float>> result;
    result.reserve(legalMoves.size());
    for (const auto& move : legalMoves) {
        result.emplace_back(move, uniformProb);
    }
    return result;
}

float UniformEvaluator::evaluateValue(const PenteGame& game) {
    return 0.0f; // No position knowledge
}

// ============================================================================
// HeuristicEvaluator Implementation
// ============================================================================

std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> HeuristicEvaluator::evaluate(const PenteGame& game) {
    
    // auto policy = evaluatePolicy(game);
    // skip calculating policy for now, do lazy loading in selection to save on heuristic evals
    std::vector<std::pair<PenteGame::Move, float>> policy = {};
    float value = evaluateValue(game);
    return {policy, value};
}

std::vector<std::pair<PenteGame::Move, float>> HeuristicEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("HeurEval::evaluatePolicy");
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

    if (legalMoves.empty()) {
        return {};
    }

    std::vector<std::pair<PenteGame::Move, float>> policyScores;
    policyScores.reserve(legalMoves.size());

    float totalScore = 0.0f;
    for (const auto& move : legalMoves) {
        float score = game.evaluateMove(move);
        policyScores.emplace_back(move, score);
        totalScore += score;
    }

    // Normalize to probabilities
    for (auto& [move, score] : policyScores) {
        score /= totalScore;
    }

    // Sort from largest to smallest 
    // (order doesnt matter, puct still loops)
    // std::sort(policyScores.begin(), policyScores.end(),
    //           [](const auto& a, const auto& b) {
    //               return a.second > b.second;
    //           });
    
    return policyScores;
}


float HeuristicEvaluator::evaluateValue(const PenteGame& game) {
    return game.evaluatePosition() * -1; // invert for opponent's perspective
}


