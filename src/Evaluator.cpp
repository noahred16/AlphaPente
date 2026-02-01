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

std::vector<float> UniformEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("UniformEvaluator::evaluatePolicy");
    // get moves from legalMoves
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();
    
    if (legalMoves.empty()) {
        return {};
    }
    
    float uniformProb = 1.0f / legalMoves.size();
    
    std::vector<float> result;
    result.reserve(legalMoves.size());
    result.assign(legalMoves.size(), uniformProb);
    return result;
}

float UniformEvaluator::evaluateValue(const PenteGame& game) {
    return 0.0f; // No position knowledge
}

// ============================================================================
// HeuristicEvaluator Implementation
// ============================================================================

std::vector<float> HeuristicEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("HeuristicEvaluator::evaluatePolicy");
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

    if (legalMoves.empty()) {
        return {};
    }

    std::vector<float> scores;
    scores.reserve(legalMoves.size());

    float totalScore = 0.0f;
    for (const auto& move : legalMoves) {
        float score = game.evaluateMove(move);
        scores.push_back(score);
        totalScore += score;
        // TEMP
        // std::cout << "Moves: " 
        //           << GameUtils::displayMove(move.x, move.y)
        //           << " Score: " << score << "\n";
    }
    // exit(1);

    // Normalize to probabilities
    for (float& score : scores) {
        score /= totalScore;
        // print out score and move
    }

    return scores;
}

float HeuristicEvaluator::evaluateValue(const PenteGame& game) {
    return 0.0f; // No position knowledge
}


