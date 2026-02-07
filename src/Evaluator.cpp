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

std::array<std::array<float, PenteGame::BOARD_SIZE>, PenteGame::BOARD_SIZE> UniformEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("UniformEvaluator::evaluatePolicy");
    // get moves from legalMoves
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();
    
    if (legalMoves.empty()) {
        return {};
    }
    
    float uniformProb = 1.0f / legalMoves.size();
    
    std::array<std::array<float, PenteGame::BOARD_SIZE>, PenteGame::BOARD_SIZE> result{};
    for (const auto& move : legalMoves) {
        result[move.x][move.y] = uniformProb;
    }
    return result;
}

float UniformEvaluator::evaluateValue(const PenteGame& game) {
    return 0.0f; // No position knowledge
}

// ============================================================================
// HeuristicEvaluator Implementation
// ============================================================================

std::array<std::array<float, PenteGame::BOARD_SIZE>, PenteGame::BOARD_SIZE> HeuristicEvaluator::evaluatePolicy(const PenteGame& game) {
    PROFILE_SCOPE("HeurEval::evaluatePolicy");
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

    if (legalMoves.empty()) {
        return {};
    }

    std::array<std::array<float, PenteGame::BOARD_SIZE>, PenteGame::BOARD_SIZE> scores;
    for (auto& row : scores) {
        row.fill(-1.0f);
    }
    // scores.reserve(legalMoves.size());

    float totalScore = 0.0f;
    for (const auto& move : legalMoves) {
        float score = game.evaluateMove(move);
        scores[move.x][move.y] = score;
        totalScore += score;
        // TEMP
        // std::cout << "Moves: " 
        //           << GameUtils::displayMove(move.x, move.y)
        //           << " Score: " << score << " Policy: " << (score / 110.0f) << "\n";
    }
    // std::cout << "Total Score: " << totalScore << "\n";
    // exit(1);

    // Normalize to probabilities
    for (auto& row : scores) {
        for (float& score : row) {
            if (score < 0.0f) continue;
            score /= totalScore;
        }
    }

    return scores;
}

float HeuristicEvaluator::evaluateValue(const PenteGame& game) {
    return game.evaluatePosition() * -1; // invert for opponent's perspective
}


