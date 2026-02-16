#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

// ============================================================================
// Base Evaluator - rollout helper
// ============================================================================
float Evaluator::rollout(const PenteGame &game) {
    PROFILE_SCOPE("Evaluator::rollout");
    PenteGame simGame = game;
    PenteGame::Player startPlayer = simGame.getCurrentPlayer();
    PenteGame::Player winner = PenteGame::NONE;
    int depth = 0;

    while ((winner = simGame.getWinner()) == PenteGame::NONE && depth < maxRolloutDepth_) {
        PenteGame::Move move = simGame.getRandomLegalMove();
        simGame.makeMove(move.x, move.y);
        depth++;
    }

    if (winner != PenteGame::NONE) {
        return (winner == startPlayer) ? -1.0f : 1.0f;
    }
    return 0.0f;
}

// ============================================================================
// UniformEvaluator Implementation
// ============================================================================
std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> UniformEvaluator::evaluate(const PenteGame &game) {
    auto policy = evaluatePolicy(game);
    float value = evaluateValue(game);
    return {policy, value};
}

std::vector<std::pair<PenteGame::Move, float>> UniformEvaluator::evaluatePolicy(const PenteGame &game) {
    PROFILE_SCOPE("UniformEvaluator::evaluatePolicy");
    // get moves from legalMoves
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

    if (legalMoves.empty()) {
        return {};
    }

    float uniformProb = 1.0f / legalMoves.size();

    std::vector<std::pair<PenteGame::Move, float>> result;
    result.reserve(legalMoves.size());
    for (const auto &move : legalMoves) {
        result.emplace_back(move, uniformProb);
    }
    return result;
}

float UniformEvaluator::evaluateValue(const PenteGame &game) { return rollout(game); }

// ============================================================================
// HeuristicEvaluator Implementation
// ============================================================================

std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> HeuristicEvaluator::evaluate(const PenteGame &game) {

    // auto policy = evaluatePolicy(game);
    // skip calculating policy for now, do lazy loading in selection to save on
    // heuristic evals
    std::vector<std::pair<PenteGame::Move, float>> policy = {};
    float value = evaluateValue(game);
    return {policy, value};
}

std::vector<std::pair<PenteGame::Move, float>> HeuristicEvaluator::evaluatePolicy(const PenteGame &game) {
    PROFILE_SCOPE("HeurEval::evaluatePolicy");
    std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

    if (legalMoves.empty()) {
        return {};
    }

    std::vector<std::pair<PenteGame::Move, float>> policyScores;
    policyScores.reserve(legalMoves.size());

    float totalScore = 0.0f;
    for (const auto &move : legalMoves) {
        float score = game.evaluateMove(move);
        policyScores.emplace_back(move, score);
        totalScore += score;
    }

    // Normalize to probabilities
    for (auto &[move, score] : policyScores) {
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

float HeuristicEvaluator::evaluateValue(const PenteGame &game) {
    float value = game.evaluatePosition() * -1; // invert for opponent's perspective
    if (value != 0.0f) {
        return value;
    }
    return rollout(game);
}
