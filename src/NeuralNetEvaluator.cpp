#ifdef WITH_TORCH

#include "NeuralNetEvaluator.hpp"
#include <algorithm>

NeuralNetEvaluator::NeuralNetEvaluator(const PenteNetConfig &cfg)
    : model_(cfg) {
    model_->eval();
}

void NeuralNetEvaluator::loadWeights(const std::string &path) {
    model_->loadFromFile(path);
    model_->eval();
}

void NeuralNetEvaluator::saveWeights(const std::string &path) const {
    model_->saveToFile(path);
}

std::pair<std::vector<float>, float> NeuralNetEvaluator::forward(const PenteGame &game) {
    float rawInput[NNInputEncoder::TOTAL_SIZE];
    NNInputEncoder::encode(game, rawInput);

    auto inputTensor = torch::from_blob(rawInput,
        {1, NNInputEncoder::NUM_PLANES, NNInputEncoder::BOARD_SIZE, NNInputEncoder::BOARD_SIZE})
        .clone();

    torch::NoGradGuard noGrad;
    auto [logits, valueT] = model_->forward(inputTensor);

    // Softmax over all 361 positions
    auto probs = torch::softmax(logits.squeeze(0), 0); // [361]
    std::vector<float> policy(probs.data_ptr<float>(),
                              probs.data_ptr<float>() + NNInputEncoder::PLANE_SIZE);

    float value = valueT.item<float>();
    return {policy, value};
}

std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>
NeuralNetEvaluator::evaluate(const PenteGame &game) {
    auto [rawPolicy, value] = forward(game);

    const auto &legalMoves = game.getLegalMoves();

    // Filter to legal moves and renormalize
    std::vector<std::pair<PenteGame::Move, float>> policy;
    policy.reserve(legalMoves.size());

    float total = 0.0f;
    for (const auto &move : legalMoves) {
        float prob = rawPolicy[move.y * NNInputEncoder::BOARD_SIZE + move.x];
        policy.emplace_back(move, prob);
        total += prob;
    }

    if (total > 1e-8f) {
        for (auto &[m, p] : policy) p /= total;
    } else {
        // Network assigned near-zero mass to all legal moves — fall back to uniform
        float uniform = 1.0f / float(policy.size());
        for (auto &[m, p] : policy) p = uniform;
    }

    std::sort(policy.begin(), policy.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    return {policy, value};
}

std::vector<std::pair<PenteGame::Move, float>>
NeuralNetEvaluator::evaluatePolicy(const PenteGame &game) {
    return evaluate(game).first;
}

float NeuralNetEvaluator::evaluateValue(const PenteGame &game) {
    return evaluate(game).second;
}

#endif // WITH_TORCH
