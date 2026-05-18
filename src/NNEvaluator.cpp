#ifdef WITH_TORCH
#include "Evaluator.hpp"
#include "NNModel.hpp"
#include <algorithm>
#include <stdexcept>
#include <torch/torch.h>

struct NNEvaluator::Impl {
    AlphaNet model{nullptr};
    torch::Device device{torch::kCPU};

    explicit Impl(const std::string &path) {
        model = AlphaNet(64, 5);
        torch::load(model, path);
        model->eval();
        model->to(device);
    }
};

NNEvaluator::NNEvaluator(const std::string &modelPath)
    : impl_(std::make_unique<Impl>(modelPath)) {}

NNEvaluator::~NNEvaluator() = default;

// Build input tensors from game state, from current player's perspective.
static std::pair<torch::Tensor, torch::Tensor> gameToTensors(const PenteGame &game) {
    constexpr int B = PenteGame::BOARD_SIZE;
    auto planes = torch::zeros({3, B, B});
    auto acc = planes.accessor<float, 3>();

    bool blackToMove = (game.getCurrentPlayer() == PenteGame::BLACK);
    const BitBoard &myBB  = blackToMove ? game.getBlackBitBoard() : game.getWhiteBitBoard();
    const BitBoard &oppBB = blackToMove ? game.getWhiteBitBoard() : game.getBlackBitBoard();

    myBB.forEachSetBit([&](int cell) { acc[0][cell / B][cell % B] = 1.0f; });
    oppBB.forEachSetBit([&](int cell) { acc[1][cell / B][cell % B] = 1.0f; });
    for (int y = 0; y < B; y++)
        for (int x = 0; x < B; x++)
            if (!myBB.getBitUnchecked(x, y) && !oppBB.getBitUnchecked(x, y))
                acc[2][y][x] = 1.0f;

    int myCaptures  = blackToMove ? game.getBlackCaptures() : game.getWhiteCaptures();
    int oppCaptures = blackToMove ? game.getWhiteCaptures() : game.getBlackCaptures();
    float maxCap    = static_cast<float>(game.getConfig().capturesToWin);

    auto captures = torch::zeros({2});
    captures[0] = myCaptures  / maxCap;
    captures[1] = oppCaptures / maxCap;

    return {planes, captures};
}

std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>
NNEvaluator::evaluate(const PenteGame &game) {
    auto [planes, captures] = gameToTensors(game);

    torch::NoGradGuard no_grad;
    auto [logPolicy, valueTensor] = impl_->model->forward(
        planes.unsqueeze(0).to(impl_->device),
        captures.unsqueeze(0).to(impl_->device));

    auto probs = torch::exp(logPolicy).squeeze(0).cpu(); // [361]
    auto probsAcc = probs.accessor<float, 1>();

    constexpr int B = PenteGame::BOARD_SIZE;
    const auto &legalMoves = game.getLegalMoves();
    std::vector<std::pair<PenteGame::Move, float>> policy;
    policy.reserve(legalMoves.size());

    float total = 0.0f;
    for (const auto &move : legalMoves) {
        float p = probsAcc[move.y * B + move.x];
        policy.emplace_back(move, p);
        total += p;
    }
    if (total > 0.0f)
        for (auto &[m, p] : policy) p /= total;
    else
        for (auto &[m, p] : policy) p = 1.0f / static_cast<float>(legalMoves.size());

    std::sort(policy.begin(), policy.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    return {policy, valueTensor.item<float>()};
}

std::vector<std::pair<PenteGame::Move, float>>
NNEvaluator::evaluatePolicy(const PenteGame &game) {
    return evaluate(game).first;
}

float NNEvaluator::evaluateValue(const PenteGame &game) {
    return evaluate(game).second;
}

#endif // WITH_TORCH
