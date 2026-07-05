#ifdef WITH_TORCH
#include "Evaluator.hpp"
#include "NNModel.hpp"
#include <algorithm>
#include <stdexcept>
#include <torch/torch.h>

struct NNEvaluator::Impl {
    AlphaNet model{nullptr};
    torch::Device device{torch::cuda::is_available() ? torch::kCUDA : torch::kCPU};
    // fp32 throughout. fp16 (kHalf) was tested but hurt throughput on this model
    // (64 channels, 6 res blocks) — too small to saturate Tensor Cores, and pure
    // fp16 destabilizes BatchNorm. Worth revisiting at 128+ channels, ideally with
    // PyTorch AMP rather than a blunt model->to(kHalf).
    // Future: torch_tensorrt can compile a fixed-batch-size engine for 2-4x gains
    // by fusing ops and eliminating PyTorch dispatch overhead.
    torch::ScalarType dtype{torch::kFloat};

    Impl() {
        model = AlphaNet(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
        model->eval();
        model->to(device);
    }

    explicit Impl(const std::string &path) {
        model = AlphaNet(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
        torch::load(model, path);
        model->eval();
        model->to(device);
    }
};

NNEvaluator::NNEvaluator()
    : impl_(std::make_unique<Impl>()) {}

NNEvaluator::NNEvaluator(const std::string &modelPath)
    : impl_(std::make_unique<Impl>(modelPath)) {}

NNEvaluator::~NNEvaluator() = default;

// Build input tensors from game state, from current player's perspective.
std::pair<torch::Tensor, torch::Tensor> NNEvaluator::gameToTensors(const PenteGame &game) {
    constexpr int B = PenteGame::BOARD_SIZE;
    auto planes = torch::zeros({AlphaNetImpl::kInputPlanes, B, B});
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

    float myCapNorm  = myCaptures  / maxCap;
    float oppCapNorm = oppCaptures / maxCap;
    for (int y = 0; y < B; y++)
        for (int x = 0; x < B; x++) {
            acc[3][y][x] = myCapNorm;
            acc[4][y][x] = oppCapNorm;
        }

    auto captures = torch::zeros({2});
    captures[0] = myCapNorm;
    captures[1] = oppCapNorm;

    return {planes, captures};
}

std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>
NNEvaluator::evaluate(const PenteGame &game) {
    auto [planes, captures] = gameToTensors(game);

    torch::NoGradGuard no_grad;
    auto [logPolicy, valueTensor] = impl_->model->forward(
        planes.unsqueeze(0).to(impl_->device, impl_->dtype),
        captures.unsqueeze(0).to(impl_->device, impl_->dtype));

    auto probs = torch::exp(logPolicy).squeeze(0).to(torch::kFloat).cpu(); // [361]
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

    // NN is trained on previous-player (mover) perspective — matches MCTS backprop directly.
    return {policy, valueTensor.item<float>()};
}

std::vector<std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>>
NNEvaluator::evaluateBatch(const std::vector<PenteGame> &games) {
    if (games.empty()) return {};

    constexpr int B = PenteGame::BOARD_SIZE;
    int N = (int)games.size();

    std::vector<torch::Tensor> planeVec, captureVec;
    planeVec.reserve(N);
    captureVec.reserve(N);
    for (const auto &g : games) {
        auto [planes, captures] = gameToTensors(g);
        planeVec.push_back(std::move(planes));
        captureVec.push_back(std::move(captures));
    }

    auto batchPlanes   = torch::stack(planeVec,   0).to(impl_->device, impl_->dtype);  // [N, 5, 19, 19]
    auto batchCaptures = torch::stack(captureVec, 0).to(impl_->device, impl_->dtype);  // [N, 2]

    torch::NoGradGuard no_grad;
    auto [logPolicy, valueTensor] = impl_->model->forward(batchPlanes, batchCaptures);

    auto probs  = torch::exp(logPolicy).to(torch::kFloat).cpu();  // [N, 361]
    auto values = valueTensor.to(torch::kFloat).cpu();             // [N, 1]

    std::vector<std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>> results;
    results.reserve(N);

    for (int i = 0; i < N; i++) {
        auto row = probs[i];
        auto probsAcc = row.accessor<float, 1>();
        const auto &legalMoves = games[i].getLegalMoves();

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
            for (auto &[m, p] : policy) p = 1.0f / (float)legalMoves.size();

        std::sort(policy.begin(), policy.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        results.emplace_back(std::move(policy), values[i][0].item<float>());
    }

    return results;
}

std::vector<std::pair<PenteGame::Move, float>>
NNEvaluator::evaluatePolicy(const PenteGame &game) {
    return evaluate(game).first;
}

float NNEvaluator::evaluateValue(const PenteGame &game) {
    return evaluate(game).second;
}

#endif // WITH_TORCH
