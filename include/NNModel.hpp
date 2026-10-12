#pragma once
#ifdef WITH_TORCH
#include <torch/torch.h>
#include <stdexcept>
#include <string>
#include <utility>

// Residual block: Conv->BN->ReLU->Conv->BN + skip->ReLU
struct ResBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    explicit ResBlockImpl(int channels);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResBlock);

// AlphaZero-style dual-head ResNet.
// Inputs:  planes   [B, 5, 19, 19]  — current player, opponent, empty,
//                                      my_captures/max (const plane), opp_captures/max (const plane)
//          captures [B, 2]           — same capture scalars fed directly into value head
// Outputs: log_policy [B, 361], value [B, 1]
struct AlphaNetImpl : torch::nn::Module {
    static constexpr int BOARD        = 19;
    static constexpr int kChannels    = 128;
    static constexpr int kResBlocks   = 8;
    static constexpr int kInputPlanes = 5;

    torch::nn::Conv2d inputConv{nullptr}, policyConv{nullptr}, valueConv{nullptr};
    torch::nn::BatchNorm2d inputBn{nullptr}, policyBn{nullptr}, valueBn{nullptr};
    torch::nn::ModuleList resBlocks{nullptr};
    torch::nn::Linear policyFc{nullptr}, valueFc1{nullptr}, valueFc2{nullptr}, valueFc3{nullptr};

    AlphaNetImpl(int channels = kChannels, int numResBlocks = kResBlocks);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor planes, torch::Tensor captures);
};
TORCH_MODULE(AlphaNet);

// torch::load(model, path) does NOT error on a channels mismatch — it silently
// resizes model's tensors to match whatever is in the archive, so the loaded
// model silently ends up at the *file's* channel count, not the one it was
// constructed with. (A res_blocks/topology mismatch does throw on its own,
// since that changes which named submodules exist.) Call this right after
// torch::load to catch the channels case explicitly instead of silently
// training/evaluating at the wrong width.
inline void checkArchitectureMatches(const AlphaNet &model, int expectedChannels,
                                      const std::string &path) {
    int actualChannels = (int)model->inputConv->weight.size(0);
    if (actualChannels != expectedChannels) {
        throw std::runtime_error(
            "Architecture mismatch loading " + path + ": requested " +
            std::to_string(expectedChannels) + " channels, but torch::load silently resized to " +
            std::to_string(actualChannels) + " channels (the file's actual shape).");
    }
}

#endif // WITH_TORCH
