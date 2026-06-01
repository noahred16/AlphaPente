#pragma once
#ifdef WITH_TORCH
#include <torch/torch.h>
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
// Inputs:  planes   [B, 3, 19, 19]  — current player, opponent, empty
//          captures [B, 2]           — (my captures, opp captures) normalized to [0,1]
// Outputs: log_policy [B, 361], value [B, 1]
struct AlphaNetImpl : torch::nn::Module {
    static constexpr int BOARD        = 19;
    static constexpr int kChannels    = 32;
    static constexpr int kResBlocks   = 3;

    torch::nn::Conv2d inputConv{nullptr}, policyConv{nullptr}, valueConv{nullptr};
    torch::nn::BatchNorm2d inputBn{nullptr}, policyBn{nullptr}, valueBn{nullptr};
    torch::nn::ModuleList resBlocks{nullptr};
    torch::nn::Linear policyFc{nullptr}, valueFc1{nullptr}, valueFc2{nullptr};

    AlphaNetImpl(int channels = kChannels, int numResBlocks = kResBlocks);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor planes, torch::Tensor captures);
};
TORCH_MODULE(AlphaNet);

#endif // WITH_TORCH
