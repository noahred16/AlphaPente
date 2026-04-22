#ifndef NN_MODEL_HPP
#define NN_MODEL_HPP

#ifdef WITH_TORCH

#include "NNInputEncoder.hpp"
#include <torch/torch.h>
#include <string>

// ============================================================================
// Residual Block
// ============================================================================

struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int filters);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};
TORCH_MODULE(ResBlock);

// ============================================================================
// PenteNet — dual-headed ResNet
//
// Input:  [B, NUM_PLANES, 19, 19]  (as produced by NNInputEncoder)
// Output: tuple of
//   policy_logits  [B, 361]   — raw logits, apply softmax + legal-move mask outside
//   value          [B, 1]     — tanh output in (-1, 1)
// ============================================================================

struct PenteNetConfig {
    int numResBlocks = 10;
    int numFilters   = 128;
};

struct PenteNetImpl : torch::nn::Module {
    explicit PenteNetImpl(const PenteNetConfig &cfg = {});

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    void saveToFile(const std::string &path) const;
    void loadFromFile(const std::string &path);

    // Body
    torch::nn::Conv2d      inputConv{nullptr};
    torch::nn::BatchNorm2d inputBn{nullptr};
    torch::nn::ModuleList  resBlocks;

    // Policy head
    torch::nn::Conv2d      policyConv{nullptr};
    torch::nn::BatchNorm2d policyBn{nullptr};
    torch::nn::Linear      policyFc{nullptr};

    // Value head
    torch::nn::Conv2d      valueConv{nullptr};
    torch::nn::BatchNorm2d valueBn{nullptr};
    torch::nn::Linear      valueFc1{nullptr}, valueFc2{nullptr};
};
TORCH_MODULE(PenteNet);

#endif // WITH_TORCH
#endif // NN_MODEL_HPP
