#ifdef WITH_TORCH
#include "NNModel.hpp"

ResBlockImpl::ResBlockImpl(int ch) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(1).bias(false)));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(ch));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(ch));
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    auto skip = x;
    x = torch::relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    return torch::relu(x + skip);
}

AlphaNetImpl::AlphaNetImpl(int ch, int numBlocks) {
    inputConv = register_module("inputConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(kInputPlanes, ch, 3).padding(1).bias(false)));
    inputBn = register_module("inputBn", torch::nn::BatchNorm2d(ch));

    resBlocks = register_module("resBlocks", torch::nn::ModuleList());
    for (int i = 0; i < numBlocks; i++)
        resBlocks->push_back(ResBlock(ch));

    policyConv = register_module("policyConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, 4, 1).bias(false)));
    policyBn = register_module("policyBn", torch::nn::BatchNorm2d(4));
    policyFc  = register_module("policyFc",
        torch::nn::Linear(4 * BOARD * BOARD, BOARD * BOARD));

    valueConv = register_module("valueConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, 1, 1).bias(false)));
    valueBn  = register_module("valueBn",  torch::nn::BatchNorm2d(1));
    valueFc1 = register_module("valueFc1",
        torch::nn::Linear(BOARD * BOARD + 2, 256)); // +2 for capture scalars
    valueFc2 = register_module("valueFc2", torch::nn::Linear(256, 64));
    valueFc3 = register_module("valueFc3", torch::nn::Linear(64, 1));
}

std::pair<torch::Tensor, torch::Tensor> AlphaNetImpl::forward(
    torch::Tensor planes, torch::Tensor captures) {
    // Shared trunk
    auto x = torch::relu(inputBn(inputConv(planes)));
    for (const auto& block : *resBlocks)
        x = block->as<ResBlockImpl>()->forward(x);

    // Policy head: [B, 4, 19, 19] -> [B, 361] log-probabilities
    auto p = torch::relu(policyBn(policyConv(x)));
    p = torch::log_softmax(policyFc(p.flatten(1)), /*dim=*/1);

    // Value head: [B, 1, 19, 19] + captures [B, 2] -> [B, 1]
    auto v = torch::relu(valueBn(valueConv(x)));
    v = torch::cat({v.flatten(1), captures}, /*dim=*/1);
    v = torch::relu(valueFc1(v));
    v = torch::relu(valueFc2(v));
    v = torch::tanh(valueFc3(v));

    return {p, v};
}

#endif // WITH_TORCH
