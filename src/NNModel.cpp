#ifdef WITH_TORCH

#include "NNModel.hpp"

// ============================================================================
// ResBlock
// ============================================================================

ResBlockImpl::ResBlockImpl(int filters) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false)));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(filters));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(filters));
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    auto residual = x;
    x = torch::relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    return torch::relu(x + residual);
}

// ============================================================================
// PenteNet
// ============================================================================

static constexpr int PLANES  = NNInputEncoder::NUM_PLANES;  // 6
static constexpr int N_CELLS = NNInputEncoder::PLANE_SIZE;  // 361

PenteNetImpl::PenteNetImpl(const PenteNetConfig &cfg) {
    int f = cfg.numFilters;

    // Input conv: PLANES → f, 3×3
    inputConv = register_module("inputConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(PLANES, f, 3).padding(1).bias(false)));
    inputBn = register_module("inputBn", torch::nn::BatchNorm2d(f));

    // Residual tower
    resBlocks = register_module("resBlocks", torch::nn::ModuleList());
    for (int i = 0; i < cfg.numResBlocks; ++i)
        resBlocks->push_back(ResBlock(f));

    // Policy head: f → 2 filters (1×1), then flatten → 361 logits
    policyConv = register_module("policyConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(f, 2, 1).bias(false)));
    policyBn = register_module("policyBn", torch::nn::BatchNorm2d(2));
    policyFc  = register_module("policyFc",  torch::nn::Linear(2 * N_CELLS, N_CELLS));

    // Value head: f → 1 filter (1×1), flatten → 256 → 1 (tanh)
    valueConv = register_module("valueConv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(f, 1, 1).bias(false)));
    valueBn  = register_module("valueBn",  torch::nn::BatchNorm2d(1));
    valueFc1 = register_module("valueFc1", torch::nn::Linear(N_CELLS, 256));
    valueFc2 = register_module("valueFc2", torch::nn::Linear(256, 1));
}

std::tuple<torch::Tensor, torch::Tensor> PenteNetImpl::forward(torch::Tensor x) {
    // Body
    x = torch::relu(inputBn(inputConv(x)));
    for (const auto &block : *resBlocks)
        x = block->as<ResBlockImpl>()->forward(x);

    // Policy head
    auto p = torch::relu(policyBn(policyConv(x)));
    p = p.flatten(1);        // [B, 2*361]
    p = policyFc(p);         // [B, 361] — raw logits, no softmax here

    // Value head
    auto v = torch::relu(valueBn(valueConv(x)));
    v = v.flatten(1);        // [B, 361]
    v = torch::relu(valueFc1(v));
    v = torch::tanh(valueFc2(v)); // [B, 1] in (-1, 1)

    return {p, v};
}

void PenteNetImpl::saveToFile(const std::string &path) const {
    torch::serialize::OutputArchive archive;
    torch::nn::Module::save(archive);
    archive.save_to(path);
}

void PenteNetImpl::loadFromFile(const std::string &path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::nn::Module::load(archive);
}

#endif // WITH_TORCH
