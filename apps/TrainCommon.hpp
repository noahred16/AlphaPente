#pragma once
#ifdef WITH_TORCH

#include <filesystem>
#include <iostream>
#include <torch/torch.h>

// Buffer counts are unique positions; the 8 board symmetries are applied at
// sample time (augmentBatch), not stored. Values below match the effective
// window of the previous 8x-stored scheme (500k / 5k stored samples).
static constexpr int   BUFFER_SIZE       = 62'500;
static constexpr int   MIN_BUFFER_SIZE   = 625;
static constexpr int   BATCH_SIZE        = 256;
static constexpr int   MIN_GRAD_STEPS    = 200;
static constexpr float LR                = 0.01f;
static constexpr float WEIGHT_DECAY      = 1e-4f;
static constexpr float VALUE_LOSS_WEIGHT = 1.0f;
static constexpr double VAL_FRACTION       = 0.05;  // held out from training, used to gate best_model.pt promotion
static constexpr double VAL_GATE_TOLERANCE = 1.02;  // candidate must not regress val loss by more than this factor

struct ReplayBuffer {
    torch::Tensor states;    // [N, 2, 19, 19] uint8 — my stones, opp stones
    torch::Tensor captures;  // [N, 2]
    torch::Tensor policies;  // [N, 361] float16
    torch::Tensor values;    // [N, 1]
    int64_t size() const { return states.defined() ? states.size(0) : 0; }
};

// Reconstruct the 5-plane float input the net expects (my, opp, empty,
// my_caps/max, opp_caps/max) from the compact stored form: empty is derived
// from the stone planes, capture planes are broadcast from `captures`.
inline torch::Tensor decodeStates(const torch::Tensor &stones, const torch::Tensor &captures) {
    auto s     = stones.to(torch::kFloat);
    auto empty = 1.0f - s.sum(1, /*keepdim=*/true);
    auto caps  = captures.to(torch::kFloat)
                     .view({-1, 2, 1, 1})
                     .expand({s.size(0), 2, s.size(2), s.size(3)});
    return torch::cat({s, empty, caps}, 1);
}

// Apply board symmetry k (0-7: bit 2 = horizontal flip, bits 0-1 = quarter
// turns) to a batch of planes [N, C, B, B] and policies [N, B*B], keeping the
// policy grid consistent with the board.
inline std::pair<torch::Tensor, torch::Tensor>
applySymmetry(torch::Tensor planes, torch::Tensor policies, int k) {
    int64_t B = planes.size(2);
    auto pol  = policies.view({-1, B, B});
    if (k & 4) { planes = planes.flip(3); pol = pol.flip(2); }
    if (k & 3) { planes = torch::rot90(planes, k & 3, {2, 3});
                 pol    = torch::rot90(pol,    k & 3, {1, 2}); }
    return {planes, pol.reshape({-1, B * B})};
}

// Re-orient each sample in a batch with a random board symmetry (training-time
// augmentation, replacing the 8x stored copies). Samples are grouped by
// symmetry and concatenated, so batch order is shuffled — irrelevant for SGD.
inline void augmentBatch(torch::Tensor &states, torch::Tensor &captures,
                         torch::Tensor &policies, torch::Tensor &values) {
    auto ks = torch::randint(0, 8, {states.size(0)}, torch::kInt64);
    std::vector<torch::Tensor> s, c, p, v;
    for (int k = 0; k < 8; k++) {
        auto sel = (ks == k).nonzero().squeeze(1).to(states.device());
        if (sel.size(0) == 0) continue;
        auto [ts, tp] = applySymmetry(states.index_select(0, sel),
                                      policies.index_select(0, sel), k);
        s.push_back(ts);
        p.push_back(tp);
        c.push_back(captures.index_select(0, sel));
        v.push_back(values.index_select(0, sel));
    }
    states   = torch::cat(s, 0);
    captures = torch::cat(c, 0);
    policies = torch::cat(p, 0);
    values   = torch::cat(v, 0);
}

inline ReplayBuffer loadBuffer(const std::string &path) {
    ReplayBuffer buf;
    if (!std::filesystem::exists(path)) return buf;
    try {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        ar.read("states",   buf.states);
        ar.read("captures", buf.captures);
        ar.read("policies", buf.policies);
        ar.read("values",   buf.values);
        if (buf.states.defined() && buf.states.size(1) == 5) {
            // Legacy full-float format — convert to compact form.
            buf.states   = buf.states.slice(1, 0, 2).to(torch::kU8).contiguous();
            buf.policies = buf.policies.to(torch::kHalf);
        }
    } catch (const std::exception &e) {
        std::cerr << "Warning: failed to load buffer (" << e.what() << ")\n";
    }
    return buf;
}

inline void saveBuffer(const ReplayBuffer &buf, const std::string &path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    torch::serialize::OutputArchive ar;
    ar.write("states",   buf.states);
    ar.write("captures", buf.captures);
    ar.write("policies", buf.policies);
    ar.write("values",   buf.values);
    ar.save_to(path);
}

inline ReplayBuffer mergeAndTrim(ReplayBuffer existing,
                                  torch::Tensor newStates,
                                  torch::Tensor newCaptures,
                                  torch::Tensor newPolicies,
                                  torch::Tensor newValues,
                                  int64_t maxSize = BUFFER_SIZE) {
    if (!existing.states.defined()) {
        existing = {newStates, newCaptures, newPolicies, newValues};
    } else {
        existing.states   = torch::cat({existing.states,   newStates},   0);
        existing.captures = torch::cat({existing.captures, newCaptures}, 0);
        existing.policies = torch::cat({existing.policies, newPolicies}, 0);
        existing.values   = torch::cat({existing.values,   newValues},   0);
    }
    int64_t n = existing.states.size(0);
    if (maxSize > 0 && n > maxSize) {
        int64_t drop = n - maxSize;
        existing.states   = existing.states.slice(0, drop);
        existing.captures = existing.captures.slice(0, drop);
        existing.policies = existing.policies.slice(0, drop);
        existing.values   = existing.values.slice(0, drop);
    }
    return existing;
}

inline int nextIterNumber(const std::string &dir) {
    int maxIter = 0;
    if (!std::filesystem::exists(dir)) return 1;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        const std::string stem = entry.path().stem().string();
        if (stem.rfind("model_iter", 0) == 0) {
            try { maxIter = std::max(maxIter, std::stoi(stem.substr(10))); }
            catch (...) {}
        }
    }
    return maxIter + 1;
}

#endif // WITH_TORCH
