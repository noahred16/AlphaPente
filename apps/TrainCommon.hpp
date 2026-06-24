#pragma once
#ifdef WITH_TORCH

#include <filesystem>
#include <iostream>
#include <torch/torch.h>

static constexpr int   BUFFER_SIZE       = 100'000;
static constexpr int   MIN_BUFFER_SIZE   = 5'000;
static constexpr int   BATCH_SIZE        = 256;
static constexpr int   MIN_GRAD_STEPS    = 200;
static constexpr float LR                = 0.01f;
static constexpr float WEIGHT_DECAY      = 1e-4f;
static constexpr float VALUE_LOSS_WEIGHT = 1.0f;

struct ReplayBuffer {
    torch::Tensor states;    // [N, 5, 19, 19]  — my stones, opp stones, empty, my_captures/max (const), opp_captures/max (const)
    torch::Tensor captures;  // [N, 2]
    torch::Tensor policies;  // [N, 361]
    torch::Tensor values;    // [N, 1]
    int64_t size() const { return states.defined() ? states.size(0) : 0; }
};

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
