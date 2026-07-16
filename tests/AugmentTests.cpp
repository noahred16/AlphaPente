#include "doctest.h"

#ifdef WITH_TORCH

#include "PenteGame.hpp"
#include "TrainCommon.hpp"
#include <torch/torch.h>

static constexpr int B = PenteGame::BOARD_SIZE;

TEST_CASE("applySymmetry - 8 distinct transforms preserve stones") {
    // K10 = x9,y9  L9 = x10,y8  L7 = x10,y6  (tensor indexed [channel][row][col])
    auto planes = torch::zeros({1, 5, B, B});
    planes[0][0][9][9]  = 1.0f;  // K10
    planes[0][0][8][10] = 1.0f;  // L9
    planes[0][0][6][10] = 1.0f;  // L7
    auto policy = torch::zeros({1, B * B});

    std::vector<torch::Tensor> all8;
    for (int k = 0; k < 8; k++)
        all8.push_back(applySymmetry(planes, policy, k).first);

    // k = 0 must be the identity.
    CHECK(torch::equal(all8[0], planes));

    // All 8 transforms must be distinct.
    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++)
            CHECK_FALSE(torch::equal(all8[i], all8[j]));

    // Piece count must be preserved across all transforms.
    float origCount = all8[0].sum().item<float>();
    for (int i = 1; i < 8; i++)
        CHECK(all8[i].sum().item<float>() == doctest::Approx(origCount));
}

TEST_CASE("applySymmetry - policy transforms with the board") {
    // Marker stone and a one-hot policy on the same cell: after any symmetry
    // the policy peak must still sit on the marker.
    const int y = 3, x = 7;
    auto planes = torch::zeros({1, 2, B, B});
    planes[0][0][y][x] = 1.0f;
    auto policy = torch::zeros({1, B * B});
    policy[0][y * B + x] = 1.0f;

    for (int k = 0; k < 8; k++) {
        auto [tPlanes, tPolicy] = applySymmetry(planes, policy, k);
        CHECK(tPolicy.sum().item<float>() == doctest::Approx(1.0f));
        int peak = (int)tPolicy.argmax(1).item<int64_t>();
        CHECK(tPlanes[0][0][peak / B][peak % B].item<float>() == 1.0f);
    }
}

TEST_CASE("augmentBatch - rows stay consistent across all four tensors") {
    // Each sample: one stone, a one-hot policy on it, and a row id stored in
    // both captures and values. After augmentation the policy peak must still
    // sit on the stone, and captures/values must still travel with their row.
    const int N = 64;
    auto states   = torch::zeros({N, 2, B, B}, torch::kU8);
    auto policies = torch::zeros({N, B * B});
    auto captures = torch::zeros({N, 2});
    auto values   = torch::zeros({N, 1});

    for (int i = 0; i < N; i++) {
        int y = i % B, x = (i * 5 + 3) % B;
        states[i][0][y][x] = 1;
        policies[i][y * B + x] = 1.0f;
        captures[i][0] = (float)i;
        values[i][0]   = (float)i;
    }

    augmentBatch(states, captures, policies, values);

    REQUIRE(states.size(0) == N);
    REQUIRE(captures.size(0) == N);
    REQUIRE(policies.size(0) == N);
    REQUIRE(values.size(0) == N);

    // Row ids are a permutation of 0..N-1, and captures/values stay paired.
    CHECK(torch::equal(captures.select(1, 0), values.select(1, 0)));
    auto sortedIds = std::get<0>(captures.select(1, 0).sort());
    CHECK(torch::equal(sortedIds, torch::arange(N, torch::kFloat)));

    for (int i = 0; i < N; i++) {
        CHECK(states[i].sum().item<int64_t>() == 1);
        int peak = (int)policies[i].argmax().item<int64_t>();
        CHECK(states[i][0][peak / B][peak % B].item<uint8_t>() == 1);
    }
}

#endif // WITH_TORCH
