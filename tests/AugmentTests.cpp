#include "doctest.h"

#ifdef WITH_TORCH

#include "PenteGame.hpp"
#include <torch/torch.h>

static constexpr int B = PenteGame::BOARD_SIZE;

static torch::Tensor applyTransform(const torch::Tensor &planes, int flip, int rot) {
    auto t = flip ? torch::flip(planes, {2}) : planes;
    if (rot > 0) t = torch::rot90(t, rot, {1, 2});
    return t;
}

TEST_CASE("Augmentation symmetries - L shape K10 L9 L7") {
    // K10 = x9,y9  L9 = x10,y8  L7 = x10,y6  (tensor indexed [channel][row][col])
    auto planes = torch::zeros({5, B, B});
    planes[0][9][9]  = 1.0f;  // K10
    planes[0][8][10] = 1.0f;  // L9
    planes[0][6][10] = 1.0f;  // L7

    std::vector<torch::Tensor> all8;
    for (int flip = 0; flip < 2; flip++) {
        for (int rot = 0; rot < 4; rot++) {
            all8.push_back(applyTransform(planes, flip, rot));
        }
    }

    // All 8 transforms must be distinct.
    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++)
            CHECK_FALSE(torch::equal(all8[i], all8[j]));

    // Piece count must be preserved across all transforms.
    float origCount = all8[0].sum().item<float>();
    for (int i = 1; i < 8; i++)
        CHECK(all8[i].sum().item<float>() == doctest::Approx(origCount));
}

#endif // WITH_TORCH
