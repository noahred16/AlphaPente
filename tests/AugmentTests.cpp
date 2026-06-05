#include "doctest.h"

#ifdef WITH_TORCH

#include "PenteGame.hpp"
#include <iostream>
#include <torch/torch.h>

static constexpr int B = PenteGame::BOARD_SIZE;

// Print a window of the board centered around the action (rows 4-13, cols 6-13).
static void printBoard(const torch::Tensor &planes) {
    auto acc = planes.accessor<float, 3>();
    std::cout << "   ";
    for (int c = 6; c <= 13; c++) std::cout << c % 10;
    std::cout << "\n";
    for (int r = 4; r <= 13; r++) {
        std::cout << (r < 10 ? " " : "") << r << " ";
        for (int c = 6; c <= 13; c++)
            std::cout << (acc[0][r][c] > 0.5f ? "X" : ".");
        std::cout << "\n";
    }
}

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
    int idx = 0;
    for (int flip = 0; flip < 2; flip++) {
        for (int rot = 0; rot < 4; rot++) {
            auto t = applyTransform(planes, flip, rot);
            all8.push_back(t);
            std::cout << "\n[" << idx++ << "] flip=" << flip << " rot=" << rot << "\n";
            printBoard(t);
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
