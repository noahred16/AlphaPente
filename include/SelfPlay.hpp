#pragma once
#ifdef WITH_TORCH

#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <random>
#include <torch/torch.h>
#include <vector>

struct SelfPlayExample {
    torch::Tensor planes;     // [3, 19, 19] — current/opponent/empty planes
    torch::Tensor captures;   // [2] — capture counts for each player
    torch::Tensor policy;     // [361] — MCTS visit proportions
    PenteGame::Player player; // whose turn it was
    float value = 0.0f;       // game outcome from this player's perspective (+1/-1/0)
};

struct SelfPlayConfig {
    int   simulations        = 100;
    int   tempDropoff        = 15;   // moves before switching from proportional to greedy
    float explorationC       = 3.0f;
    float dirichletAlpha     = 0.3f;
    float dirichletEpsilon   = 0.25f;
    int   numWorkerThreads   = 4;
    int   numEvalThreads     = 1;
};

// Run one self-play game. Returns one SelfPlayExample per move made.
// Values are filled retroactively from the game outcome.
std::vector<SelfPlayExample> runGame(Evaluator &eval,
                                      const PenteGame::Config &gameConfig,
                                      const SelfPlayConfig &cfg,
                                      std::mt19937 &rng);

#endif // WITH_TORCH
