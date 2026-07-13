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
    int   explorationDropoff = 15;   // moves after which Dirichlet noise and temp sampling are disabled
    float explorationC       = 3.0f;
    float dirichletAlpha     = 0.3f;
    float dirichletEpsilon   = 0.25f;
    int   numWorkerThreads   = 4;
    int   numEvalThreads     = 1;
};

// Build the policy-target visit counts from raw root-child visits. If a proven
// winning child exists it takes all the mass (at least 1 visit, even when the
// search recorded none — e.g. the root arrived already solved). Other children,
// including proven losses, keep their raw counts: PUCT stops visiting a child
// once it is proven lost, and in a lost position the raw distribution is the
// best-resistance target.
std::vector<int> policyTargetVisits(std::vector<int> visits, int solvedWinIdx);

// Run one self-play game. Returns one SelfPlayExample per move made.
// Values are filled retroactively from the game outcome.
std::vector<SelfPlayExample> runGame(Evaluator &eval,
                                      const PenteGame::Config &gameConfig,
                                      const SelfPlayConfig &cfg,
                                      std::mt19937 &rng);

#endif // WITH_TORCH
