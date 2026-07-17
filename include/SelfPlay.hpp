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
    float value = 0.0f;       // training target: blend of game outcome and root Q (+1/-1/0 scale)
    float rootValue = 0.0f;   // root's own totalValue/visits at search time, same perspective as value
};

struct SelfPlayConfig {
    int   simulations        = 100;
    int   explorationDropoff = 8;    // moves after which Dirichlet noise and temp sampling are disabled
    float explorationC       = 3.0f;
    float dirichletAlpha     = 0.3f;
    float dirichletEpsilon   = 0.25f;
    int   numWorkerThreads   = 4;
    int   numEvalThreads     = 1;
    // Value target = valueBlendAlpha * z + (1 - valueBlendAlpha) * rootQ, where z is
    // the final game outcome and rootQ is this position's own root value estimate at
    // search time. Pure z (alpha=1) gives every position in a game an identical,
    // maximally-confident target regardless of how far it is from the decisive endgame;
    // blending in the position's own search estimate tempers that for early/uncertain
    // positions. See KataGo/AlphaZero-descendant value target design.
    float valueBlendAlpha    = 0.6f;
};

// Build the policy-target visit counts from raw root-child visits. If a proven
// winning child exists it takes all the mass (at least 1 visit, even when the
// search recorded none — e.g. the root arrived already solved). Other children,
// including proven losses, keep their raw counts: PUCT stops visiting a child
// once it is proven lost, and in a lost position the raw distribution is the
// best-resistance target.
std::vector<int> policyTargetVisits(std::vector<int> visits, int solvedWinIdx);

// Run one self-play game. Returns one SelfPlayExample per move made.
// Values are filled retroactively as a blend of game outcome and each
// position's own root Q estimate (see SelfPlayConfig::valueBlendAlpha).
std::vector<SelfPlayExample> runGame(Evaluator &eval,
                                      const PenteGame::Config &gameConfig,
                                      const SelfPlayConfig &cfg,
                                      std::mt19937 &rng);

#endif // WITH_TORCH
