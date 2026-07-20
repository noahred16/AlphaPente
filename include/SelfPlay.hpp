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
    float outcome = 0.0f;     // final game result z (+1/-1/0), previous-player perspective
    float rootValue = 0.0f;   // root's own totalValue/visits at search time, same perspective as outcome
};

struct SelfPlayConfig {
    int   simulations        = 100;
    int   explorationDropoff = 8;    // moves after which Dirichlet noise and temp sampling are disabled
    // Exploration constant is not a fixed knob here — runGame scales it by game
    // phase via GameUtils::explorationConstantForMoveCount, matching the pente
    // app's tuning (a limited sim budget needs more exploitation-focused search
    // early/tactically, and can afford broader exploration once the game settles).
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
// The final game outcome z and each position's own root Q estimate are kept
// as separate fields; the value target is blended from them at train time
// (see blendValueTargets in TrainCommon.hpp).
std::vector<SelfPlayExample> runGame(Evaluator &eval,
                                      const PenteGame::Config &gameConfig,
                                      const SelfPlayConfig &cfg,
                                      std::mt19937 &rng);

#endif // WITH_TORCH
