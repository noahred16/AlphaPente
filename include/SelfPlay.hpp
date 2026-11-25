#pragma once
#ifdef WITH_TORCH

#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <functional>
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
    float policyEntropy = 0.0f;  // entropy of the MCTS visit distribution, normalized by log(legal moves) to [0, 1]
    int   moveIndex = 0;      // this position's true ply index within its game, counting
                               // from the game's real start even if runGame later trims
                               // that start away (see the exploration-dropoff comment below)
};

// PUCT runs through runGame() (ParallelMCTS, one game at a time, Dirichlet-noise
// exploration). GUMBEL runs through runGamesGumbel() (GumbelSearch, many games
// batched together -- see that function's comment for why it needs its own
// driver rather than a branch inside runGame()).
enum class SelfPlaySearchMode { PUCT, GUMBEL };

struct SelfPlayConfig {
    SelfPlaySearchMode searchMode = SelfPlaySearchMode::PUCT;

    int   simulations        = 100;
    // PUCT only. Upper bound on a per-game randomized exploration cutoff
    // (runGame draws a fresh Uniform(0, explorationDropoff) for each game) --
    // not a fixed cutoff itself. See runGame's own comment: positions before
    // that game's draw are dropped from its returned examples, not merely
    // explored. Gumbel's chosen move IS the search's own conclusion (no
    // separate temperature-sampling step layered on top of it), so there's no
    // equivalent value-target contamination to guard against -- runGamesGumbel
    // ignores this and keeps every position.
    int   explorationDropoff = 8;
    // PUCT only. Exploration constant is not a fixed knob here — runGame scales
    // it by game phase via GameUtils::explorationConstantForMoveCount, matching
    // the pente app's tuning (a limited sim budget needs more exploitation-
    // focused search early/tactically, and can afford broader exploration once
    // the game settles). Gumbel-Top-k supplies its own exploration (see
    // gumbelNoise below) and ignores dirichletAlpha/dirichletEpsilon.
    float dirichletAlpha     = 0.3f;
    float dirichletEpsilon   = 0.25f;
    int   numWorkerThreads   = 4;
    int   numEvalThreads     = 1;

    // Gumbel only (searchMode == GUMBEL) -- see GumbelTree::Config.
    int   gumbelMaxConsideredActions = 16;
    bool  gumbelNoise                = true;
    // Games batched together per GumbelSearch::searchBatch() call -- this is
    // what drives NN batch size for Gumbel (PUCT gets its batch size from many
    // threads sharing ONE game's tree instead; Gumbel parallelizes across
    // independent games, not within one). Needs to be large enough to saturate
    // the GPU; see runGamesGumbel.
    int   gumbelBatchGames           = 64;
};

// Throws std::invalid_argument, with a message naming the offending field, if
// cfg has a value that can never be sane (e.g. zero simulations) or a
// combination that doesn't make sense together (e.g. a Dirichlet knob moved
// off its default under GUMBEL mode, which ignores it, or vice versa). Call
// this right after building a SelfPlayConfig from CLI flags, before starting
// any games.
void validateSelfPlayConfig(const SelfPlayConfig &cfg);

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

// Run `numGames` self-play games with Gumbel search, continuously batching
// cfg.gumbelBatchGames of them together: every call to GumbelSearch::searchBatch
// advances that many games by one ply at once, and the instant any game finishes
// its slot is immediately refilled with a fresh game, so the batch -- and the NN
// eval batch size riding on it -- stays full for the whole run instead of
// shrinking as individual games finish at different times. This is the real
// difference from runGame(): PUCT gets its NN batch size from many threads
// cooperating on ONE game's tree, which has no Gumbel equivalent (GumbelTree is
// single-threaded per game), so Gumbel self-play has to get it from many games
// in flight instead, which only a dedicated multi-game driver can do.
//
// Returns one vector<SelfPlayExample> per completed game. If onGameCompleted is
// set, it's called synchronously right after each game finishes (and before its
// slot is refilled), with the 0-based index of that completion (0..numGames-1,
// in completion order, not game-start order) and its examples -- for progress
// reporting/accounting the way Generate.cpp's per-game loop does for runGame().
//
// No rng parameter: unlike runGame (which uses one for the exploration-dropoff
// draw and temperature sampling), every tree seeds itself non-deterministically
// -- Gumbel's exploration is the fresh per-tree Gumbel noise, not anything this
// driver samples itself.
std::vector<std::vector<SelfPlayExample>> runGamesGumbel(
    Evaluator &eval, const PenteGame::Config &gameConfig, const SelfPlayConfig &cfg, int numGames,
    const std::function<void(int, const std::vector<SelfPlayExample> &)> &onGameCompleted = nullptr);

#endif // WITH_TORCH
