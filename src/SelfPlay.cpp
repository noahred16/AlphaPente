#ifdef WITH_TORCH

#include "SelfPlay.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "NNModel.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <algorithm>
#include <cassert>

std::vector<int> policyTargetVisits(std::vector<int> visits, int solvedWinIdx) {
    if (solvedWinIdx >= 0) {
        int total = 0;
        for (int v : visits) total += v;
        visits[solvedWinIdx] = std::max(total, 1);
    }
    return visits;
}

std::vector<SelfPlayExample> runGame(Evaluator &eval,
                                      const PenteGame::Config &gameConfig,
                                      const SelfPlayConfig &cfg,
                                      std::mt19937 &rng) {
    PenteGame game(gameConfig);
    game.reset();

    ParallelMCTS::Config mctsConfig;
    mctsConfig.maxIterations      = cfg.simulations;
    mctsConfig.evaluator          = &eval;
    mctsConfig.numWorkerThreads   = cfg.numWorkerThreads;
    mctsConfig.numEvalThreads     = cfg.numEvalThreads;
    mctsConfig.seed               = static_cast<uint32_t>(rng());
    mctsConfig.dirichletAlpha     = cfg.dirichletAlpha;
    mctsConfig.dirichletEpsilon   = cfg.dirichletEpsilon;
    ParallelMCTS mcts(mctsConfig);

    constexpr int B = PenteGame::BOARD_SIZE;
    std::vector<SelfPlayExample> examples;

    while (!game.isGameOver()) {
        // Both scale with game phase: exploration constant tapers continuously
        // (see GameUtils::explorationConstantForMoveCount — matches the pente
        // app's tuning), while Dirichlet noise is a one-time on/off switch at
        // explorationDropoff.
        auto c = mcts.getConfig();
        c.explorationConstant = GameUtils::explorationConstantForMoveCount(game.getMoveCount());
        c.dirichletEpsilon    = (game.getMoveCount() >= cfg.explorationDropoff) ? 0.0f : cfg.dirichletEpsilon;
        mcts.setConfig(c);
        mcts.search(game);

        const auto *root = mcts.getRoot();
        int cap          = static_cast<int>(root->childCapacity);

        std::vector<int> visits(cap, 0);
        int solvedWinIdx = -1;
        for (int i = 0; i < cap; i++) {
            if (!root->children[i]) continue;
            visits[i] = root->children[i]->visits.load();
            if (solvedWinIdx < 0 &&
                root->children[i]->solvedStatus.load(std::memory_order_acquire) ==
                    ParallelMCTS::SolvedStatus::SOLVED_WIN)
                solvedWinIdx = i;
        }

        visits = policyTargetVisits(std::move(visits), solvedWinIdx);
        int totalVisits = 0;
        for (int i = 0; i < cap; i++) totalVisits += visits[i];

        // Should not happen: search always records visits unless it ran zero
        // iterations (e.g. arena exhausted). If violated, the recorded policy
        // target is all-zero and the chosen move arbitrary.
        assert(totalVisits > 0 && "search recorded no root visits");

        auto policyTensor = torch::zeros({B * B});
        if (totalVisits > 0) {
            auto acc = policyTensor.accessor<float, 1>();
            for (int i = 0; i < cap; i++) {
                if (visits[i] > 0) {
                    const auto &mv = root->moves[i];
                    acc[mv.y * B + mv.x] = (float)visits[i] / (float)totalVisits;
                }
            }
        }

        int32_t rootVisits = root->visits.load(std::memory_order_relaxed);
        float rootQ = rootVisits > 0
            ? static_cast<float>(root->totalValue.load(std::memory_order_relaxed) / rootVisits)
            : 0.0f;

        auto [planes, captures] = NNEvaluator::gameToTensors(game);
        examples.push_back({planes, captures, policyTensor, game.getCurrentPlayer(), 0.0f, rootQ});

        int chosen = 0;
        if (solvedWinIdx >= 0) {
            // A proven win exists — always play it regardless of exploration mode.
            // Sampling a different move here would corrupt the value target for this position.
            chosen = solvedWinIdx;
        } else if (game.getMoveCount() >= cfg.explorationDropoff || totalVisits == 0) {
            chosen = (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
        } else {
            std::uniform_int_distribution<int> dist(0, totalVisits - 1);
            int r = dist(rng), cum = 0;
            for (int i = 0; i < cap; i++) {
                cum += visits[i];
                if (r < cum) { chosen = i; break; }
            }
        }

        PenteGame::Move mv = root->moves[chosen];
        game.makeMove(mv.x, mv.y);
        mcts.reuseSubtree(mv);
    }

    PenteGame::Player winner = game.getWinner();
    for (auto &ex : examples) {
        // Convention: z = +1 if the player who MOVED INTO this position wins
        // (previous-player perspective), matching HeuristicEvaluator and MCTS backprop.
        // ex.rootValue was captured in the same perspective at search time, so the two
        // can be blended directly at train time without sign adjustment.
        ex.outcome = (winner == PenteGame::NONE) ? 0.0f
                   : (ex.player == winner)        ? -1.0f   // current player won → previous player lost
                                                  :  1.0f;  // current player lost → previous player won
    }

    return examples;
}

#endif // WITH_TORCH
