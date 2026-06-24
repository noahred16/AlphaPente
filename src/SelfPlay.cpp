#ifdef WITH_TORCH

#include "SelfPlay.hpp"
#include "Evaluator.hpp"
#include "NNModel.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <algorithm>

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
    mctsConfig.explorationConstant = cfg.explorationC;
    mctsConfig.dirichletAlpha     = cfg.dirichletAlpha;
    mctsConfig.dirichletEpsilon   = cfg.dirichletEpsilon;
    ParallelMCTS mcts(mctsConfig);

    constexpr int B = PenteGame::BOARD_SIZE;
    std::vector<SelfPlayExample> examples;

    while (!game.isGameOver()) {
        if (game.getMoveCount() == cfg.explorationDropoff) {
            auto c = mcts.getConfig();
            c.dirichletAlpha = 0.0f;
            mcts.setConfig(c);
        }
        mcts.search(game);

        const auto *root = mcts.getRoot();
        int cap          = static_cast<int>(root->childCapacity);

        std::vector<int> visits(cap, 0);
        int totalVisits = 0;
        for (int i = 0; i < cap; i++) {
            if (root->children[i])
                visits[i] = root->children[i]->visits.load();
            totalVisits += visits[i];
        }

        // If any child is a proved win, boost its visit count to totalVisits so it
        // dominates the policy target without erasing the signal from other moves.
        // SOLVED_LOSS children are zeroed — the policy should never point toward a proven loss.
        int solvedWinIdx = -1;
        for (int i = 0; i < cap; i++) {
            if (!root->children[i]) continue;
            auto st = root->children[i]->solvedStatus.load(std::memory_order_acquire);
            if (st == ParallelMCTS::SolvedStatus::SOLVED_WIN && solvedWinIdx < 0)
                solvedWinIdx = i;
            else if (st == ParallelMCTS::SolvedStatus::SOLVED_LOSS)
                visits[i] = 0;
        }
        if (solvedWinIdx >= 0)
            visits[solvedWinIdx] = totalVisits;
        // Recompute totalVisits after any adjustments
        totalVisits = 0;
        for (int i = 0; i < cap; i++) totalVisits += visits[i];

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

        auto [planes, captures] = NNEvaluator::gameToTensors(game);
        examples.push_back({planes, captures, policyTensor, game.getCurrentPlayer(), 0.0f});

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
    for (auto &ex : examples)
        ex.value = (winner == PenteGame::NONE) ? 0.0f
                 : (ex.player == winner)        ? 1.0f
                                                : -1.0f;

    return examples;
}

#endif // WITH_TORCH
