#ifdef WITH_TORCH

#include "SelfPlay.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "NNModel.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

std::vector<int> policyTargetVisits(std::vector<int> visits, int solvedWinIdx) {
    if (solvedWinIdx >= 0) {
        int total = 0;
        for (int v : visits) total += v;
        visits[solvedWinIdx] = std::max(total, 1);
    }
    return visits;
}

std::vector<float> gumbelPolicyTarget(std::vector<float> logits, const std::vector<float> &completedQ,
                                       float cScale, int solvedWinIdx) {
    for (size_t i = 0; i < logits.size(); ++i) logits[i] += cScale * completedQ[i];

    if (solvedWinIdx >= 0) {
        std::vector<float> out(logits.size(), 0.0f);
        out[solvedWinIdx] = 1.0f;
        return out;
    }

    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float &v : logits) {
        v = std::exp(v - maxLogit);
        sum += v;
    }
    for (float &v : logits) v /= sum;
    return logits;
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
    mctsConfig.searchMode         = cfg.searchMode;
    mctsConfig.gumbelM            = cfg.gumbelM;
    mctsConfig.gumbelCScale       = cfg.gumbelCScale;
    ParallelMCTS mcts(mctsConfig);

    constexpr int B = PenteGame::BOARD_SIZE;
    std::vector<SelfPlayExample> examples;

    // Exploration (Dirichlet noise + temperature sampling below) makes the
    // *chosen* move occasionally worse than the search's own best estimate —
    // fine for move diversity, but every position *after* that chosen move
    // then has its z outcome partly determined by a choice that wasn't really
    // the engine's judgment. Instead of a fixed per-game cutoff, draw a fresh
    // one per game: positions before it are exploratory and get dropped from
    // the returned examples entirely (see the erase() at the bottom), so
    // every example that's actually kept has deterministic (argmax) play the
    // rest of the way to the game's end — its z is never contaminated by a
    // sampling event that hadn't happened yet when it was recorded.
    std::uniform_int_distribution<int> dropoffDist(0, cfg.explorationDropoff);
    int gameDropoff = dropoffDist(rng);

    while (!game.isGameOver()) {
        // Both scale with game phase: exploration constant tapers continuously
        // (see GameUtils::explorationConstantForMoveCount — matches the pente
        // app's tuning), while Dirichlet noise is a one-time on/off switch at
        // gameDropoff.
        auto c = mcts.getConfig();
        c.explorationConstant = GameUtils::explorationConstantForMoveCount(game.getMoveCount());
        if (cfg.searchMode == ParallelMCTS::SearchMode::GUMBEL) {
            // Gumbel-Top-k + Sequential Halving is the exploration mechanism in
            // this mode; zero these rather than leave a stale non-zero alpha
            // from mctsConfig's construction, which would trip search()'s
            // mutual-exclusion warning every move for no reason.
            c.dirichletAlpha   = 0.0f;
            c.dirichletEpsilon = 0.0f;
        } else {
            c.dirichletEpsilon = (game.getMoveCount() >= gameDropoff) ? 0.0f : cfg.dirichletEpsilon;
        }
        mcts.setConfig(c);
        PenteGame::Move searchMove = mcts.search(game);

        const auto *root = mcts.getRoot();
        int cap          = static_cast<int>(root->childCapacity);

        // root->moves[] is in canonical (symmetry-folded) coordinates within the
        // first few plies (ParallelMCTS::Config::canonicalHashDepth) -- un-rotate
        // before using them as board cell indices or as the move actually played.
        int rootSym = -1;
        if (root->canonicalSym >= 0) {
            game.getCanonicalHash(rootSym);
        }
        auto toPhysical = [&](const PenteGame::Move &m) -> PenteGame::Move {
            if (rootSym < 0) return m;
            int px, py;
            Zobrist::instance().applyInverseSym(rootSym, m.x, m.y, px, py);
            return PenteGame::Move(px, py);
        };

        std::vector<int> visits(cap, 0);
        int solvedWinIdx = -1;
        for (int i = 0; i < cap; i++) {
            const auto *child = root->children[i].load(std::memory_order_acquire);
            if (!child) continue;
            visits[i] = child->visits.load();
            if (solvedWinIdx < 0 &&
                child->solvedStatus.load(std::memory_order_acquire) ==
                    ParallelMCTS::SolvedStatus::SOLVED_WIN)
                solvedWinIdx = i;
        }

        auto policyTensor = torch::zeros({B * B});
        float entropy = 0.0f;
        bool gumbel = (cfg.searchMode == ParallelMCTS::SearchMode::GUMBEL);

        if (gumbel) {
            // Policy target is softmax(logit(a) + cScale*completedQ(a)) over every
            // root child, visited or not -- NOT a function of visit counts.
            // root->priors are already-softmaxed, legal-move-renormalized
            // probabilities rather than raw logits, but log(prob) differs from
            // the true logit only by a per-node additive constant that the
            // softmax below is invariant to.
            std::vector<float> logits(cap), completedQ(cap);
            for (int i = 0; i < cap; i++) {
                logits[i] = std::log(std::max(root->priors[i], 1e-9f));
                const auto *child = root->children[i].load(std::memory_order_acquire);
                int32_t v = child ? child->visits.load(std::memory_order_relaxed) : 0;
                completedQ[i] = (child && v > 0)
                    ? static_cast<float>(child->totalValue.load(std::memory_order_relaxed) / v)
                    : -root->value;
            }
            auto target = gumbelPolicyTarget(std::move(logits), completedQ, cfg.gumbelCScale, solvedWinIdx);

            auto acc = policyTensor.accessor<float, 1>();
            for (int i = 0; i < cap; i++) {
                if (target[i] > 0.0f) {
                    PenteGame::Move mv = toPhysical(root->moves[i]);
                    acc[mv.y * B + mv.x] = target[i];
                    entropy -= target[i] * std::log(target[i]);
                }
            }
        } else {
            visits = policyTargetVisits(std::move(visits), solvedWinIdx);
            int totalVisits = 0;
            for (int i = 0; i < cap; i++) totalVisits += visits[i];

            // Should not happen: search always records visits unless it ran zero
            // iterations (e.g. arena exhausted). If violated, the recorded policy
            // target is all-zero and the chosen move arbitrary.
            assert(totalVisits > 0 && "search recorded no root visits");

            if (totalVisits > 0) {
                auto acc = policyTensor.accessor<float, 1>();
                for (int i = 0; i < cap; i++) {
                    if (visits[i] > 0) {
                        PenteGame::Move mv = toPhysical(root->moves[i]);
                        float p = (float)visits[i] / (float)totalVisits;
                        acc[mv.y * B + mv.x] = p;
                        entropy -= p * std::log(p);
                    }
                }
            }
        }
        // Normalized by log(legal moves) so it's comparable across game phase
        // (fewer legal moves late in the game) rather than depending on board size.
        float policyEntropy = (cap > 1) ? entropy / std::log((float)cap) : 0.0f;

        int32_t rootVisits = root->visits.load(std::memory_order_relaxed);
        float rootQ = rootVisits > 0
            ? static_cast<float>(root->totalValue.load(std::memory_order_relaxed) / rootVisits)
            : 0.0f;

        auto [planes, captures] = NNEvaluator::gameToTensors(game);
        examples.push_back({planes, captures, policyTensor, game.getCurrentPlayer(), 0.0f, rootQ, policyEntropy,
                             game.getMoveCount()});

        PenteGame::Move mv;
        if (gumbel) {
            // Gumbel-Top-k at the root is itself the exploration mechanism --
            // always play whatever the search returned (the Sequential-Halving
            // winner, or the engine's own proven-win move) rather than
            // resampling by visit count on top of it.
            mv = searchMove;
        } else {
            int chosen = 0;
            int totalVisits = 0;
            for (int v : visits) totalVisits += v;
            if (solvedWinIdx >= 0) {
                // A proven win exists — always play it regardless of exploration mode.
                // Sampling a different move here would corrupt the value target for this position.
                chosen = solvedWinIdx;
            } else if (game.getMoveCount() >= gameDropoff || totalVisits == 0) {
                chosen = (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
            } else {
                std::uniform_int_distribution<int> dist(0, totalVisits - 1);
                int r = dist(rng), cum = 0;
                for (int i = 0; i < cap; i++) {
                    cum += visits[i];
                    if (r < cum) { chosen = i; break; }
                }
            }
            mv = toPhysical(root->moves[chosen]);
        }

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

    // Drop this game's exploratory prefix (see gameDropoff above) — if the
    // whole game ended before gameDropoff, every position in it was
    // exploratory, so nothing from it is usable.
    size_t drop = std::min((size_t)gameDropoff, examples.size());
    examples.erase(examples.begin(), examples.begin() + drop);

    return examples;
}

#endif // WITH_TORCH
