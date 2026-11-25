#ifdef WITH_TORCH

#include "SelfPlay.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "GumbelMCTS.hpp"
#include "NNModel.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <algorithm>
#include <cassert>
#include <stdexcept>

void validateSelfPlayConfig(const SelfPlayConfig &cfg) {
    if (cfg.simulations <= 0)
        throw std::invalid_argument("SelfPlayConfig: simulations must be > 0");
    if (cfg.numWorkerThreads <= 0)
        throw std::invalid_argument("SelfPlayConfig: numWorkerThreads must be > 0");
    if (cfg.numEvalThreads < 0)
        throw std::invalid_argument("SelfPlayConfig: numEvalThreads must be >= 0");

    if (cfg.searchMode == SelfPlaySearchMode::PUCT) {
        if (cfg.explorationDropoff < 0)
            throw std::invalid_argument("SelfPlayConfig: explorationDropoff must be >= 0");
        if (cfg.dirichletAlpha <= 0.0f)
            throw std::invalid_argument("SelfPlayConfig: dirichletAlpha must be > 0");
        if (cfg.dirichletEpsilon < 0.0f || cfg.dirichletEpsilon > 1.0f)
            throw std::invalid_argument("SelfPlayConfig: dirichletEpsilon must be in [0, 1]");
        if (cfg.gumbelMaxConsideredActions != 16 || cfg.gumbelNoise != true || cfg.gumbelBatchGames != 64)
            throw std::invalid_argument(
                "SelfPlayConfig: Gumbel knobs (gumbelMaxConsideredActions/gumbelNoise/gumbelBatchGames) "
                "are set but searchMode is PUCT, which ignores them -- they'd silently have no effect");
    } else { // GUMBEL
        if (cfg.gumbelMaxConsideredActions <= 0)
            throw std::invalid_argument("SelfPlayConfig: gumbelMaxConsideredActions must be > 0");
        if (cfg.gumbelBatchGames <= 0)
            throw std::invalid_argument("SelfPlayConfig: gumbelBatchGames must be > 0");
        if (cfg.dirichletAlpha != 0.3f || cfg.dirichletEpsilon != 0.25f || cfg.explorationDropoff != 8)
            throw std::invalid_argument(
                "SelfPlayConfig: PUCT knobs (dirichletAlpha/dirichletEpsilon/explorationDropoff) "
                "are set but searchMode is GUMBEL, which ignores them -- they'd silently have no effect");
    }
}

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
        c.dirichletEpsilon    = (game.getMoveCount() >= gameDropoff) ? 0.0f : cfg.dirichletEpsilon;
        mcts.setConfig(c);
        mcts.search(game);

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

        visits = policyTargetVisits(std::move(visits), solvedWinIdx);
        int totalVisits = 0;
        for (int i = 0; i < cap; i++) totalVisits += visits[i];

        // Should not happen: search always records visits unless it ran zero
        // iterations (e.g. arena exhausted). If violated, the recorded policy
        // target is all-zero and the chosen move arbitrary.
        assert(totalVisits > 0 && "search recorded no root visits");

        auto policyTensor = torch::zeros({B * B});
        float entropy = 0.0f;
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

        int chosen = 0;
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

        PenteGame::Move mv = toPhysical(root->moves[chosen]);
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

std::vector<std::vector<SelfPlayExample>> runGamesGumbel(
    Evaluator &eval, const PenteGame::Config &gameConfig, const SelfPlayConfig &cfg, int numGames,
    const std::function<void(int, const std::vector<SelfPlayExample> &)> &onGameCompleted) {
    std::vector<std::vector<SelfPlayExample>> completedGames;
    if (numGames <= 0) return completedGames;
    completedGames.reserve(numGames);

    GumbelSearch::Config searchConfig;
    searchConfig.tree.numSimulations        = cfg.simulations;
    searchConfig.tree.maxConsideredActions  = cfg.gumbelMaxConsideredActions;
    searchConfig.tree.gumbelNoise           = cfg.gumbelNoise;
    searchConfig.tree.seed                  = 0; // non-deterministic: a fresh random_device seed per tree
    searchConfig.numThreads                 = cfg.numWorkerThreads;
    GumbelSearch search(searchConfig, &eval);

    constexpr int B = PenteGame::BOARD_SIZE;
    const int slots = std::min(cfg.gumbelBatchGames, numGames);

    auto freshGame = [&] {
        PenteGame g(gameConfig);
        g.reset();
        return g;
    };

    // Continuous batching: `slots` games are always in flight (until fewer than
    // `numGames` remain to be started), so the batch GumbelSearch::searchBatch
    // sees -- and the NN eval batch size riding on it -- never shrinks just
    // because one particular game happened to finish early.
    std::vector<PenteGame> slotGame;
    slotGame.reserve(slots);
    for (int i = 0; i < slots; i++) slotGame.push_back(freshGame());
    std::vector<std::vector<SelfPlayExample>> slotExamples(slots);
    std::vector<bool> slotLive(slots, true);

    int started = slots;
    int finished = 0;

    std::vector<PenteGame> positions;
    std::vector<int> slotForPos;
    while (finished < numGames) {
        positions.clear();
        slotForPos.clear();
        for (int i = 0; i < slots; i++) {
            if (slotLive[i]) {
                positions.push_back(slotGame[i]);
                slotForPos.push_back(i);
            }
        }
        assert(!positions.empty() && "live slots exhausted before numGames finished");

        auto results = search.searchBatch(positions);

        for (size_t j = 0; j < results.size(); j++) {
            const int i = slotForPos[j];
            const GumbelTree::Result &r = results[j];
            PenteGame &game = slotGame[i];

            const int k = static_cast<int>(r.moves.size());
            auto policyTensor = torch::zeros({B * B});
            float entropy = 0.0f;
            auto acc = policyTensor.accessor<float, 1>();
            for (int a = 0; a < k; a++) {
                const float p = r.improvedPolicy[a];
                if (p > 0.0f) {
                    const PenteGame::Move &mv = r.moves[a]; // no canonical un-rotation -- GumbelTree does none
                    acc[mv.y * B + mv.x] = p;
                    entropy -= p * std::log(p);
                }
            }
            const float policyEntropy = (k > 1) ? entropy / std::log(static_cast<float>(k)) : 0.0f;

            auto [planes, captures] = NNEvaluator::gameToTensors(game);
            slotExamples[i].push_back({planes, captures, policyTensor, game.getCurrentPlayer(), 0.0f,
                                        r.improvedValue, policyEntropy, game.getMoveCount()});

            game.makeMove(r.move.x, r.move.y);

            if (game.isGameOver()) {
                const PenteGame::Player winner = game.getWinner();
                for (auto &ex : slotExamples[i]) {
                    // Same previous-mover-perspective convention as runGame().
                    ex.outcome = (winner == PenteGame::NONE) ? 0.0f
                               : (ex.player == winner)        ? -1.0f
                                                               :  1.0f;
                }
                ++finished;
                if (onGameCompleted) onGameCompleted(finished - 1, slotExamples[i]);
                completedGames.push_back(std::move(slotExamples[i]));

                if (started < numGames) {
                    slotGame[i] = freshGame();
                    slotExamples[i].clear();
                    ++started;
                } else {
                    slotLive[i] = false;
                }
            }
        }
    }

    return completedGames;
}

#endif // WITH_TORCH
