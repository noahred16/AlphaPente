#ifdef WITH_TORCH

// Include torch-dependent headers BEFORE doctest: the c10 logging header
// (pulled in via Evaluator.hpp) defines its own CHECK macro, which would
// silently shadow doctest's CHECK and turn every assertion into a no-op.
#include "Evaluator.hpp"
#include "SelfPlay.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#undef CHECK

#include "doctest.h"

TEST_CASE("policyTargetVisits - no solved win leaves visits untouched") {
    const std::vector<int> raw{10, 5, 0, 3};
    CHECK(policyTargetVisits(raw, -1) == raw);
    const std::vector<int> empty{0, 0, 0};
    CHECK(policyTargetVisits(empty, -1) == empty);
}

TEST_CASE("policyTargetVisits - proven win takes all the mass") {
    // Win child gets the pre-boost total (18), dominating without erasing the rest
    const std::vector<int> raw{10, 5, 3};
    const std::vector<int> expected{10, 18, 3};
    CHECK(policyTargetVisits(raw, 1) == expected);
}

TEST_CASE("policyTargetVisits - proven win with zero recorded visits still gets mass") {
    // Root arrived already solved and the search recorded no visits: the winning
    // move must still carry the policy target (this was the all-zero-policy bug)
    const std::vector<int> raw{0, 0, 0};
    const std::vector<int> expected{0, 0, 1};
    CHECK(policyTargetVisits(raw, 2) == expected);
}

TEST_CASE("validateSelfPlayConfig - default config is valid") {
    CHECK_NOTHROW(validateSelfPlayConfig(SelfPlayConfig{}));
}

TEST_CASE("validateSelfPlayConfig - rejects nonsensical scalar values") {
    SelfPlayConfig cfg;
    cfg.simulations = 0;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.explorationDropoff = -1;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.numWorkerThreads = 0;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.numEvalThreads = -1;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.dirichletAlpha = 0.0f;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.dirichletEpsilon = 1.5f;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);
}

TEST_CASE("validateSelfPlayConfig - rejects Gumbel knobs set under PUCT mode") {
    SelfPlayConfig cfg;
    cfg.gumbelMaxConsideredActions = 32;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.gumbelNoise = false;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.gumbelBatchGames = 32;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);
}

TEST_CASE("validateSelfPlayConfig - GUMBEL mode with default knobs is valid") {
    SelfPlayConfig cfg;
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    CHECK_NOTHROW(validateSelfPlayConfig(cfg));
}

TEST_CASE("validateSelfPlayConfig - rejects nonsensical Gumbel scalar values") {
    SelfPlayConfig cfg;
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    cfg.gumbelMaxConsideredActions = 0;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    cfg.gumbelBatchGames = 0;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);
}

TEST_CASE("validateSelfPlayConfig - rejects PUCT knobs set under GUMBEL mode") {
    SelfPlayConfig cfg;
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    cfg.dirichletAlpha = 0.5f;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    cfg.dirichletEpsilon = 0.1f;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);

    cfg = SelfPlayConfig{};
    cfg.searchMode = SelfPlaySearchMode::GUMBEL;
    cfg.explorationDropoff = 3;
    CHECK_THROWS_AS(validateSelfPlayConfig(cfg), std::invalid_argument);
}

TEST_CASE("SelfPlay - bootstrap examples form valid training targets") {
    // End-to-end data-quality check on the bootstrap generation path
    // (heuristic evaluator, as used by generate -b)
    HeuristicEvaluator eval;
    SelfPlayConfig cfg;
    cfg.simulations      = 400;
    cfg.numWorkerThreads = 2;
    cfg.numEvalThreads   = 0;
    std::mt19937 rng(7);
    auto examples = runGame(eval, PenteGame::Config::pente(), cfg, rng);

    REQUIRE(!examples.empty());
    constexpr int B = PenteGame::BOARD_SIZE;
    for (const auto &ex : examples) {
        // Policy is a proper distribution — in particular never all-zero,
        // which solved endgame positions used to produce
        float psum = ex.policy.sum().item<float>();
        CHECK(std::abs(psum - 1.0f) < 1e-4f);
        CHECK(ex.policy.min().item<float>() >= 0.0f);

        // Mass only on empty cells
        auto emptyPlane = ex.planes[2].reshape({B * B});
        CHECK((((ex.policy > 0.0f) & (emptyPlane < 0.5f))).sum().item<int64_t>() == 0);

        // Outcome must be a valid game result and root Q a bounded value
        // estimate — the two components of the train-time value target.
        CHECK((ex.outcome == 1.0f || ex.outcome == 0.0f || ex.outcome == -1.0f));
        CHECK(ex.rootValue >= -1.0f);
        CHECK(ex.rootValue <= 1.0f);
    }
}

TEST_CASE("runGamesGumbel - batched self-play examples form valid training targets") {
    // Same data-quality checks as the PUCT bootstrap test above, but through
    // the batched Gumbel driver, with gumbelBatchGames < numGames so slot
    // refill (a finished game's slot immediately starting a new one) actually
    // exercises itself rather than every game running in its own slot.
    HeuristicEvaluator eval;
    SelfPlayConfig cfg;
    cfg.searchMode              = SelfPlaySearchMode::GUMBEL;
    cfg.simulations              = 64;
    cfg.gumbelMaxConsideredActions = 8;
    cfg.gumbelBatchGames         = 3;
    cfg.numWorkerThreads         = 2;

    int completedCount = 0;
    auto games = runGamesGumbel(eval, PenteGame::Config::pente(), cfg, /*numGames=*/7,
                                 [&](int idx, const std::vector<SelfPlayExample> &) {
                                     CHECK(idx == completedCount);
                                     ++completedCount;
                                 });

    REQUIRE(games.size() == 7);
    CHECK(completedCount == 7);

    constexpr int B = PenteGame::BOARD_SIZE;
    for (const auto &examples : games) {
        REQUIRE(!examples.empty());
        for (const auto &ex : examples) {
            float psum = ex.policy.sum().item<float>();
            CHECK(std::abs(psum - 1.0f) < 1e-3f);
            CHECK(ex.policy.min().item<float>() >= 0.0f);

            auto emptyPlane = ex.planes[2].reshape({B * B});
            CHECK((((ex.policy > 0.0f) & (emptyPlane < 0.5f))).sum().item<int64_t>() == 0);

            CHECK((ex.outcome == 1.0f || ex.outcome == 0.0f || ex.outcome == -1.0f));
            CHECK(ex.rootValue >= -1.0f);
            CHECK(ex.rootValue <= 1.0f);
        }
    }
}

#endif // WITH_TORCH
