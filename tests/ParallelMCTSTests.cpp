#include "doctest.h"
#include "Evaluator.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <thread>

TEST_CASE("prepareRoot expands root with evaluated policy and value") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    ParallelMCTS::Config config;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);
    mcts.prepareRoot(game);

    const auto *root = mcts.getRoot();
    REQUIRE(root != nullptr);
    CHECK(root->expanded.load() == true);
    CHECK(root->childCount > 0);
    CHECK(root->value >= -1.0f);
    CHECK(root->value <= 1.0f);

    float priorSum = 0.0f;
    for (int i = 0; i < root->childCount; ++i) {
        CHECK(root->priors[i] >= 0.0f);
        priorSum += root->priors[i];
    }
    CHECK(priorSum == doctest::Approx(1.0f).epsilon(0.01f));

    // Only root is allocated — child nodes are created lazily during search
    CHECK(mcts.getTreeSize() == 1);
}

TEST_CASE("EvalPool processes request and pushes result to backprop queue") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    ParallelMCTS::Config config;
    config.numEvalThreads = 1;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);

    ParallelMCTS::EvaluationRequest req;
    req.node = nullptr;
    req.gameState = game;
    req.searchPath = {};
    mcts.pushEvalRequest(req);

    mcts.startEvalThreads();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    mcts.stopEvalThreads();

    auto results = mcts.drainBackpropQueue();
    REQUIRE(results.size() == 1);
    CHECK(results[0].node == nullptr);
    CHECK(results[0].value >= -1.0f);
    CHECK(results[0].value <= 1.0f);
    CHECK(!results[0].policy.empty());
}
