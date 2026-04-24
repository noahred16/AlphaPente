#include "doctest.h"
#include "Evaluator.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <thread>

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
