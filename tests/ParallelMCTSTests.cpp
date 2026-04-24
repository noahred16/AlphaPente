#include "doctest.h"
#include "Evaluator.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

TEST_CASE("search returns a valid move after completing all iterations") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    ParallelMCTS::Config config;
    config.numWorkerThreads = 2;
    config.numEvalThreads = 1;
    config.maxIterations = 10;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);
    PenteGame::Move move = mcts.search(game);

    CHECK(move.x >= 0);
    CHECK(move.x < PenteGame::BOARD_SIZE);
    CHECK(move.y >= 0);
    CHECK(move.y < PenteGame::BOARD_SIZE);
    CHECK(mcts.getTreeSize() > 1);
    CHECK(mcts.getRoot()->visits.load() > 0);
}

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

TEST_CASE("Worker thread selects leaf, applies virtual loss, pushes to eval queue") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    ParallelMCTS::Config config;
    config.numWorkerThreads = 1;
    config.numEvalThreads = 0;
    config.maxIterations = 1;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);
    mcts.prepareRoot(game);  // root expanded with N children slots (all null)

    mcts.startWorkerThreads();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    mcts.stopWorkerThreads();

    auto requests = mcts.drainEvalQueue();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].node != nullptr);
    CHECK(requests[0].node->virtualLosses.load() == 1);
    CHECK(mcts.getTreeSize() == 2);  // root + 1 allocated child leaf
}

TEST_CASE("Worker completes full iteration: select, eval, expand, backprop") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    ParallelMCTS::Config config;
    config.numWorkerThreads = 1;
    config.numEvalThreads = 1;
    config.maxIterations = 1;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);
    mcts.prepareRoot(game);

    mcts.startEvalThreads();
    mcts.startWorkerThreads();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mcts.stopWorkerThreads();
    mcts.stopEvalThreads();

    const auto *root = mcts.getRoot();
    CHECK(root->visits.load() == 1);    // backprop propagated up to root

    // Find the one child node that was allocated during selection
    const ParallelMCTS::ThreadSafeNode *child = nullptr;
    for (int i = 0; i < root->childCapacity; ++i) {
        if (root->children[i] != nullptr) { child = root->children[i]; break; }
    }
    REQUIRE(child != nullptr);
    CHECK(child->visits.load() == 1);
    CHECK(child->virtualLosses.load() == 0);  // removed during backprop
    CHECK(child->expanded.load() == true);    // expanded with policy during backprop
    CHECK(child->childCount > 0);
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

TEST_CASE("Benchmark: parallel speedup across worker counts") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    const int iterations = 1000;

    struct Result {
        int workers;
        int evalThreads;
        double itersPerSec;
        double wallSec;
    };

    auto runConfig = [&](int workers, int evalThreads) -> Result {
        ParallelMCTS::Config config;
        config.numWorkerThreads = workers;
        config.numEvalThreads   = evalThreads;
        config.maxIterations    = iterations;
        config.evaluator        = &evaluator;
        config.arenaSize        = 512 * 1024 * 1024;  // 512 MB, enough for 10k iters

        ParallelMCTS mcts(config);
        auto start = std::chrono::high_resolution_clock::now();
        PenteGame::Move move = mcts.search(game);
        auto end   = std::chrono::high_resolution_clock::now();

        CHECK(move.x >= 0);
        CHECK(move.x < PenteGame::BOARD_SIZE);

        double wallSec = std::chrono::duration<double>(end - start).count();
        return {workers, evalThreads, iterations / wallSec, wallSec};
    };

    std::vector<Result> results = {
        runConfig(1, 1),
        runConfig(2, 1),
        runConfig(4, 1),
        runConfig(4, 2),
        runConfig(4, 4),
    };

    std::cout << "\n--- Parallel MCTS Benchmark (" << iterations << " iterations) ---\n";
    std::cout << std::fixed;
    for (auto &r : results) {
        std::cout << std::setprecision(1)
                  << "  " << r.workers << "w / " << r.evalThreads << "e: "
                  << std::setw(8) << r.itersPerSec << " iters/sec"
                  << "  (" << std::setprecision(3) << r.wallSec << "s)\n";
    }

    double baseline = results[0].itersPerSec;
    std::cout << "\n  Speedup vs 1w/1e baseline:\n";
    for (auto &r : results) {
        std::cout << std::setprecision(2)
                  << "    " << r.workers << "w / " << r.evalThreads << "e: "
                  << r.itersPerSec / baseline << "x\n";
    }
    std::cout << "\n";
}
