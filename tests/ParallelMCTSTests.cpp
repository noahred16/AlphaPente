#include "Evaluator.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#ifdef WITH_TORCH
#include <torch/torch.h>
#include <unordered_map>
#endif

// doctest must come AFTER the torch-dependent headers: the c10 logging header
// (pulled in via Evaluator.hpp when WITH_TORCH is set) defines its own CHECK
// macro, which silently shadows doctest's CHECK and turns assertions into no-ops.
#ifdef CHECK
#undef CHECK
#endif
#include "doctest.h"

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
    config.numEvalThreads = 1;  // queue mode: worker pushes to eval queue
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

TEST_CASE("reuseSubtree preserves child visit counts across searches") {
    PenteGame game(PenteGame::Config::pente());
    game.reset();
    game.makeMove("K10");

    HeuristicEvaluator evaluator;
    ParallelMCTS::Config config;
    config.numWorkerThreads = 2;
    config.numEvalThreads = 0;
    config.maxIterations = 200;
    config.evaluator = &evaluator;

    ParallelMCTS mcts(config);
    mcts.search(game);

    PenteGame::Move best = mcts.getBestMove();
    REQUIRE(best.x >= 0);

    // Find the visit count of the best child before reuse
    const auto *root = mcts.getRoot();
    int rootVisits = root->visits.load();
    int childVisits = 0;
    for (int i = 0; i < root->childCapacity; ++i) {
        if (root->moves[i].x == best.x && root->moves[i].y == best.y && root->children[i]) {
            childVisits = root->children[i]->visits.load();
            break;
        }
    }
    REQUIRE(childVisits > 0);

    // Reuse: root_ pivots to the best child without touching the arena
    mcts.reuseSubtree(best);
    game.makeMove(best.x, best.y);

    // getTotalVisits() immediately reflects the child's prior visit count
    REQUIRE(mcts.getTotalVisits() == childVisits);

    // Undo restores the original root
    REQUIRE(mcts.undoSubtree());
    REQUIRE(mcts.getTotalVisits() == rootVisits);
}

// Slow (compares many worker configs) - skipped by default, run with --no-skip
TEST_CASE("Benchmark: parallel speedup across worker counts" * doctest::skip()) {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator evaluator;

    const int iterations = 100000;

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
        config.arenaSize        = 2ull * 1024 * 1024 * 1024;  // 2 GB, enough for 100k iters

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
        // Inline eval (numEvalThreads=0): workers evaluate, expand, backprop directly
        runConfig(1, 0),
        runConfig(2, 0),
        runConfig(4, 0),
        runConfig(6, 0),
        runConfig(8, 0),
        // Queue eval (numEvalThreads>0): producer/consumer pipeline for NN/GPU
        // runConfig(1, 1),
        // runConfig(2, 1),
        // runConfig(4, 2),
        // runConfig(5, 3),
        // runConfig(6, 2),
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
    std::cout << "\n  Speedup vs 1w/0e baseline:\n";
    for (auto &r : results) {
        std::cout << std::setprecision(2)
                  << "    " << r.workers << "w / " << r.evalThreads << "e: "
                  << r.itersPerSec / baseline << "x\n";
    }
    std::cout << "\n";
}

#ifdef WITH_TORCH
// Each MCTS worker thread gets its own NNEvaluator — no shared model, no eval queue.
// Works with numEvalThreads=0 (inline mode): each worker evaluates independently.
class PerWorkerNNEvaluator : public Evaluator {
    static NNEvaluator &local() {
        thread_local NNEvaluator worker;
        return worker;
    }
public:
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>
    evaluate(const PenteGame &game) override { return local().evaluate(game); }
    std::vector<std::pair<PenteGame::Move, float>>
    evaluatePolicy(const PenteGame &game) override { return local().evaluatePolicy(game); }
    float evaluateValue(const PenteGame &game) override { return local().evaluateValue(game); }
    std::vector<std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>>
    evaluateBatch(const std::vector<PenteGame> &games) override { return local().evaluateBatch(games); }
};

// Slow (compares many thread configs) - skipped by default, run with --no-skip
TEST_CASE("Benchmark: NN eval throughput across thread configs" * doctest::skip()) {
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    NNEvaluator sharedEval;
    PerWorkerNNEvaluator perWorkerEval;

    const int iterations = 2000;

    struct Result {
        std::string label;
        int workers;
        int evalThreads;
        int torchThreads;
        double simsPerSec;
        double wallSec;
    };

    auto runWith = [&](Evaluator &eval, int workers, int evalThreads, int torchThreads) -> Result {
        torch::set_num_threads(torchThreads);

        ParallelMCTS::Config config;
        config.numWorkerThreads    = workers;
        config.numEvalThreads      = evalThreads;
        config.evaluationBatchSize = 32;
        config.maxIterations       = iterations;
        config.evaluator           = &eval;

        ParallelMCTS mcts(config);
        auto start = std::chrono::high_resolution_clock::now();
        mcts.search(game);
        auto end = std::chrono::high_resolution_clock::now();

        double wallSec = std::chrono::duration<double>(end - start).count();
        return {"", workers, evalThreads, torchThreads, iterations / wallSec, wallSec};
    };

    std::cout << "\n--- NN Eval Throughput Benchmark (" << iterations << " sims) ---\n";
    std::cout << std::setw(12) << "evaluator"
              << std::setw(8)  << "workers"
              << std::setw(8)  << "eval-t"
              << std::setw(8)  << "torch-t"
              << std::setw(12) << "sims/sec"
              << std::setw(10) << "wall(s)"
              << "\n" << std::string(58, '-') << "\n";

    std::vector<Result> results;

    auto run = [&](int w, int e, int t) {
        auto r = runWith(sharedEval, w, e, t);
        r.label = "shared";
        results.push_back(r);
        std::cout << std::fixed
                  << std::setw(12) << r.label
                  << std::setw(8)  << r.workers
                  << std::setw(8)  << r.evalThreads
                  << std::setw(8)  << r.torchThreads
                  << std::setw(12) << std::setprecision(1) << r.simsPerSec
                  << std::setw(10) << std::setprecision(3) << r.wallSec
                  << "\n";
    };

    auto runPW = [&](int w, int t) {
        auto r = runWith(perWorkerEval, w, 0, t);
        r.label = "per-worker";
        results.push_back(r);
        std::cout << std::fixed
                  << std::setw(12) << r.label
                  << std::setw(8)  << r.workers
                  << std::setw(8)  << r.evalThreads
                  << std::setw(8)  << r.torchThreads
                  << std::setw(12) << std::setprecision(1) << r.simsPerSec
                  << std::setw(10) << std::setprecision(3) << r.wallSec
                  << "\n";
    };

    // Top contenders from initial sweep (all >290 sims/sec at 400 iterations)
    run(1, 4, 2);   // 338 sims/sec — previous best
    run(1, 3, 3);   // 329
    run(1, 6, 1);   // 325
    run(1, 4, 1);   // 317
    run(1, 3, 2);   // 315
    run(1, 2, 2);   // 312
    run(2, 3, 2);   // 308
    run(2, 4, 1);   // 303
    run(2, 2, 3);   // 299
    run(2, 4, 2);   // 294

    // Per-worker: N workers each own their model — no eval queue, no contention
    std::cout << "\n";
    runPW(4, 1);    // 4 workers × 1 torch thread = 4 cores
    runPW(6, 1);    // 6 workers × 1 torch thread = 6 cores
    runPW(8, 1);    // 8 workers × 1 torch thread = 8 cores
    runPW(4, 2);    // 4 workers × 2 torch threads = 8 cores

    double baseline = results[0].simsPerSec;
    std::cout << "\n  Speedup vs first config:\n";
    for (auto &r : results) {
        std::cout << std::setprecision(2)
                  << "    " << r.label << " " << r.workers << "w/" << r.evalThreads << "e"
                  << " torch=" << r.torchThreads << ": "
                  << r.simsPerSec / baseline << "x\n";
    }
    std::cout << "\n";

    torch::set_num_threads(1);
}
#endif
