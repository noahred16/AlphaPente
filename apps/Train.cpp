#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include "ParallelMCTS.hpp"
#include "Profiler.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Training AlphaPente..." << std::endl;

    // Setup
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator heuristicEvaluator;

    ParallelMCTS::Config config;
    config.maxIterations = 100;
    config.explorationConstant = 1.7;
    config.searchMode = ParallelMCTS::SearchMode::PUCT;
    config.seed = 42;
    config.evaluator = &heuristicEvaluator;
    config.canonicalHashDepth = 0;
    config.numWorkers = 4;
    
    // Search
    ParallelMCTS parallelMcts(config);
    parallelMcts.search(game);

    std::cout << "Training complete!" << std::endl;

    return 0;
}
