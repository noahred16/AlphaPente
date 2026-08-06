// Standalone repro harness: force MCTS arena exhaustion repeatedly with a tiny
// arena so ThreadSanitizer can catch the race near exhaustion quickly, instead
// of waiting for a real 29GB arena to fill.
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
    int trials = argc > 1 ? std::atoi(argv[1]) : 30;
    size_t arenaBytes = argc > 2 ? static_cast<size_t>(std::atoll(argv[2])) : (8ull * 1024 * 1024);
    int numThreads = argc > 3 ? std::atoi(argv[3]) : 4;

    const char *hardCodedGame =
        "1.  H8 H7  2.  H6 J7  3.  G7 J9  4.  J5 K4  5.  F8 E9  6.  G8 J8  "
        "7.  J10 H9  8.  G9 G6  9.  F5 G10  10.  K7 F6  11.  K9 E10  12.  E8 D8 ";

    std::vector<std::string> moves = GameUtils::parseGameString(hardCodedGame);

    for (int t = 0; t < trials; ++t) {
        PenteGame::Config renjuConfig = PenteGame::Config::renju();
        renjuConfig.numOffsets = 16;
        PenteGame game(renjuConfig);
        game.reset();
        for (const auto &moveStr : moves) game.makeMove(moveStr.c_str());

        HeuristicEvaluator heuristicEvaluator;

        ParallelMCTS::Config config;
        config.maxIterations = 50000000;  // intentionally far beyond what the tiny arena can hold
        config.explorationConstant = 1.414;
        config.numWorkerThreads = numThreads;
        config.numEvalThreads = 0;
        config.arenaSize = arenaBytes;
        config.evaluator = &heuristicEvaluator;

        std::cout << "=== trial " << t << " (arena=" << arenaBytes << " bytes, threads=" << numThreads << ") ===" << std::endl;
        ParallelMCTS mcts(config);
        PenteGame::Move best = mcts.search(game);
        std::cout << "trial " << t << " OK, best=(" << best.x << "," << best.y << ")" << std::endl;
    }
    std::cout << "ALL TRIALS COMPLETED" << std::endl;
    return 0;
}
