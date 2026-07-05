#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <cstring>
#include <iostream>
#include <unistd.h>

// How to run: ./renju "1. K10 L9 2. K12 M10" 100000 [-o <numOffsets>] [-n] [-s]
// Renju: 15x15 board, 5-in-a-row only, no captures (forbidden-move rules not yet implemented)
int main(int argc, char *argv[]) {
    std::cout << "Playing Renju (15x15, 5-in-a-row, no captures)..." << std::endl;

    int numOffsets = 16;
    bool nonInteractive = false;
    bool useSerial = false;
    int opt;
    while ((opt = getopt(argc, argv, "no:s")) != -1) {
        if (opt == 'o') numOffsets = std::atoi(optarg);
        else if (opt == 'n') nonInteractive = true;
        else if (opt == 's') useSerial = true;
    }

    const char *hardCodedGame = "1. K10 L9 2. K12 M10";

    const char *gameDataStr = optind < argc ? argv[optind] : hardCodedGame;
    int mctsIterations = optind + 1 < argc ? std::atoi(argv[optind + 1]) : 100000;

    // Parse the game data string using GameUtils
    std::vector<std::string> moves = GameUtils::parseGameString(gameDataStr);

    // Show iterations with comma formatting
    std::cout << "Iterations: " << GameUtils::formatWithCommas(mctsIterations) << std::endl;

    // Show parsed moves on same line
    std::cout << "Parsed moves: ";
    for (const auto &moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << std::endl;

    // Game time - use Renju config (15x15, no captures)
    PenteGame::Config renjuConfig = PenteGame::Config::renju();
    renjuConfig.numOffsets = numOffsets;
    std::cout << "Num offsets: " << numOffsets << std::endl;
    PenteGame game(renjuConfig);
    game.reset();

    // Replay the moves
    for (const auto &moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    HeuristicEvaluator heuristicEvaluator;

    if (useSerial) {
        MCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = 1.414;
        config.searchMode = MCTS::SearchMode::PUCT;
        config.arenaSize = GameUtils::arenaSizeFromEnv();
        config.evaluator = &heuristicEvaluator;

        MCTS mcts(config);
        if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    } else {
        ParallelMCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = 1.414;
        config.numWorkerThreads = GameUtils::numThreadsFromEnv();
        config.numEvalThreads = 0;
        config.arenaSize = GameUtils::arenaSizeFromEnv(2);
        config.evaluator = &heuristicEvaluator;

        ParallelMCTS mcts(config);
        if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    }

    return 0;
}
