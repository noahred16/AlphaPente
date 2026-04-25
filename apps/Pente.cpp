#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <unistd.h>

// How to run: ./pente "1. K10 L9 2. K12 M10" 100000 [-o <numOffsets>] [-n] [-p]
int main(int argc, char *argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    int numOffsets = 16;
    bool nonInteractive = false;
    bool useParallel = false;
    int opt;
    while ((opt = getopt(argc, argv, "no:p")) != -1) {
        if (opt == 'o') numOffsets = std::atoi(optarg);
        else if (opt == 'n') nonInteractive = true;
        else if (opt == 'p') useParallel = true;
    }

    if (useParallel && !nonInteractive) {
        std::cerr << "Error: -p (parallel) requires -n (non-interactive) mode\n";
        return 1;
    }

    const char *hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. "
                                "K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

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

    // Game time - use Pente config (default)
    PenteGame::Config penteConfig = PenteGame::Config::pente();
    penteConfig.numOffsets = numOffsets;
    std::cout << "Num offsets: " << numOffsets << std::endl;
    PenteGame game(penteConfig);
    game.reset();

    // Replay the moves
    for (const auto &moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    HeuristicEvaluator heuristicEvaluator;

    if (useParallel) {
        ParallelMCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = 1.414;
        config.numWorkerThreads = 5;
        config.numEvalThreads = 0;  // 0 = inline eval (CPU heuristic); set >0 for NN/GPU
        config.arenaSize = GameUtils::arenaSizeFromEnv(1);  // 1 GB default; override with ARENA_SIZE_GB
        config.evaluator = &heuristicEvaluator;

        auto wallStart = std::chrono::high_resolution_clock::now();
        ParallelMCTS mcts(config);
        PenteGame::Move bestMove = mcts.search(game);
        auto wallEnd = std::chrono::high_resolution_clock::now();
        double wallElapsed = std::chrono::duration<double>(wallEnd - wallStart).count();

        mcts.printStats(wallElapsed);
        mcts.printBestMoves(15);
        std::cout << "MCTS selected move: " << GameUtils::displayMove(bestMove.x, bestMove.y) << std::endl;
    } else {
        MCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = 1.414;
        config.searchMode = MCTS::SearchMode::PUCT;
        config.seed = 42;
        config.arenaSize = GameUtils::arenaSizeFromEnv();
        config.evaluator = &heuristicEvaluator;

        MCTS mcts(config);
        if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    }

    return 0;
}
