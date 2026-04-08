#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include <iostream>
#include <unistd.h>

// How to run: ./keryopente "1. K10 K9 2. K6 L11 3. M8 J11" 100000 [-o <numOffsets>]
// Keryo-Pente: captures enabled, 3-stone captures, 15 captures to win
int main(int argc, char *argv[]) {
    std::cout << "Playing Keryo-Pente (3-stone captures, 15 to win)..." << std::endl;

    int numOffsets = 16;
    int opt;
    while ((opt = getopt(argc, argv, "o:")) != -1) {
        if (opt == 'o') numOffsets = std::atoi(optarg);
    }

    const char *hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10";

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

    // Game time - use Keryo-Pente config
    PenteGame::Config keryoConfig = PenteGame::Config::keryoPente();
    keryoConfig.numOffsets = numOffsets;
    std::cout << "Num offsets: " << numOffsets << std::endl;
    PenteGame game(keryoConfig);
    game.reset();

    // Replay the moves
    for (const auto &moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    // MCTS configuration
    MCTS::Config config;
    config.maxIterations = mctsIterations;
    config.explorationConstant = 1.414;
    config.searchMode = MCTS::SearchMode::PUCT;
    // UniformEvaluator uniformEvaluator;
    // config.evaluator = &uniformEvaluator;
    HeuristicEvaluator heuristicEvaluator;
    config.evaluator = &heuristicEvaluator;

    MCTS mcts(config);
    GameUtils::interactiveSearchLoop(mcts, game);

    return 0;
}
