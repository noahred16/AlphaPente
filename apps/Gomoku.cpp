#include "MCTS.hpp"
#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include <iostream>
#include <cstring>

// How to run: ./gomoku "1. K10 K9 2. K6 L11 3. M8 J11" 100000
//             ./gomoku -d 2 "1. K10 K9" 100000
// Gomoku: 5-in-a-row only, no captures
int main(int argc, char* argv[]) {
    std::cout << "Playing Gomoku (5-in-a-row, no captures)..." << std::endl;

    const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10";

    // Parse optional flags, collect positional args
    int dilationDistance = -1; // -1 = use default
    std::vector<const char*> positional;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            dilationDistance = std::atoi(argv[++i]);
        } else {
            positional.push_back(argv[i]);
        }
    }

    const char* gameDataStr = positional.size() >= 1 ? positional[0] : hardCodedGame;
    int mctsIterations = positional.size() >= 2 ? std::atoi(positional[1]) : 100000;

    // Parse the game data string using GameUtils
    std::vector<std::string> moves = GameUtils::parseGameString(gameDataStr);

    // Show iterations with comma formatting
    std::cout << "Iterations: " << GameUtils::formatWithCommas(mctsIterations) << std::endl;

    // Show parsed moves on same line
    std::cout << "Parsed moves: ";
    for (const auto& moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << std::endl;

    // Game time - use Gomoku config (no captures)
    PenteGame::Config gameConfig = PenteGame::Config::gomoku();
    if (dilationDistance >= 0) gameConfig.dilationDistance = dilationDistance;
    PenteGame game(gameConfig);
    game.reset();

    // Replay the moves
    for (const auto& moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    // MCTS configuration
    MCTS::Config config;
    config.maxIterations = mctsIterations;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    UniformEvaluator uniformEvaluator;
    config.evaluator = &uniformEvaluator;

    MCTS mcts(config);
    GameUtils::interactiveSearchLoop(mcts, game);

    return 0;
}
