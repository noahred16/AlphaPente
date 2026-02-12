#include "MCTS.hpp"
#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include <iostream>
#include <cstring>

// How to run: ./pente "1. K10 L9 2. K12 M10" 100000
//             ./pente -d 2 "1. K10 L9" 100000
int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

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

    // Game time - use Pente config (default)
    PenteGame::Config gameConfig = PenteGame::Config::pente();
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
    // config.explorationConstant = 1.414;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    UniformEvaluator uniformEvaluator;
    config.evaluator = &uniformEvaluator;
    // HeuristicEvaluator heuristicEvaluator;
    // config.evaluator = &heuristicEvaluator;

    MCTS mcts(config);
    GameUtils::interactiveSearchLoop(mcts, game);

    return 0;
}
