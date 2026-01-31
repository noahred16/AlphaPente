#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <iostream>
#include <chrono>

// How to run: ./pente "1. K10 K9 2. K6 L11 3. M8 J11 4. L7 N9 5. J5 H4" 100000
int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

    // use argv[1] if provided, else use hardcoded
    const char* gameDataStr = (argc >= 2) ? argv[1] : hardCodedGame;

    // use argv[2] if provided, else default to 100,000
    int mctsIterations = (argc >= 3) ? std::atoi(argv[2]) : 100000;

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
    PenteGame game(PenteGame::Config::pente());
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

    MCTS mcts(config);
    auto start = std::chrono::high_resolution_clock::now();
    mcts.search(game);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    int minutes = elapsed.count() / 60;
    int seconds = elapsed.count() % 60;
    std::cout << "Search took: " << minutes << " min " << seconds << " sec." << std::endl;
    mcts.printStats();
    mcts.printBestMoves(15);

    // make move
    PenteGame::Move bestMove = mcts.getBestMove();
    std::string bestMoveStr = GameUtils::displayMove(bestMove.x, bestMove.y);
    std::cout << "MCTS selected move: " << bestMoveStr << std::endl;

    std::cout << '\a' << std::flush; // Terminal bell

    // Print profiler report
    Profiler::instance().printReport();

    return 0;
}
