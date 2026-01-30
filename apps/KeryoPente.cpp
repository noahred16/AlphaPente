#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <iostream>
#include <chrono>

// How to run: ./keryopente "1. K10 K9 2. K6 L11 3. M8 J11" 100000
// Keryo-Pente: captures enabled, 3-stone captures, 15 captures to win
int main(int argc, char* argv[]) {
    std::cout << "Playing Keryo-Pente (3-stone captures, 15 to win)..." << std::endl;

    const char* hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10";

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

    // Game time - use Keryo-Pente config
    PenteGame game(PenteGame::Config::keryoPente());
    game.reset();

    // Replay the moves
    for (const auto& moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    // MCTS configuration
    MCTS::Config config;
    config.maxIterations = mctsIterations;
    config.explorationConstant = 1.414;

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
