#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "Profiler.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

static constexpr bool VERBOSE = false;

int main(int argc, char *argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    PenteGame game(PenteGame::Config::pente());
    game.reset();

    // const int initialIterations = 1000000;
    // const int targetVisits = 200000;
    const int initialIterations = 10000;
    const int targetVisits = 10000;

    MCTS::Config config;
    config.maxIterations = initialIterations;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    config.seed = 42;
    HeuristicEvaluator heuristicEvaluator;
    config.evaluator = &heuristicEvaluator;
    MCTS mcts(config);

    double blackTotalTime = 0.0;
    double whiteTotalTime = 0.0;

    std::vector<std::string> moves;
    // Black opens at center
    game.makeMove("K10");
    moves.push_back("K10");

    while (!game.isGameOver()) {
        if (VERBOSE) GameUtils::printGameState(game);
        auto t0 = std::chrono::high_resolution_clock::now();

        int currentVisits = mcts.getTotalVisits();
        int needed = (currentVisits == 0)
            ? initialIterations
            : std::max(0, targetVisits - currentVisits);
        if (VERBOSE) std::cout << "  root visits: " << currentVisits << ", running " << needed << " more\n";
        config.maxIterations = needed;
        mcts.setConfig(config);
        mcts.search(game);
        PenteGame::Move move = mcts.getBestMove();

        double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        if (game.getCurrentPlayer() == PenteGame::BLACK) {
            if (VERBOSE) std::cout << "Black's turn (Heuristic)\n";
            blackTotalTime += elapsed;
        } else {
            if (VERBOSE) std::cout << "White's turn (Heuristic)\n";
            whiteTotalTime += elapsed;
        }
        if (VERBOSE) {
            std::cout << "  move time: " << elapsed << "s"
                      << "  arena: " << std::fixed << std::setprecision(1)
                      << (mcts.getArenaUsedBytes() / (1024.0 * 1024.0)) << " MB"
                      << " (" << std::setprecision(1) << mcts.getArenaUtilization() << "%)\n";
            mcts.printBestMoves(3);
            std::cout << "Selected move: " << GameUtils::displayMove(move.x, move.y) << "\n";
        }

        moves.push_back(GameUtils::displayMove(move.x, move.y));
        game.makeMove(move.x, move.y);
        mcts.reuseSubtree(move);
    }

    GameUtils::printGameState(game);

    // Print moves
    std::cout << "Moves (" << moves.size() << "): ";
    for (const auto &moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << "Number of moves: " << moves.size() << "\n\n";

    // Print result
    PenteGame::Player winner = game.getWinner();
    if (winner == PenteGame::BLACK) {
        std::cout << "Winner: Black\n";
    } else if (winner == PenteGame::WHITE) {
        std::cout << "Winner: White\n";
    } else {
        std::cout << "Draw\n";
    }

    std::cout << "Black total time: " << blackTotalTime << "s\n";
    std::cout << "White total time: " << whiteTotalTime << "s\n";
    std::cout << "Total time: " << blackTotalTime + whiteTotalTime << "s\n";
    std::cout << "Arena final: " << std::fixed << std::setprecision(1)
              << (mcts.getArenaUsedBytes() / (1024.0 * 1024.0)) << " MB used"
              << " (" << mcts.getArenaUtilization() << "% of total)\n";

    Profiler::instance().printReport();

    return 0;
}
