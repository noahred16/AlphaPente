#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "Profiler.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    bool VERBOSE = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") VERBOSE = true;
    }
    std::cout << "Playing Pente: Uniform (Black) vs Heuristic (White)..." << std::endl;

    PenteGame game(PenteGame::Config::pente());
    game.reset();

    const int warmupIterations = 1000000;
    const int initialIterations = 100000;
    const int targetVisits = 1000000;

    UniformEvaluator uniformEvaluator;
    HeuristicEvaluator heuristicEvaluator;
    size_t arenaSize = GameUtils::arenaSizeFromEnv();

    // Black: uniform policy (flat priors)
    MCTS::Config blackConfig;
    blackConfig.maxIterations = warmupIterations;
    blackConfig.explorationConstant = 1.7;
    blackConfig.searchMode = MCTS::SearchMode::PUCT;
    blackConfig.seed = 42;
    blackConfig.evaluator = &uniformEvaluator;
    blackConfig.arenaSize = arenaSize;
    MCTS blackMcts(blackConfig);

    // White: heuristic policy (evaluateMove priors)
    MCTS::Config whiteConfig;
    whiteConfig.maxIterations = warmupIterations;
    whiteConfig.explorationConstant = 1.7;
    whiteConfig.searchMode = MCTS::SearchMode::PUCT;
    whiteConfig.seed = 42;
    whiteConfig.evaluator = &heuristicEvaluator;
    whiteConfig.arenaSize = arenaSize;
    MCTS whiteMcts(whiteConfig);

    double blackTotalTime = 0.0;
    double whiteTotalTime = 0.0;

    std::vector<std::string> moves;
    // Black opens at center
    game.makeMove("K10");
    moves.push_back("K10");

    // Warmup: both engines build a tree from the opening position.
    // This seeds this->game inside each engine so reuseSubtree works correctly later.
    blackMcts.search(game);
    whiteMcts.search(game);

    while (!game.isGameOver()) {
        if (VERBOSE) GameUtils::printGameState(game);
        auto t0 = std::chrono::high_resolution_clock::now();

        bool isBlackTurn = (game.getCurrentPlayer() == PenteGame::BLACK);
        MCTS &activeMcts = isBlackTurn ? blackMcts : whiteMcts;
        MCTS::Config &activeConfig = isBlackTurn ? blackConfig : whiteConfig;
        MCTS &idleMcts = isBlackTurn ? whiteMcts : blackMcts;
        MCTS::Config &idleConfig = isBlackTurn ? whiteConfig : blackConfig;

        int currentVisits = activeMcts.getTotalVisits();
        int needed = (currentVisits == 0)
            ? initialIterations
            : std::max(0, targetVisits - currentVisits);
        if (VERBOSE) std::cout << "  root visits: " << currentVisits << ", running " << needed << " more\n";
        activeConfig.maxIterations = needed;
        activeMcts.setConfig(activeConfig);
        activeMcts.search(game);
        PenteGame::Move move = activeMcts.getBestMove();

        double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        if (isBlackTurn) {
            if (VERBOSE) std::cout << "Black's turn (Uniform)\n";
            blackTotalTime += elapsed;
        } else {
            if (VERBOSE) std::cout << "White's turn (Heuristic)\n";
            whiteTotalTime += elapsed;
        }
        if (VERBOSE) {
            std::cout << "  move time: " << elapsed << "s"
                      << "  arena: " << std::fixed << std::setprecision(1)
                      << (activeMcts.getArenaUsedBytes() / (1024.0 * 1024.0)) << " MB"
                      << " (" << std::setprecision(1) << activeMcts.getArenaUtilization() << "%)\n";
            activeMcts.printBestMoves(3);
            std::cout << "Selected move: " << GameUtils::displayMove(move.x, move.y) << "\n";
        }

        moves.push_back(GameUtils::displayMove(move.x, move.y));

        // Sync the idle engine's internal game state (0 iters) so its this->game matches
        // the pre-move position before we call reuseSubtree on it.
        idleConfig.maxIterations = 0;
        idleMcts.setConfig(idleConfig);
        idleMcts.search(game);
        idleConfig.maxIterations = targetVisits;
        idleMcts.setConfig(idleConfig);

        game.makeMove(move.x, move.y);
        activeMcts.reuseSubtree(move);
        idleMcts.reuseSubtree(move);
    }

    // if verbose print the list of the moves. format: 1. K10 L10 2. J10 J11 ...
    if (VERBOSE) {
        std::cout << "\nGame over! Final moves:\n";
        for (size_t i = 0; i < moves.size(); i += 2) {
            std::cout << (i / 2 + 1) << ". " << moves[i];
            if (i + 1 < moves.size()) std::cout << " " << moves[i + 1];
            std::cout << "\n";
        }
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
        std::cout << "Winner: Black (Uniform)\n";
    } else if (winner == PenteGame::WHITE) {
        std::cout << "Winner: White (Heuristic)\n";
    } else {
        std::cout << "Draw\n";
    }

    std::cout << "Black (Uniform) total time: " << blackTotalTime << "s\n";
    std::cout << "White (Heuristic) total time: " << whiteTotalTime << "s\n";
    std::cout << "Total time: " << blackTotalTime + whiteTotalTime << "s\n";
    std::cout << "Black arena: " << std::fixed << std::setprecision(1)
              << (blackMcts.getArenaUsedBytes() / (1024.0 * 1024.0)) << " MB used"
              << " (" << blackMcts.getArenaUtilization() << "% of total)\n";
    std::cout << "White arena: " << std::fixed << std::setprecision(1)
              << (whiteMcts.getArenaUsedBytes() / (1024.0 * 1024.0)) << " MB used"
              << " (" << whiteMcts.getArenaUtilization() << "% of total)\n";

    Profiler::instance().printReport();

    return 0;
}
