#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include <iostream>
#include <chrono>


int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    PenteGame game(PenteGame::Config::pente());
    game.reset();

    // Uniform goes first (BLACK) to give it the advantage
    MCTS::Config configUniform;
    configUniform.maxIterations = 100000;
    configUniform.explorationConstant = 1.7;
    configUniform.searchMode = MCTS::SearchMode::PUCT;
    UniformEvaluator uniformEvaluator;
    configUniform.evaluator = &uniformEvaluator;
    MCTS mctsUniform(configUniform);

    MCTS::Config configHeuristic;
    configHeuristic.maxIterations = 100000;
    configHeuristic.explorationConstant = 1.7;
    configHeuristic.searchMode = MCTS::SearchMode::PUCT;
    HeuristicEvaluator heuristicEvaluator;
    configHeuristic.evaluator = &heuristicEvaluator;
    MCTS mctsHeuristic(configHeuristic);

    MCTS* blackPlayer = &mctsUniform;
    MCTS* whitePlayer = &mctsHeuristic;
    const char* blackPlayerName = "Uniform";
    const char* whitePlayerName = "Heuristic";

    double blackTotalTime = 0.0;
    double whiteTotalTime = 0.0;

    std::vector<std::string> moves;
    // Opening moves
    // TODO fix legal moves bug
    game.makeMove("K10"); // Black
    game.makeMove("L9");  // White
    game.makeMove("N10"); // Black
    moves.push_back("K10");
    moves.push_back("L9");
    moves.push_back("N10");

    while (!game.isGameOver()) {
        GameUtils::printGameState(game);
        PenteGame::Move move;
        auto t0 = std::chrono::high_resolution_clock::now();

        if (game.getCurrentPlayer() == PenteGame::BLACK) {
            std::cout << "Black's turn (" << blackPlayerName << ")" << std::endl;
            blackPlayer->clearTree();
            blackPlayer->search(game);
            move = blackPlayer->getBestMove();
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            blackTotalTime += elapsed;
            std::cout << "  move time: " << elapsed << "s\n";
        } else {
            std::cout << "White's turn (" << whitePlayerName << ")" << std::endl;
            whitePlayer->clearTree();
            whitePlayer->search(game);
            move = whitePlayer->getBestMove();
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            whiteTotalTime += elapsed;
            std::cout << "  move time: " << elapsed << "s\n";
        }

        std::cout << "Selected move: " << GameUtils::displayMove(move.x, move.y) << "\n";
        moves.push_back(GameUtils::displayMove(move.x, move.y));
        game.makeMove(move.x, move.y);
    }

    GameUtils::printGameState(game);

    // Print moves
    std::cout << "Moves: ";
    for (const auto& moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << "\n\n";

    // Print result
    PenteGame::Player winner = game.getWinner();
    if (winner == PenteGame::BLACK) {
        std::cout << "Winner: Black (" << blackPlayerName << ")\n";
    } else if (winner == PenteGame::WHITE) {
        std::cout << "Winner: White (" << whitePlayerName << ")\n";
    } else {
        std::cout << "Draw\n";
    }

    std::cout << "Black (" << blackPlayerName << ") total time: " << blackTotalTime << "s\n";
    std::cout << "White (" << whitePlayerName << ") total time: " << whiteTotalTime << "s\n";

    return 0;
}