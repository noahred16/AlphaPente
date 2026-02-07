#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include <iostream>


int main(int argc, char* argv[]) {
    std::cout << "Playing Pente..." << std::endl;

    // ============================================
    // TOGGLE THIS: true = Heuristic goes first (BLACK)
    //              false = Uniform goes first (BLACK)
    // ============================================
    bool heuristicGoesFirst = false;
    
    PenteGame game(PenteGame::Config::pente());
    game.reset();

    // Setup MCTS with Heuristic Evaluator
    MCTS::Config config1;
    config1.maxIterations = 100000;
    config1.explorationConstant = 1.7;
    config1.searchMode = MCTS::SearchMode::PUCT;
    HeuristicEvaluator heuristicEvaluator;
    config1.evaluator = &heuristicEvaluator;
    MCTS mctsHeuristic(config1);

    // Setup MCTS with Uniform Evaluator
    MCTS::Config config2;
    config2.maxIterations = 100000;
    config2.explorationConstant = 1.7;
    config2.searchMode = MCTS::SearchMode::PUCT;
    UniformEvaluator uniformEvaluator;
    config2.evaluator = &uniformEvaluator;
    MCTS mctsUniform(config2);

    // Assign players based on who goes first
    MCTS* blackPlayer = heuristicGoesFirst ? &mctsHeuristic : &mctsUniform;
    MCTS* whitePlayer = heuristicGoesFirst ? &mctsUniform : &mctsHeuristic;
    std::string blackPlayerName = heuristicGoesFirst ? "HeuristicEvaluator" : "UniformEvaluator";
    std::string whitePlayerName = heuristicGoesFirst ? "UniformEvaluator" : "HeuristicEvaluator";


    // string array of moves to print at each level
    // variable called moves initialized to empty
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
        if (game.getCurrentPlayer() == PenteGame::BLACK) {
            std::cout << "Black's turn (" << blackPlayerName << ")" << std::endl;
            blackPlayer->search(game);
            move = blackPlayer->getBestMove();
        } else {
            std::cout << "White's turn (" << whitePlayerName << ")" << std::endl;
            whitePlayer->search(game);
            move = whitePlayer->getBestMove();
        }
        std::cout << "Selected move: " << GameUtils::displayMove(move.x, move.y) << "\n";
        moves.push_back(GameUtils::displayMove(move.x, move.y));
        game.makeMove(move.x, move.y);
    }
    
    GameUtils::printGameState(game);
    std::cout << "Game over! ";

    // print out moves
    std::cout << "Moves played: ";
    for (const auto& moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << std::endl;

    PenteGame::Player winner = game.getWinner();
    if (winner == PenteGame::BLACK) {
        std::cout << "Black wins! (" << blackPlayerName << ")" << std::endl;
    } else if (winner == PenteGame::WHITE) {
        std::cout << "White wins! (" << whitePlayerName << ")" << std::endl;
    } else {
        std::cout << "It's a draw!" << std::endl;
    }

    return 0;
}