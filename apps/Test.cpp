#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "BitBoard.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Testing AlphaPente..." << std::endl;

    // TODO: Implement test logic
    PenteGame game;
    game.reset();

    // white open 3 test
    game.makeMove(9, 9); // Black
    game.makeMove(9, 10); // White
    game.makeMove(9, 8); // Black
    game.makeMove(4, 4); // White
    game.makeMove(9, 7); // Black
    game.makeMove(4, 14); // White
    game.makeMove(9, 6); // Black
    game.makeMove(15, 14); // White if white doesnt cover black wins


    // print
    game.print();

    // Test getting the current player
    // PenteGame::Player currentPlayer = game.getCurrentPlayer();
    // std::cout << "Current player: " << (currentPlayer == PenteGame::BLACK ? "Black" : "White") << std::endl;


    MCTS::Config config;
    config.maxIterations = 10000;
    config.explorationConstant = 1.414;

    MCTS mcts(config);
    PenteGame::Move bestMove = mcts.search(game);
    mcts.printStats();
    mcts.printBestMoves(10);
    mcts.printBranch(10, 6, 10);

    return 0;
}
