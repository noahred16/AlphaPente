#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "BitBoard.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Testing AlphaPente..." << std::endl;

    // TODO: Implement test logic
    PenteGame game;
    game.reset();

    game.makeMove(9, 9); // Black
    game.makeMove(10, 9); // White

    // print
    game.print();

    // undo
    game.undoMove();
    game.print();


    // Test getting the current player
    PenteGame::Player currentPlayer = game.getCurrentPlayer();
    std::cout << "Current player: " << (currentPlayer == PenteGame::BLACK ? "Black" : "White") << std::endl;

    return 0;
}
