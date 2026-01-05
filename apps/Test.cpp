#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "BitBoard.hpp"
#include <iostream>

void moveTwoAnalysis(PenteGame& game);
void setupSimpleOpenThreeThreat(PenteGame& game);
void setupOneSidedFourThreat(PenteGame& game);

int main(int argc, char* argv[]) {
    std::cout << "Testing AlphaPente..." << std::endl;

    // TODO: Implement test logic
    PenteGame game;
    game.reset();


    moveTwoAnalysis(game);
    char expectedMove[] = "K10";
    
    // setupSimpleOpenThreeThreat(game);
    // char expectedMove[] = "M10";

    // setupOneSidedFourThreat(game);
    // char expectedMove[] = "K7";

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
    mcts.printBranch(expectedMove, 10);

    return 0;
}

void moveTwoAnalysis(PenteGame& game) {
    game.makeMove("J10"); // Black
    game.makeMove("K11"); // White
}

void setupSimpleOpenThreeThreat(PenteGame& game) {
    // white one-sided four threat test
    game.makeMove("J10"); // Black
    game.makeMove("C17"); // White
    game.makeMove("K10"); // Black
    game.makeMove("E5"); // White
    game.makeMove("L10"); // Black
    game.makeMove("E15"); // White

    // needs to cover either I10 or M10 to block black win
}

void setupOneSidedFourThreat(PenteGame& game) {
    // white one-sided four threat test
    game.makeMove("J10"); // Black
    game.makeMove("J11"); // White
    game.makeMove("J9"); // Black
    game.makeMove("E5"); // White
    game.makeMove("J8"); // Black
    game.makeMove("E15"); // White

    game.makeMove("J7"); // Black
    game.makeMove("P15"); // White if white doesnt cover black wins
    // K7 is winning move for black
}

