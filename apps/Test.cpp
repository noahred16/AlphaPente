#include "BitBoard.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Testing AlphaPente..." << std::endl;

    // TODO: Implement test logic
    PenteGame game;
    game.reset();

    // K10 L9 F10 J9 H10 J10 J8 J11 G10 H9 E10 D10 K9 K11 G9 H9 G8 F8 G9 G7 G11 G12 E9 E11 H6 H8 G9 G7 F10 E8 E9 D8 J13
    // H12 G13 H12 J12

    // const char *moves[] = {"K10", "L9", "F10", "J9", "H10", "J10", "J8",  "J11", "G10", "H9",  "E10", "D10", "K9",
    //                        "K11", "G9", "H9",  "G8", "F8",  "G9",  "G7",  "G11", "G12", "E9",  "E11", "H6",  "H8",
    //                        "G9",  "G7", "F10", "E8", "E9",  "D8",  "J13", "H12", "G13", "H12", "J12"};
    // const char *moves[] = {"K10", "L9", "F10", "J9", "H10", "J10", "J8",  "J11", "G10", "H9",  "E10", "D10", "K9",
    //                        "K11", "G9", "H9",  "G8", "F8",  "G9",  "G7",  "G11", "G12", "E9",  "E11", "H6",  "H8",
    //                        "G9",  "G7", "F10", "E8", "E9",  "D8",  "J13", "H12", "G13"};
    // const char *moves[] = {"K10"};
    const char *moves[] = {"K10", "K9"};
    // const char *moves[] = {"K10", "K9", "K7"};
    // const char *moves[] = {"K10", "K9", "K7", "K6"};
    for (const char *move : moves) {
        game.makeMove(move);
    }

    // print
    GameUtils::printGameState(game);

    // Test getting the current player
    // PenteGame::Player currentPlayer = game.getCurrentPlayer();
    // std::cout << "Current player: " << (currentPlayer == PenteGame::BLACK ? "Black" : "White") << std::endl;

    MCTS::Config config;
    // config.maxIterations = 100000;
    config.maxIterations = 10000;
    // config.explorationConstant = 1.414;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    config.seed = 42;
    HeuristicEvaluator heuristicEvaluator;
    config.evaluator = &heuristicEvaluator;

    std::cout << "TEST: Running MCTS search..." << std::endl;
    MCTS mcts(config);
    mcts.search(game);
    mcts.printStats();
    mcts.printBestMoves(10);
    // mcts.printBranch("K10", 10);

    // round n - do 10 loops

    for (int i = 0; i < 10; i++) {
        std::cout << "\n=== Round " << (i + 1) << " ===\n";
        mcts.setConfig(config);
        mcts.search(game);
        mcts.printStats();
        mcts.printBestMoves(10);
    }

    return 0;
}
