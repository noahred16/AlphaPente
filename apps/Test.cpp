#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "BitBoard.hpp"
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include <iostream>
#include <chrono>

void moveTwoAnalysis(PenteGame& game);
void setupSimpleOpenThreeThreat(PenteGame& game);
void setupOneSidedFourThreat(PenteGame& game);

int main(int argc, char* argv[]) {
    std::cout << "Testing AlphaPente..." << std::endl;

    // TODO: Implement test logic
    PenteGame game;
    game.reset();


    const char* moves[] = {"K10", "K9", "K7"};
    for (const char* move : moves) {
        game.makeMove(move);
    }


    GameUtils::printGameState(game);

    MCTS::Config config;
    // config.maxIterations = 5;
    config.maxIterations = 1000;
    // config.maxIterations = 10000;
    config.explorationConstant = 1.414;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    HeuristicEvaluator heuristicEvaluator;
    config.evaluator = &heuristicEvaluator;

    MCTS mcts(config);
    // starting at {curr_time} using built in cpp time 
    std::cout << "Starting at " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl; 
    std::cout << "Running MCTS search..." << std::endl;
    mcts.search(game);
    mcts.printStats();
    mcts.printBestMoves(10);
    std::cout << "Finished at " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
    // mcts.printBranch("L10", 10);

    return 0;
}

