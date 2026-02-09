#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "Evaluator.hpp"
#include "Profiler.hpp"
#include <iostream>
#include <thread>

int main(int argc, char* argv[]) {
    std::cout << "Parallel search test..." << std::endl;
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;

    PenteGame game(PenteGame::Config::pente());
    game.reset();
    std::vector<std::string> openingMoves = {
        // "K10", "M9", "N10", "L10", "N8", "K11", "N9", "N7", "N11", "N12", "M12", "M11", "O13", "L13", "O10", "J12", "H13", "L11", "P9", "Q8", "L9", "L12", "L14", "J11", "K13", "M8", "L9", "J10", "J9", "L12", "H9"
        "K10", "M9", "N10", "L10", "N8", "K11", "N9", "N7", "N11", "N12", "M12", "M11", "O13", "L13", "O10", "J12", "H13", "L11", "P9", "Q8", "L9", "L12", "L14", "J11", "K13", "M8", "L9", 
    };
    for (const auto& moveStr : openingMoves) {
        game.makeMove(moveStr.c_str());
    }


    MCTS::Config config;
    // config.maxIterations = 50000;
    config.maxIterations = 500000;
    config.explorationConstant = 1.7;
    config.searchMode = MCTS::SearchMode::PUCT;
    // config.arenaSize = 64 * 1024 * 1024; // 64 MB


    UniformEvaluator uniformEvaluator;
    config.evaluator = &uniformEvaluator;
    // HeuristicEvaluator evaluator;
    // config.evaluator = &evaluator;

    MCTS mcts(config);

    MCTS::ParallelConfig pconfig;
    pconfig.numWorkers = 2;
    // pconfig.numWorkers = 6;
    pconfig.useInferenceThread = false;

    std::cout << "Starting parallelSearch with " << pconfig.numWorkers << " workers..." << std::endl;
    PenteGame::Move move = mcts.parallelSearch(game, pconfig);
    std::cout << "Done. Move: (" << (int)move.x << ", " << (int)move.y << ")" << std::endl;

    std::cout << "Total visits: " << mcts.getTotalVisits() << std::endl;
    std::cout << "Tree size: " << mcts.getTreeSize() << std::endl;

    mcts.printStats();
    mcts.printBestMoves(5);

    Profiler::instance().printReport();

    return 0;
}
