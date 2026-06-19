#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unistd.h>

int main(int argc, char *argv[]) {
    bool verbose     = false;
    int  simulations = 20000;
    std::string nnPath;

    int opt;
    while ((opt = getopt(argc, argv, "vs:Np:")) != -1) {
        if      (opt == 'v') verbose     = true;
        else if (opt == 's') simulations = std::stoi(optarg);
        else if (opt == 'N') nnPath      = PROJECT_ROOT "/checkpoints/pente/best_model.pt";
        else if (opt == 'p') nnPath      = optarg;
    }

    std::string evalName = nnPath.empty() ? "HeuristicEvaluator" : "NNEvaluator";
    std::cout << "Playing Pente (" << evalName << ", " << simulations << " sims/move)\n\n";

    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator hEval;
    Evaluator *evalPtr = &hEval;

#ifdef WITH_TORCH
    std::unique_ptr<NNEvaluator> nnEval;
    if (!nnPath.empty()) {
        nnEval  = std::make_unique<NNEvaluator>(nnPath);
        evalPtr = nnEval.get();
    }
#endif

    ParallelMCTS::Config cfg;
    cfg.evaluator           = evalPtr;
    cfg.maxIterations       = simulations;
    cfg.explorationConstant = 1.7;
    cfg.numWorkerThreads    = GameUtils::numThreadsFromEnv();
    cfg.numEvalThreads      = nnPath.empty() ? 0 : 1;
    cfg.evaluationBatchSize = 512;
    cfg.arenaSize           = GameUtils::arenaSizeFromEnv();
    cfg.seed                = 42;

    ParallelMCTS mcts(cfg);

    std::vector<std::string> moves;
    game.makeMove("K10");
    moves.push_back("K10");

    double totalTime = 0.0;

    while (!game.isGameOver()) {
        if (verbose) GameUtils::printGameState(game);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Top up to target rather than restarting from zero
        int currentVisits = mcts.getTotalVisits();
        int needed        = std::max(0, simulations - currentVisits);
        if (verbose)
            std::cout << "  root visits: " << currentVisits << ", running " << needed << " more\n";

        cfg.maxIterations = needed;
        mcts.setConfig(cfg);
        mcts.search(game);

        PenteGame::Move move = mcts.getBestMove();
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        totalTime += elapsed;

        if (verbose) {
            std::cout << "  time: " << std::fixed << std::setprecision(3) << elapsed << "s  "
                      << "visits: " << mcts.getTotalVisits() << "\n";
            mcts.printBestMoves(3);
            std::cout << "  selected: " << GameUtils::displayMove(move.x, move.y) << "\n\n";
        }

        moves.push_back(GameUtils::displayMove(move.x, move.y));
        game.makeMove(move.x, move.y);
        mcts.reuseSubtree(move);
    }

    GameUtils::printGameState(game);

    std::cout << "Moves (" << moves.size() << "): ";
    for (const auto &m : moves) std::cout << m << " ";
    std::cout << "\n\n";

    PenteGame::Player winner = game.getWinner();
    if      (winner == PenteGame::BLACK) std::cout << "Winner: Black\n";
    else if (winner == PenteGame::WHITE) std::cout << "Winner: White\n";
    else                                 std::cout << "Draw\n";

    std::cout << "Total time : " << std::fixed << std::setprecision(2) << totalTime << "s\n";
    std::cout << "Avg/move   : " << std::setprecision(3) << totalTime / moves.size() << "s\n";

    return 0;
}
