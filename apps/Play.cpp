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
    bool verbose          = false;
    bool heuristicOpponent = false;
    int  simulations      = 800;
    std::string nnPath;

    auto usage = [&]() {
        std::cout <<
            "Usage: play [-s sims] [-v] [-N] [-p model.pt] [-H]\n"
            "\n"
            "  -s <n>      MCTS simulations per move (default: " << simulations << ")\n"
            "  -v          verbose: print board, visit counts, and top moves each turn\n"
            "  -N          use best NN model (checkpoints/pente/best_model.pt)\n"
            "  -p <path>   use a specific .pt model file\n"
            "  -H          heuristic opponent: Black=NN/heuristic, White=heuristic\n"
            "  -h          show this help\n"
            "\n"
            "Runs an AI-vs-AI Pente game so you can inspect the training data distribution.\n";
    };

    int opt;
    while ((opt = getopt(argc, argv, "vs:Np:Hh")) != -1) {
        if      (opt == 'v') verbose            = true;
        else if (opt == 's') simulations        = std::stoi(optarg);
        else if (opt == 'N') nnPath             = PROJECT_ROOT "/checkpoints/pente/best_model.pt";
        else if (opt == 'p') nnPath             = optarg;
        else if (opt == 'H') heuristicOpponent  = true;
        else if (opt == 'h') { usage(); return 0; }
        else                 { usage(); return 1; }
    }

    std::string evalName = nnPath.empty() ? "HeuristicEvaluator" : "NNEvaluator";
    std::string p2Name   = heuristicOpponent ? "HeuristicEvaluator" : evalName;
    std::cout << "Playing Pente (Black: " << evalName << " vs White: " << p2Name
              << ", " << simulations << " sims/move)\n\n";

    PenteGame game(PenteGame::Config::pente());
    game.reset();

    HeuristicEvaluator hEval;
    Evaluator *blackEval = &hEval;

#ifdef WITH_TORCH
    std::unique_ptr<NNEvaluator> nnEval;
    if (!nnPath.empty()) {
        nnEval    = std::make_unique<NNEvaluator>(nnPath);
        blackEval = nnEval.get();
    }
#endif

    auto makeCfg = [&](Evaluator *eval) {
        ParallelMCTS::Config cfg;
        cfg.evaluator           = eval;
        cfg.maxIterations       = simulations;
        cfg.explorationConstant = 1.7;
        cfg.numWorkerThreads    = GameUtils::numThreadsFromEnv();
        cfg.numEvalThreads      = (eval != &hEval) ? 1 : 0;
        cfg.evaluationBatchSize = 512;
        cfg.arenaSize           = GameUtils::arenaSizeFromEnv();
        cfg.seed                = 42;
        return cfg;
    };

    ParallelMCTS::Config blackCfg = makeCfg(blackEval);
    ParallelMCTS::Config whiteCfg = makeCfg(&hEval);

    ParallelMCTS blackMCTS(blackCfg);
    ParallelMCTS whiteMCTS(heuristicOpponent ? whiteCfg : blackCfg);

    std::vector<std::string> moves;
    game.makeMove("K10");
    moves.push_back("K10");

    double totalTime = 0.0;

    while (!game.isGameOver()) {
        if (verbose) GameUtils::printGameState(game);

        bool isBlack = (game.getCurrentPlayer() == PenteGame::BLACK);
        ParallelMCTS &mcts    = (isBlack || !heuristicOpponent) ? blackMCTS : whiteMCTS;
        ParallelMCTS::Config &cfg = (isBlack || !heuristicOpponent) ? blackCfg : whiteCfg;

        auto t0 = std::chrono::high_resolution_clock::now();

        int currentVisits = mcts.getTotalVisits();
        int needed        = std::max(0, simulations - currentVisits);
        if (verbose)
            std::cout << "  [" << (isBlack ? "Black" : "White") << "] "
                      << "root visits: " << currentVisits << ", running " << needed << " more\n";

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
        blackMCTS.reuseSubtree(move);
        if (heuristicOpponent) whiteMCTS.reuseSubtree(move);
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
