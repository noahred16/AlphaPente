#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <unistd.h>

// How to run: ./pente "1. K10 L9 2. K12 M10" 100000 [-o <numOffsets>] [-n] [-s]
int main(int argc, char *argv[]) {
    int numOffsets = 16;
    int batchSize = 512;
    bool nonInteractive = false;
    bool useSerial = false;
    bool useUniform = false;
    std::string nnPath;
    int opt;
    while ((opt = getopt(argc, argv, "no:suNp:b:h")) != -1) {
        if (opt == 'o') numOffsets = std::atoi(optarg);
        else if (opt == 'n') nonInteractive = true;
        else if (opt == 's') useSerial = true;
        else if (opt == 'u') useUniform = true;
        else if (opt == 'N') nnPath = PROJECT_ROOT "/checkpoints/pente/best_model.pt";
        else if (opt == 'p') nnPath = optarg;
        else if (opt == 'b') batchSize = std::atoi(optarg);
        else if (opt == 'h') {
            std::cout <<
                "Usage: pente [options] [\"move string\"] [iterations]\n"
                "\n"
                "  \"move string\"   PGN-style moves, e.g. \"1. K10 L9 2. K12 M10\"\n"
                "  iterations      MCTS iterations (default: 100000)\n"
                "\n"
                "Options:\n"
                "  -N              Use NN evaluator (checkpoints/pente/best_model.pt)\n"
                "  -p <path>       Use NN evaluator at custom path\n"
                "  -b <size>       Eval batch size (default: 512)\n"
                "  -n              Non-interactive: run search once and exit\n"
                "  -s              Use serial (single-threaded) MCTS\n"
                "  -u              Use uniform random evaluator\n"
                "  -o <n>          Number of move offsets for heuristic (default: 16)\n"
                "  -h              Show this help\n"
                "\n"
                "Environment:\n"
                "  NUM_THREADS     Worker threads for parallel MCTS (default: nproc)\n"
                "  ARENA_SIZE_GB   Tree arena size in GB (default: 2)\n";
            return 0;
        }
    }

    std::cout << "Playing Pente..." << std::endl;

    const char *hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. "
                                "K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

    const char *gameDataStr = optind < argc ? argv[optind] : hardCodedGame;
    int mctsIterations = optind + 1 < argc ? std::atoi(argv[optind + 1]) : 100000;

    // Parse the game data string using GameUtils
    std::vector<std::string> moves = GameUtils::parseGameString(gameDataStr);

    // Show iterations with comma formatting
    std::cout << "Iterations: " << GameUtils::formatWithCommas(mctsIterations) << std::endl;

    // Show parsed moves on same line
    std::cout << "Parsed moves: ";
    for (const auto &moveStr : moves) {
        std::cout << moveStr << " ";
    }
    std::cout << std::endl;

    // Game time - use Pente config (default)
    PenteGame::Config penteConfig = PenteGame::Config::pente();
    penteConfig.numOffsets = numOffsets;
    std::cout << "Num offsets: " << numOffsets << std::endl;
    if (!nnPath.empty()) std::cout << "Evaluator: NN (" << nnPath << ")" << std::endl;
    PenteGame game(penteConfig);
    game.reset();

    // Replay the moves
    for (const auto &moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    GameUtils::printGameState(game);

    // Scale exploration constant based on game phase
    int mc = game.getMoveCount();
    double explorationConstant = mc <= 10 ? 2.5 : mc <= 18 ? 1.8 : 1.414;
    std::cout << "Exploration constant: " << explorationConstant
              << " (move " << mc << ")\n";

    HeuristicEvaluator heuristicEvaluator;
    UniformEvaluator uniformEvaluator;
    Evaluator *evaluator = useUniform ? static_cast<Evaluator *>(&uniformEvaluator) : &heuristicEvaluator;

#ifdef WITH_TORCH
    std::unique_ptr<NNEvaluator> nnEval;
    if (!nnPath.empty()) {
        nnEval = std::make_unique<NNEvaluator>(nnPath);
        evaluator = nnEval.get();
    }
#endif

    if (useSerial) {
        MCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = explorationConstant;
        config.searchMode = MCTS::SearchMode::PUCT;
        config.seed = 42;
        config.arenaSize = GameUtils::arenaSizeFromEnv();
        config.evaluator = evaluator;

        MCTS mcts(config);
        if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    } else {
        ParallelMCTS::Config config;
        config.maxIterations = mctsIterations;
        config.explorationConstant = explorationConstant;
        config.numWorkerThreads = GameUtils::numThreadsFromEnv();
        config.numEvalThreads = nnPath.empty() ? 0 : 1;  // NN: serialize evals through one thread to avoid BLAS conflicts
        config.evaluationBatchSize = batchSize;
        config.arenaSize = GameUtils::arenaSizeFromEnv(2);  // 2 GB default; override with ARENA_SIZE_GB
        config.evaluator = evaluator;
        config.seed = 42;
        // config.warmupIterations = std::min(100000, mctsIterations / 20);
        config.warmupIterations = 100000;

        ParallelMCTS mcts(config);
        if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    }

    return 0;
}
