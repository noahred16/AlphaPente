#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>

// How to run: ./pente "1. K10 L9 2. K12 M10" 100000 [-o <numOffsets>] [-n] [-s]
int main(int argc, char *argv[]) {
    int numOffsets = 16;
    int batchSize = 512;
    int boardSize = 19;
    bool nonInteractive = false;
    bool useSerial = false;
    bool useUniform = false;
    bool jsonOutput = false;
    bool boardOnly = false;
    std::string promisingMovesArg;
    std::string nnPath;
    int opt;
    while ((opt = getopt(argc, argv, "no:suNp:b:B:jDM:h")) != -1) {
        if (opt == 'o') numOffsets = std::atoi(optarg);
        else if (opt == 'n') nonInteractive = true;
        else if (opt == 's') useSerial = true;
        else if (opt == 'j') { jsonOutput = true; nonInteractive = true; }
        else if (opt == 'D') boardOnly = true;
        else if (opt == 'M') promisingMovesArg = optarg;
        else if (opt == 'u') useUniform = true;
        else if (opt == 'N') nnPath = PROJECT_ROOT "/checkpoints/pente/best_model.pt";
        else if (opt == 'p') nnPath = optarg;
        else if (opt == 'b') batchSize = std::atoi(optarg);
        else if (opt == 'B') boardSize = std::max(1, std::min(19, std::atoi(optarg)));
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
                "  -B <size>       Board size, NxN centered board (default: 19)\n"
                "  -n              Non-interactive: run search once and exit\n"
                "  -s              Use serial (single-threaded) MCTS\n"
                "  -u              Use uniform random evaluator\n"
                "  -o <n>          Number of move offsets for heuristic (default: 16)\n"
                "  -j              Print one JSON object (search stats + top moves) instead\n"
                "                  of human-readable text; implies -n\n"
                "  -D              Print the board for the given position and exit (no search)\n"
                "  -M <moves>      With -D, comma-separated moves to mark on the board as\n"
                "                  ranked candidates, e.g. -M \"K13,H13,G10\"\n"
                "  -h              Show this help\n"
                "\n"
                "Environment:\n"
                "  NUM_THREADS     Worker threads for parallel MCTS (default: nproc)\n"
                "  ARENA_SIZE_GB   Tree arena size in GB (default: 2)\n";
            return 0;
        }
    }

    bool quiet = jsonOutput || boardOnly;

    if (!quiet) std::cout << "Playing Pente..." << std::endl;

    const char *hardCodedGame = "1. K10 L9 2. G10 L7 3. M10 L8 4. L10 J10 5. J12 L6 6. L5 K9 7. H11 K13 8. K11 K12 9. "
                                "K11 M9 10. F9 E8 11. K14 K13 12. H13 G14 13. N9 M7 14. N6 K7 15. N10";

    const char *gameDataStr = optind < argc ? argv[optind] : hardCodedGame;
    int mctsIterations = optind + 1 < argc ? std::atoi(argv[optind + 1]) : 100000;

    // Parse the game data string using GameUtils
    std::vector<std::string> moves = GameUtils::parseGameString(gameDataStr);

    if (!quiet) {
        // Show iterations with comma formatting
        std::cout << "Iterations: " << GameUtils::formatWithCommas(mctsIterations) << std::endl;

        // Show parsed moves on same line
        std::cout << "Parsed moves: ";
        for (const auto &moveStr : moves) {
            std::cout << moveStr << " ";
        }
        std::cout << std::endl;
    }

    // Game time - use Pente config (default)
    PenteGame::Config penteConfig = PenteGame::Config::pente();
    penteConfig.numOffsets = numOffsets;
    penteConfig.boardSize = boardSize;
    if (boardSize < 7 && penteConfig.tournamentRule) {
        // Tournament rule (3rd-move restriction) is a fixed distance-3 ring around
        // center; it doesn't fit inside a board smaller than 7x7.
        penteConfig.tournamentRule = false;
        if (!quiet) std::cout << "Board size " << boardSize << " < 7: tournament rule doesn't fit, disabling it.\n";
    }
    if (!quiet) {
        if (boardSize != 19) std::cout << "Board size: " << boardSize << "x" << boardSize << std::endl;
        std::cout << "Num offsets: " << numOffsets << std::endl;
        if (!nnPath.empty()) std::cout << "Evaluator: NN (" << nnPath << ")" << std::endl;
    }
    PenteGame game(penteConfig);
    game.reset();

    // Replay the moves
    for (const auto &moveStr : moves) {
        game.makeMove(moveStr.c_str());
    }

    if (boardOnly) {
        std::vector<std::pair<int, int>> promising;
        std::stringstream ss(promisingMovesArg);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto [px, py] = GameUtils::parseMove(token.c_str());
            if (px >= 0 && py >= 0) promising.push_back({px, py});
        }
        PenteGame::Move lastMove = game.getLastMove();
        GameUtils::printGameState(game, lastMove.x, lastMove.y, promising);
        return 0;
    }

    if (!quiet) {
        PenteGame::Move lastMove = game.getLastMove();
        GameUtils::printGameState(game, lastMove.x, lastMove.y);
    }

    // Scale exploration constant based on game phase
    int mc = game.getMoveCount();
    double explorationConstant = GameUtils::explorationConstantForMoveCount(mc);
    if (!quiet)
        std::cout << "Exploration constant: " << explorationConstant
                  << " (move " << mc << ")\n" << std::flush;

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
        if (jsonOutput)
            GameUtils::runSearchAndReportJSON(mcts, game);
        else if (nonInteractive)
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

        ParallelMCTS mcts(config);
        if (jsonOutput)
            GameUtils::runSearchAndReportJSON(mcts, game);
        else if (nonInteractive)
            GameUtils::runSearchAndReport(mcts, game);
        else
            GameUtils::interactiveSearchLoop(mcts, game);
    }

    return 0;
}
