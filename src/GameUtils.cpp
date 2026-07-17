#include "GameUtils.hpp"
#include "MCTS.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include "Profiler.hpp"
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <csignal>
#include <fstream>
#include <unistd.h>

namespace {

struct CrashContext {
    MCTS *mcts = nullptr;
    std::chrono::high_resolution_clock::time_point wallStart;
    std::clock_t cpuStart = 0;
    bool active = false;
} g_crashCtx;

struct ParallelCrashContext {
    ParallelMCTS *mcts = nullptr;
    std::chrono::high_resolution_clock::time_point wallStart;
    bool active = false;
} g_parallelCrashCtx;

void printCrashSummary() {
    if (!g_crashCtx.active || !g_crashCtx.mcts) return;
    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallElapsed = std::chrono::duration<double>(wallEnd - g_crashCtx.wallStart).count();
    double cpuElapsed = static_cast<double>(std::clock() - g_crashCtx.cpuStart) / CLOCKS_PER_SEC;
    g_crashCtx.mcts->printStats(wallElapsed, cpuElapsed);
    g_crashCtx.mcts->printBestMoves(5);
    PenteGame::Move best = g_crashCtx.mcts->getBestMove();
    std::string bestStr = GameUtils::displayMove(best.x, best.y);
    std::cout << "MCTS selected move: " << bestStr << std::endl;
}

void crashSignalHandler(int sig) {
    const char *msg = "\n[Signal] Printing search summary before exit:\n";
    if (write(STDERR_FILENO, msg, 48) < 0) {}  // best-effort, ignore errors in signal handler
    printCrashSummary();
    signal(sig, SIG_DFL);
    raise(sig);
}

void printParallelCrashSummary() {
    if (!g_parallelCrashCtx.active || !g_parallelCrashCtx.mcts) return;
    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallElapsed = std::chrono::duration<double>(wallEnd - g_parallelCrashCtx.wallStart).count();
    g_parallelCrashCtx.mcts->stopWorkerThreads();
    g_parallelCrashCtx.mcts->stopEvalThreads();
    g_parallelCrashCtx.mcts->printStats(wallElapsed);
    g_parallelCrashCtx.mcts->printBestMoves(5);
    PenteGame::Move best = g_parallelCrashCtx.mcts->getBestMove();
    std::cout << "MCTS selected move: " << GameUtils::displayMove(best.x, best.y) << std::endl;
}

void parallelCrashSignalHandler(int sig) {
    const char *msg = "\n[Signal] Printing search summary before exit:\n";
    if (write(STDERR_FILENO, msg, 48) < 0) {}
    printParallelCrashSummary();
    signal(sig, SIG_DFL);
    raise(sig);
}

} // namespace

static size_t parseGbValue(const char *val) {
    size_t gb = static_cast<size_t>(std::atoll(val));
    return gb > 0 ? gb * 1024ULL * 1024 * 1024 : 0;
}

static const char *readKeyFromDotEnv(const std::string &key, std::string &out) {
    for (const char *path : {".env", "../.env"}) {
        std::ifstream f(path);
        if (!f) continue;
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind(key, 0) == 0 && line.size() > key.size()) {
                out = line.substr(key.size());
                return out.c_str();
            }
        }
    }
    return nullptr;
}

static size_t readFromDotEnv() {
    std::string val;
    const char *p = readKeyFromDotEnv("ARENA_SIZE_GB=", val);
    if (p) {
        size_t bytes = parseGbValue(p);
        if (bytes > 0) return bytes;
    }
    return 0;
}

int GameUtils::numThreadsFromEnv() {
    const char *val = std::getenv("NUM_THREADS");
    if (val && std::strlen(val) > 0) {
        int n = std::atoi(val);
        if (n > 0) return n;
    }
    std::string buf;
    const char *p = readKeyFromDotEnv("NUM_THREADS=", buf);
    if (p) {
        int n = std::atoi(p);
        if (n > 0) return n;
    }
    std::cerr << "Error: NUM_THREADS not set. Add it to .env or set the environment variable.\n";
    std::exit(1);
}

double GameUtils::explorationConstantForMoveCount(int moveCount) {
    return moveCount <= 10 ? 2.5 : moveCount <= 18 ? 1.8 : 1.414;
}

size_t GameUtils::arenaSizeFromEnv(size_t defaultGb) {
    const char *val = std::getenv("ARENA_SIZE_GB");
    if (val && std::strlen(val) > 0) {
        size_t bytes = parseGbValue(val);
        if (bytes > 0) return bytes;
    }
    size_t bytes = readFromDotEnv();
    if (bytes > 0) return bytes;
    return defaultGb * 1024ULL * 1024 * 1024;
}

std::pair<int, int> GameUtils::parseMove(const char *move) {
    if (strlen(move) < 2) {
        return {-1, -1};
    }

    char colChar = move[0];
    if (colChar >= 'I') {
        colChar--; // Skip 'I'
    }
    int x = colChar - 'A';

    int y = std::atoi(move + 1) - 1;

    return {x, y};
}

std::string GameUtils::displayMove(int x, int y) {
    char colChar = 'A' + x;
    if (colChar >= 'I') {
        colChar++; // Skip 'I'
    }
    return std::string(1, colChar) + std::to_string(y + 1);
}

std::vector<std::string> GameUtils::parseGameString(const char *gameStr) {
    std::vector<std::string> moves;
    char *gameDataCopy = strdup(gameStr);
    char *token = std::strtok(gameDataCopy, " \t\n\r");

    while (token != nullptr) {
        // Skip move numbers (e.g. "1.", "2.", "1", "2") and dash separators (e.g. "-")
        bool isMoveNumber = std::isdigit((unsigned char)token[0]);
        bool isDash = std::strcmp(token, "-") == 0;
        if (!isMoveNumber && !isDash) {
            moves.push_back(std::string(token));
        }
        token = std::strtok(nullptr, " \t\n\r");
    }
    free(gameDataCopy);

    return moves;
}

void GameUtils::printBoard(const PenteGame &game) {
    const auto &legalMoves = game.getLegalMoves();

    // Helper to handle skipping 'I'
    auto getColChar = [](int x) {
        char c = (char)('A' + x);
        return (c >= 'I') ? (char)(c + 1) : c;
    };

    std::cout << "   ";
    for (int x = 0; x < PenteGame::BOARD_SIZE; x++) {
        std::cout << getColChar(x) << " ";
    }
    std::cout << "\n";

    for (int y = PenteGame::BOARD_SIZE - 1; y >= 0; y--) {
        std::cout << (y < 9 ? " " : "") << (y + 1) << " ";
        for (int x = 0; x < PenteGame::BOARD_SIZE; x++) {
            PenteGame::Player stone = game.getStoneAt(x, y);
            if (stone == PenteGame::BLACK) {
                std::cout << "\u25CB "; // White circle for Black stones
            } else if (stone == PenteGame::WHITE) {
                std::cout << "\u25CF "; // Black circle for White stones
            } else {
                bool isLegal = false;
                for (const auto &move : legalMoves) {
                    if (move.x == x && move.y == y) {
                        isLegal = true;
                        break;
                    }
                }
                std::cout << (isLegal ? "  " : "\u00B7 ");
            }
        }
        std::cout << (y + 1) << "\n";
    }

    std::cout << "   ";
    for (int x = 0; x < PenteGame::BOARD_SIZE; x++) {
        std::cout << getColChar(x) << " ";
    }
    std::cout << "\n";
}

void GameUtils::printGameState(const PenteGame &game) {
    printBoard(game);

    const PenteGame::Config &config = game.getConfig();

    if (config.capturesEnabled) {
        std::cout << game.getBlackCaptures() << "/" << config.capturesToWin << " Black \u25CB, "
                  << game.getWhiteCaptures() << "/" << config.capturesToWin << " White \u25CF, ";
    }

    std::cout << "Current player: " << (game.getCurrentPlayer() == PenteGame::BLACK ? "Black" : "White") << "\n";
}

std::string GameUtils::formatWithCommas(int value) {
    std::string num = std::to_string(value);
    std::string result;
    int count = 0;
    for (int i = num.length() - 1; i >= 0; --i) {
        if (count > 0 && count % 3 == 0)
            result = ',' + result;
        result = num[i] + result;
        ++count;
    }
    return result;
}

void GameUtils::runSearchAndReport(ParallelMCTS &mcts, const PenteGame &game) {
    auto wallStart = std::chrono::high_resolution_clock::now();

    g_parallelCrashCtx = {&mcts, wallStart, true};
    signal(SIGTERM, parallelCrashSignalHandler);
    signal(SIGINT,  parallelCrashSignalHandler);

    PenteGame::Move bestMove = mcts.search(game);

    signal(SIGTERM, SIG_DFL);
    signal(SIGINT,  SIG_DFL);
    g_parallelCrashCtx.active = false;

    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallElapsed = std::chrono::duration<double>(wallEnd - wallStart).count();

    mcts.printStats(wallElapsed);
    mcts.printBestMoves(5);
    std::cout << "MCTS selected move: " << displayMove(bestMove.x, bestMove.y) << std::endl;
    std::cout << '\a' << std::flush;
}

void GameUtils::runSearchAndReport(MCTS &mcts, const PenteGame &game) {
    auto wallStart = std::chrono::high_resolution_clock::now();
    std::clock_t cpuStart = std::clock();

    g_crashCtx = {&mcts, wallStart, cpuStart, true};
    signal(SIGTERM, crashSignalHandler);
    signal(SIGABRT, crashSignalHandler);
    signal(SIGSEGV, crashSignalHandler);

    try {
        mcts.search(game);
    } catch (const std::exception &e) {
        std::cerr << "\nSearch interrupted: " << e.what() << "\n";
    }

    signal(SIGTERM, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    g_crashCtx.active = false;

    std::clock_t cpuEnd = std::clock();
    auto wallEnd = std::chrono::high_resolution_clock::now();

    double wallElapsed = std::chrono::duration<double>(wallEnd - wallStart).count();
    double cpuElapsed = static_cast<double>(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    mcts.printStats(wallElapsed, cpuElapsed);
    mcts.printBestMoves(5);
    PenteGame::Move bestMove = mcts.getBestMove();
    std::string bestMoveStr = displayMove(bestMove.x, bestMove.y);
    std::cout << "MCTS selected move: " << bestMoveStr << std::endl;
    std::cout << '\a' << std::flush;
}

template <typename MCTSType>
void GameUtils::interactiveSearchLoop(MCTSType &mcts, PenteGame game) {
    runSearchAndReport(mcts, game);

    Profiler::instance().printReport();

    int iterationsToAdd = mcts.getConfig().maxIterations;
    std::vector<PenteGame> gameHistory;

    while (iterationsToAdd > 0) {
        std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        std::cout << "Enter iterations (0 to quit, 1-9 for millions, enter for default "
                  << formatWithCommas(iterationsToAdd) << "), move (e.g. K10), or -1 to undo: ";
        std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

        std::string input;
        std::getline(std::cin, input);

        if (input.empty()) {
            // Default: run more iterations
        } else if (std::isalpha(input[0])) {
            // Input looks like a move
            auto [mx, my] = parseMove(input.c_str());
            if (mx < 0 || my < 0 || !game.isLegalMove(mx, my)) {
                std::cout << "Invalid or illegal move: " << input << std::endl;
                continue;
            }
            gameHistory.push_back(game);
            PenteGame::Move move(mx, my);
            game.makeMove(mx, my);
            mcts.reuseSubtree(move);
            std::cout << "Played " << input << ", reusing subtree." << std::endl;
            printGameState(game);
        } else {
            try {
                int val = std::stoi(input);
                if (val == -1) {
                    if (mcts.undoSubtree() && !gameHistory.empty()) {
                        game = gameHistory.back();
                        gameHistory.pop_back();
                        std::cout << "Undid last move." << std::endl;
                        printGameState(game);
                    } else {
                        std::cout << "Nothing to undo." << std::endl;
                    }
                    continue;
                } else if (val >= 1 && val <= 9) {
                    iterationsToAdd = val * 1000000;
                } else {
                    iterationsToAdd = val;
                }
            } catch (const std::invalid_argument &) {
                std::cout << "Invalid input." << std::endl;
                continue;
            }
        }

        if (iterationsToAdd <= 0)
            break;

        std::cout << "RUNNING SEARCH with " << formatWithCommas(iterationsToAdd) << " iterations..." << std::endl;

        auto config = mcts.getConfig();
        config.maxIterations = iterationsToAdd;
        mcts.setConfig(config);
        runSearchAndReport(mcts, game);
    }
}

// Explicit instantiations so the template body can live in the .cpp file.
template void GameUtils::interactiveSearchLoop<MCTS>(MCTS &, PenteGame);
template void GameUtils::interactiveSearchLoop<ParallelMCTS>(ParallelMCTS &, PenteGame);
