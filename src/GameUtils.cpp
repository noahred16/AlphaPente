#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include "MCTS.hpp"
#include "Profiler.hpp"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <chrono>

std::pair<int, int> GameUtils::parseMove(const char* move) {
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

std::vector<std::string> GameUtils::parseGameString(const char* gameStr) {
    std::vector<std::string> moves;
    char* gameDataCopy = strdup(gameStr);
    char* token = std::strtok(gameDataCopy, " \t\n\r");

    int tokenIndex = 0;
    while (token != nullptr) {
        // Skip move numbers (every 3rd token: "1.", "2.", etc.)
        if (tokenIndex % 3 != 0) {
            moves.push_back(std::string(token));
        }
        token = std::strtok(nullptr, " \t\n\r");
        tokenIndex++;
    }
    free(gameDataCopy);

    return moves;
}

void GameUtils::printBoard(const PenteGame& game) {
    const std::vector<PenteGame::Move> legalMoves = game.getLegalMoves();

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
                std::cout << "\u25CB ";  // White circle for Black stones
            } else if (stone == PenteGame::WHITE) {
                std::cout << "\u25CF ";  // Black circle for White stones
            } else {
                bool isLegal = false;
                for (const auto& move : legalMoves) {
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

void GameUtils::printGameState(const PenteGame& game) {
    printBoard(game);

    const PenteGame::Config& config = game.getConfig();

    if (config.capturesEnabled) {
        std::cout << game.getBlackCaptures() << "/" << config.capturesToWin << " Black \u25CB, "
                  << game.getWhiteCaptures() << "/" << config.capturesToWin << " White \u25CF, ";
    }

    std::cout << "Current player: "
              << (game.getCurrentPlayer() == PenteGame::BLACK ? "Black" : "White") << "\n";
}

std::string GameUtils::formatWithCommas(int value) {
    std::string num = std::to_string(value);
    std::string result;
    int count = 0;
    for (int i = num.length() - 1; i >= 0; --i) {
        if (count > 0 && count % 3 == 0) result = ',' + result;
        result = num[i] + result;
        ++count;
    }
    return result;
}

void GameUtils::runSearchAndReport(MCTS& mcts, const PenteGame& game) {
    auto start = std::chrono::high_resolution_clock::now();
    mcts.search(game);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    int minutes = elapsed.count() / 60;
    int seconds = elapsed.count() % 60;
    std::cout << "Search took: " << minutes << " min " << seconds << " sec." << std::endl;
    mcts.printStats();
    mcts.printBestMoves(15);
    PenteGame::Move bestMove = mcts.getBestMove();
    std::string bestMoveStr = displayMove(bestMove.x, bestMove.y);
    std::cout << "MCTS selected move: " << bestMoveStr << std::endl;
    std::cout << '\a' << std::flush;
}

void GameUtils::interactiveSearchLoop(MCTS& mcts, PenteGame game) {
    runSearchAndReport(mcts, game);

    Profiler::instance().printReport();

    int iterationsToAdd = mcts.getConfig().maxIterations;
    std::vector<PenteGame> gameHistory;

    while (iterationsToAdd > 0) {
        std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        std::cout << "Enter iterations (0 to quit, 1-5 for millions, enter for default "
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
                } else if (val >= 1 && val <= 5) {
                    iterationsToAdd = val * 1000000;
                } else {
                    iterationsToAdd = val;
                }
            } catch (const std::invalid_argument&) {
                std::cout << "Invalid input." << std::endl;
                continue;
            }
        }

        if (iterationsToAdd <= 0) break;

        std::cout << "RUNNING SEARCH with " << formatWithCommas(iterationsToAdd) << " iterations..." << std::endl;

        MCTS::Config config = mcts.getConfig();
        config.maxIterations = iterationsToAdd;
        mcts.setConfig(config);
        runSearchAndReport(mcts, game);
    }
}
