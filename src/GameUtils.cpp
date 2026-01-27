#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include <cstring>
#include <cstdlib>
#include <iostream>

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
                  << game.getWhiteCaptures() << "/" << config.capturesToWin << " White \u25CF\n";
    }

    std::cout << "Current player: "
              << (game.getCurrentPlayer() == PenteGame::BLACK ? "Black" : "White") << "\n";
}
