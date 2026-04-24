#ifndef GAMEUTILS_HPP
#define GAMEUTILS_HPP

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Forward declarations
class PenteGame;
class MCTS;

class GameUtils {
  public:
    // Move parsing/display
    static std::pair<int, int> parseMove(const char *move);
    static std::string displayMove(int x, int y);

    // Game string parsing - returns move strings only (filters out move numbers)
    static std::vector<std::string> parseGameString(const char *gameStr);

    // Board printing
    static void printBoard(const PenteGame &game);
    static void printGameState(const PenteGame &game);

    // Number formatting
    static std::string formatWithCommas(int value);

    // Search utilities
    static void runSearchAndReport(MCTS &mcts, const PenteGame &game);
    static void interactiveSearchLoop(MCTS &mcts, PenteGame game);

    // Read ARENA_SIZE_GB from env var or .env file; falls back to defaultGb if unset or invalid.
    static size_t arenaSizeFromEnv(size_t defaultGb = 8);
};

#endif // GAMEUTILS_HPP
