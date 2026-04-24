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

    // Read ARENA_SIZE_GB from environment; falls back to defaultGb if unset or invalid.
    static size_t arenaSizeFromEnv(size_t defaultGb = 8) {
        const char *val = std::getenv("ARENA_SIZE_GB");
        if (val && std::strlen(val) > 0) {
            size_t gb = static_cast<size_t>(std::atoll(val));
            if (gb > 0) return gb * 1024ULL * 1024 * 1024;
        }
        return defaultGb * 1024ULL * 1024 * 1024;
    }
};

#endif // GAMEUTILS_HPP
