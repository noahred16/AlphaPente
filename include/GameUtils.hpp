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
class ParallelMCTS;

class GameUtils {
  public:
    // Move parsing/display
    static std::pair<int, int> parseMove(const char *move);
    static std::string displayMove(int x, int y);

    // Game string parsing - returns move strings only (filters out move numbers)
    static std::vector<std::string> parseGameString(const char *gameStr);

    // Board printing. highlightX/Y (if a cell on the board) is shown in reverse
    // video -- meant for the last move played. promising is a list of (x, y)
    // empty-cell coords shown as a ranked, colored move label (1-based index
    // into the list) instead of blank -- meant for candidate moves under
    // consideration. Plain ints rather than PenteGame::Move to avoid this
    // header needing a full (circular) PenteGame.hpp include.
    static void printBoard(const PenteGame &game, int highlightX = -1, int highlightY = -1,
                            const std::vector<std::pair<int, int>> &promising = {});
    static void printGameState(const PenteGame &game, int highlightX = -1, int highlightY = -1,
                                const std::vector<std::pair<int, int>> &promising = {});

    // Number formatting
    static std::string formatWithCommas(int value);

    // Search utilities - overloaded per engine type
    static void runSearchAndReport(MCTS &mcts, const PenteGame &game);
    static void runSearchAndReport(ParallelMCTS &mcts, const PenteGame &game);

    // Same as runSearchAndReport, but prints a single JSON object (mcts.toJSON())
    // to stdout instead of the human-readable stats/best-moves text. For scripted
    // consumption, e.g. running the search N times and aggregating across runs.
    static void runSearchAndReportJSON(MCTS &mcts, const PenteGame &game);
    static void runSearchAndReportJSON(ParallelMCTS &mcts, const PenteGame &game);

    // Interactive search loop - shared template; explicit instantiations for MCTS and ParallelMCTS
    template <typename MCTSType>
    static void interactiveSearchLoop(MCTSType &mcts, PenteGame game);

    // Read ARENA_SIZE_GB from env var or .env file; falls back to defaultGb if unset or invalid.
    static size_t arenaSizeFromEnv(size_t defaultGb = 8);

    // Read NUM_THREADS from env var or .env file; aborts if unset or invalid.
    static int numThreadsFromEnv();

    // PUCT exploration constant, scaled down as the game progresses: a wide-open
    // early/tactical position needs more exploitation-focused search to
    // concentrate a limited sim budget on forcing lines, vs. a settled late-game
    // position where broader exploration is cheaper to afford.
    static double explorationConstantForMoveCount(int moveCount);
};

#endif // GAMEUTILS_HPP
