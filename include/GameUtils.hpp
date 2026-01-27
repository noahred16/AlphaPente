#ifndef GAMEUTILS_HPP
#define GAMEUTILS_HPP

#include <string>
#include <vector>
#include <utility>

// Forward declaration
class PenteGame;

class GameUtils {
public:
    // Move parsing/display
    static std::pair<int, int> parseMove(const char* move);
    static std::string displayMove(int x, int y);

    // Game string parsing - returns move strings only (filters out move numbers)
    static std::vector<std::string> parseGameString(const char* gameStr);

    // Board printing
    static void printBoard(const PenteGame& game);
    static void printGameState(const PenteGame& game);
};

#endif // GAMEUTILS_HPP
