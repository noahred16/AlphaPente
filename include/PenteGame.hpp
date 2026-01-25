#ifndef PENTEGAME_HPP
#define PENTEGAME_HPP

#include "BitBoard.hpp"
#include <vector>
#include <cstdint>
#include <stack>
#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>

class PenteGame {
public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int CAPTURES_TO_WIN = 10;
    static constexpr bool KERYO_RULES = false;
    static constexpr bool CAPTURES_ENABLED = true;
    
    enum Player {
        NONE = 0,
        BLACK = 1,
        WHITE = 2
    };
    
    struct Move {
        int x, y;
        Move() : x(-1), y(-1) {}
        Move(int x, int y) : x(x), y(y) {}
        bool isValid() const { return x >= 0 && y >= 0; }
    };

    struct MoveInfo {
        Move move;
        uint16_t captureMask; // 16 bits: 8 directions * 2 bits each
        Player player;
        uint8_t totalCapturedStones; // Helpful for quick score updates
    };

private:
    BitBoard blackStones;
    BitBoard whiteStones;
    Player currentPlayer;
    int blackCaptures;
    int whiteCaptures;
    int moveCount;

    // Move history stack for undo support
    std::vector<MoveInfo> moveHistory;
    
    // Helper functions
    bool checkFiveInRow(int x, int y) const;
    MoveInfo checkAndCapture(int x, int y);
    int countConsecutive(const BitBoard& stones, int x, int y, int dx, int dy) const;

public:
    PenteGame();
    
    // Core game functions
    void reset();
    bool makeMove(const char* move); // Overloaded to accept string moves like "J11"
    bool makeMove(int x, int y);  // Returns false if illegal
    void undoMove();               // Undo last move using stack
    
    // Game state queries
    Player getCurrentPlayer() const { return currentPlayer; }
    Player getWinner() const;
    bool isGameOver() const;
    bool isLegalMove(int x, int y) const;
    std::vector<Move> getLegalMoves() const;
    
    // State access
    int getBlackCaptures() const { return blackCaptures; }
    int getWhiteCaptures() const { return whiteCaptures; }
    Move getLastMove() const { 
        return moveHistory.empty() ? Move() : moveHistory.back().move; 
    }
    int getMoveCount() const { return moveCount; }
    bool canUndo() const { return !moveHistory.empty(); }
    
    // For MCTS
    Move getRandomMove() const;
    PenteGame clone() const;
    uint64_t getHash() const;
    
    // Debug
    void print() const;
    Player getStoneAt(int x, int y) const;

    // Utils
    // convert move string like "J11" to x,y
    static std::pair<int, int> parseMove(const char* move) {
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

    // convert x,y to move string like "J11" (skips over 'I')
    static std::string displayMove(int x, int y) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++; // Skip 'I'
        }
        return std::string(1, colChar) + std::to_string(y + 1);
    }
};

#endif // PENTE_HPP
