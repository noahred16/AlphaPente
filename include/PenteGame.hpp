#ifndef PENTEGAME_HPP
#define PENTEGAME_HPP

#include "BitBoard.hpp"
#include <vector>
#include <cstdint>
#include <stack>

class PenteGame {
public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int CAPTURES_TO_WIN = 5;
    
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
        int capturedPairs;
        uint8_t captureDirections;  // Bit flags for which of 8 directions had captures
        Player player;              // Who made this move
    };

private:
    BitBoard blackStones;
    BitBoard whiteStones;
    Player currentPlayer;
    int blackCaptures;
    int whiteCaptures;
    int moveCount;
    
    // Move history stack for undo support
    std::stack<MoveInfo> moveHistory;
    
    // Helper functions
    bool checkFiveInRow(int x, int y) const;
    MoveInfo checkAndCapture(int x, int y);
    int countConsecutive(const BitBoard& stones, int x, int y, int dx, int dy) const;

public:
    PenteGame();
    
    // Core game functions
    void reset();
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
        return moveHistory.empty() ? Move() : moveHistory.top().move; 
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
};

#endif // PENTE_HPP
