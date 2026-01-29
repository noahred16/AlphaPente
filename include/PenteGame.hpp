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
#include <array>

class PenteGame {
public:
    static constexpr int BOARD_SIZE = 19;

    // Runtime-configurable game settings
    struct Config {
        int capturesToWin = 10;       // Pente: 10, Keryo: 15
        bool keryoRules = false;       // Keryo: true (3-stone captures)
        bool capturesEnabled = true;   // Gomoku: false
        bool tournamentRule = true;    // 3rd move restriction

        // Factory methods for presets
        static Config pente() { return Config{}; }
        static Config gomoku() { return Config{10, false, false, true}; }
        static Config keryoPente() { return Config{15, true, true, true}; }
    };

    enum Player : uint8_t {
        NONE = 0,
        BLACK = 1,
        WHITE = 2
    };
    
    struct Move {
        uint8_t x, y;  // 2 bytes total, sufficient for 19x19
        static constexpr uint8_t INVALID = 255;
        Move() : x(INVALID), y(INVALID) {}
        Move(int x_, int y_) : x(static_cast<uint8_t>(x_)), y(static_cast<uint8_t>(y_)) {}
    };

    struct MoveInfo {
        Move move;
        uint16_t captureMask; // 16 bits: 8 directions * 2 bits each
        Player player;
        uint8_t totalCapturedStones; // Helpful for quick score updates
    };

private:
    Config config_;
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

    std::vector<Move> legalMovesVector;
    std::vector<Move> legalMovesVectorPrevious; // For undo
    std::array<size_t, BOARD_SIZE * BOARD_SIZE> moveIndex;  // -1 = not present
    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);  // Max size_t value

    size_t encodePos(int x, int y) const { 
        return static_cast<size_t>(y * BOARD_SIZE + x); 
    }

    // Add a legal move - O(1)
    void setLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        if (moveIndex[pos] != INVALID_INDEX) return;
        
        legalMovesVector.emplace_back(x, y);
        moveIndex[pos] = legalMovesVector.size() - 1;
    }
    
    // Remove a legal move - O(1)
    void clearLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        size_t idx = moveIndex[pos];
        
        if (idx == INVALID_INDEX) return;
        
        if (idx != legalMovesVector.size() - 1) {  // No warning!
            Move lastMove = legalMovesVector.back();
            legalMovesVector[idx] = lastMove;
            moveIndex[encodePos(lastMove.x, lastMove.y)] = idx;
        }
        
        legalMovesVector.pop_back();
        moveIndex[pos] = INVALID_INDEX;
    }

public:
    PenteGame(const Config& config = Config::pente());
    
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
    std::vector<Move> getPromisingMoves(int distance) const;
    
    // State access
    int getBlackCaptures() const { return blackCaptures; }
    int getWhiteCaptures() const { return whiteCaptures; }
    Move getLastMove() const { 
        return moveHistory.empty() ? Move() : moveHistory.back().move; 
    }
    int getMoveCount() const { return moveCount; }
    bool canUndo() const { return !moveHistory.empty(); }
    
    // For MCTS
    Move getRandomMove(const std::vector<Move>& moves) const;
    PenteGame clone() const;
    void syncFrom(const PenteGame& other);
    uint64_t getHash() const;
    
    // Debug
    void print() const;
    Player getStoneAt(int x, int y) const;

    // Config access
    const Config& getConfig() const { return config_; }
};

#endif // PENTE_HPP
