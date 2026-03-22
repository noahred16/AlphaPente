#ifndef PENTEGAME_HPP
#define PENTEGAME_HPP

#include "BitBoard.hpp"
#include "GameUtils.hpp"
#include "Zobrist.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stack>
#include <string>
#include <vector>

class PenteGame {
  public:
    static constexpr int BOARD_SIZE = 19;

    // Runtime-configurable game settings
    struct Config {
        int capturesToWin = 10;      // Pente: 10, Keryo: 15
        bool keryoRules = false;     // Keryo: true (3-stone captures)
        bool capturesEnabled = true; // Gomoku: false
        bool tournamentRule = true;  // 3rd move restriction
        uint32_t seed = 0;           // 0 = non-deterministic, non-zero = deterministic

        // Factory methods for presets
        static Config pente() { return Config{}; }
        static Config gomoku() { return Config{10, false, false, false}; }
        static Config keryoPente() { return Config{15, true, true, true}; }
    };

    enum Player : uint8_t { NONE = 0, BLACK = 1, WHITE = 2 };

    struct Move {
        uint8_t x, y; // 2 bytes total, sufficient for 19x19
        static constexpr uint8_t INVALID = 255;
        Move() : x(INVALID), y(INVALID) {}
        Move(int x_, int y_) : x(static_cast<uint8_t>(x_)), y(static_cast<uint8_t>(y_)) {}
    };

    // struct MoveInfo {
    //     Move move;
    //     uint16_t captureMask; // 16 bits: 8 directions * 2 bits each
    //     Player player;
    //     uint8_t totalCapturedStones; // Helpful for quick score updates
    // };

  private:
    Config config_;
    BitBoard blackStones;
    BitBoard whiteStones;
    Player currentPlayer;
    int blackCaptures;
    int whiteCaptures;
    int moveCount;

    // Move history stack for undo support
    // std::vector<MoveInfo> moveHistory;

    Move lastMove;
    uint64_t hash_;
    mutable std::mt19937 rng_;

    // Helper functions
    bool checkFiveInRow(int x, int y) const;
    int checkAndCapture(int x, int y);
    int countConsecutive(const BitBoard &stones, int x, int y, int dx, int dy) const;

    std::vector<Move> promisingMovesVector;                     // empty squares within distance 1 of any stone
    std::array<size_t, BOARD_SIZE * BOARD_SIZE> promisingMoveIndex;
    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1); // Max size_t value

    size_t encodePos(int x, int y) const { return static_cast<size_t>(y * BOARD_SIZE + x); }

    // Add a legal move - O(1). Called when captured stones are returned to the board.
    void setLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        if (promisingMoveIndex[pos] == INVALID_INDEX) {
            promisingMovesVector.emplace_back(x, y);
            promisingMoveIndex[pos] = promisingMovesVector.size() - 1;
        }
    }

    // Remove a legal move - O(1). Called when a stone is placed.
    void clearLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        size_t promisingIdx = promisingMoveIndex[pos];

        // Remove from promising
        if (promisingIdx != INVALID_INDEX) {
            size_t lastPromisingIdx = promisingMovesVector.size() - 1;
            if (promisingIdx != lastPromisingIdx) {
                Move lastMove = promisingMovesVector.back();
                promisingMovesVector[promisingIdx] = lastMove;
                promisingMoveIndex[encodePos(lastMove.x, lastMove.y)] = promisingIdx;
            }
            promisingMovesVector.pop_back();
            promisingMoveIndex[pos] = INVALID_INDEX;
        }

        // Dilate: add empty distance-1 neighbors to promising
        static const int dirs[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
        // TODO Dilate more aggressively, and fix known bug
        for (int i = 0; i < 8; i++) {
            int nx = x + dirs[i][0], ny = y + dirs[i][1];
            if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                if (!blackStones.getBitUnchecked(nx, ny) && !whiteStones.getBitUnchecked(nx, ny)) {
                    size_t npos = encodePos(nx, ny);
                    if (promisingMoveIndex[npos] == INVALID_INDEX) {
                        promisingMovesVector.emplace_back(nx, ny);
                        promisingMoveIndex[npos] = promisingMovesVector.size() - 1;
                    }
                }
            }
        }
    }

  public:
    PenteGame(const Config &config = Config::pente());

    // Core game functions
    void reset();
    bool makeMove(const char *move); // Overloaded to accept string moves like "J11"
    bool makeMove(int x, int y);     // Returns false if illegal
    // void undoMove();               // Undo last move using stack

    // Game state queries
    Player getCurrentPlayer() const { return currentPlayer; }
    Player getWinner() const;
    bool isGameOver() const;
    bool isLegalMove(int x, int y) const;
    const std::vector<Move> &getLegalMoves() const;
    std::vector<Move> getPromisingMoves(int distance) const;

    // State access
    int getBlackCaptures() const { return blackCaptures; }
    int getWhiteCaptures() const { return whiteCaptures; }
    // Move getLastMove() const {
    //     return moveHistory.empty() ? Move() : moveHistory.back().move;
    // }
    int getMoveCount() const { return moveCount; }
    // bool canUndo() const { return !moveHistory.empty(); }

    // For MCTS
    Move getRandomPromisingMove() const;
    PenteGame clone() const;
    void syncFrom(const PenteGame &other);
    uint64_t computeHash() const;
    uint64_t getHash() const;
    uint64_t getCanonicalHash(int &outSym) const;

    // Debug
    void print() const;
    Player getStoneAt(int x, int y) const;

    // Heuristic evaluation
    float evaluateMove(Move move) const;
    float evaluatePosition() const;
    int countOpenFours(Player player) const;

    // Config access
    const Config &getConfig() const { return config_; }
};

#endif // PENTE_HPP
