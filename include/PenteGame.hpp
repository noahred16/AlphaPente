#ifndef PENTEGAME_HPP
#define PENTEGAME_HPP

#include "BitBoard.hpp"
#include "Zobrist.hpp"
#include <array>
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
        int dilationDistance = 1;    // Legal move dilation radius (1 or 2)
        uint32_t seed = 0;          // 0 = non-deterministic, non-zero = deterministic

        // Factory methods for presets
        static Config pente() { return Config{}; }
        static Config gomoku() { return Config{10, false, false, true, 1}; }
        static Config keryoPente() { return Config{15, true, true, true, 2}; }
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

    std::vector<Move> legalMovesVector;
    std::array<size_t, BOARD_SIZE * BOARD_SIZE> moveIndex;           // -1 = not present
    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1); // Max size_t value

    size_t encodePos(int x, int y) const { return static_cast<size_t>(y * BOARD_SIZE + x); }

    // Add a legal move - O(1)
    // Happens during captures. Both involve pieces being taken off the board, therefore freeing up legal moves.
    void setLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        if (moveIndex[pos] != INVALID_INDEX)
            return;

        legalMovesVector.emplace_back(x, y);
        moveIndex[pos] = legalMovesVector.size() - 1;
    }

    // Remove a legal move - O(1)
    // Happens during makeMove. involve pieces being placed on the board, therefore removing legal moves.
    void clearLegalMove(int x, int y) {
        size_t pos = encodePos(x, y);
        size_t idx = moveIndex[pos];

        if (idx == INVALID_INDEX)
            return;

        if (idx != legalMovesVector.size() - 1) { // No warning!
            Move lastMove = legalMovesVector.back();
            legalMovesVector[idx] = lastMove;
            moveIndex[encodePos(lastMove.x, lastMove.y)] = idx;
        }

        legalMovesVector.pop_back();
        moveIndex[pos] = INVALID_INDEX;

        // Dilate legal moves around the placed stone (0 = no dilation)
        if (config_.dilationDistance > 0) {
            static const int dirs1[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
            static const int dirs2[24][2] = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1},
                                             {1, -1},  {2, -1},  {-2, 0}, {-1, 0}, {1, 0},  {2, 0},   {-2, 1},  {-1, 1},
                                             {0, 1},   {1, 1},   {2, 1},  {-2, 2}, {-1, 2}, {0, 2},   {1, 2},   {2, 2}};

            const auto *dirs = (config_.dilationDistance >= 2) ? dirs2 : dirs1;
            int size = (config_.dilationDistance >= 2) ? 24 : 8;

            for (int i = 0; i < size; i++) {
                int nx = x + dirs[i][0];
                int ny = y + dirs[i][1];
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                    if (!blackStones.getBitUnchecked(nx, ny) && !whiteStones.getBitUnchecked(nx, ny)) {
                        setLegalMove(nx, ny);
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
    std::vector<Move> getLegalMoves() const;
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
    Move getRandomMove(const std::vector<Move> &moves) const;
    Move getRandomLegalMove() const; // Zero-copy version for simulations
    PenteGame clone() const;
    void syncFrom(const PenteGame &other);
    uint64_t computeHash() const;
    uint64_t getHash() const;
    uint64_t getCanonicalHash() const;

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
