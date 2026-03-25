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

        enum class PromisingMoveConsideration { Chebyshev1, Chebyshev2, Chebyshev2ExcludingKnight };
        // PromisingMoveConsideration moveConsideration = PromisingMoveConsideration::Chebyshev1;
        PromisingMoveConsideration moveConsideration = PromisingMoveConsideration::Chebyshev2ExcludingKnight;

        // Factory methods for presets
        static Config pente() { return Config{}; }
        static Config gomoku() { return Config{10, false, false, false}; }
        static Config keryoPente() { return Config{15, true, true, true}; }

        // 3 types of move considerations (Chebyshev_d = 1, d = 2, or d = 2 but excluding knight movement type distances up 2 right 1 and 2 right 1)
        

        // Centralized logic for offsets
        struct Offset { int dx, dy; };
        static const std::vector<Offset>& getPromisingOffsets(PromisingMoveConsideration type) {
            static const std::vector<Offset> ch1 = {
                {-1,-1}, {0,-1}, {1,-1}, {-1,0}, {1,0}, {-1,1}, {0,1}, {1,1}
            };
            static const std::vector<Offset> ch2 = {
                {-2,-2}, {-1,-2}, {0,-2}, {1,-2}, {2,-2},
                {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1},
                {-2, 0}, {-1, 0},          {1, 0}, {2, 0},
                {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1},
                {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2}
            };
            static const std::vector<Offset> ch2NoKnight = {
                {-2,-2},          {-2, 0},          {-2, 2},
                        {-1,-1}, {-1, 0}, {-1, 1},
                { 0,-2}, { 0,-1},          { 0, 1}, { 0, 2},
                        { 1,-1}, { 1, 0}, { 1, 1},
                { 2,-2},          { 2, 0},          { 2, 2}
            };

            switch(type) {
                case PromisingMoveConsideration::Chebyshev1: return ch1;
                case PromisingMoveConsideration::Chebyshev2: return ch2;
                case PromisingMoveConsideration::Chebyshev2ExcludingKnight: return ch2NoKnight;
                default: return ch1;
            }
        }
    };

    struct Move {
        uint8_t x, y; // 2 bytes total, sufficient for 19x19
        static constexpr uint8_t INVALID = 255;
        Move() : x(INVALID), y(INVALID) {}
        Move(int x_, int y_) : x(static_cast<uint8_t>(x_)), y(static_cast<uint8_t>(y_)) {}
    };

    enum Player : uint8_t { NONE = 0, BLACK = 1, WHITE = 2 };

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
    mutable std::vector<Move> tournamentRulePerimeterBuffer;    // filtered perimeter for move 3 rule
    std::array<size_t, BOARD_SIZE * BOARD_SIZE> promisingMoveIndex;
    const Config::Offset* offsetData_;                         // raw pointer into static offset array, cached at construction
    int offsetCount_;                                          // number of offsets (compile-time constant per config)
    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1); // Max size_t value

    // how does the order work in the vector? 
    size_t encodePos(int x, int y) const { return static_cast<size_t>(y * BOARD_SIZE + x); }

    // Hard-code tournament perimeter for move 3 (first move is always center),
    // then filter out occupied perimeter squares for the current position.
    const std::vector<Move> &getTournamentRulePerimeter() const {
        static const std::vector<Move> allPerimeterMoves = [] {
            std::vector<Move> out;
            int center = BOARD_SIZE / 2;
            int dist = 3;

            // Generate the boundary of the 7x7 square (distance 3 from center)
            for (int i = -dist; i <= dist; ++i) {
                // Top and bottom rows
                out.emplace_back(center + i, center - dist);
                out.emplace_back(center + i, center + dist);
                // Left and right columns (skip corners already added)
                if (i > -dist && i < dist) {
                    out.emplace_back(center - dist, center + i);
                    out.emplace_back(center + dist, center + i);
                }
            }
            return out;
        }();

        tournamentRulePerimeterBuffer.clear();
        tournamentRulePerimeterBuffer.reserve(allPerimeterMoves.size());

        for (const Move &move : allPerimeterMoves) {
            if (!blackStones.getBitUnchecked(move.x, move.y) &&
                !whiteStones.getBitUnchecked(move.x, move.y)) {
                tournamentRulePerimeterBuffer.push_back(move);
            }
        }

        return tournamentRulePerimeterBuffer;
    }

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
        PROFILE_SCOPE("PenteGame::clearLegalMove");
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
        for (int i = 0; i < offsetCount_; i++) { const auto& offset = offsetData_[i];
            int nx = x + offset.dx, ny = y + offset.dy;
            // int nx = x + dirs[i][0], ny = y + dirs[i][1];
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

    void recalculatePromisingMoves() {
        PROFILE_SCOPE("PenteGame::recalculatePromisingMoves");
        // 1. Reset state without reallocating vector capacity
        promisingMovesVector.clear();
        promisingMoveIndex.fill(INVALID_INDEX);

        // 2. Combine both bitboards to get all occupied squares
        BitBoard occupied = blackStones | whiteStones;

        // 3. Use the optimized bit-scanner to iterate only over stones
        occupied.forEachSetBit([&](int cell) {
            // Convert the flat bit index back to coordinates
            int x = cell % BOARD_SIZE;
            int y = cell / BOARD_SIZE;

            // Apply dilation for this stone
            for (int i = 0; i < offsetCount_; i++) { const auto& off = offsetData_[i];
                int nx = x + off.dx;
                int ny = y + off.dy;

                // Bounds check
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                    // If the neighbor is empty
                    if (!blackStones.getBitUnchecked(nx, ny) && !whiteStones.getBitUnchecked(nx, ny)) {
                        size_t npos = encodePos(nx, ny);
                        
                        // If not already added to the promising list
                        if (promisingMoveIndex[npos] == INVALID_INDEX) {
                            promisingMovesVector.emplace_back(nx, ny);
                            promisingMoveIndex[npos] = promisingMovesVector.size() - 1;
                        }
                    }
                }
            }
        });
    }

    void patchPromisingMovesAfterCaptures(const BitBoard &capturedBits);

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
