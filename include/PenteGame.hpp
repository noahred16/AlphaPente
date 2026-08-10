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
        int boardSize = 19;          // logical play area, centered within the physical BOARD_SIZE grid
        int numOffsets = 16;
        uint32_t seed = 0;           // 0 = non-deterministic, non-zero = deterministic
        bool renjuForbiddenMoves = false; // Renju forbidden-move rules (overline/double-four/double-three), Black only

        // Factory methods for presets
        static Config pente() { return Config{}; }
        static Config gomoku() { return Config{10, false, false, false}; }
        static Config keryoPente() { return Config{15, true, true, true}; }
        static Config renju() { return Config{10, false, false, false, 15, 16, 0, true}; }
    };

    enum Player : uint8_t { NONE = 0, BLACK = 1, WHITE = 2 };

    struct Move {
        uint8_t x, y; // 2 bytes total, sufficient for 19x19
        static constexpr uint8_t INVALID = 255;
        Move() : x(INVALID), y(INVALID) {}
        Move(int x_, int y_) : x(static_cast<uint8_t>(x_)), y(static_cast<uint8_t>(y_)) {}
    };

  private:
    Config config_;
    BitBoard blackStones;
    BitBoard whiteStones;
    Player currentPlayer;
    int blackCaptures;
    int whiteCaptures;
    int moveCount;

    static constexpr int dirs[24][2] = {
        // First 8: ch1
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
        // Next 8: in ch2NoKnight but not in ch1
        {-2, -2}, {-2, 0}, {-2, 2}, {0, -2}, {0, 2}, {2, -2}, {2, 0}, {2, 2},
        // Last 8: in ch2 but not in ch2NoKnight (the knight-move offsets)
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}
    };

    // Move history stack for undo support
    // std::vector<MoveInfo> moveHistory;

    Move lastMove;
    uint64_t hash_;
    mutable std::mt19937 rng_;

    // Helper functions
    bool checkFiveInRow(int x, int y) const;
    int checkAndCapture(int x, int y);
    int countConsecutive(const BitBoard &stones, int x, int y, int dx, int dy) const;

    // Renju forbidden-move rules (Black only). Builds a fresh finder from the current board each
    // time - cheap relative to node expansion/evaluation, and avoids keeping a second incrementally
    // maintained board in sync through captures/clone/syncFrom.
    class RenjuForbiddenPointFinder buildRenjuFinder() const;
    bool isRenjuForbidden(int x, int y) const; // false unless renjuForbiddenMoves is set and it's Black's turn
    const std::vector<Move> &getRenjuLegalMoves() const;

    std::vector<Move> promisingMovesVector;                     // empty squares within distance 1 of any stone
    mutable std::vector<Move> tournamentRulePerimeterBuffer;    // filtered perimeter for move 3 rule
    mutable std::vector<Move> renjuLegalMovesBuffer;            // promising moves minus Black's forbidden points
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

        // Handle offsets
        for (int i = 0; i < config_.numOffsets; i++) {
            int nx = x + dirs[i][0], ny = y + dirs[i][1];
            if (nx >= minIdx() && nx < maxIdx() && ny >= minIdx() && ny < maxIdx()) {
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

    void patchPromisingMovesAfterCaptures(const BitBoard &capturedBits);

  public:
    PenteGame(const Config &config = Config::pente());

    // Core game functions
    void reset();
    bool makeMove(const char *move); // Overloaded to accept string moves like "J11"
    bool makeMove(int x, int y);     // Returns false if illegal
    // void undoMove();               // Undo last move using stack

    // Directly sets this game's state to the given position (window-relative,
    // row-major cells - same layout PositionKey::Unpacked uses) and rebuilds
    // every derived internal structure (hash, promising-move index,
    // moveCount) from scratch. Unlike makeMove(), does NOT run capture
    // detection - the state is assumed already valid/consistent (e.g.
    // unpacked from an exact PositionKey rather than reached via incremental
    // play). `cells` must have exactly (maxIdx()-minIdx())^2 entries. Used
    // for PNS checkpoint/resume (src/PNS.cpp), where reconstructing a
    // position directly from its own key is far simpler than replaying an
    // arbitrary move history that was never actually recorded (a DAG node
    // may be reachable via many different move orders, none of which are
    // stored - only the resulting position matters, per PNS.hpp's own
    // no-true-cycles argument).
    void loadRawState(const Player *cells, Player sideToMove, int blackCapturesIn, int whiteCapturesIn);

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

    // Logical play area is a square of config_.boardSize centered within the physical BOARD_SIZE grid.
    int minIdx() const { return (BOARD_SIZE - config_.boardSize) / 2; }
    // Exclusive. minIdx() + boardSize, not BOARD_SIZE - minIdx(): those only
    // coincide when boardSize is odd (BOARD_SIZE=19 is odd, so
    // (BOARD_SIZE-boardSize)/2 only divides evenly for odd boardSize) - for
    // even boardSize, BOARD_SIZE-minIdx() silently widens the window by one
    // cell instead of giving exactly boardSize cells.
    int maxIdx() const { return minIdx() + config_.boardSize; }

    // Config access
    const Config &getConfig() const { return config_; }
    const BitBoard &getBlackBitBoard() const { return blackStones; }
    const BitBoard &getWhiteBitBoard() const { return whiteStones; }
};

#endif // PENTE_HPP
