#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include "Profiler.hpp"
#include <random>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

PenteGame::PenteGame(const Config& config) : config_(config) {
    reset();
}

void PenteGame::reset() {
    blackStones.clear();
    whiteStones.clear();

    legalMovesVector.clear();
    legalMovesVector.reserve(361);
    
    // TODO replace with config option for hueuristic
    if (true) { // TODO replace with config option for faster but less accurate MCTS
        moveIndex.fill(INVALID_INDEX);
    } else {
        // add all positions as legal moves
        for (int y = 0; y < BOARD_SIZE; y++) {
            for (int x = 0; x < BOARD_SIZE; x++) {
                legalMovesVector.emplace_back(x, y);
                moveIndex[encodePos(x, y)] = legalMovesVector.size() - 1;
            }
        }
    }

    // only the center move is legal at the start
    int center = BOARD_SIZE / 2;
    setLegalMove(center, center);

    currentPlayer = BLACK;
    blackCaptures = 0;
    whiteCaptures = 0;
    moveCount = 0;
    lastMove = Move();
}

bool PenteGame::makeMove(const char* move) {
    auto [x, y] = GameUtils::parseMove(move);
    if (blackStones.getBit(x, y) || whiteStones.getBit(x, y)) {
        return false;
    }
    setLegalMove(x, y);
    return makeMove(x, y);
}

bool PenteGame::makeMove(int x, int y) {
    PROFILE_SCOPE("PenteGame::makeMove");
    // Place stone
    if (currentPlayer == BLACK) {
        blackStones.setBit(x, y);
    } else {
        whiteStones.setBit(x, y);
    }
    clearLegalMove(x, y);

    // Check and perform captures
    if (config_.capturesEnabled) {
        if (currentPlayer == BLACK) {
            blackCaptures += checkAndCapture(x, y);
        } else {
            whiteCaptures += checkAndCapture(x, y);
        }
    }
    
    // Update state
    lastMove = Move(x, y);
    moveCount++;
    currentPlayer = (currentPlayer == BLACK) ? WHITE : BLACK;
    
    return true;
}

int PenteGame::checkAndCapture(int x, int y) {
    int totalCapturedStones = 0;
    
    BitBoard& myStones = (currentPlayer == BLACK) ? blackStones : whiteStones;
    BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;
    
    static const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1}, 
                                   {0,-1}, {-1,0}, {-1,-1}, {1,-1}};
    
    for (int i = 0; i < 8; i++) {
        int dx = dirs[i][0];
        int dy = dirs[i][1];

        // 1. Check for Keryo-style capture of 3 (X O O O X)
        if (config_.keryoRules) {
            int x4 = x + dx * 4;
            int y4 = y + dy * 4;
            
            if (x4 >= 0 && x4 < BOARD_SIZE && y4 >= 0 && y4 < BOARD_SIZE) {
                if (oppStones.getBit(x + dx, y + dy) && 
                    oppStones.getBit(x + dx * 2, y + dy * 2) && 
                    oppStones.getBit(x + dx * 3, y + dy * 3) && 
                    myStones.getBit(x4, y4)) 
                {
                    // Capture 3!
                    oppStones.clearBit(x + dx, y + dy);
                    oppStones.clearBit(x + dx * 2, y + dy * 2);
                    oppStones.clearBit(x + dx * 3, y + dy * 3);

                    setLegalMove(x + dx, y + dy);
                    setLegalMove(x + dx * 2, y + dy * 2);
                    setLegalMove(x + dx * 3, y + dy * 3);
                    
                    totalCapturedStones += 3;
                    continue; // Move to next direction
                }
            }
        }

        // 2. Check for Standard capture of 2 (X O O X)
        int x3 = x + dx * 3;
        int y3 = y + dy * 3;
        if (x3 >= 0 && x3 < BOARD_SIZE && y3 >= 0 && y3 < BOARD_SIZE) {
            if (oppStones.getBit(x + dx, y + dy) && 
                oppStones.getBit(x + dx * 2, y + dy * 2) && 
                myStones.getBit(x3, y3)) 
            {
                // Capture 2!
                oppStones.clearBit(x + dx, y + dy);
                oppStones.clearBit(x + dx * 2, y + dy * 2);

                setLegalMove(x + dx, y + dy);
                setLegalMove(x + dx * 2, y + dy * 2);
                
                totalCapturedStones += 2;
            }
        }
    }
    
    return totalCapturedStones;
}


bool PenteGame::isLegalMove(int x, int y) const {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
        return false;
    }
    
    // First move (black) must be in center
    if (moveCount == 0) {
        return x == BOARD_SIZE / 2 && y == BOARD_SIZE / 2;
    }

    // TODO: Tournament rule for 3rd move? might not even need this function
    // 
    std::cerr << "isLegalMove: Using BitBoard legalMoves check. We shouldnt worry.\n";

    // 
    return moveIndex[encodePos(x, y)] != INVALID_INDEX;
}

std::vector<PenteGame::Move> PenteGame::getLegalMoves() const {
    PROFILE_SCOPE("PenteGame::getLegalMoves");


    if (config_.tournamentRule && moveCount == 2) {
        // start empty
        std::vector<Move> moves = legalMovesVector;

        int center = BOARD_SIZE / 2;
        moves.erase(std::remove_if(moves.begin(), moves.end(),
                                   [center](const Move& m) {
                                       int distX = std::abs(m.x - center);
                                       int distY = std::abs(m.y - center);
                                       return distX < 3 && distY < 3;
                                   }),
                    moves.end());
        // if (moves.empty()) {
        // TODO: maybe add others? 
        std::vector<std::string> presetMoves = { 
            "K7", "L7", "M7", "N7", 
            "N8", "N9", "N10", "N11", "N12", "N13",
            "O10", "M6", "K6"
            };
        std::vector<Move> presetMovesVec;
        for (const auto& moveStr : presetMoves) {
            auto [x, y] = GameUtils::parseMove(moveStr.c_str());
            presetMovesVec.emplace_back(x, y);
        }
        // }
        return presetMovesVec;
    }

    return legalMovesVector;

    // getPromisingMoves
    // return getPromisingMoves(2);
    // legalMovesVector = getPromisingMoves(2);
    // return legalMovesVector;
    return getPromisingMoves(1);
    // return getPromisingMoves(15);
    // return getPromisingMoves(2);



    /*
    std::vector<Move> moves;
    
    // For first move, only center is legal
    if (moveCount == 0) {
        moves.emplace_back(BOARD_SIZE / 2, BOARD_SIZE / 2);
        return moves;
    }
    
    // Find all empty squares near existing stones (within distance 2)
    BitBoard occupied = blackStones | whiteStones;
    
    // int distance = 1; // Could be more dynamic. Hardcoded for now.
    int distance = 2; // Could be more dynamic. Hardcoded for now.
    if (moveCount <= 3) distance = 2;


    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (!occupied.getBit(x, y)) {
                // Check if there's a stone within distance 2
                bool nearStone = false;
                for (int dx = -distance; dx <= distance && !nearStone; dx++) {
                    for (int dy = -distance; dy <= distance && !nearStone; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                            if (occupied.getBit(nx, ny)) {
                                nearStone = true;
                            }
                        }
                    }
                }
                
                if (nearStone) {
                    // moves.emplace_back(x, y);

                }
            }
        }
    }

    // Tournament rule: if moveCount == 2, moves must be at least 3 away from center
    if (config_.tournamentRule && moveCount == 2) {
        int center = BOARD_SIZE / 2;
        moves.erase(std::remove_if(moves.begin(), moves.end(),
                                   [center](const Move& m) {
                                       int distX = std::abs(m.x - center);
                                       int distY = std::abs(m.y - center);
                                       return distX < 3 && distY < 3;
                                   }),
                    moves.end());
        if (moves.empty()) {
            // TODO: maybe add others? 
            std::vector<std::string> presetMoves = { 
                "K7", "L7", "M7", "N7", 
                "N8", "N9", "N10", "N11", "N12", "N13",
                "O10", "M6", "K6"
             };
            for (const auto& moveStr : presetMoves) {
                auto [x, y] = GameUtils::parseMove(moveStr.c_str());
                moves.emplace_back(x, y);
            }
        }
    }
    
    return moves;
    */
}

PenteGame::Player PenteGame::getWinner() const {
    PROFILE_SCOPE("PenteGame::getWinner");
    // get last move from history
    // Move lastMove;
    // if (!moveHistory.empty()) {
    //     lastMove = moveHistory.back().move;
    // }
    // Move lastMove = lastMove.move;
    // lastMove

    // Check for capture wins
    if (blackCaptures >= config_.capturesToWin) return BLACK;
    if (whiteCaptures >= config_.capturesToWin) return WHITE;

    // Check for five in a row
    if (currentPlayer == WHITE && checkFiveInRow(lastMove.x, lastMove.y)) {
        return BLACK;  // Black just moved and won
    } else if (currentPlayer == BLACK && checkFiveInRow(lastMove.x, lastMove.y)) {
        return WHITE;  // White just moved and won
    }
    
    return NONE;
}

bool PenteGame::isGameOver() const {
    PROFILE_SCOPE("PenteGame::isGameOver");
    return getWinner() != NONE;
}

bool PenteGame::checkFiveInRow(int x, int y) const {
    // Get the stones of the player who just moved
    const BitBoard& stones = (currentPlayer == WHITE) ? blackStones : whiteStones;
    
    // Check all 4 directions through the last move
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    
    for (auto [dx, dy] : dirs) {
        int count = 1;  // Count the stone we just placed
        
        // Count in positive direction
        count += countConsecutive(stones, x, y, dx, dy);
        
        // Count in negative direction
        count += countConsecutive(stones, x, y, -dx, -dy);
        
        if (count >= 5) {
            return true;
        }
    }
    
    return false;
}

int PenteGame::countConsecutive(const BitBoard& stones, int x, int y, int dx, int dy) const {
    int count = 0;
    int nx = x + dx;
    int ny = y + dy;
    
    while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && stones.getBit(nx, ny)) {
        count++;
        nx += dx;
        ny += dy;
    }
    
    return count;
}

std::vector<PenteGame::Move> PenteGame::getPromisingMoves(int distance) const {
    PROFILE_SCOPE("PenteGame::getPromisingMoves");

    if (distance != 1 && distance != 2 && distance != 15) {
        throw std::invalid_argument("getPromisingMoves: distance must be 1, 2, or 15");
    }

    BitBoard occupied = blackStones | whiteStones;
    BitBoard nearby;
    if (distance == 1) {
        nearby = occupied.dilate();
    } else if (distance == 15) {
        nearby = occupied.dilate1_5();
    } else {
        nearby = occupied.dilate2();
    }

    // Remove occupied squares
    nearby = nearby & ~occupied;

    return nearby.getSetPositions<Move>();
} 


PenteGame::Move PenteGame::getRandomMove(const std::vector<Move>& moves) const {
    PROFILE_SCOPE("PenteGame::getRandomMove");
    if (moves.empty()) {
        return Move();  // Invalid move
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, moves.size() - 1);

    return moves[dis(gen)];
}

PenteGame::Move PenteGame::getRandomLegalMove() const {
    PROFILE_SCOPE("PenteGame::getRandomLegalMove");

    // Handle tournament rule (3rd move restriction) - rare case, ok to copy
    if (config_.tournamentRule && moveCount == 2) {
        return getRandomMove(getLegalMoves());
    }

    // Fast path: pick directly from pre-computed vector (no copy)
    if (legalMovesVector.empty()) {
        return Move();
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, legalMovesVector.size() - 1);

    return legalMovesVector[dis(gen)];
}

PenteGame PenteGame::clone() const {
    return *this;  // Default copy constructor handles everything
}

void PenteGame::syncFrom(const PenteGame& other) {
    config_ = other.config_;
    blackStones = other.blackStones;
    whiteStones = other.whiteStones;
    legalMovesVector = other.legalMovesVector;
    moveIndex = other.moveIndex;
    currentPlayer = other.currentPlayer;
    blackCaptures = other.blackCaptures;
    whiteCaptures = other.whiteCaptures;
    moveCount = other.moveCount;
    // moveHistory = other.moveHistory;
    lastMove = other.lastMove;
}

uint64_t PenteGame::getHash() const {
    // Simple hash combining board states and game state
    // For a production system, you'd want Zobrist hashing
    uint64_t hash = 0;
    
    // Mix in stone positions
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (blackStones.getBit(i, j)) {
                hash ^= ((uint64_t)(i * BOARD_SIZE + j) << 1);
            }
            if (whiteStones.getBit(i, j)) {
                hash ^= ((uint64_t)(i * BOARD_SIZE + j) << 2);
            }
        }
    }
    
    // Mix in game state
    hash ^= ((uint64_t)currentPlayer << 32);
    hash ^= ((uint64_t)blackCaptures << 40);
    hash ^= ((uint64_t)whiteCaptures << 44);
    
    return hash;
}

PenteGame::Player PenteGame::getStoneAt(int x, int y) const {
    if (blackStones.getBit(x, y)) return BLACK;
    if (whiteStones.getBit(x, y)) return WHITE;
    return NONE;
}
