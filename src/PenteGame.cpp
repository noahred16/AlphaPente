#include "PenteGame.hpp"
#include <random>
#include <iostream>
#include <sstream>
#include <algorithm>

PenteGame::PenteGame() {
    reset();
}

void PenteGame::reset() {
    blackStones.clear();
    whiteStones.clear();
    currentPlayer = BLACK;
    blackCaptures = 0;
    whiteCaptures = 0;
    moveCount = 0;
    moveHistory.clear();
    moveHistory.reserve(361); // Pre-allocate for performance
}

bool PenteGame::makeMove(const char* move) {
    auto [x, y] = parseMove(move);
    return makeMove(x, y);
}

bool PenteGame::makeMove(int x, int y) {
    if (!isLegalMove(x, y)) {
        return false;
    }
    
    // Place stone
    if (currentPlayer == BLACK) {
        blackStones.setBit(x, y);
    } else {
        whiteStones.setBit(x, y);
    }
    
    // Check and perform captures
    MoveInfo info;
    if (CAPTURES_ENABLED) {
        info = checkAndCapture(x, y);
    } else {
        info.move = Move(x, y);
        info.player = currentPlayer;
        info.totalCapturedStones = 0;
        info.captureMask = 0;
    }
    
    // Update capture count
    if (currentPlayer == BLACK) {
        blackCaptures += info.totalCapturedStones;
    } else {
        whiteCaptures += info.totalCapturedStones;
    }
    
    // Push to history BEFORE changing current player
    moveHistory.push_back(info);
    
    // Update state
    moveCount++;
    currentPlayer = (currentPlayer == BLACK) ? WHITE : BLACK;
    
    return true;
}

void PenteGame::undoMove() {
    if (moveHistory.empty()) {
        return; 
    }
    
    // 1. Pop the last move
    MoveInfo lastMove = moveHistory.back();
    
    // 2. Revert the current player to the one who made the move
    currentPlayer = lastMove.player;
    
    // 3. Remove the stone placed during this move
    if (currentPlayer == BLACK) {
        blackStones.clearBit(lastMove.move.x, lastMove.move.y);
        blackCaptures -= lastMove.totalCapturedStones;
    } else {
        whiteStones.clearBit(lastMove.move.x, lastMove.move.y);
        whiteCaptures -= lastMove.totalCapturedStones;
    }
    
    // 4. Restore captured stones using the 2-bit mask
    if (lastMove.totalCapturedStones > 0) {
        BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;
        
        static const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1}, 
                                       {0,-1}, {-1,0}, {-1,-1}, {1,-1}};
        
        for (int i = 0; i < 8; i++) {
            // Extract the 2 bits for direction i:
            // 01 (1) = Pair capture, 10 (2) = Triplet capture
            int captureType = (lastMove.captureMask >> (i * 2)) & 0x03; // lol bitmasking magic
            
            if (captureType > 0) {
                int dx = dirs[i][0];
                int dy = dirs[i][1];
                
                // Restore 2 stones (Standard)
                oppStones.setBit(lastMove.move.x + dx, lastMove.move.y + dy);
                oppStones.setBit(lastMove.move.x + dx * 2, lastMove.move.y + dy * 2);
                
                // If it was a Keryo capture, restore the 3rd stone
                if (captureType == 2) {
                    oppStones.setBit(lastMove.move.x + dx * 3, lastMove.move.y + dy * 3);
                }
            }
        }
    }
    
    // remove from history
    moveHistory.pop_back();
    moveCount--;
}

PenteGame::MoveInfo PenteGame::checkAndCapture(int x, int y) {
    MoveInfo info;
    info.move = Move(x, y);
    info.player = currentPlayer;
    info.totalCapturedStones = 0;
    info.captureMask = 0;
    
    BitBoard& myStones = (currentPlayer == BLACK) ? blackStones : whiteStones;
    BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;
    
    static const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1}, 
                                   {0,-1}, {-1,0}, {-1,-1}, {1,-1}};
    
    for (int i = 0; i < 8; i++) {
        int dx = dirs[i][0];
        int dy = dirs[i][1];

        // 1. Check for Keryo-style capture of 3 (X O O O X)
        if (KERYO_RULES) {
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
                    
                    info.totalCapturedStones += 3;
                    // Set bits to 10 (binary) for this direction
                    info.captureMask |= (2 << (i * 2));
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
                
                info.totalCapturedStones += 2;
                // Set bits to 01 (binary) for this direction
                info.captureMask |= (1 << (i * 2));
            }
        }
    }
    
    return info;
}


bool PenteGame::isLegalMove(int x, int y) const {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
        return false;
    }
    
    // Check if square is empty
    if (blackStones.getBit(x, y) || whiteStones.getBit(x, y)) {
        return false;
    }
    
    // First move (black) must be in center
    if (moveCount == 0) {
        return x == BOARD_SIZE / 2 && y == BOARD_SIZE / 2;
    }
    
    return true;
}

std::vector<PenteGame::Move> PenteGame::getLegalMoves() const {
    std::vector<Move> moves;
    
    // For first move, only center is legal
    if (moveCount == 0) {
        moves.emplace_back(BOARD_SIZE / 2, BOARD_SIZE / 2);
        return moves;
    }
    
    // Find all empty squares near existing stones (within distance 2)
    BitBoard occupied = blackStones | whiteStones;
    
    int distance = 1; // Could be more dynamic. Hardcoded for now.
    // int distance = 2; // Could be more dynamic. Hardcoded for now.
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
                    moves.emplace_back(x, y);
                }
            }
        }
    }

    // TODO, make configurable.
    // tournament rule. if moveCount == 2, moves must be at least 3 away from center
    if (moveCount == 2) {
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
                auto [x, y] = parseMove(moveStr.c_str());
                moves.emplace_back(x, y);
            }
        }
    }
    
    return moves;
}

PenteGame::Player PenteGame::getWinner() const {
    // get last move from history
    Move lastMove;
    if (!moveHistory.empty()) {
        lastMove = moveHistory.back().move;
    }

    // Check for five in a row
    if (lastMove.isValid()) {
        if (currentPlayer == WHITE && checkFiveInRow(lastMove.x, lastMove.y)) {
            return BLACK;  // Black just moved and won
        } else if (currentPlayer == BLACK && checkFiveInRow(lastMove.x, lastMove.y)) {
            return WHITE;  // White just moved and won
        }
    }
    
    // Check for capture wins
    if (blackCaptures >= CAPTURES_TO_WIN) return BLACK;
    if (whiteCaptures >= CAPTURES_TO_WIN) return WHITE;
    
    return NONE;
}

bool PenteGame::isGameOver() const {
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

PenteGame::Move PenteGame::getRandomMove() const {
    auto moves = getLegalMoves();
    if (moves.empty()) {
        return Move();  // Invalid move
    }
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, moves.size() - 1);
    
    return moves[dis(gen)];
}

PenteGame PenteGame::clone() const {
    return *this;  // Default copy constructor handles everything
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

void PenteGame::print() const {
    // get from root
    const std::vector<Move> legalMoves = getLegalMoves();

    // Helper to handle skipping 'I'
    auto getColChar = [](int x) {
        char c = (char)('A' + x);
        return (c >= 'I') ? (char)(c + 1) : c;
    };

    std::cout << "   ";
    for (int x = 0; x < BOARD_SIZE; x++) {
        std::cout << getColChar(x) << " ";
    }
    std::cout << "\n";
    
    for (int y = BOARD_SIZE - 1; y >= 0; y--) {
        std::cout << (y < 9 ? " " : "") << (y + 1) << " ";
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (blackStones.getBit(x, y)) {
                std::cout << "○ ";
            } else if (whiteStones.getBit(x, y)) {
                std::cout << "● ";
            } else {
                bool isLegal = false;
                for (const auto& move : legalMoves) {
                    if (move.x == x && move.y == y) {
                        isLegal = true;
                        break;
                    }
                }
                std::cout << (isLegal ? "  " : "· ");
            }
        }
        std::cout << (y + 1) << "\n";
    }
    
    std::cout << "   ";
    for (int x = 0; x < BOARD_SIZE; x++) {
        std::cout << getColChar(x) << " ";
    }
    std::cout << "\n";
    std::cout << blackCaptures << " Black ○, " << whiteCaptures << " White ●\n";
    std::cout << "Current player: " << (currentPlayer == BLACK ? "Black" : "White") << "\n";
}