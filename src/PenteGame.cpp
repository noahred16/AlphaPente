#include "PenteGame.hpp"
#include <random>
#include <iostream>

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
    
    // Clear the history stack
    while (!moveHistory.empty()) {
        moveHistory.pop();
    }
}

bool PenteGame::makeMove(int x, int y) {
    if (!isLegalMove(x, y)) {
        return false;
    }
    
    // Create move info for history
    MoveInfo info;
    info.move = Move(x, y);
    info.player = currentPlayer;
    
    // Place stone
    if (currentPlayer == BLACK) {
        blackStones.setBit(x, y);
    } else {
        whiteStones.setBit(x, y);
    }
    
    // Check and perform captures
    MoveInfo captureInfo = checkAndCapture(x, y);
    info.capturedPairs = captureInfo.capturedPairs;
    info.captureDirections = captureInfo.captureDirections;
    
    // Update capture count
    if (currentPlayer == BLACK) {
        blackCaptures += info.capturedPairs;
    } else {
        whiteCaptures += info.capturedPairs;
    }
    
    // Push to history BEFORE changing current player
    moveHistory.push(info);
    
    // Update state
    moveCount++;
    currentPlayer = (currentPlayer == BLACK) ? WHITE : BLACK;
    
    return true;
}

void PenteGame::undoMove() {
    if (moveHistory.empty()) {
        return;  // Nothing to undo
    }
    
    // Get the last move info
    MoveInfo lastMove = moveHistory.top();
    moveHistory.pop();
    
    // Switch back to the player who made the move
    currentPlayer = lastMove.player;
    
    // Remove the stone
    if (currentPlayer == BLACK) {
        blackStones.clearBit(lastMove.move.x, lastMove.move.y);
        blackCaptures -= lastMove.capturedPairs;
    } else {
        whiteStones.clearBit(lastMove.move.x, lastMove.move.y);
        whiteCaptures -= lastMove.capturedPairs;
    }
    
    // Restore captured stones
    if (lastMove.capturedPairs > 0) {
        BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;
        const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1}, 
                                {0,-1}, {-1,0}, {-1,-1}, {1,-1}};
        
        for (int i = 0; i < 8; i++) {
            if (lastMove.captureDirections & (1 << i)) {
                // This direction had a capture - restore the two stones
                int dx = dirs[i][0];
                int dy = dirs[i][1];
                oppStones.setBit(lastMove.move.x + dx, lastMove.move.y + dy);
                oppStones.setBit(lastMove.move.x + dx * 2, lastMove.move.y + dy * 2);
            }
        }
    }
    
    moveCount--;
}

PenteGame::MoveInfo PenteGame::checkAndCapture(int x, int y) {
    MoveInfo info;
    info.capturedPairs = 0;
    info.captureDirections = 0;
    
    BitBoard& myStones = (currentPlayer == BLACK) ? blackStones : whiteStones;
    BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;
    
    const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1}, 
                            {0,-1}, {-1,0}, {-1,-1}, {1,-1}};
    
    for (int i = 0; i < 8; i++) {
        int dx = dirs[i][0];
        int dy = dirs[i][1];
        int x1 = x + dx;
        int y1 = y + dy;
        int x2 = x + dx * 2;
        int y2 = y + dy * 2;
        int x3 = x + dx * 3;
        int y3 = y + dy * 3;
        
        if (x3 >= 0 && x3 < BOARD_SIZE && y3 >= 0 && y3 < BOARD_SIZE) {
            if (oppStones.getBit(x1, y1) && oppStones.getBit(x2, y2) && 
                myStones.getBit(x3, y3)) {
                // Capture!
                oppStones.clearBit(x1, y1);
                oppStones.clearBit(x2, y2);
                info.capturedPairs++;
                info.captureDirections |= (1 << i);  // Mark this direction
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
    
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (!occupied.getBit(x, y)) {
                // Check if there's a stone within distance 2
                bool nearStone = false;
                int distance = 1; // Could be more dynamic. Hardcoded for now.
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
    
    return moves;
}

PenteGame::Player PenteGame::getWinner() const {
    // get last move from history
    Move lastMove;
    if (!moveHistory.empty()) {
        lastMove = moveHistory.top().move;
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
    std::cout << "   ";
    for (int x = 0; x < BOARD_SIZE; x++) {
        std::cout << (char)('A' + x) << " ";
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
                std::cout << "· ";
            }
        }
        std::cout << (y + 1) << "\n";
    }
    
    std::cout << "  ";
    for (int x = 0; x < BOARD_SIZE; x++) {
        std::cout << (char)('A' + x) << " ";
    }
    std::cout << "\n\n";
    
    std::cout << "Black captures: " << blackCaptures << " pairs\n";
    std::cout << "White captures: " << whiteCaptures << " pairs\n";
    std::cout << "Current player: " << (currentPlayer == BLACK ? "Black" : "White") << "\n";
}