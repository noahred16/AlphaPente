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

    // Hoist config check outside loop for better branch prediction
    if (config_.keryoRules) {
        // Keryo path: check 3-stone captures first, then fall back to 2-stone
        for (int i = 0; i < 8; i++) {
            int dx = dirs[i][0];
            int dy = dirs[i][1];

            // Pre-compute all offsets once per direction
            int x1 = x + dx,     y1 = y + dy;
            int x2 = x + dx * 2, y2 = y + dy * 2;
            int x3 = x + dx * 3, y3 = y + dy * 3;
            int x4 = x + dx * 4, y4 = y + dy * 4;

            // 1. Try Keryo-style capture of 3 (X O O O X)
            if (x4 >= 0 && x4 < BOARD_SIZE && y4 >= 0 && y4 < BOARD_SIZE) {
                // Use unchecked access - bounds already validated for furthest point
                if (oppStones.getBitUnchecked(x1, y1) &&
                    oppStones.getBitUnchecked(x2, y2) &&
                    oppStones.getBitUnchecked(x3, y3) &&
                    myStones.getBitUnchecked(x4, y4))
                {
                    oppStones.clearBitUnchecked(x1, y1);
                    oppStones.clearBitUnchecked(x2, y2);
                    oppStones.clearBitUnchecked(x3, y3);

                    setLegalMove(x1, y1);
                    setLegalMove(x2, y2);
                    setLegalMove(x3, y3);

                    totalCapturedStones += 3;
                    continue;
                }
            }

            // 2. Fall back to standard capture of 2 (X O O X)
            if (x3 >= 0 && x3 < BOARD_SIZE && y3 >= 0 && y3 < BOARD_SIZE) {
                if (oppStones.getBitUnchecked(x1, y1) &&
                    oppStones.getBitUnchecked(x2, y2) &&
                    myStones.getBitUnchecked(x3, y3))
                {
                    oppStones.clearBitUnchecked(x1, y1);
                    oppStones.clearBitUnchecked(x2, y2);

                    setLegalMove(x1, y1);
                    setLegalMove(x2, y2);

                    totalCapturedStones += 2;
                }
            }
        }
    } else {
        // Standard path: only 2-stone captures
        for (int i = 0; i < 8; i++) {
            int dx = dirs[i][0];
            int dy = dirs[i][1];

            // Pre-compute offsets
            int x1 = x + dx,     y1 = y + dy;
            int x2 = x + dx * 2, y2 = y + dy * 2;
            int x3 = x + dx * 3, y3 = y + dy * 3;

            if (x3 >= 0 && x3 < BOARD_SIZE && y3 >= 0 && y3 < BOARD_SIZE) {
                if (oppStones.getBitUnchecked(x1, y1) &&
                    oppStones.getBitUnchecked(x2, y2) &&
                    myStones.getBitUnchecked(x3, y3))
                {
                    oppStones.clearBitUnchecked(x1, y1);
                    oppStones.clearBitUnchecked(x2, y2);

                    setLegalMove(x1, y1);
                    setLegalMove(x2, y2);

                    totalCapturedStones += 2;
                }
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


    // optimization for moveCount 1

    if (moveCount == 1) {
        std::vector<std::string> presetMoves = { 
            "L9", "L10", "M9", "M10", "N9", "N8", "O9", "O8", "O7", "O6", "P6", "P7", "P9",             
        };
        std::vector<Move> presetMovesVec;
        for (const auto& moveStr : presetMoves) {
            auto [x, y] = GameUtils::parseMove(moveStr.c_str());
            presetMovesVec.emplace_back(x, y);
        }
        return presetMovesVec;
    }

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
    PROFILE_SCOPE("PenteGame::getRandLegalMove");

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

float PenteGame::evaluateMove(Move move) const {
    // Scoring weights
    constexpr float DEFAULT_SCORE = 1.0f;
    constexpr float CAPTURE_SCORE = 6.0f;
    constexpr float BLOCK_CAPTURE_SCORE = 4.0f;
    constexpr float CREATE_OPEN_THREE_SCORE = 15.0f;
    constexpr float BLOCK_OPEN_THREE_SCORE = 20.0f;
    constexpr float VULNERABLE_MOVE_PENALTY = -20.0f;
    constexpr float CREATE_FIVE_THREAT_SCORE = 20.0f;
    constexpr float BLOCK_FIVE_THREAT_SCORE = 20.0f;

    int x = move.x;
    int y = move.y;
    int captureCount = 0;
    int blockCaptureCount = 0;
    int createOpenThreeCount = 0;
    int blockOpenThreeCount = 0;
    int createFiveThreatCount = 0;
    int blockFiveThreatCount = 0;
    bool isVulnerableMove = false;

    const BitBoard& myStones = (currentPlayer == BLACK) ? blackStones : whiteStones;
    const BitBoard& oppStones = (currentPlayer == BLACK) ? whiteStones : blackStones;

    // Helper lambdas
    auto inBounds = [](int px, int py) {
        return px >= 0 && px < BOARD_SIZE && py >= 0 && py < BOARD_SIZE;
    };
    auto isEmpty = [&](int px, int py) {
        return inBounds(px, py) &&
               !myStones.getBitUnchecked(px, py) &&
               !oppStones.getBitUnchecked(px, py);
    };
    auto hasMy = [&](int px, int py) {
        return inBounds(px, py) && myStones.getBitUnchecked(px, py);
    };
    auto hasOpp = [&](int px, int py) {
        return inBounds(px, py) && oppStones.getBitUnchecked(px, py);
    };

    // 8 directions for capture checks
    static const int dirs[8][2] = {{0,1}, {1,0}, {1,1}, {-1,1},
                                   {0,-1}, {-1,0}, {-1,-1}, {1,-1}};

    for (int i = 0; i < 8; i++) {
        int dx = dirs[i][0];
        int dy = dirs[i][1];

        int x1 = x + dx,     y1 = y + dy;
        int x2 = x + dx * 2, y2 = y + dy * 2;
        int x3 = x + dx * 3, y3 = y + dy * 3;

        if (x3 >= 0 && x3 < BOARD_SIZE && y3 >= 0 && y3 < BOARD_SIZE) {
            // Capture: myStone - oppStone - oppStone - _ (we complete capture)
            if (oppStones.getBitUnchecked(x1, y1) &&
                oppStones.getBitUnchecked(x2, y2) &&
                myStones.getBitUnchecked(x3, y3))
            {
                captureCount++;
            }
            // Block capture: oppStone - myStone - myStone - _ (we prevent their capture)
            else if (myStones.getBitUnchecked(x1, y1) &&
                     myStones.getBitUnchecked(x2, y2) &&
                     oppStones.getBitUnchecked(x3, y3))
            {
                blockCaptureCount++;
            }
        }

        // NEW: Check if this move creates a capturable position
        // Pattern: opponent can capture if: O [P] M _
        // where [P] is our proposed move, M is our existing stone
        // After we place at x,y, opponent could capture us
        
        // Check if placing here creates: O P M _ (opponent - us - myStone - empty)
        int xBack = x - dx, yBack = y - dy;
        if (inBounds(xBack, yBack) && hasOpp(xBack, yBack) &&
            hasMy(x1, y1) && isEmpty(x2, y2))
        {
            isVulnerableMove = true;
        }
        
        // Also check reverse: _ M P O (empty - myStone - us - opponent)
        if (hasMy(xBack, yBack) && isEmpty(x - dx*2, y - dy*2) &&
            hasOpp(x1, y1))
        {
            isVulnerableMove = true;
        }

        // For Keryo rules, also check 3-stone captures
        if (config_.keryoRules) {
            int x4 = x + dx * 4;
            int y4 = y + dy * 4;
            
            // O P M M _ (opponent could capture 3)
            if (inBounds(x4, y4) && hasOpp(xBack, yBack) &&
                hasMy(x1, y1) && hasMy(x2, y2) && isEmpty(x3, y3))
            {
                isVulnerableMove = true;
            }
            
            // _ M M P O (reverse 3-stone capture)
            if (hasMy(xBack, yBack) && hasMy(x - dx*2, y - dy*2) &&
                isEmpty(x - dx*3, y - dy*3) && hasOpp(x1, y1))
            {
                isVulnerableMove = true;
            }
        }
    }

    // 4 line directions for open three and five threat checks
    static const int lineDirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};

    for (int i = 0; i < 4; i++) {
        int dx = lineDirs[i][0];
        int dy = lineDirs[i][1];

        // Count consecutive stones from position (not including position itself)
        int posCount = countConsecutive(myStones, x, y, dx, dy);
        int negCount = countConsecutive(myStones, x, y, -dx, -dy);
        int total = 1 + posCount + negCount;

        // === CREATE FIVE THREAT (OPEN FOUR) DETECTION ===
        // An open four means the opponent must block or we win next turn
        
        // Pattern 1: Solid four with one open end: X X X X _ or _ X X X X
        if (total == 4) {
            int posEndX = x + dx * (posCount + 1);
            int posEndY = y + dy * (posCount + 1);
            int negEndX = x - dx * (negCount + 1);
            int negEndY = y - dy * (negCount + 1);

            // At least one end must be open
            if (isEmpty(posEndX, posEndY) || isEmpty(negEndX, negEndY)) {
                createFiveThreatCount++;
            }
        }

        // Pattern 2: X X X _ X (gap in position 4)
        if (posCount == 3 && isEmpty(x + dx*4, y + dy*4) && hasMy(x + dx*5, y + dy*5)) {
            createFiveThreatCount++;
        }
        if (negCount == 3 && isEmpty(x - dx*4, y - dy*4) && hasMy(x - dx*5, y - dy*5)) {
            createFiveThreatCount++;
        }

        // Pattern 3: X X _ X X (gap in position 3)
        if (posCount == 2 && isEmpty(x + dx*3, y + dy*3) && 
            hasMy(x + dx*4, y + dy*4) && hasMy(x + dx*5, y + dy*5)) {
            createFiveThreatCount++;
        }
        if (negCount == 2 && isEmpty(x - dx*3, y - dy*3) && 
            hasMy(x - dx*4, y - dy*4) && hasMy(x - dx*5, y - dy*5)) {
            createFiveThreatCount++;
        }

        // Pattern 4: X _ X X X (gap in position 2)
        if (posCount == 1 && isEmpty(x + dx*2, y + dy*2) && 
            hasMy(x + dx*3, y + dy*3) && hasMy(x + dx*4, y + dy*4) && hasMy(x + dx*5, y + dy*5)) {
            createFiveThreatCount++;
        }
        if (negCount == 1 && isEmpty(x - dx*2, y - dy*2) && 
            hasMy(x - dx*3, y - dy*3) && hasMy(x - dx*4, y - dy*4) && hasMy(x - dx*5, y - dy*5)) {
            createFiveThreatCount++;
        }

        // Pattern 5: _ X X X X (gap at position 1, looking backward)
        if (negCount == 4 && isEmpty(x - dx*5, y - dy*5)) {
            createFiveThreatCount++;
        }
        if (posCount == 4 && isEmpty(x + dx*5, y + dy*5)) {
            createFiveThreatCount++;
        }

        // === BLOCK OPPONENT'S FIVE THREAT ===
        int oppPosCount = countConsecutive(oppStones, x, y, dx, dy);
        int oppNegCount = countConsecutive(oppStones, x, y, -dx, -dy);
        // int oppTotal = 1 + oppPosCount + oppNegCount;

        // Blocking solid four: O O O O P or P O O O O
        if (oppPosCount == 4) {
            blockFiveThreatCount++;
        }
        if (oppNegCount == 4) {
            blockFiveThreatCount++;
        }

        // Block O O O _ O patterns (filling the gap blocks the threat)
        // Pattern: P is placed in the gap of O O O _ O
        if (oppPosCount == 0 && oppNegCount == 3 && hasOpp(x + dx, y + dy)) {
            // Check pattern: O O O P O (we're filling gap)
            if (inBounds(x + dx*2, y + dy*2) && hasOpp(x + dx*2, y + dy*2)) {
                blockFiveThreatCount++;
            }
        }
        if (oppNegCount == 0 && oppPosCount == 3 && hasOpp(x - dx, y - dy)) {
            // Check pattern: O P O O O (we're filling gap)
            if (inBounds(x - dx*2, y - dy*2) && hasOpp(x - dx*2, y - dy*2)) {
                blockFiveThreatCount++;
            }
        }

        // Block O O _ O O (P fills the middle gap)
        if (oppPosCount == 2 && oppNegCount == 2) {
            blockFiveThreatCount++;
        }

        // Block O _ O O O (P fills gap at position 2)
        if (oppPosCount == 3 && oppNegCount == 0 && hasOpp(x - dx, y - dy)) {
            blockFiveThreatCount++;
        }
        if (oppNegCount == 3 && oppPosCount == 0 && hasOpp(x + dx, y + dy)) {
            blockFiveThreatCount++;
        }

        // Block O O O _ O (P fills gap at position 4)
        if (oppNegCount == 3 && isEmpty(x + dx, y + dy) && hasOpp(x + dx*2, y + dy*2)) {
            blockFiveThreatCount++;
        }
        if (oppPosCount == 3 && isEmpty(x - dx, y - dy) && hasOpp(x - dx*2, y - dy*2)) {
            blockFiveThreatCount++;
        }

        // === OPEN THREE DETECTION (existing code) ===
        // Solid open three: _ X X X _ (total == 3 with both ends open)
        if (total == 3) {
            int posEndX = x + dx * (posCount + 1);
            int posEndY = y + dy * (posCount + 1);
            int negEndX = x - dx * (negCount + 1);
            int negEndY = y - dy * (negCount + 1);

            if (isEmpty(posEndX, posEndY) && isEmpty(negEndX, negEndY)) {
                createOpenThreeCount++;
            }
        }

        // Gap open three patterns (X_XX and XX_X with open ends)
        // Pattern 1: P _ X X (place, gap, two stones)
        if (isEmpty(x + dx, y + dy) && hasMy(x + dx*2, y + dy*2) && hasMy(x + dx*3, y + dy*3)) {
            if (isEmpty(x - dx, y - dy) && isEmpty(x + dx*4, y + dy*4)) {
                createOpenThreeCount++;
            }
        }
        // Pattern 2: X _ P X (stone, gap, place, stone)
        if (hasMy(x - dx*2, y - dy*2) && isEmpty(x - dx, y - dy) && hasMy(x + dx, y + dy)) {
            if (isEmpty(x - dx*3, y - dy*3) && isEmpty(x + dx*2, y + dy*2)) {
                createOpenThreeCount++;
            }
        }
        // Pattern 3: X _ X P (stone, gap, stone, place)
        if (hasMy(x - dx*3, y - dy*3) && isEmpty(x - dx*2, y - dy*2) && hasMy(x - dx, y - dy)) {
            if (isEmpty(x - dx*4, y - dy*4) && isEmpty(x + dx, y + dy)) {
                createOpenThreeCount++;
            }
        }
        // Pattern 4: P X _ X (place, stone, gap, stone)
        if (hasMy(x + dx, y + dy) && isEmpty(x + dx*2, y + dy*2) && hasMy(x + dx*3, y + dy*3)) {
            if (isEmpty(x - dx, y - dy) && isEmpty(x + dx*4, y + dy*4)) {
                createOpenThreeCount++;
            }
        }
        // Pattern 5: X P _ X (stone, place, gap, stone)
        if (hasMy(x - dx, y - dy) && isEmpty(x + dx, y + dy) && hasMy(x + dx*2, y + dy*2)) {
            if (isEmpty(x - dx*2, y - dy*2) && isEmpty(x + dx*3, y + dy*3)) {
                createOpenThreeCount++;
            }
        }
        // Pattern 6: X X _ P (stone, stone, gap, place)
        if (hasMy(x - dx*3, y - dy*3) && hasMy(x - dx*2, y - dy*2) && isEmpty(x - dx, y - dy)) {
            if (isEmpty(x - dx*4, y - dy*4) && isEmpty(x + dx, y + dy)) {
                createOpenThreeCount++;
            }
        }

        // Block opponent's solid open three: P O O O _ or _ O O O P
        if (oppPosCount == 3 && isEmpty(x + dx*4, y + dy*4)) {
            blockOpenThreeCount++;
        }
        if (oppNegCount == 3 && isEmpty(x - dx*4, y - dy*4)) {
            blockOpenThreeCount++;
        }

        // Block opponent's gap open three
        // P _ O O (blocking O_OO pattern at left end)
        if (isEmpty(x + dx, y + dy) && hasOpp(x + dx*2, y + dy*2) && hasOpp(x + dx*3, y + dy*3)) {
            if (hasOpp(x + dx*4, y + dy*4) && isEmpty(x + dx*5, y + dy*5)) {
                blockOpenThreeCount++;  // Pattern: P _ O O O _ (blocking O_OO)
            }
        }
        // O O _ P (blocking OO_O pattern at right end)
        if (isEmpty(x - dx, y - dy) && hasOpp(x - dx*2, y - dy*2) && hasOpp(x - dx*3, y - dy*3)) {
            if (hasOpp(x - dx*4, y - dy*4) && isEmpty(x - dx*5, y - dy*5)) {
                blockOpenThreeCount++;  // Pattern: _ O O O _ P (blocking OO_O)
            }
        }
        // P O _ O O (blocking O_OO at far left)
        if (hasOpp(x + dx, y + dy) && isEmpty(x + dx*2, y + dy*2) &&
            hasOpp(x + dx*3, y + dy*3) && hasOpp(x + dx*4, y + dy*4)) {
            if (isEmpty(x + dx*5, y + dy*5)) {
                blockOpenThreeCount++;
            }
        }
        // O O _ O P (blocking OO_O at far right)
        if (hasOpp(x - dx, y - dy) && isEmpty(x - dx*2, y - dy*2) &&
            hasOpp(x - dx*3, y - dy*3) && hasOpp(x - dx*4, y - dy*4)) {
            if (isEmpty(x - dx*5, y - dy*5)) {
                blockOpenThreeCount++;
            }
        }
        // P O O _ O (blocking OO_O at left)
        if (hasOpp(x + dx, y + dy) && hasOpp(x + dx*2, y + dy*2) &&
            isEmpty(x + dx*3, y + dy*3) && hasOpp(x + dx*4, y + dy*4)) {
            if (isEmpty(x + dx*5, y + dy*5)) {
                blockOpenThreeCount++;
            }
        }
        // O _ O O P (blocking O_OO at right)
        if (hasOpp(x - dx, y - dy) && hasOpp(x - dx*2, y - dy*2) &&
            isEmpty(x - dx*3, y - dy*3) && hasOpp(x - dx*4, y - dy*4)) {
            if (isEmpty(x - dx*5, y - dy*5)) {
                blockOpenThreeCount++;
            }
        }
    }

    float score = DEFAULT_SCORE;
    score += captureCount * CAPTURE_SCORE;
    score += blockCaptureCount * BLOCK_CAPTURE_SCORE;
    score += createOpenThreeCount * CREATE_OPEN_THREE_SCORE;
    score += blockOpenThreeCount * BLOCK_OPEN_THREE_SCORE;
    score += createFiveThreatCount * CREATE_FIVE_THREAT_SCORE;
    score += blockFiveThreatCount * BLOCK_FIVE_THREAT_SCORE;
    
    // Apply penalty if this move is vulnerable to capture
    if (isVulnerableMove) {
        score += VULNERABLE_MOVE_PENALTY;
    }
    
    // return max of 1 and score;
    return std::max(0.5f, score);
}

int PenteGame::countOpenFours(Player player) const {
    const BitBoard& stones = (player == BLACK) ? blackStones : whiteStones;
    const BitBoard& oppStones = (player == BLACK) ? whiteStones : blackStones;

    int openFourCount = 0;

    // 4 line directions
    static const int lineDirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};

    // Scan the board for open four patterns: _XXXX_
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            // Only check from stones of the player
            if (!stones.getBit(x, y)) continue;

            for (int d = 0; d < 4; d++) {
                int dx = lineDirs[d][0];
                int dy = lineDirs[d][1];

                // Check for 4 consecutive stones starting at (x,y)
                // Pattern: _XXXX_ where first X is at (x,y)
                int x1 = x + dx, y1 = y + dy;
                int x2 = x + dx*2, y2 = y + dy*2;
                int x3 = x + dx*3, y3 = y + dy*3;

                // Check all 4 stones exist
                if (x3 < 0 || x3 >= BOARD_SIZE || y3 < 0 || y3 >= BOARD_SIZE) continue;
                if (!stones.getBit(x1, y1) || !stones.getBit(x2, y2) || !stones.getBit(x3, y3)) continue;

                // Check both ends are empty (open)
                int beforeX = x - dx, beforeY = y - dy;
                int afterX = x + dx*4, afterY = y + dy*4;

                bool beforeOpen = (beforeX >= 0 && beforeX < BOARD_SIZE &&
                                   beforeY >= 0 && beforeY < BOARD_SIZE &&
                                   !stones.getBit(beforeX, beforeY) &&
                                   !oppStones.getBit(beforeX, beforeY));

                bool afterOpen = (afterX >= 0 && afterX < BOARD_SIZE &&
                                  afterY >= 0 && afterY < BOARD_SIZE &&
                                  !stones.getBit(afterX, afterY) &&
                                  !oppStones.getBit(afterX, afterY));

                if (beforeOpen && afterOpen) {
                    openFourCount++;
                }
            }
        }
    }

    // Each open four is counted once (from the first stone in the direction)
    return openFourCount;
}

float PenteGame::evaluatePosition() const {
    PROFILE_SCOPE("PenteGame::evaluatePosition");
    // return 0.0f;
    // Scoring weights
    // constexpr float CAPTURE_WEIGHT = 0.08f;
    // constexpr float OPEN_FOUR_WEIGHT = 0.45f;

    // // Count open fours for each side
    int myOpenFours = countOpenFours(currentPlayer);
    if (myOpenFours >= 1) {
        // DEBUG game utils print state and exit
        // GameUtils::printGameState(*this);
        // // print current player
        // std::cout << "Immediate win detected in evaluatePosition!" << std::endl;
        // exit(1);
        return 1.0f; // immediate win
    }
    int oppOpenFours = countOpenFours((currentPlayer == BLACK) ? WHITE : BLACK);
    if (oppOpenFours >= 1) {
        // GameUtils::printGameState(*this);
        // std::cout << "Immediate loss detected in evaluatePosition!" << std::endl;
        // exit(1);
        return -1.0f; // immediate loss
    }



    // // Get captures for current player and opponent
    int myCaptures, oppCaptures;
    if (currentPlayer == BLACK) {
        myCaptures = blackCaptures;
        oppCaptures = whiteCaptures;
    } else {
        myCaptures = whiteCaptures;
        oppCaptures = blackCaptures;
    }

    // if opp one person is leading by more than 2, consider it a win. 
    if (myCaptures - oppCaptures > 5) {
        // DEBUG game utils print state and exit
        // GameUtils::printGameState(*this);
        // std::cout << "Strong advantage detected in evaluatePosition!" << std::endl;
        // exit(1);
        return 0.7f;
    }
    if (oppCaptures - myCaptures > 5) {
        // DEBUG game utils print state and exit
        // GameUtils::printGameState(*this);
        // std::cout << "Strong disadvantage detected in evaluatePosition!" << std::endl;
        // exit(1);
        return -0.7f;
    }

    // TODO check if opp has two open 3s
    // TODO, do we ever find positives? 

    return 0.0f; // neutral position

    // // Calculate score
    // float score = 0.0f;
    // score += (myCaptures - oppCaptures) * CAPTURE_WEIGHT;
    // score += (myOpenFours - oppOpenFours) * OPEN_FOUR_WEIGHT;

    // // Clamp to [-1, 1]
    // if (score > 1.0f) score = 1.0f;
    // if (score < -1.0f) score = -1.0f;

    // return score;
}
