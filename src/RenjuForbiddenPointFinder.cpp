#include "RenjuForbiddenPointFinder.hpp"

// Faithful C++ port of CForbiddenPointFinder (Wenzhe Lu). Internal storage is 1-based with a
// one-cell BORDER ring so line scans terminate naturally at the edge without bounds checks.
//
// Convention shared by every public query below: (x, y) must be EMPTY on entry. Each function
// evaluates the pattern as if the relevant stone were placed at (x, y), given whatever real stones
// already sit on the board, without mutating the board.
//
// Performance note: isDoubleThree/isDoubleFour ultimately need up to isOpenThree -> isOpenFour ->
// isFive (up to 81 five-checks per direction for a three) - hot enough to matter, since
// getLegalMoves() calls this once per MCTS node expansion. Two things keep that cheap:
//   1. Each public query extracts its line into a small local array ONCE and runs the whole
//      recursive check against that array, instead of re-deriving the same cells from the live
//      board (with its bounds checks) at every level of the recursion.
//   2. hasNearby() bails out of a direction immediately when there's no same-color stone within
//      reach - true for most (candidate, direction) pairs in a real game, since stones cluster.
// Same algorithm, same results either way (see the isFive/isFour/isOpenFour/isOpenThree tests).

namespace {
// The 4 canonical line directions.
constexpr int kDirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

// Radius (in cells, either side of the candidate) covered by the extracted line. Growth points
// are always empty *board* cells (BORDER fails the emptiness check), so no reachable cell - however
// deep the isOpenThree -> isOpenFour -> isFive recursion goes - can ever be farther from the
// candidate than the board itself is wide. 15 covers the documented max board size (see the
// class's 17x17 `b_`) with a 1-cell margin.
constexpr int kRadius = 15;
constexpr int kLen = kRadius * 2 + 1;

// Counts consecutive `c` in `line`, starting one step from idx, stepping by `step` (+-1).
int countOnLine(const char *line, int idx, char c, int step) {
    int count = 0;
    int i = idx + step;
    while (i >= 0 && i < kLen && line[i] == c) {
        count++;
        i += step;
    }
    return count;
}

// A four/open-four/three can only form where at least one same-color stone already sits within
// reach (a completion needs 3-4 stones total, one of which is the candidate itself). Bailing out
// early when a direction is "empty" skips the expensive nested search entirely for the common case
// of a candidate that isn't near any of its own color along that particular line.
bool hasNearby(const char *line, int idx, char c, int radius) {
    for (int i = -radius; i <= radius; i++) {
        if (i == 0) continue;
        int p = idx + i;
        if (p >= 0 && p < kLen && line[p] == c) return true;
    }
    return false;
}

bool isFiveOnLine(char *line, int idx, int nColor) {
    char c = (nColor == 0) ? RenjuForbiddenPointFinder::BLACK : RenjuForbiddenPointFinder::WHITE;
    char saved = line[idx];
    line[idx] = c;
    int count = 1 + countOnLine(line, idx, c, +1) + countOnLine(line, idx, c, -1);
    line[idx] = saved;
    // Black needs exactly five: a run of six or more is an overline, not a win.
    return (nColor == 0) ? (count == 5) : (count >= 5);
}

bool isFourOnLine(char *line, int idx, int nColor) {
    char c = (nColor == 0) ? RenjuForbiddenPointFinder::BLACK : RenjuForbiddenPointFinder::WHITE;
    char saved = line[idx];
    line[idx] = c;

    if (!hasNearby(line, idx, c, 4)) {
        line[idx] = saved;
        return false;
    }

    bool result = false;
    for (int i = -4; i <= 4 && !result; i++) {
        if (i == 0) continue;
        int p = idx + i;
        if (p < 0 || p >= kLen || line[p] != RenjuForbiddenPointFinder::EMPTY) continue;
        if (isFiveOnLine(line, p, nColor)) result = true;
    }

    line[idx] = saved;
    return result;
}

int isOpenFourOnLine(char *line, int idx, int nColor) {
    char c = (nColor == 0) ? RenjuForbiddenPointFinder::BLACK : RenjuForbiddenPointFinder::WHITE;
    char saved = line[idx];
    line[idx] = c;

    if (!hasNearby(line, idx, c, 4)) {
        line[idx] = saved;
        return 0;
    }

    int count = 0;
    for (int i = -4; i <= 4 && count < 2; i++) {
        if (i == 0) continue;
        int p = idx + i;
        if (p < 0 || p >= kLen || line[p] != RenjuForbiddenPointFinder::EMPTY) continue;
        if (isFiveOnLine(line, p, nColor)) count++;
    }

    line[idx] = saved;
    return count; // loop stops as soon as count reaches 2, so already capped
}

bool isOpenThreeOnLine(char *line, int idx, int nColor) {
    char c = (nColor == 0) ? RenjuForbiddenPointFinder::BLACK : RenjuForbiddenPointFinder::WHITE;
    char saved = line[idx];
    line[idx] = c;

    if (!hasNearby(line, idx, c, 4)) {
        line[idx] = saved;
        return false;
    }

    bool result = false;
    for (int i = -4; i <= 4 && !result; i++) {
        if (i == 0) continue;
        int p = idx + i;
        if (p < 0 || p >= kLen || line[p] != RenjuForbiddenPointFinder::EMPTY) continue;
        if (isOpenFourOnLine(line, p, nColor) == 2) result = true;
    }

    line[idx] = saved;
    return result;
}

// Extracts the board's line through (x, y) along `dir` into `line[kLen]`, with (x, y) itself at
// index kRadius. (x, y) is copied as-is (EMPTY, per the class-wide precondition).
void extractLine(const RenjuForbiddenPointFinder &finder, int x, int y, int dir, char *line) {
    int dx = kDirs[dir][0], dy = kDirs[dir][1];
    for (int i = -kRadius; i <= kRadius; i++) {
        line[i + kRadius] = finder.getStone(x + dx * i, y + dy * i);
    }
}
} // namespace

RenjuForbiddenPointFinder::RenjuForbiddenPointFinder(int size) : size_(size) { clear(); }

void RenjuForbiddenPointFinder::clear() {
    for (int y = 0; y < 17; y++) {
        for (int x = 0; x < 17; x++) {
            bool inBounds = x >= 1 && x <= size_ && y >= 1 && y <= size_;
            b_[y][x] = inBounds ? EMPTY : BORDER;
        }
    }
}

void RenjuForbiddenPointFinder::setStone(int x, int y, char stone) const {
    b_[y + 1][x + 1] = stone;
}

char RenjuForbiddenPointFinder::getStone(int x, int y) const {
    int ix = x + 1, iy = y + 1;
    if (ix < 0 || ix > 16 || iy < 0 || iy > 16) return BORDER;
    return b_[iy][ix];
}

// Counts consecutive stones of color `c` starting one step from (x, y), walking in direction
// `dir` (0..3 forward, 4..7 the reverse of 0..3). Does not include (x, y) itself. Kept for
// isOverline, which is already O(1) and doesn't need the line-extraction machinery below.
int RenjuForbiddenPointFinder::countLine(int x, int y, char c, int dir) const {
    int ddx, ddy;
    if (dir < 4) {
        ddx = kDirs[dir][0];
        ddy = kDirs[dir][1];
    } else {
        ddx = -kDirs[dir - 4][0];
        ddy = -kDirs[dir - 4][1];
    }

    int count = 0;
    int nx = x + ddx, ny = y + ddy;
    while (getStone(nx, ny) == c) {
        count++;
        nx += ddx;
        ny += ddy;
    }
    return count;
}

bool RenjuForbiddenPointFinder::isFive(int x, int y, int nColor, int dir) const {
    char line[kLen];
    extractLine(*this, x, y, dir, line);
    return isFiveOnLine(line, kRadius, nColor);
}

bool RenjuForbiddenPointFinder::isFive(int x, int y, int nColor) const {
    for (int dir = 0; dir < 4; dir++) {
        if (isFive(x, y, nColor, dir)) return true;
    }
    return false;
}

bool RenjuForbiddenPointFinder::isOverline(int x, int y) const {
    setStone(x, y, BLACK);
    bool result = false;
    for (int dir = 0; dir < 4 && !result; dir++) {
        int count = 1 + countLine(x, y, BLACK, dir) + countLine(x, y, BLACK, dir + 4);
        if (count >= 6) result = true;
    }
    setStone(x, y, EMPTY);
    return result;
}

// True if placing nColor at (x, y) creates a "four" along dir: some empty point within the
// 5-in-a-row window through (x, y) which, if also filled by nColor, completes a real five (per
// isFive's exact-five rule for black). This naturally excludes fours whose only completion would
// be an overline for black - no special-casing needed.
bool RenjuForbiddenPointFinder::isFour(int x, int y, int nColor, int dir) const {
    char line[kLen];
    extractLine(*this, x, y, dir, line);
    return isFourOnLine(line, kRadius, nColor);
}

// Counts how many distinct empty points along dir would complete a five if filled (0, 1, or 2).
// 2 means a genuine open four (e.g. "_XXXX_"): both completions are live, so it cannot be blocked
// with a single stone.
int RenjuForbiddenPointFinder::isOpenFour(int x, int y, int nColor, int dir) const {
    char line[kLen];
    extractLine(*this, x, y, dir, line);
    return isOpenFourOnLine(line, kRadius, nColor);
}

// True if placing nColor at (x, y) creates a real "three" along dir: an empty point within reach
// which, if filled by nColor, produces a genuine open four (isOpenFour == 2). A three whose only
// follow-up is a simple (one-sided) four doesn't count - the opponent could block it with one move.
bool RenjuForbiddenPointFinder::isOpenThree(int x, int y, int nColor, int dir) const {
    char line[kLen];
    extractLine(*this, x, y, dir, line);
    return isOpenThreeOnLine(line, kRadius, nColor);
}

// Double-four: two or more distinct four-lines through (x, y). Counting by direction (rather than
// by gain-point) is what keeps a single open four ("_XXXX_", which has two gain points but is one
// line) from being miscounted as a double-four.
bool RenjuForbiddenPointFinder::isDoubleFour(int x, int y) const {
    int count = 0;
    for (int dir = 0; dir < 4 && count < 2; dir++) {
        if (isFour(x, y, 0, dir)) count++;
    }
    return count >= 2;
}

bool RenjuForbiddenPointFinder::isDoubleThree(int x, int y) const {
    int count = 0;
    for (int dir = 0; dir < 4 && count < 2; dir++) {
        if (isOpenThree(x, y, 0, dir)) count++;
    }
    return count >= 2;
}

// A move that completes an exact five always wins outright, even if it also happens to form an
// overline/double-four/double-three along another line - five-in-a-row takes priority over the
// forbidden-move rules.
bool RenjuForbiddenPointFinder::isForbidden(int x, int y) const {
    if (getStone(x, y) != EMPTY) return false;
    if (isFive(x, y, 0)) return false;
    if (isOverline(x, y)) return true;
    if (isDoubleFour(x, y)) return true;
    if (isDoubleThree(x, y)) return true;
    return false;
}

std::vector<RenjuCoord> RenjuForbiddenPointFinder::findForbiddenPoints() const {
    std::vector<RenjuCoord> result;
    for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
            if (getStone(x, y) == EMPTY && isForbidden(x, y)) {
                result.push_back({x, y});
            }
        }
    }
    return result;
}
