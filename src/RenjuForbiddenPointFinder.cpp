#include "RenjuForbiddenPointFinder.hpp"
#include <algorithm>

// Faithful C++ port of CForbiddenPointFinder (Wenzhe Lu). Internal storage is 1-based with a
// one-cell BORDER ring so line scans terminate naturally at the edge without bounds checks.
//
// Convention shared by every public query below: (x, y) must be EMPTY on entry. Each function
// virtually places the relevant stone at (x, y), evaluates the pattern using whatever real/virtual
// stones already sit in the grid, then reverts (x, y) back to EMPTY before returning. This lets
// the functions nest (isDoubleThree -> isOpenThree -> isOpenFour -> isFive) with each level only
// ever touching its own single cell, and lets every concept be unit-tested in isolation by simply
// building a board and calling the query directly.

namespace {
// The 4 canonical line directions; index 0..3 = forward, 4..7 = the same lines reversed.
constexpr int kDirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
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
// `dir` (0..3 forward, 4..7 the reverse of 0..3). Does not include (x, y) itself.
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
    char c = (nColor == 0) ? BLACK : WHITE;
    setStone(x, y, c);
    int count = 1 + countLine(x, y, c, dir) + countLine(x, y, c, dir + 4);
    setStone(x, y, EMPTY);

    // Black needs exactly five: a run of six or more is an overline, not a win.
    return (nColor == 0) ? (count == 5) : (count >= 5);
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
    char c = (nColor == 0) ? BLACK : WHITE;
    setStone(x, y, c);

    bool result = false;
    int ddx = kDirs[dir][0], ddy = kDirs[dir][1];
    for (int i = -4; i <= 4 && !result; i++) {
        if (i == 0) continue;
        int px = x + ddx * i, py = y + ddy * i;
        if (getStone(px, py) != EMPTY) continue;
        if (isFive(px, py, nColor, dir)) result = true;
    }

    setStone(x, y, EMPTY);
    return result;
}

// Counts how many distinct empty points along dir would complete a five if filled (0, 1, or 2).
// 2 means a genuine open four (e.g. "_XXXX_"): both completions are live, so it cannot be blocked
// with a single stone.
int RenjuForbiddenPointFinder::isOpenFour(int x, int y, int nColor, int dir) const {
    char c = (nColor == 0) ? BLACK : WHITE;
    setStone(x, y, c);

    int count = 0;
    int ddx = kDirs[dir][0], ddy = kDirs[dir][1];
    for (int i = -4; i <= 4; i++) {
        if (i == 0) continue;
        int px = x + ddx * i, py = y + ddy * i;
        if (getStone(px, py) != EMPTY) continue;
        if (isFive(px, py, nColor, dir)) count++;
    }

    setStone(x, y, EMPTY);
    return std::min(count, 2);
}

// True if placing nColor at (x, y) creates a real "three" along dir: an empty point within reach
// which, if filled by nColor, produces a genuine open four (isOpenFour == 2). A three whose only
// follow-up is a simple (one-sided) four doesn't count - the opponent could block it with one move.
bool RenjuForbiddenPointFinder::isOpenThree(int x, int y, int nColor, int dir) const {
    char c = (nColor == 0) ? BLACK : WHITE;
    setStone(x, y, c);

    bool result = false;
    int ddx = kDirs[dir][0], ddy = kDirs[dir][1];
    for (int i = -4; i <= 4 && !result; i++) {
        if (i == 0) continue;
        int px = x + ddx * i, py = y + ddy * i;
        if (getStone(px, py) != EMPTY) continue;
        if (isOpenFour(px, py, nColor, dir) == 2) result = true;
    }

    setStone(x, y, EMPTY);
    return result;
}

// Double-four: two or more distinct four-lines through (x, y). Counting by direction (rather than
// by gain-point) is what keeps a single open four ("_XXXX_", which has two gain points but is one
// line) from being miscounted as a double-four.
bool RenjuForbiddenPointFinder::isDoubleFour(int x, int y) const {
    int count = 0;
    for (int dir = 0; dir < 4; dir++) {
        if (isFour(x, y, 0, dir)) count++;
    }
    return count >= 2;
}

bool RenjuForbiddenPointFinder::isDoubleThree(int x, int y) const {
    int count = 0;
    for (int dir = 0; dir < 4; dir++) {
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
