#pragma once
#include <vector>

struct RenjuCoord { int x, y; };

// Detects Renju forbidden points for black: overline, double-four, double-three.
// Pure board logic. 0-based external coordinates. Faithful C++ port of CForbiddenPointFinder (Wenzhe Lu).
class RenjuForbiddenPointFinder {
public:
    static constexpr char BLACK  = 'X';
    static constexpr char WHITE  = 'O';
    static constexpr char EMPTY  = '.';
    static constexpr char BORDER = '$';

    explicit RenjuForbiddenPointFinder(int size = 15);

    void clear();
    void setStone(int x, int y, char stone);
    char getStone(int x, int y) const;
    int  getSize() const { return size_; }

    // nColor 0 = black (exactly 5), 1 = white (>= 5)
    bool isFive(int x, int y, int nColor) const;
    bool isFive(int x, int y, int nColor, int dir) const;
    bool isOverline(int x, int y) const;
    bool isFour(int x, int y, int nColor, int dir) const;
    int  isOpenFour(int x, int y, int nColor, int dir) const; // 0/1/2
    bool isDoubleFour(int x, int y) const;
    bool isOpenThree(int x, int y, int nColor, int dir) const;
    bool isDoubleThree(int x, int y) const;
    bool isForbidden(int x, int y) const;

    std::vector<RenjuCoord> findForbiddenPoints() const;

private:
    int size_;
    mutable char b_[17][17]; // (size+2)x(size+2), 1-based playable region; max size=15 -> 17x17

    int countLine(int x, int y, char c, int dir) const;
};
