#include "RenjuRules.hpp"

namespace RenjuRules {

bool isOverline(const BitBoard &black, int x, int y) {
    static constexpr int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

    for (auto &d : dirs) {
        int dx = d[0], dy = d[1];
        int count = 1; // the virtual stone at (x, y)

        int nx = x + dx, ny = y + dy;
        while (black.getBit(nx, ny)) {
            count++;
            nx += dx;
            ny += dy;
        }

        nx = x - dx;
        ny = y - dy;
        while (black.getBit(nx, ny)) {
            count++;
            nx -= dx;
            ny -= dy;
        }

        if (count >= 6) return true;
    }

    return false;
}

} // namespace RenjuRules
