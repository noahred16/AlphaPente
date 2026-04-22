#include "NNInputEncoder.hpp"
#include <algorithm>
#include <cstring>

void NNInputEncoder::encode(const PenteGame &game, float *out) {
    std::memset(out, 0, TOTAL_SIZE * sizeof(float));

    PenteGame::Player current  = game.getCurrentPlayer();
    PenteGame::Player opponent = (current == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;

    // --- Planes 0 & 1: stone positions ---
    float *myStones  = out + 0 * PLANE_SIZE;
    float *oppStones = out + 1 * PLANE_SIZE;

    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            PenteGame::Player stone = game.getStoneAt(x, y);
            int idx = y * BOARD_SIZE + x;
            if (stone == current)
                myStones[idx] = 1.0f;
            else if (stone == opponent)
                oppStones[idx] = 1.0f;
        }
    }

    // --- Planes 2-5: scalar features broadcast to constant planes ---
    int capturesToWin = game.getConfig().capturesToWin;
    int myCaps        = (current == PenteGame::BLACK) ? game.getBlackCaptures() : game.getWhiteCaptures();
    int oppCaps       = (current == PenteGame::BLACK) ? game.getWhiteCaptures() : game.getBlackCaptures();

    float scalars[4] = {
        (current == PenteGame::BLACK) ? 1.0f : 0.0f,                        // plane 2: color
        (capturesToWin > 0) ? float(myCaps)  / float(capturesToWin) : 0.0f, // plane 3: my caps
        (capturesToWin > 0) ? float(oppCaps) / float(capturesToWin) : 0.0f, // plane 4: opp caps
        float(game.getMoveCount()) / float(BOARD_SIZE * BOARD_SIZE),         // plane 5: move count
    };

    for (int p = 0; p < 4; ++p) {
        float *plane = out + (2 + p) * PLANE_SIZE;
        std::fill(plane, plane + PLANE_SIZE, scalars[p]);
    }
}

std::vector<float> NNInputEncoder::encode(const PenteGame &game) {
    std::vector<float> out(TOTAL_SIZE);
    encode(game, out.data());
    return out;
}
