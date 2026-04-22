#ifndef NN_INPUT_ENCODER_HPP
#define NN_INPUT_ENCODER_HPP

#include "PenteGame.hpp"
#include <vector>

// Encodes a PenteGame state into a float tensor for neural network input.
//
// Layout: [NUM_PLANES × BOARD_SIZE × BOARD_SIZE], channel-first (CHW),
// matching PyTorch's default tensor format. All values are in [0.0, 1.0].
//
// Planes — always from the current player's perspective:
//   0: current player's stones (1.0 where stone exists, 0.0 elsewhere)
//   1: opponent's stones
//   2: color indicator — 1.0 if current player is Black, 0.0 if White (constant plane)
//   3: current player's captures / capturesToWin (constant plane)
//   4: opponent's captures / capturesToWin (constant plane)
//   5: move count / (BOARD_SIZE * BOARD_SIZE) (constant plane)
class NNInputEncoder {
  public:
    static constexpr int BOARD_SIZE = PenteGame::BOARD_SIZE; // 19
    static constexpr int NUM_PLANES = 6;
    static constexpr int PLANE_SIZE = BOARD_SIZE * BOARD_SIZE; // 361
    static constexpr int TOTAL_SIZE = NUM_PLANES * PLANE_SIZE; // 2166

    // Fills `out` (caller must provide TOTAL_SIZE floats) with the encoded state.
    static void encode(const PenteGame &game, float *out);

    // Convenience overload — returns a heap-allocated vector.
    static std::vector<float> encode(const PenteGame &game);
};

#endif // NN_INPUT_ENCODER_HPP
