// Native Python bindings (pybind11) into the C++ engine - see api/zobrist.py
// for the Python-side wrapper. Kept deliberately tiny: this exists so cheap,
// frequent operations (hashing a position, computing its canonical form)
// can call the real Zobrist logic in-process instead of paying subprocess
// overhead for every call, the way api/engine.py's run_search() does for
// actual (expensive) MCTS search.
#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include "Zobrist.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

PenteGame::Config makeConfig(int boardSize) {
    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = std::max(1, std::min(19, boardSize));
    if (config.boardSize < 7 && config.tournamentRule) {
        config.tournamentRule = false; // doesn't fit on a board this small - see apps/Pente.cpp
    }
    return config;
}

PenteGame replayMoves(const std::vector<std::string> &moves, int boardSize) {
    PenteGame game(makeConfig(boardSize));
    game.reset();

    for (const auto &move : moves) {
        if (!game.makeMove(move.c_str())) {
            throw std::invalid_argument("Illegal move: " + move);
        }
    }
    return game;
}

std::string computeHash(const std::vector<std::string> &moves, int boardSize) {
    PenteGame game = replayMoves(moves, boardSize);
    return GameUtils::hashToHex(game.getHash());
}

std::pair<std::string, int> computeCanonicalHash(const std::vector<std::string> &moves, int boardSize) {
    PenteGame game = replayMoves(moves, boardSize);
    int sym = -1;
    uint64_t hash = game.getCanonicalHash(sym);
    return {GameUtils::hashToHex(hash), sym};
}

std::string applySymmetry(const std::string &move, int sym, bool inverse) {
    auto [x, y] = GameUtils::parseMove(move.c_str());
    if (x < 0 || y < 0) {
        throw std::invalid_argument("Illegal move: " + move);
    }

    int ox = -1, oy = -1;
    const Zobrist &zob = Zobrist::instance();
    if (inverse) {
        zob.applyInverseSym(sym, x, y, ox, oy);
    } else {
        zob.applySymToMove(sym, x, y, ox, oy);
    }
    return GameUtils::displayMove(ox, oy);
}

} // namespace

PYBIND11_MODULE(pente_native, m) {
    m.doc() = "Native bindings into the AlphaPente C++ engine";
    m.def("compute_hash", &computeHash, py::arg("moves"), py::arg("board_size") = 19,
          "Replay `moves` (e.g. [\"K10\", \"L9\"]) from an empty board and return the "
          "resulting position's Zobrist hash as a lowercase hex string. Raises "
          "ValueError on an illegal move.");
    m.def("compute_canonical_hash", &computeCanonicalHash, py::arg("moves"), py::arg("board_size") = 19,
          "Like compute_hash, but returns (hash, sym): the position's canonical-form "
          "hash - invariant under the 8 board symmetries, so positions that are "
          "rotations/reflections of each other share the same value - and which "
          "symmetry (0-7) maps this specific physical position to that canonical "
          "form. Raises ValueError on an illegal move.");
    m.def("apply_symmetry", &applySymmetry, py::arg("move"), py::arg("sym"), py::arg("inverse") = false,
          "Apply symmetry `sym` (or its inverse, if inverse=True) to a move label, "
          "returning the transformed label - e.g. to re-express a move found in one "
          "orientation in another orientation's coordinate frame. Raises ValueError "
          "if `move` isn't a well-formed move label.");
}
