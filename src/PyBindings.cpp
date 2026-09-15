// Native Python bindings (pybind11) into the C++ engine - see api/zobrist.py
// for the Python-side wrapper. Kept deliberately tiny: this exists so a
// cheap, frequent operation (hashing a position) can call the real Zobrist
// logic in-process instead of paying subprocess overhead for every call, the
// way api/engine.py's run_search() does for actual (expensive) MCTS search.
#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::string computeHash(const std::vector<std::string> &moves, int boardSize) {
    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = std::max(1, std::min(19, boardSize));
    if (config.boardSize < 7 && config.tournamentRule) {
        config.tournamentRule = false; // doesn't fit on a board this small - see apps/Pente.cpp
    }

    PenteGame game(config);
    game.reset();

    for (const auto &move : moves) {
        if (!game.makeMove(move.c_str())) {
            throw std::invalid_argument("Illegal move: " + move);
        }
    }

    return GameUtils::hashToHex(game.getHash());
}

} // namespace

PYBIND11_MODULE(pente_native, m) {
    m.doc() = "Native bindings into the AlphaPente C++ engine";
    m.def("compute_hash", &computeHash, py::arg("moves"), py::arg("board_size") = 19,
          "Replay `moves` (e.g. [\"K10\", \"L9\"]) from an empty board and return the "
          "resulting position's Zobrist hash as a lowercase hex string. Raises "
          "ValueError on an illegal move.");
}
