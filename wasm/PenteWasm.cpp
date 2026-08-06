// Embind wrapper exposing PenteGame + single-threaded MCTS (HeuristicEvaluator)
// to the browser. Built separately from the native app via scripts/build_wasm.sh
// (Emscripten) — not part of the native CMake build.
#include "Evaluator.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include <emscripten/bind.h>

using namespace emscripten;

// Coordinates crossing this API are "local" (0..boardSize-1), not the physical
// 0..18 grid PenteGame centers the logical play area within.
class WasmGame {
  public:
    explicit WasmGame(int boardSize, int simulations)
        : origin_((PenteGame::BOARD_SIZE - boardSize) / 2), simulations_(simulations),
          game_(makeConfig(boardSize)), mcts_(makeMctsConfig()) {}

    void reset() {
        game_.reset();
        mcts_.clearTree();
    }

    // Changes AI search strength without touching the board/game state.
    void setSimulations(int simulations) {
        simulations_ = simulations;
        mcts_.setConfig(makeMctsConfig());
    }

    bool makeMove(int lx, int ly) {
        bool ok = game_.makeMove(lx + origin_, ly + origin_);
        if (ok)
            mcts_.clearTree(); // last search's tree no longer matches the position
        return ok;
    }

    // A full board with no legal moves left is a draw; PenteGame::isGameOver()
    // only checks for 5-in-a-row/capture wins, so also check for that here.
    bool isGameOver() const { return game_.isGameOver() || game_.getLegalMoves().empty(); }
    int getWinner() const { return static_cast<int>(game_.getWinner()); }
    int getCurrentPlayer() const { return static_cast<int>(game_.getCurrentPlayer()); }
    int getBoardSize() const { return game_.getConfig().boardSize; }
    int getBlackCaptures() const { return game_.getBlackCaptures(); }
    int getWhiteCaptures() const { return game_.getWhiteCaptures(); }

    int getStoneAt(int lx, int ly) const {
        return static_cast<int>(game_.getStoneAt(lx + origin_, ly + origin_));
    }

    // Runs a fresh search and returns the chosen move as {x, y}; does NOT apply
    // it. Call makeMove() separately once the caller is ready to commit it —
    // this leaves the searched tree in place so getTopMoves() reflects it.
    // Returns {x: -1, y: -1} if the game is already over (nothing to search).
    val computeAIMove() {
        if (isGameOver()) {
            val out = val::object();
            out.set("x", -1);
            out.set("y", -1);
            return out;
        }
        PenteGame::Move best = mcts_.search(game_);
        val out = val::object();
        out.set("x", best.x - origin_);
        out.set("y", best.y - origin_);
        return out;
    }

    // Top N candidate moves from the most recent computeAIMove() search.
    val getTopMoves(int topN) const {
        val out = val::array();
        for (const MCTS::TopMove &m : mcts_.getTopMoves(topN)) {
            val entry = val::object();
            entry.set("x", m.move.x - origin_);
            entry.set("y", m.move.y - origin_);
            entry.set("visits", m.visits);
            entry.set("value", m.avgValue);
            entry.set("puct", m.puct);
            entry.set("status", m.solvedStatus == MCTS::SolvedStatus::SOLVED_WIN    ? "WIN"
                                 : m.solvedStatus == MCTS::SolvedStatus::SOLVED_LOSS ? "LOSS"
                                 : m.solvedStatus == MCTS::SolvedStatus::SOLVED_DRAW ? "DRAW"
                                                                                     : "-");
            out.call<void>("push", entry);
        }
        return out;
    }

  private:
    static PenteGame::Config makeConfig(int boardSize) {
        PenteGame::Config cfg = PenteGame::Config::pente();
        cfg.boardSize = boardSize;
        // Tournament rule (3rd-move restriction) is a fixed distance-3 ring around
        // center; it doesn't fit inside a board smaller than 7x7.
        if (boardSize < 7)
            cfg.tournamentRule = false;
        return cfg;
    }

    MCTS::Config makeMctsConfig() {
        MCTS::Config cfg;
        cfg.evaluator = &evaluator_;
        cfg.maxIterations = simulations_;
        cfg.arenaSize = 64ull * 1024 * 1024; // 64MB is ample for a small board
        cfg.seed = 0;
        return cfg;
    }

    int origin_;
    int simulations_;
    PenteGame game_;
    HeuristicEvaluator evaluator_;
    MCTS mcts_;
};

EMSCRIPTEN_BINDINGS(pente_module) {
    class_<WasmGame>("Game")
        .constructor<int, int>()
        .function("reset", &WasmGame::reset)
        .function("setSimulations", &WasmGame::setSimulations)
        .function("makeMove", &WasmGame::makeMove)
        .function("isGameOver", &WasmGame::isGameOver)
        .function("getWinner", &WasmGame::getWinner)
        .function("getCurrentPlayer", &WasmGame::getCurrentPlayer)
        .function("getBoardSize", &WasmGame::getBoardSize)
        .function("getBlackCaptures", &WasmGame::getBlackCaptures)
        .function("getWhiteCaptures", &WasmGame::getWhiteCaptures)
        .function("getStoneAt", &WasmGame::getStoneAt)
        .function("computeAIMove", &WasmGame::computeAIMove)
        .function("getTopMoves", &WasmGame::getTopMoves);
}
