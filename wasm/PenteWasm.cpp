// Embind wrapper exposing PenteGame + single-threaded MCTS (HeuristicEvaluator)
// to the browser. Built separately from the native app via scripts/build_wasm.sh
// (Emscripten) — not part of the native CMake build.
#include "Evaluator.hpp"
#include "MCTS.hpp"
#include "PenteGame.hpp"
#include "PositionBook.hpp"
#include <algorithm>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;

namespace {

// Move enumeration must match PNS's own exhaustive enumeration exactly
// (every empty cell in the logical window, forced center on move 0) - the
// book was built by PNS::solveExhaustive() using exactly this, not
// PenteGame::getLegalMoves() (a heuristic neighborhood restriction - see
// PNS.hpp's class comment for why that's unsound for a solved-position
// lookup, even though it's fine for MCTS's own heuristic play elsewhere in
// this same file).
std::vector<PenteGame::Move> enumerateLegalMoves(const PenteGame &g) {
    std::vector<PenteGame::Move> moves;
    if (g.getMoveCount() == 0) {
        int c = PenteGame::BOARD_SIZE / 2;
        moves.emplace_back(c, c);
        return moves;
    }
    for (int y = g.minIdx(); y < g.maxIdx(); ++y) {
        for (int x = g.minIdx(); x < g.maxIdx(); ++x) {
            if (g.getStoneAt(x, y) == PenteGame::NONE) moves.emplace_back(x, y);
        }
    }
    return moves;
}

// Book entries are always stored from Black's perspective (Black is always
// the root player a book was generated from the empty board with). Ranks a
// result from the perspective of whoever is actually choosing: 2=mover
// wins, 1=draw, 0=mover loses.
int rankForMover(PNS::Outcome outcome, bool moverIsBlack) {
    if (outcome == PNS::Outcome::DRAW) return 1;
    bool blackWins = (outcome == PNS::Outcome::WIN);
    return (blackWins == moverIsBlack) ? 2 : 0;
}

} // namespace

// Coordinates crossing this API are "local" (0..boardSize-1), not the physical
// 0..18 grid PenteGame centers the logical play area within.
class WasmGame {
  public:
    explicit WasmGame(int boardSize, int simulations)
        : origin_((PenteGame::BOARD_SIZE - boardSize) / 2), simulations_(simulations),
          game_(makeConfig(boardSize)), mcts_(makeMctsConfig()) {}

    // Loads a solved book (docs/data/book4x4.bin today) from raw bytes the
    // JS side fetched over HTTP - deliberately NOT auto-loaded via
    // Emscripten's --preload-file, which would force the whole ~67MB book to
    // download before the WASM module is even ready, on every page load,
    // regardless of which board size the user actually picks. JS is
    // expected to only fetch+call this when boardSize()==4 is selected, and
    // can cache the bytes across Game instances to skip re-fetching. Returns
    // whether the book parsed successfully; usingBook() reflects the result.
    bool loadBookFromBytes(val jsBytes) {
        std::vector<uint8_t> bytes = vecFromJSArray<uint8_t>(jsBytes);
        hasBook_ = book_.loadFromMemory(bytes.data(), bytes.size());
        return hasBook_;
    }

    void reset() {
        game_.reset();
        mcts_.clearTree();
    }

    // Changes AI search strength without touching the board/game state.
    // No effect when a book is backing this board size (lookups are exact
    // and instant either way) - kept a no-op rather than an error so the UI
    // doesn't need to special-case disabling the control.
    void setSimulations(int simulations) {
        simulations_ = simulations;
        mcts_.setConfig(makeMctsConfig());
    }

    bool makeMove(int lx, int ly) {
        int x = lx + origin_, y = ly + origin_;
        // PenteGame::makeMove(x, y) is a trusted low-level primitive that skips
        // legality checks (relied on by MCTS/tests); the UI boundary must check
        // isLegalMove() itself, e.g. to enforce the center-only opening move.
        if (!game_.isLegalMove(x, y))
            return false;
        bool ok = game_.makeMove(x, y);
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
    bool usingBook() const { return hasBook_; }

    int getStoneAt(int lx, int ly) const {
        return static_cast<int>(game_.getStoneAt(lx + origin_, ly + origin_));
    }

    // Book-driven lookup: instant, exact, no "search" involved - every legal
    // reply is already resolved. Returns Move::INVALID if the current
    // position somehow isn't book-covered (shouldn't happen once hasBook_ is
    // true, given the book is a full solve, but defensive nonetheless).
    PenteGame::Move bestBookMove() const {
        bool moverIsBlack = (game_.getCurrentPlayer() == PenteGame::BLACK);
        PenteGame::Move best;
        int bestRank = -1;
        uint16_t bestDepth = 0;
        for (const auto &move : enumerateLegalMoves(game_)) {
            PenteGame child = game_.clone();
            child.makeMove(move.x, move.y);
            auto entry = book_.lookup(child);
            if (!entry) continue;
            int rank = rankForMover(entry->outcome, moverIsBlack);
            bool better = bestRank < 0 || rank > bestRank ||
                          (rank == bestRank && rank != 0 && entry->depth < bestDepth) || // fastest win/draw
                          (rank == bestRank && rank == 0 && entry->depth > bestDepth);   // slowest loss
            if (better) {
                best = move;
                bestRank = rank;
                bestDepth = entry->depth;
            }
        }
        return best;
    }

    // Runs a fresh search (or, with a book, an instant lookup) and returns
    // the chosen move as {x, y}; does NOT apply it. Call makeMove()
    // separately once the caller is ready to commit it — this leaves the
    // searched tree (or book selection) in place so getTopMoves() reflects
    // it. Returns {x: -1, y: -1} if the game is already over (nothing to
    // search).
    val computeAIMove() {
        if (isGameOver()) {
            val out = val::object();
            out.set("x", -1);
            out.set("y", -1);
            return out;
        }
        PenteGame::Move best = hasBook_ ? bestBookMove() : mcts_.search(game_);
        val out = val::object();
        out.set("x", best.x - origin_);
        out.set("y", best.y - origin_);
        return out;
    }

    // Top N candidate moves. From the book when available (every legal
    // reply, ranked by outcome then by depth - see rankForMover), otherwise
    // from the most recent computeAIMove() MCTS search.
    val getTopMoves(int topN) const {
        val out = val::array();
        if (hasBook_) {
            bool moverIsBlack = (game_.getCurrentPlayer() == PenteGame::BLACK);
            std::vector<std::pair<PenteGame::Move, PositionBook::Entry>> scored;
            for (const auto &move : enumerateLegalMoves(game_)) {
                PenteGame child = game_.clone();
                child.makeMove(move.x, move.y);
                auto entry = book_.lookup(child);
                if (entry) scored.emplace_back(move, *entry);
            }
            std::sort(scored.begin(), scored.end(), [moverIsBlack](const auto &a, const auto &b) {
                int ra = rankForMover(a.second.outcome, moverIsBlack);
                int rb = rankForMover(b.second.outcome, moverIsBlack);
                if (ra != rb) return ra > rb;
                return ra == 0 ? a.second.depth > b.second.depth : a.second.depth < b.second.depth;
            });
            int n = std::min<int>(topN, static_cast<int>(scored.size()));
            for (int i = 0; i < n; ++i) {
                const auto &[move, entry] = scored[static_cast<size_t>(i)];
                int rank = rankForMover(entry.outcome, moverIsBlack);
                val out_entry = val::object();
                out_entry.set("x", move.x - origin_);
                out_entry.set("y", move.y - origin_);
                out_entry.set("visits", 0);
                out_entry.set("value", 0.0);
                out_entry.set("puct", 0.0);
                out_entry.set("depth", static_cast<int>(entry.depth));
                out_entry.set("status", rank == 2 ? "WIN" : rank == 1 ? "DRAW" : "LOSS");
                out.call<void>("push", out_entry);
            }
            return out;
        }
        for (const MCTS::TopMove &m : mcts_.getTopMoves(topN)) {
            val entry = val::object();
            entry.set("x", m.move.x - origin_);
            entry.set("y", m.move.y - origin_);
            entry.set("visits", m.visits);
            entry.set("value", m.avgValue);
            entry.set("puct", m.puct);
            entry.set("depth", 0);
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
    PositionBook book_;
    bool hasBook_ = false;
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
        .function("usingBook", &WasmGame::usingBook)
        .function("loadBookFromBytes", &WasmGame::loadBookFromBytes)
        .function("getStoneAt", &WasmGame::getStoneAt)
        .function("computeAIMove", &WasmGame::computeAIMove)
        .function("getTopMoves", &WasmGame::getTopMoves);
}
