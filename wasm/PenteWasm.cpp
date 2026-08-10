// Embind wrapper exposing PenteGame + single-threaded MCTS (HeuristicEvaluator)
// to the browser. Built separately from the native app via scripts/build_wasm.sh
// (Emscripten) — not part of the native CMake build.
#include "Evaluator.hpp"
#include "MCTS.hpp"
#include "PNS.hpp"
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

// Book/live-solve entries are always from Black's perspective (Black is
// always the root player either was built/solved from). Ranks a result from
// the perspective of whoever is actually choosing: 2=mover wins, 1=draw,
// 0=mover loses.
int rankForMover(PNS::Outcome outcome, bool moverIsBlack) {
    if (outcome == PNS::Outcome::DRAW) return 1;
    bool blackWins = (outcome == PNS::Outcome::WIN);
    return (blackWins == moverIsBlack) ? 2 : 0;
}

struct RankedMove {
    PenteGame::Move move;
    PNS::Outcome outcome;
    uint16_t depth;
};

// Shared by the book path and the live-PNS-fallback path (see WasmGame
// below) - both are "exact outcome+depth per legal reply", just from
// different sources. `lookupChild(childGame) -> optional<PositionBook::Entry>`
// is the only thing that differs between them. Empty return means none of
// the current position's legal replies were covered by whatever source
// `lookupChild` draws from.
template <typename LookupFn>
std::vector<RankedMove> rankMoves(const PenteGame &game, LookupFn &&lookupChild) {
    bool moverIsBlack = (game.getCurrentPlayer() == PenteGame::BLACK);
    std::vector<RankedMove> scored;
    for (const auto &move : enumerateLegalMoves(game)) {
        PenteGame child = game.clone();
        child.makeMove(move.x, move.y);
        auto entry = lookupChild(child);
        if (entry) scored.push_back({move, entry->outcome, entry->depth});
    }
    std::sort(scored.begin(), scored.end(), [moverIsBlack](const RankedMove &a, const RankedMove &b) {
        int ra = rankForMover(a.outcome, moverIsBlack);
        int rb = rankForMover(b.outcome, moverIsBlack);
        if (ra != rb) return ra > rb;
        return ra == 0 ? a.depth > b.depth : a.depth < b.depth; // slowest loss, else fastest win/draw
    });
    return scored;
}

} // namespace

// Coordinates crossing this API are "local" (0..boardSize-1), not the physical
// 0..18 grid PenteGame centers the logical play area within.
class WasmGame {
  public:
    explicit WasmGame(int boardSize, int simulations)
        : origin_((PenteGame::BOARD_SIZE - boardSize) / 2), simulations_(simulations),
          game_(makeConfig(boardSize)), mcts_(makeMctsConfig()), livePns_(makeLivePnsConfig()) {}

    // Loads a solved book (docs/data/book4x4.bin today) from raw bytes the
    // JS side fetched over HTTP - deliberately NOT auto-loaded via
    // Emscripten's --preload-file, which would force the whole book to
    // download before the WASM module is even ready, on every page load,
    // regardless of which board size the user actually picks. JS is
    // expected to only fetch+call this when boardSize()==4 is selected, and
    // can cache the bytes across Game instances to skip re-fetching. Returns
    // whether the book parsed successfully; usingBook() reflects the result.
    //
    // The book itself may be TRIMMED to a shallow moveCount (see
    // apps/Solve5x5.cpp's -m flag) - once play runs past its coverage,
    // rankedMoves() below falls back to a live PNS::solve() from the
    // current position, which the calibration notes in the project's
    // solve-5x5 issue doc found to be cheap (sub-25ms even from a fairly
    // early, "open" position) for a board this size.
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
    // No effect when a book is backing this board size (lookups/live-solve
    // fallback are exact either way) - kept a no-op rather than an error so
    // the UI doesn't need to special-case disabling the control.
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

    // Runs a fresh search (book lookup, live-solve fallback, or MCTS) and
    // returns the chosen move as {x, y}; does NOT apply it. Call makeMove()
    // separately once the caller is ready to commit it — this leaves the
    // searched tree/ranking in place so getTopMoves() reflects it. Returns
    // {x: -1, y: -1} if the game is already over (nothing to search).
    val computeAIMove() {
        if (isGameOver()) {
            val out = val::object();
            out.set("x", -1);
            out.set("y", -1);
            return out;
        }
        auto ranked = rankedMoves();
        PenteGame::Move best = ranked.empty() ? mcts_.search(game_) : ranked.front().move;
        val out = val::object();
        out.set("x", best.x - origin_);
        out.set("y", best.y - origin_);
        return out;
    }

    // Top N candidate moves: book (or, past its coverage, a live PNS solve -
    // see rankedMoves()) when available, otherwise the most recent
    // computeAIMove() MCTS search.
    val getTopMoves(int topN) const {
        val out = val::array();
        auto ranked = rankedMoves();
        if (!ranked.empty()) {
            bool moverIsBlack = (game_.getCurrentPlayer() == PenteGame::BLACK);
            int n = std::min<int>(topN, static_cast<int>(ranked.size()));
            for (int i = 0; i < n; ++i) {
                const RankedMove &r = ranked[static_cast<size_t>(i)];
                int rank = rankForMover(r.outcome, moverIsBlack);
                val entry = val::object();
                entry.set("x", r.move.x - origin_);
                entry.set("y", r.move.y - origin_);
                entry.set("visits", 0);
                entry.set("value", 0.0);
                entry.set("puct", 0.0);
                entry.set("depth", static_cast<int>(r.depth));
                entry.set("status", rank == 2 ? "WIN" : rank == 1 ? "DRAW" : "LOSS");
                out.call<void>("push", entry);
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

    // Generous relative to what a live query actually needs (see this
    // class's loadBookFromBytes() comment: sub-25ms/under 3000 nodes even
    // from an early, unfavorable position in practice) - these are safety
    // caps against a pathological case freezing the browser tab, not
    // expected to bind in normal play. maxRecursionDepth must stay within
    // what scripts/build_wasm.sh's -s STACK_SIZE actually grants (each level
    // takes a full PenteGame - 8KB+ - by value; see PNS::Config's own
    // comment for the underlying reason this exists at all).
    static PNS::Config makeLivePnsConfig() {
        PNS::Config cfg;
        cfg.maxNodes = 5'000'000;
        cfg.maxSeconds = 5.0;
        cfg.maxRecursionDepth = 60;
        return cfg;
    }

    // Book lookup first; if the current position's replies aren't covered
    // (either no book at all, or - if the book was trimmed, see
    // apps/Solve5x5.cpp's -m flag - play has gone past its depth), and a
    // book exists at all (so we're on a board size PNS actually supports),
    // falls back to a fresh live PNS::solve() from the current position.
    // Empty return means neither source covers it (no book at all - the
    // MCTS callers already handle that).
    std::vector<RankedMove> rankedMoves() const {
        if (!hasBook_) return {};

        auto bookRanked = rankMoves(game_, [this](const PenteGame &child) { return book_.lookup(child); });
        if (!bookRanked.empty()) return bookRanked;

        if (!livePns_.solve(game_)) return {}; // budget exceeded (see makeLivePnsConfig) - falls back to MCTS
        return rankMoves(game_, [this](const PenteGame &child) -> std::optional<PositionBook::Entry> {
            PNS::Outcome outcome = livePns_.getOutcome(child);
            if (outcome == PNS::Outcome::UNKNOWN) return std::nullopt;
            return PositionBook::Entry{outcome, static_cast<uint16_t>(livePns_.getDepth(child))};
        });
    }

    int origin_;
    int simulations_;
    PenteGame game_;
    HeuristicEvaluator evaluator_;
    MCTS mcts_;
    PositionBook book_;
    bool hasBook_ = false;
    mutable PNS livePns_; // rankedMoves() is logically read-only but solve() mutates PNS's internal DAG
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
