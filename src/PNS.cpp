#include "PNS.hpp"
#include "Zobrist.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

namespace {
const char *outcomeToString(PNS::Outcome o) {
    switch (o) {
    case PNS::Outcome::WIN:
        return "WIN";
    case PNS::Outcome::LOSS:
        return "LOSS";
    case PNS::Outcome::DRAW:
        return "DRAW";
    default:
        return "UNKNOWN";
    }
}
} // namespace

PNS::PNS(const Config &config) : config_(config) {}

std::vector<PenteGame::Move> PNS::enumerateLegalMoves(const PenteGame &game) {
    std::vector<PenteGame::Move> moves;
    if (game.getMoveCount() == 0) {
        int c = PenteGame::BOARD_SIZE / 2;
        moves.emplace_back(c, c);
        return moves;
    }
    const int lo = game.minIdx();
    const int hi = game.maxIdx();
    for (int y = lo; y < hi; ++y) {
        for (int x = lo; x < hi; ++x) {
            if (game.getStoneAt(x, y) == PenteGame::NONE) {
                moves.emplace_back(x, y);
            }
        }
    }
    return moves;
}

PNS::Node *PNS::getOrCreateNode(const PenteGame &game) {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    if (it != table_.end()) {
        stats_.transpositionHits++;
        return &it->second;
    }
    // std::unordered_map guarantees reference/pointer stability across
    // insertion and rehashing (only erasure invalidates), so this pointer
    // stays valid for the lifetime of the map even as more nodes are added.
    auto [insertedIt, inserted] = table_.emplace(key, Node{});
    assert(inserted);
    stats_.nodesCreated++;
    return &insertedIt->second;
}

void PNS::expandNode(Node *n, const PenteGame &game) {
    n->expanded = true;

    PenteGame::Player winner = game.getWinner();
    if (winner != PenteGame::NONE) {
        // The player to move at `game` inherited a lost position: their
        // opponent's last move already won.
        n->outcome = (winner == rootPlayer_) ? Outcome::WIN : Outcome::LOSS;
        n->pn = (n->outcome == Outcome::WIN) ? 0 : INF;
        n->dn = (n->outcome == Outcome::WIN) ? INF : 0;
        return;
    }

    std::vector<PenteGame::Move> moves = enumerateLegalMoves(game);
    if (moves.empty()) {
        // Every cell in the logical window is occupied, no winner: draw.
        n->outcome = Outcome::DRAW;
        n->pn = INF;
        n->dn = 0;
        return;
    }

    // Order best-first by the mover's own heuristic score, purely to help
    // df-pn's descent find proofs/disproofs faster - every legal move is
    // still present in childMoves, so this doesn't affect completeness.
    std::vector<std::pair<PenteGame::Move, float>> scored;
    scored.reserve(moves.size());
    for (const auto &m : moves) scored.emplace_back(m, game.evaluateMove(m));
    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    // Store childMoves in CANONICAL coordinates, not physical: this node may
    // be shared (via the transposition table) with a parent reached through a
    // different physical orientation than the one doing this particular
    // expansion. mid() re-derives each visiting parent's own symmetry fresh
    // and un-rotates before calling makeMove() - see the comment there. This
    // mirrors MCTS::expand()'s identical rotation-to-canonical step.
    int canonSym = -1;
    PositionKey::canonical(game, canonSym);
    const auto &zob = Zobrist::instance();
    n->childMoves.reserve(scored.size());
    for (const auto &entry : scored) {
        int cx, cy;
        zob.applySymToMove(canonSym, entry.first.x, entry.first.y, cx, cy);
        n->childMoves.emplace_back(cx, cy);
    }
    n->childPtr.assign(n->childMoves.size(), nullptr);

    bool isOr = (game.getCurrentPlayer() == rootPlayer_);
    updatePnDn(n, isOr); // every child still untried (default pn=dn=1)
}

void PNS::updatePnDn(Node *n, bool isOrNode) const {
    if (isOrNode) {
        Number pn = INF, dn = 0;
        for (Node *c : n->childPtr) {
            pn = std::min(pn, pnOf(c));
            dn = std::min(INF, dn + dnOf(c));
        }
        n->pn = pn;
        n->dn = dn;
    } else {
        Number pn = 0, dn = INF;
        for (Node *c : n->childPtr) {
            pn = std::min(INF, pn + pnOf(c));
            dn = std::min(dn, dnOf(c));
        }
        n->pn = pn;
        n->dn = dn;
    }
}

template <typename Predicate>
uint16_t PNS::depthFrom(const std::vector<Node *> &children, bool useMax, Predicate want) {
    bool any = false;
    uint16_t best = useMax ? 0 : std::numeric_limits<uint16_t>::max();
    for (Node *c : children) {
        if (!c || !want(c)) continue;
        any = true;
        best = useMax ? std::max(best, c->depth) : std::min(best, c->depth);
    }
    assert(any);
    return static_cast<uint16_t>(best + 1);
}

void PNS::resolveOutcome(Node *n, bool isOrNode) const {
    if (n->pn == 0) {
        n->outcome = Outcome::WIN;
        // OR: root picks whichever winning reply is fastest (min). AND: every
        // reply already forces a win (pn=sum==0 needs every term 0), so the
        // guarantee is only as fast as the opponent's best (slowest) defense
        // (max).
        n->depth = depthFrom(n->childPtr, /*useMax=*/!isOrNode,
                              [](Node *c) { return c->outcome == Outcome::WIN; });
        return;
    }
    // dn == 0 here (mid()'s caller only calls this when pn==0 or dn==0).
    if (isOrNode) {
        // dn = sum(children dn) == 0 requires every child individually
        // resolved to non-WIN (a WIN child would have set pn=0 above already).
        bool anyDraw = false;
        for (Node *c : n->childPtr) {
            assert(c != nullptr && c->outcome != Outcome::UNKNOWN && c->outcome != Outcome::WIN);
            if (c->outcome == Outcome::DRAW) anyDraw = true;
        }
        // Root, moving here, simply avoids the losing replies: DRAW if any
        // move preserves one, else every move loses.
        if (anyDraw) {
            n->outcome = Outcome::DRAW;
            // Root picks the fastest draw among its options.
            n->depth = depthFrom(n->childPtr, /*useMax=*/false, [](Node *c) { return c->outcome == Outcome::DRAW; });
        } else {
            n->outcome = Outcome::LOSS;
            // Every move loses; root delays the inevitable as long as possible.
            n->depth = depthFrom(n->childPtr, /*useMax=*/true, [](Node *c) { return c->outcome == Outcome::LOSS; });
        }
    } else {
        // dn = min(children dn) == 0 needs only one resolved non-WIN child;
        // others may still be untried/unresolved - ignore them, that's the
        // point of proof numbers (don't wait for exhaustive exploration once
        // an escape is found). Opponent is adversarial to root: prefers a
        // LOSS-for-root escape over a DRAW-for-root one if both are known.
        bool anyLoss = false, anyDraw = false;
        for (Node *c : n->childPtr) {
            if (!c) continue;
            if (c->outcome == Outcome::LOSS) anyLoss = true;
            else if (c->outcome == Outcome::DRAW) anyDraw = true;
        }
        assert(anyLoss || anyDraw);
        if (anyLoss) {
            n->outcome = Outcome::LOSS;
            // Opponent takes their fastest win (root's fastest loss).
            n->depth = depthFrom(n->childPtr, /*useMax=*/false, [](Node *c) { return c->outcome == Outcome::LOSS; });
        } else {
            n->outcome = Outcome::DRAW;
            n->depth = depthFrom(n->childPtr, /*useMax=*/false, [](Node *c) { return c->outcome == Outcome::DRAW; });
        }
    }
}

void PNS::selectChildOr(const Node *n, int &bestIdx, Number &secondPn) const {
    Number best = INF, second = INF;
    int bi = -1;
    for (size_t i = 0; i < n->childPtr.size(); ++i) {
        Number p = pnOf(n->childPtr[i]);
        if (p < best) {
            second = best;
            best = p;
            bi = static_cast<int>(i);
        } else if (p < second) {
            second = p;
        }
    }
    bestIdx = bi;
    secondPn = second;
}

void PNS::selectChildAnd(const Node *n, int &bestIdx, Number &secondDn) const {
    Number best = INF, second = INF;
    int bi = -1;
    for (size_t i = 0; i < n->childPtr.size(); ++i) {
        Number d = dnOf(n->childPtr[i]);
        if (d < best) {
            second = best;
            best = d;
            bi = static_cast<int>(i);
        } else if (d < second) {
            second = d;
        }
    }
    bestIdx = bi;
    secondDn = second;
}

void PNS::mid(Node *n, PenteGame game, Number thpn, Number thdn) {
    if (stopRequested_) return;
    stats_.midCalls++;

    if (!n->expanded) {
        expandNode(n, game);
        return;
    }
    if (n->outcome != Outcome::UNKNOWN) return;

    const bool isOr = (game.getCurrentPlayer() == rootPlayer_);

    // childMoves are stored in canonical coordinates (see expandNode); this
    // parent's own `game` may be a different physical orientation than
    // whichever parent first expanded this (possibly shared) node, so always
    // re-derive the current orientation's symmetry fresh here - never assume
    // it matches whatever orientation created the node. Fixed for the
    // duration of this call (game doesn't change outside the loop below).
    int currentSym = -1;
    PositionKey::canonical(game, currentSym);
    const auto &zob = Zobrist::instance();

    while (true) {
        if (stopRequested_) return;

        updatePnDn(n, isOr);
        if (n->pn == 0 || n->dn == 0) {
            resolveOutcome(n, isOr);
            return;
        }
        if (n->pn >= thpn || n->dn >= thdn) return;

        int bestIdx = -1;
        Number secondVal = INF;
        if (isOr) selectChildOr(n, bestIdx, secondVal);
        else selectChildAnd(n, bestIdx, secondVal);
        assert(bestIdx >= 0);
        const size_t bi = static_cast<size_t>(bestIdx);

        int physX, physY;
        zob.applyInverseSym(currentSym, n->childMoves[bi].x, n->childMoves[bi].y, physX, physY);
        PenteGame childGame = game;
        childGame.makeMove(physX, physY);

        if (!n->childPtr[bi]) {
            if (table_.size() >= config_.maxNodes) {
                stopRequested_ = true;
                return;
            }
            n->childPtr[bi] = getOrCreateNode(childGame);
        }
        Node *child = n->childPtr[bi];

        Number childThPn, childThDn;
        if (isOr) {
            childThPn = (secondVal >= INF) ? thpn : std::min(thpn, secondVal + 1);
            childThDn = (thdn >= INF) ? INF : std::min(INF, thdn - n->dn + dnOf(child));
        } else {
            childThDn = (secondVal >= INF) ? thdn : std::min(thdn, secondVal + 1);
            childThPn = (thpn >= INF) ? INF : std::min(INF, thpn - n->pn + pnOf(child));
        }

        mid(child, std::move(childGame), childThPn, childThDn);
        // Loop back: re-derive this node's pn/dn from (possibly now-updated)
        // children and either resolve, bail on thresholds, or pick again.
    }
}

bool PNS::solve(const PenteGame &rootGame) {
    assert(!rootGame.getConfig().tournamentRule &&
           "PNS requires tournamentRule disabled (matches apps/Pente.cpp's boardSize<7 auto-disable convention) "
           "since its exhaustive move enumeration doesn't reproduce the tournament-rule perimeter restriction");
    assert(!rootGame.getConfig().renjuForbiddenMoves && "PNS does not support Renju forbidden-move rules");
    assert(rootGame.getConfig().boardSize <= PositionKey::kMaxBoardSize &&
           "PositionKey packing only supports boardSize <= 5");

    table_.clear();
    stats_ = Stats{};
    stopRequested_ = false;
    rootPlayer_ = rootGame.getCurrentPlayer();

    rootNode_ = getOrCreateNode(rootGame);
    // mid() returns after merely expanding a not-yet-expanded node (standard
    // df-pn "MID" behavior - see the class-level comment), relying on its
    // caller to re-invoke it to actually descend. Every recursive call gets
    // this for free from its parent's own while(true) loop; the root has no
    // such parent, so solve() must play that role itself.
    while (rootNode_->outcome == Outcome::UNKNOWN && !stopRequested_) {
        mid(rootNode_, rootGame, INF, INF);
    }

    return rootNode_->outcome != Outcome::UNKNOWN;
}

PNS::Outcome PNS::getRootOutcome() const { return rootNode_ ? rootNode_->outcome : Outcome::UNKNOWN; }

int PNS::getRootDepth() const { return rootNode_ ? static_cast<int>(rootNode_->depth) : 0; }

PNS::Outcome PNS::getOutcome(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    return (it != table_.end()) ? it->second.outcome : Outcome::UNKNOWN;
}

int PNS::getDepth(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    return (it != table_.end()) ? static_cast<int>(it->second.depth) : 0;
}

uint64_t PNS::getNodeCount() const { return table_.size(); }

std::vector<PNS::Record> PNS::exportResolved() const {
    std::vector<Record> out;
    out.reserve(table_.size());
    for (const auto &entry : table_) {
        if (entry.second.outcome != Outcome::UNKNOWN) {
            out.push_back({entry.first, entry.second.outcome, entry.second.depth});
        }
    }
    return out;
}

void PNS::printProofStats() const {
    std::cout << "PNS: nodes=" << table_.size() << " midCalls=" << stats_.midCalls
              << " transpositionHits=" << stats_.transpositionHits;
    if (rootNode_) {
        std::cout << " root(pn=" << rootNode_->pn << ", dn=" << rootNode_->dn
                   << ", outcome=" << outcomeToString(rootNode_->outcome) << ", depth=" << rootNode_->depth << ")";
    }
    std::cout << "\n";
}
