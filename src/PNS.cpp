#include "PNS.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
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

uint32_t PNS::getOrCreateNodeIdx(const PenteGame &game) {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    if (it != table_.end()) {
        stats_.transpositionHits++;
        return it->second;
    }
    uint32_t idx = static_cast<uint32_t>(nodeArena_.size());
    nodeArena_.emplace_back();
    table_.emplace(key, idx);
    stats_.nodesCreated++;
    return idx;
}

PNS::Node *PNS::getOrCreateNode(const PenteGame &game) { return &nodeArena_[getOrCreateNodeIdx(game)]; }

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
    // still present in children, so this doesn't affect completeness.
    std::vector<std::pair<PenteGame::Move, float>> scored;
    scored.reserve(moves.size());
    for (const auto &m : moves) scored.emplace_back(m, game.evaluateMove(m));
    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    // Store each child's move in CANONICAL coordinates, not physical: this
    // node may be shared (via the transposition table) with a parent reached
    // through a different physical orientation than the one doing this
    // particular expansion. mid() re-derives each visiting parent's own
    // symmetry fresh and un-rotates before calling makeMove() - see the
    // comment there. This mirrors MCTS::expand()'s identical
    // rotation-to-canonical step.
    int canonSym = -1;
    PositionKey::canonical(game, canonSym);
    n->children.reserve(scored.size());
    for (const auto &entry : scored) {
        int cx, cy;
        PositionKey::applySymToPhysical(game, canonSym, entry.first.x, entry.first.y, cx, cy);
        n->children.push_back(Child{PenteGame::Move(cx, cy), kInvalidIdx});
    }

    bool isOr = (game.getCurrentPlayer() == rootPlayer_);
    updatePnDn(n, isOr); // every child still untried (default pn=dn=1)
}

void PNS::updatePnDn(Node *n, bool isOrNode) const {
    if (isOrNode) {
        Number pn = INF, dn = 0;
        for (const Child &c : n->children) {
            const Node *cn = nodeAt(c.idx);
            pn = std::min(pn, pnOf(cn));
            dn = std::min(INF, dn + dnOf(cn));
        }
        n->pn = pn;
        n->dn = dn;
    } else {
        Number pn = 0, dn = INF;
        for (const Child &c : n->children) {
            const Node *cn = nodeAt(c.idx);
            pn = std::min(INF, pn + pnOf(cn));
            dn = std::min(dn, dnOf(cn));
        }
        n->pn = pn;
        n->dn = dn;
    }
}

template <typename Predicate>
uint16_t PNS::depthFrom(const std::vector<Child> &children, bool useMax, Predicate want) const {
    bool any = false;
    uint16_t best = useMax ? 0 : std::numeric_limits<uint16_t>::max();
    for (const Child &c : children) {
        const Node *cn = nodeAt(c.idx);
        if (!cn || !want(cn)) continue;
        any = true;
        best = useMax ? std::max(best, cn->depth) : std::min(best, cn->depth);
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
        n->depth = depthFrom(n->children, /*useMax=*/!isOrNode,
                              [](const Node *c) { return c->outcome == Outcome::WIN; });
        return;
    }
    // dn == 0 here (mid()'s caller only calls this when pn==0 or dn==0).
    if (isOrNode) {
        // dn = sum(children dn) == 0 requires every child individually
        // resolved to non-WIN (a WIN child would have set pn=0 above already).
        bool anyDraw = false;
        for (const Child &c : n->children) {
            const Node *cn = nodeAt(c.idx);
            assert(cn != nullptr && cn->outcome != Outcome::UNKNOWN && cn->outcome != Outcome::WIN);
            if (cn->outcome == Outcome::DRAW) anyDraw = true;
        }
        // Root, moving here, simply avoids the losing replies: DRAW if any
        // move preserves one, else every move loses.
        if (anyDraw) {
            n->outcome = Outcome::DRAW;
            // Root picks the fastest draw among its options.
            n->depth =
                depthFrom(n->children, /*useMax=*/false, [](const Node *c) { return c->outcome == Outcome::DRAW; });
        } else {
            n->outcome = Outcome::LOSS;
            // Every move loses; root delays the inevitable as long as possible.
            n->depth =
                depthFrom(n->children, /*useMax=*/true, [](const Node *c) { return c->outcome == Outcome::LOSS; });
        }
    } else {
        // dn = min(children dn) == 0 needs only one resolved non-WIN child;
        // others may still be untried/unresolved - ignore them, that's the
        // point of proof numbers (don't wait for exhaustive exploration once
        // an escape is found). Opponent is adversarial to root: prefers a
        // LOSS-for-root escape over a DRAW-for-root one if both are known.
        bool anyLoss = false, anyDraw = false;
        for (const Child &c : n->children) {
            const Node *cn = nodeAt(c.idx);
            if (!cn) continue;
            if (cn->outcome == Outcome::LOSS) anyLoss = true;
            else if (cn->outcome == Outcome::DRAW) anyDraw = true;
        }
        assert(anyLoss || anyDraw);
        if (anyLoss) {
            n->outcome = Outcome::LOSS;
            // Opponent takes their fastest win (root's fastest loss).
            n->depth =
                depthFrom(n->children, /*useMax=*/false, [](const Node *c) { return c->outcome == Outcome::LOSS; });
        } else {
            n->outcome = Outcome::DRAW;
            n->depth =
                depthFrom(n->children, /*useMax=*/false, [](const Node *c) { return c->outcome == Outcome::DRAW; });
        }
    }
}

void PNS::selectChildOr(const Node *n, int &bestIdx, Number &secondPn) const {
    Number best = INF, second = INF;
    int bi = -1;
    for (size_t i = 0; i < n->children.size(); ++i) {
        Number p = pnOf(nodeAt(n->children[i].idx));
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
    for (size_t i = 0; i < n->children.size(); ++i) {
        Number d = dnOf(nodeAt(n->children[i].idx));
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

void PNS::mid(Node *n, PenteGame game, Number thpn, Number thdn, int depth) {
    if (stopRequested_) return;
    stats_.midCalls++;

    // See Config::maxRecursionDepth's comment: a genuinely deep line here
    // would otherwise silently stack-overflow (SIGSEGV, uncatchable) rather
    // than fail gracefully. Stops the whole search, not just this line - see
    // that comment for why a narrower stop isn't safe.
    if (depth >= config_.maxRecursionDepth) {
        stopRequested_ = true;
        return;
    }

    // Clock queries are relatively expensive; only check every 4096 calls to
    // keep this off the hot path. Can overshoot the budget slightly as a
    // result - that's fine, this is a coarse "stop eventually" budget, not a
    // hard deadline. Same cadence also drives periodic checkpointing (see
    // Config::checkpointPath) - one now() call covers both checks.
    if ((stats_.midCalls & 0xFFF) == 0) {
        auto now = std::chrono::steady_clock::now();
        if (config_.maxSeconds > 0) {
            double elapsed = std::chrono::duration<double>(now - startTime_).count();
            if (elapsed >= config_.maxSeconds) {
                stopRequested_ = true;
                return;
            }
        }
        if (!config_.checkpointPath.empty()) {
            double sinceCheckpoint = std::chrono::duration<double>(now - lastCheckpointTime_).count();
            if (sinceCheckpoint >= config_.checkpointIntervalSeconds) {
                saveCheckpoint(config_.checkpointPath);
                lastCheckpointTime_ = std::chrono::steady_clock::now();
            }
        }
    }

    if (!n->expanded) {
        expandNode(n, game);
        return;
    }
    if (n->outcome != Outcome::UNKNOWN) return;

    const bool isOr = (game.getCurrentPlayer() == rootPlayer_);

    // Child moves are stored in canonical coordinates (see expandNode); this
    // parent's own `game` may be a different physical orientation than
    // whichever parent first expanded this (possibly shared) node, so always
    // re-derive the current orientation's symmetry fresh here - never assume
    // it matches whatever orientation created the node. Fixed for the
    // duration of this call (game doesn't change outside the loop below).
    int currentSym = -1;
    PositionKey::canonical(game, currentSym);

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
        PositionKey::applyInverseSymToPhysical(game, currentSym, n->children[bi].move.x, n->children[bi].move.y,
                                                physX, physY);
        PenteGame childGame = game;
        childGame.makeMove(physX, physY);

        if (n->children[bi].idx == kInvalidIdx) {
            if (nodeArena_.size() >= config_.maxNodes) {
                stopRequested_ = true;
                return;
            }
            // getOrCreateNodeIdx() can grow nodeArena_ (emplace_back), which
            // is exactly why nodeArena_ is a deque, not a vector: `n` (and
            // any other Node* held across this call, e.g. by an ancestor
            // frame's own `n`) stays valid regardless - see nodeArena_'s
            // comment in PNS.hpp.
            n->children[bi].idx = getOrCreateNodeIdx(childGame);
        }
        Node *child = &nodeArena_[n->children[bi].idx];

        Number childThPn, childThDn;
        if (isOr) {
            childThPn = (secondVal >= INF) ? thpn : std::min(thpn, secondVal + 1);
            childThDn = (thdn >= INF) ? INF : std::min(INF, thdn - n->dn + dnOf(child));
        } else {
            childThDn = (secondVal >= INF) ? thdn : std::min(thdn, secondVal + 1);
            childThPn = (thpn >= INF) ? INF : std::min(INF, thpn - n->pn + pnOf(child));
        }

        mid(child, std::move(childGame), childThPn, childThDn, depth + 1);
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

    if (resuming_) {
        // loadCheckpoint() already populated table_/nodeArena_/rootPlayer_/
        // rootNode_ - reusing them (instead of clearing) is the entire point
        // of resuming.
        assert(rootPlayer_ == rootGame.getCurrentPlayer() &&
               "rootGame must match the position loadCheckpoint() was called with");
        resuming_ = false;
    } else {
        table_.clear();
        nodeArena_.clear();
        rootPlayer_ = rootGame.getCurrentPlayer();
        rootNode_ = getOrCreateNode(rootGame);
    }
    stats_ = Stats{};
    stopRequested_ = false;
    startTime_ = std::chrono::steady_clock::now();
    lastCheckpointTime_ = startTime_;

    // mid() returns after merely expanding a not-yet-expanded node (standard
    // df-pn "MID" behavior - see the class-level comment), relying on its
    // caller to re-invoke it to actually descend. Every recursive call gets
    // this for free from its parent's own while(true) loop; the root has no
    // such parent, so solve() must play that role itself.
    while (rootNode_->outcome == Outcome::UNKNOWN && !stopRequested_) {
        mid(rootNode_, rootGame, INF, INF, 0);
    }

    // Always write a final checkpoint (not just the periodic ones) so a run
    // stopped by maxNodes/maxSeconds - the exact scenario this feature exists
    // for - never loses its last bit of progress.
    if (!config_.checkpointPath.empty()) saveCheckpoint(config_.checkpointPath);

    return rootNode_->outcome != Outcome::UNKNOWN;
}

void PNS::dfsExhaustive(Node *n, PenteGame game, int depth) {
    if (n->outcome != Outcome::UNKNOWN) return; // already resolved (memoized via table_)
    if (stopRequested_) return;
    stats_.midCalls++;

    if (depth >= config_.maxRecursionDepth) {
        stopRequested_ = true;
        return;
    }
    if ((stats_.midCalls & 0xFFF) == 0) {
        auto now = std::chrono::steady_clock::now();
        if (config_.maxSeconds > 0) {
            double elapsed = std::chrono::duration<double>(now - startTime_).count();
            if (elapsed >= config_.maxSeconds) {
                stopRequested_ = true;
                return;
            }
        }
        if (!config_.checkpointPath.empty()) {
            double sinceCheckpoint = std::chrono::duration<double>(now - lastCheckpointTime_).count();
            if (sinceCheckpoint >= config_.checkpointIntervalSeconds) {
                saveCheckpoint(config_.checkpointPath);
                lastCheckpointTime_ = std::chrono::steady_clock::now();
            }
        }
    }

    if (!n->expanded) {
        expandNode(n, game); // resolves n directly (sets outcome) if terminal
    }
    if (n->outcome != Outcome::UNKNOWN) return;

    const bool isOr = (game.getCurrentPlayer() == rootPlayer_);
    int currentSym = -1;
    PositionKey::canonical(game, currentSym);

    for (size_t bi = 0; bi < n->children.size(); ++bi) {
        int physX, physY;
        PositionKey::applyInverseSymToPhysical(game, currentSym, n->children[bi].move.x, n->children[bi].move.y,
                                                physX, physY);
        PenteGame childGame = game;
        childGame.makeMove(physX, physY);

        if (n->children[bi].idx == kInvalidIdx) {
            if (nodeArena_.size() >= config_.maxNodes) {
                stopRequested_ = true;
                return;
            }
            // See mid()'s identical comment: safe across nodeArena_ growth
            // because nodeArena_ is a deque.
            n->children[bi].idx = getOrCreateNodeIdx(childGame);
        }
        dfsExhaustive(&nodeArena_[n->children[bi].idx], std::move(childGame), depth + 1);
        if (stopRequested_) return;
    }

    // Every child is now fully resolved. Provably, this always leaves n->pn
    // or n->dn exactly 0 in whichever direction resolveOutcome() expects (an
    // OR node either has a WIN child, giving pn=0, or every child is
    // LOSS/DRAW, giving dn=sum(0s)=0; symmetric for AND nodes) - so both
    // helpers work completely unchanged from df-pn's own use of them.
    updatePnDn(n, isOr);
    resolveOutcome(n, isOr);
}

bool PNS::solveExhaustive(const PenteGame &rootGame) {
    assert(!rootGame.getConfig().tournamentRule &&
           "PNS requires tournamentRule disabled (matches apps/Pente.cpp's boardSize<7 auto-disable convention) "
           "since its exhaustive move enumeration doesn't reproduce the tournament-rule perimeter restriction");
    assert(!rootGame.getConfig().renjuForbiddenMoves && "PNS does not support Renju forbidden-move rules");
    assert(rootGame.getConfig().boardSize <= PositionKey::kMaxBoardSize &&
           "PositionKey packing only supports boardSize <= 5");

    if (resuming_) {
        assert(rootPlayer_ == rootGame.getCurrentPlayer() &&
               "rootGame must match the position loadCheckpoint() was called with");
        resuming_ = false;
    } else {
        table_.clear();
        nodeArena_.clear();
        rootPlayer_ = rootGame.getCurrentPlayer();
        rootNode_ = getOrCreateNode(rootGame);
    }
    stats_ = Stats{};
    stopRequested_ = false;
    startTime_ = std::chrono::steady_clock::now();
    lastCheckpointTime_ = startTime_;

    dfsExhaustive(rootNode_, rootGame, 0);

    if (!config_.checkpointPath.empty()) saveCheckpoint(config_.checkpointPath);

    return rootNode_->outcome != Outcome::UNKNOWN;
}

PNS::Outcome PNS::getRootOutcome() const { return rootNode_ ? rootNode_->outcome : Outcome::UNKNOWN; }

int PNS::getRootDepth() const { return rootNode_ ? static_cast<int>(rootNode_->depth) : 0; }

PNS::Outcome PNS::getOutcome(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    return (it != table_.end()) ? nodeArena_[it->second].outcome : Outcome::UNKNOWN;
}

int PNS::getDepth(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    auto it = table_.find(key);
    return (it != table_.end()) ? static_cast<int>(nodeArena_[it->second].depth) : 0;
}

uint64_t PNS::getNodeCount() const { return nodeArena_.size(); }

std::vector<PNS::Record> PNS::exportResolved() const {
    std::vector<Record> out;
    out.reserve(table_.size());
    for (const auto &entry : table_) {
        const Node &n = nodeArena_[entry.second];
        if (n.outcome != Outcome::UNKNOWN) {
            out.push_back({entry.first, n.outcome, n.depth});
        }
    }
    return out;
}

namespace {
// v1: pn/dn written as 8 bytes each (from when Number was uint64_t). v2
// (current): pn/dn written as 4 bytes each (Number is now uint32_t - see
// PNS.hpp's Number/INF comments). loadCheckpoint() reads either; saved files
// are always written as the current version.
constexpr uint32_t kCheckpointVersionLegacyU64Number = 1;
constexpr uint32_t kCheckpointVersion = 2;
} // namespace

bool PNS::saveCheckpoint(const std::string &path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;

    os.write("PNSC", 4);
    os.write(reinterpret_cast<const char *>(&kCheckpointVersion), sizeof(kCheckpointVersion));
    uint8_t rootPlayerByte = static_cast<uint8_t>(rootPlayer_);
    os.write(reinterpret_cast<const char *>(&rootPlayerByte), 1);
    uint64_t nodeCount = table_.size();
    os.write(reinterpret_cast<const char *>(&nodeCount), sizeof(nodeCount));

    for (const auto &entry : table_) {
        const PositionKey &key = entry.first;
        const Node &n = nodeArena_[entry.second];
        os.write(reinterpret_cast<const char *>(&key.bits), sizeof(key.bits));
        os.write(reinterpret_cast<const char *>(&n.pn), sizeof(n.pn));
        os.write(reinterpret_cast<const char *>(&n.dn), sizeof(n.dn));
        uint8_t outcomeByte = static_cast<uint8_t>(n.outcome);
        uint8_t expandedByte = n.expanded ? 1 : 0;
        os.write(reinterpret_cast<const char *>(&outcomeByte), 1);
        os.write(reinterpret_cast<const char *>(&expandedByte), 1);
        os.write(reinterpret_cast<const char *>(&n.depth), sizeof(n.depth));
        uint16_t childCount = n.expanded ? static_cast<uint16_t>(n.children.size()) : 0;
        os.write(reinterpret_cast<const char *>(&childCount), sizeof(childCount));
        if (n.expanded) {
            for (const auto &c : n.children) {
                os.write(reinterpret_cast<const char *>(&c.move.x), 1);
                os.write(reinterpret_cast<const char *>(&c.move.y), 1);
            }
        }
    }
    os.flush();
    return static_cast<bool>(os);
}

bool PNS::loadCheckpoint(const std::string &path, const PenteGame &rootGame) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    char magic[4];
    is.read(magic, 4);
    if (!is || std::memcmp(magic, "PNSC", 4) != 0) return false;
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (!is || (version != kCheckpointVersion && version != kCheckpointVersionLegacyU64Number)) return false;
    const bool legacyU64Number = (version == kCheckpointVersionLegacyU64Number);
    uint8_t rootPlayerByte = 0;
    is.read(reinterpret_cast<char *>(&rootPlayerByte), 1);
    if (!is) return false;
    PenteGame::Player storedRootPlayer = static_cast<PenteGame::Player>(rootPlayerByte);
    // A checkpoint's proof numbers are only meaningful relative to whoever
    // was proving when it was written - refuse rather than silently resuming
    // against a mismatched root.
    if (storedRootPlayer != rootGame.getCurrentPlayer()) return false;

    uint64_t nodeCount = 0;
    is.read(reinterpret_cast<char *>(&nodeCount), sizeof(nodeCount));
    if (!is) return false;

    // Raw per-record data as read from disk (still keyed by array position,
    // not yet linked into a DAG) - childMoves only, same on-disk shape as
    // before this refactor: the file never stored raw pointers/indices, only
    // move coordinates, so the format itself didn't need to change.
    struct RawNode {
        PositionKey key;
        Number pn = 1, dn = 1;
        Outcome outcome = Outcome::UNKNOWN;
        bool expanded = false;
        uint16_t depth = 0;
        std::vector<PenteGame::Move> childMoves;
    };
    std::vector<RawNode> raw;
    raw.reserve(nodeCount);

    // v1 wrote pn/dn as 8 bytes (Number was uint64_t then, INF=1ULL<<40 -
    // bigger than the current Number type can hold); v2 writes 4 bytes
    // (current Number/INF, see PNS.hpp). Map any legacy value at/above the
    // old INF to the current (smaller) INF sentinel rather than truncating
    // into a bogus wrapped 32-bit value - real finite proof numbers observed
    // on this project's actual 5x5 runs stay in the low hundreds of
    // thousands (see issues/solve-5x5-pente.md), far below either INF, so
    // this only ever affects genuinely-infinite sentinel values.
    auto readNumber = [&is](bool legacy) -> Number {
        if (legacy) {
            uint64_t v = 0;
            is.read(reinterpret_cast<char *>(&v), sizeof(v));
            constexpr uint64_t kLegacyInf = 1ULL << 40;
            return (v >= kLegacyInf) ? PNS::INF : static_cast<Number>(v);
        }
        uint32_t v = 0;
        is.read(reinterpret_cast<char *>(&v), sizeof(v));
        return static_cast<Number>(v);
    };

    for (uint64_t i = 0; i < nodeCount; ++i) {
        RawNode rn;
        is.read(reinterpret_cast<char *>(&rn.key.bits), sizeof(rn.key.bits));
        rn.pn = readNumber(legacyU64Number);
        rn.dn = readNumber(legacyU64Number);
        uint8_t outcomeByte = 0, expandedByte = 0;
        is.read(reinterpret_cast<char *>(&outcomeByte), 1);
        is.read(reinterpret_cast<char *>(&expandedByte), 1);
        is.read(reinterpret_cast<char *>(&rn.depth), sizeof(rn.depth));
        uint16_t childCount = 0;
        is.read(reinterpret_cast<char *>(&childCount), sizeof(childCount));
        if (!is) return false;
        rn.outcome = static_cast<Outcome>(outcomeByte);
        rn.expanded = expandedByte != 0;
        if (rn.expanded) {
            rn.childMoves.reserve(childCount);
            for (uint16_t c = 0; c < childCount; ++c) {
                uint8_t x = 0, y = 0;
                is.read(reinterpret_cast<char *>(&x), 1);
                is.read(reinterpret_cast<char *>(&y), 1);
                if (!is) return false;
                rn.childMoves.emplace_back(x, y);
            }
        }
        raw.push_back(std::move(rn));
    }

    // First pass: create every Node verbatim from its raw record, in the
    // same order as `raw` - so raw[i].key is i's key throughout pass two
    // below, no separate index->key reverse-lookup needed.
    std::deque<Node> newArena;
    std::unordered_map<PositionKey, uint32_t> newTable;
    newTable.reserve(nodeCount);
    for (const auto &rn : raw) {
        Node n;
        n.pn = rn.pn;
        n.dn = rn.dn;
        n.outcome = rn.outcome;
        n.depth = rn.depth;
        n.expanded = rn.expanded;
        n.children.reserve(rn.childMoves.size());
        for (const auto &m : rn.childMoves) n.children.push_back(Child{m, kInvalidIdx});
        uint32_t idx = static_cast<uint32_t>(newArena.size());
        newArena.push_back(std::move(n));
        newTable.emplace(rn.key, idx);
    }

    // Second pass: for every expanded node, reconstruct its position (via
    // PenteGame::loadRawState(), see that method's doc comment) and wire up
    // each child's idx by re-deriving its canonical key and looking it up.
    // A node reconstructed directly from its own canonical key always yields
    // sym=0 when re-canonicalized, so each child's move - stored canonically,
    // see Node::children - can be applied directly as physical coordinates
    // here with no un-rotation needed (unlike mid()'s general case).
    const int windowSize = rootGame.maxIdx() - rootGame.minIdx();
    const PenteGame::Config &cfg = rootGame.getConfig();
    for (size_t i = 0; i < raw.size(); ++i) {
        Node &n = newArena[i];
        if (!n.expanded || n.children.empty()) continue;
        auto unpacked = PositionKey::unpack(raw[i].key, windowSize);
        PenteGame game(cfg);
        game.loadRawState(unpacked.cell.data(), unpacked.sideToMove, unpacked.blackCaptures, unpacked.whiteCaptures);
        for (auto &child : n.children) {
            PenteGame childGame = game;
            childGame.makeMove(child.move.x, child.move.y);
            int childSym = -1;
            PositionKey childKey = PositionKey::canonical(childGame, childSym);
            auto it = newTable.find(childKey);
            if (it != newTable.end()) child.idx = it->second; // else still-untried child: leave kInvalidIdx
        }
    }

    int rootSym = -1;
    PositionKey rootKey = PositionKey::canonical(rootGame, rootSym);
    auto rootIt = newTable.find(rootKey);
    if (rootIt == newTable.end()) return false; // corrupt/mismatched checkpoint: root itself missing
    uint32_t rootIdx = rootIt->second;

    // Plain integer index, not a pointer, captured before the moves below -
    // so it's trivially unaffected by whatever std::deque/unordered_map::
    // operator=(&&) does or doesn't do to element addresses; look it up
    // fresh in nodeArena_ after the move completes.
    nodeArena_ = std::move(newArena);
    table_ = std::move(newTable);
    rootNode_ = &nodeArena_[rootIdx];
    rootPlayer_ = storedRootPlayer;
    resuming_ = true;
    stopRequested_ = false;
    return true;
}

void PNS::printProofStats() const {
    std::cout << "PNS: nodes=" << nodeArena_.size() << " midCalls=" << stats_.midCalls
              << " transpositionHits=" << stats_.transpositionHits;
    if (rootNode_) {
        std::cout << " root(pn=" << rootNode_->pn << ", dn=" << rootNode_->dn
                   << ", outcome=" << outcomeToString(rootNode_->outcome) << ", depth=" << rootNode_->depth << ")";
    }
    std::cout << "\n";
}
