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
    n->childMoves.reserve(scored.size());
    for (const auto &entry : scored) {
        int cx, cy;
        PositionKey::applySymToPhysical(game, canonSym, entry.first.x, entry.first.y, cx, cy);
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

    // childMoves are stored in canonical coordinates (see expandNode); this
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
        PositionKey::applyInverseSymToPhysical(game, currentSym, n->childMoves[bi].x, n->childMoves[bi].y, physX, physY);
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
        // loadCheckpoint() already populated table_/rootPlayer_/rootNode_ -
        // reusing them (instead of clearing) is the entire point of resuming.
        assert(rootPlayer_ == rootGame.getCurrentPlayer() &&
               "rootGame must match the position loadCheckpoint() was called with");
        resuming_ = false;
    } else {
        table_.clear();
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

    for (size_t bi = 0; bi < n->childMoves.size(); ++bi) {
        int physX, physY;
        PositionKey::applyInverseSymToPhysical(game, currentSym, n->childMoves[bi].x, n->childMoves[bi].y, physX, physY);
        PenteGame childGame = game;
        childGame.makeMove(physX, physY);

        if (!n->childPtr[bi]) {
            if (table_.size() >= config_.maxNodes) {
                stopRequested_ = true;
                return;
            }
            n->childPtr[bi] = getOrCreateNode(childGame);
        }
        dfsExhaustive(n->childPtr[bi], std::move(childGame), depth + 1);
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

namespace {
constexpr uint32_t kCheckpointVersion = 1;
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
        const Node &n = entry.second;
        os.write(reinterpret_cast<const char *>(&key.bits), sizeof(key.bits));
        os.write(reinterpret_cast<const char *>(&n.pn), sizeof(n.pn));
        os.write(reinterpret_cast<const char *>(&n.dn), sizeof(n.dn));
        uint8_t outcomeByte = static_cast<uint8_t>(n.outcome);
        uint8_t expandedByte = n.expanded ? 1 : 0;
        os.write(reinterpret_cast<const char *>(&outcomeByte), 1);
        os.write(reinterpret_cast<const char *>(&expandedByte), 1);
        os.write(reinterpret_cast<const char *>(&n.depth), sizeof(n.depth));
        uint16_t childCount = n.expanded ? static_cast<uint16_t>(n.childMoves.size()) : 0;
        os.write(reinterpret_cast<const char *>(&childCount), sizeof(childCount));
        if (n.expanded) {
            for (const auto &m : n.childMoves) {
                os.write(reinterpret_cast<const char *>(&m.x), 1);
                os.write(reinterpret_cast<const char *>(&m.y), 1);
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
    if (!is || version != kCheckpointVersion) return false;
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

    for (uint64_t i = 0; i < nodeCount; ++i) {
        RawNode rn;
        is.read(reinterpret_cast<char *>(&rn.key.bits), sizeof(rn.key.bits));
        is.read(reinterpret_cast<char *>(&rn.pn), sizeof(rn.pn));
        is.read(reinterpret_cast<char *>(&rn.dn), sizeof(rn.dn));
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

    // First pass: create every Node verbatim from its raw record.
    std::unordered_map<PositionKey, Node> newTable;
    newTable.reserve(nodeCount);
    for (const auto &rn : raw) {
        Node n;
        n.pn = rn.pn;
        n.dn = rn.dn;
        n.outcome = rn.outcome;
        n.depth = rn.depth;
        n.expanded = rn.expanded;
        n.childMoves = rn.childMoves;
        n.childPtr.assign(rn.childMoves.size(), nullptr);
        newTable.emplace(rn.key, std::move(n));
    }

    // Second pass: for every expanded node, reconstruct its position (via
    // PenteGame::loadRawState(), see that method's doc comment) and wire up
    // childPtr by re-deriving each child's canonical key and looking it up.
    // A node reconstructed directly from its own canonical key always yields
    // sym=0 when re-canonicalized, so childMoves - stored canonically, see
    // Node::childMoves - can be applied directly as physical coordinates
    // here with no un-rotation needed (unlike mid()'s general case).
    const int windowSize = rootGame.maxIdx() - rootGame.minIdx();
    const PenteGame::Config &cfg = rootGame.getConfig();
    for (auto &entry : newTable) {
        Node &n = entry.second;
        if (!n.expanded || n.childMoves.empty()) continue;
        auto unpacked = PositionKey::unpack(entry.first, windowSize);
        PenteGame game(cfg);
        game.loadRawState(unpacked.cell.data(), unpacked.sideToMove, unpacked.blackCaptures, unpacked.whiteCaptures);
        for (size_t i = 0; i < n.childMoves.size(); ++i) {
            PenteGame childGame = game;
            childGame.makeMove(n.childMoves[i].x, n.childMoves[i].y);
            int childSym = -1;
            PositionKey childKey = PositionKey::canonical(childGame, childSym);
            auto it = newTable.find(childKey);
            if (it != newTable.end()) n.childPtr[i] = &it->second; // else still-untried child: leave nullptr
        }
    }

    int rootSym = -1;
    PositionKey rootKey = PositionKey::canonical(rootGame, rootSym);
    auto rootIt = newTable.find(rootKey);
    if (rootIt == newTable.end()) return false; // corrupt/mismatched checkpoint: root itself missing

    // std::unordered_map's move (equal allocators) transfers buckets/nodes
    // without touching individual elements, so every pointer captured above
    // (rootNodePtr and every childPtr wired into newTable) stays valid after
    // this move - same pointer-stability guarantee already relied on
    // elsewhere (see getOrCreateNode()'s comment).
    Node *rootNodePtr = &rootIt->second;
    table_ = std::move(newTable);
    rootNode_ = rootNodePtr;
    rootPlayer_ = storedRootPlayer;
    resuming_ = true;
    stopRequested_ = false;
    return true;
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
