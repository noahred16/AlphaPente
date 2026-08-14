#include "PNSRocks.hpp"
#ifdef WITH_ROCKSDB

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/table.h>
#include <stdexcept>

namespace {
const char *outcomeToString(PNSRocks::Outcome o) {
    switch (o) {
    case PNSRocks::Outcome::WIN:
        return "WIN";
    case PNSRocks::Outcome::LOSS:
        return "LOSS";
    case PNSRocks::Outcome::DRAW:
        return "DRAW";
    default:
        return "UNKNOWN";
    }
}
} // namespace

PNSRocks::PNSRocks(const Config &config) : config_(config) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();
    rocksdb::BlockBasedTableOptions tableOptions;
    tableOptions.block_cache = rocksdb::NewLRUCache(config_.blockCacheBytes);
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(tableOptions));

    rocksdb::DB *dbRaw = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, config_.dbPath, &dbRaw);
    if (!status.ok()) {
        throw std::runtime_error("PNSRocks: failed to open RocksDB at " + config_.dbPath + ": " + status.ToString());
    }
    db_.reset(dbRaw);
}

PNSRocks::~PNSRocks() = default;

std::vector<PenteGame::Move> PNSRocks::enumerateLegalMoves(const PenteGame &game) {
    // Identical to PNS::enumerateLegalMoves - see that copy's comment for why
    // this deliberately does NOT use PenteGame::getLegalMoves() (a sound
    // proof requires an exhaustive legal-move set, not the neighborhood
    // heuristic MCTS uses).
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

std::string PNSRocks::serialize(const Node &n) {
    std::string buf;
    buf.resize(14 + n.children.size() * 10);
    char *p = buf.data();
    std::memcpy(p, &n.pn, 4);
    p += 4;
    std::memcpy(p, &n.dn, 4);
    p += 4;
    uint8_t outcomeByte = static_cast<uint8_t>(n.outcome);
    std::memcpy(p, &outcomeByte, 1);
    p += 1;
    uint8_t expandedByte = n.expanded ? 1 : 0;
    std::memcpy(p, &expandedByte, 1);
    p += 1;
    std::memcpy(p, &n.depth, 2);
    p += 2;
    uint16_t childCount = static_cast<uint16_t>(n.children.size());
    std::memcpy(p, &childCount, 2);
    p += 2;
    for (const auto &c : n.children) {
        std::memcpy(p, &c.move.x, 1);
        p += 1;
        std::memcpy(p, &c.move.y, 1);
        p += 1;
        std::memcpy(p, &c.key.bits, 8);
        p += 8;
    }
    return buf;
}

PNSRocks::Node PNSRocks::deserialize(const std::string &value) {
    Node n;
    const char *p = value.data();
    std::memcpy(&n.pn, p, 4);
    p += 4;
    std::memcpy(&n.dn, p, 4);
    p += 4;
    uint8_t outcomeByte = 0, expandedByte = 0;
    std::memcpy(&outcomeByte, p, 1);
    p += 1;
    std::memcpy(&expandedByte, p, 1);
    p += 1;
    n.outcome = static_cast<Outcome>(outcomeByte);
    n.expanded = expandedByte != 0;
    std::memcpy(&n.depth, p, 2);
    p += 2;
    uint16_t childCount = 0;
    std::memcpy(&childCount, p, 2);
    p += 2;
    n.children.reserve(childCount);
    for (uint16_t i = 0; i < childCount; ++i) {
        PenteGame::Move mv;
        std::memcpy(&mv.x, p, 1);
        p += 1;
        std::memcpy(&mv.y, p, 1);
        p += 1;
        PositionKey k;
        std::memcpy(&k.bits, p, 8);
        p += 8;
        n.children.push_back(Child{mv, k});
    }
    return n;
}

PNSRocks::Node PNSRocks::get(const PositionKey &key) const {
    std::string value;
    rocksdb::Slice slice(reinterpret_cast<const char *>(&key.bits), sizeof(key.bits));
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), slice, &value);
    stats_.dbGets++;
    if (!status.ok()) return Node{}; // not found: untried, default pn=dn=1
    return deserialize(value);
}

void PNSRocks::put(const PositionKey &key, const Node &n) const {
    rocksdb::Slice slice(reinterpret_cast<const char *>(&key.bits), sizeof(key.bits));
    std::string value = serialize(n);
    db_->Put(rocksdb::WriteOptions(), slice, value);
    stats_.dbPuts++;
}

void PNSRocks::expandNode(Node &n, const PenteGame &game) {
    n.expanded = true;

    PenteGame::Player winner = game.getWinner();
    if (winner != PenteGame::NONE) {
        n.outcome = (winner == rootPlayer_) ? Outcome::WIN : Outcome::LOSS;
        n.pn = (n.outcome == Outcome::WIN) ? 0 : INF;
        n.dn = (n.outcome == Outcome::WIN) ? INF : 0;
        return;
    }

    std::vector<PenteGame::Move> moves = enumerateLegalMoves(game);
    if (moves.empty()) {
        n.outcome = Outcome::DRAW;
        n.pn = INF;
        n.dn = 0;
        return;
    }

    std::vector<std::pair<PenteGame::Move, float>> scored;
    scored.reserve(moves.size());
    for (const auto &m : moves) scored.emplace_back(m, game.evaluateMove(m));
    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    int canonSym = -1;
    PositionKey::canonical(game, canonSym);
    n.children.reserve(scored.size());
    for (const auto &entry : scored) {
        int cx, cy;
        PositionKey::applySymToPhysical(game, canonSym, entry.first.x, entry.first.y, cx, cy);
        // Unlike PNS::expandNode (which defers this to first descent), the
        // child's canonical key is computed here, once, by simulating the
        // move - see Node::children's comment for why: it lets every later
        // pn/dn read skip straight to a Get(), with no separate "materialize
        // this child first" step.
        PenteGame childGame = game;
        childGame.makeMove(entry.first.x, entry.first.y);
        int childSym = -1;
        PositionKey childKey = PositionKey::canonical(childGame, childSym);
        n.children.push_back(Child{PenteGame::Move(cx, cy), childKey});
    }

    bool isOr = (game.getCurrentPlayer() == rootPlayer_);
    updatePnDn(n, isOr); // every child still untried -> get() returns default pn=dn=1
}

void PNSRocks::updatePnDn(Node &n, bool isOrNode) const {
    if (isOrNode) {
        Number pn = INF, dn = 0;
        for (const Child &c : n.children) {
            Node cn = get(c.key);
            pn = std::min(pn, cn.pn);
            dn = std::min(INF, dn + cn.dn);
        }
        n.pn = pn;
        n.dn = dn;
    } else {
        Number pn = 0, dn = INF;
        for (const Child &c : n.children) {
            Node cn = get(c.key);
            pn = std::min(INF, pn + cn.pn);
            dn = std::min(dn, cn.dn);
        }
        n.pn = pn;
        n.dn = dn;
    }
}

template <typename Predicate>
uint16_t PNSRocks::depthFrom(const std::vector<Child> &children, bool useMax, Predicate want) const {
    bool any = false;
    uint16_t best = useMax ? 0 : std::numeric_limits<uint16_t>::max();
    for (const Child &c : children) {
        Node cn = get(c.key);
        if (!want(cn)) continue;
        any = true;
        best = useMax ? std::max(best, cn.depth) : std::min(best, cn.depth);
    }
    assert(any);
    (void)any;
    return static_cast<uint16_t>(best + 1);
}

void PNSRocks::resolveOutcome(Node &n, bool isOrNode) const {
    if (n.pn == 0) {
        n.outcome = Outcome::WIN;
        n.depth = depthFrom(n.children, /*useMax=*/!isOrNode,
                             [](const Node &c) { return c.outcome == Outcome::WIN; });
        return;
    }
    if (isOrNode) {
        bool anyDraw = false;
        for (const Child &c : n.children) {
            Node cn = get(c.key);
            assert(cn.outcome != Outcome::UNKNOWN && cn.outcome != Outcome::WIN);
            if (cn.outcome == Outcome::DRAW) anyDraw = true;
        }
        if (anyDraw) {
            n.outcome = Outcome::DRAW;
            n.depth =
                depthFrom(n.children, /*useMax=*/false, [](const Node &c) { return c.outcome == Outcome::DRAW; });
        } else {
            n.outcome = Outcome::LOSS;
            n.depth =
                depthFrom(n.children, /*useMax=*/true, [](const Node &c) { return c.outcome == Outcome::LOSS; });
        }
    } else {
        bool anyLoss = false, anyDraw = false;
        for (const Child &c : n.children) {
            Node cn = get(c.key);
            if (cn.outcome == Outcome::LOSS) anyLoss = true;
            else if (cn.outcome == Outcome::DRAW) anyDraw = true;
        }
        assert(anyLoss || anyDraw);
        if (anyLoss) {
            n.outcome = Outcome::LOSS;
            n.depth =
                depthFrom(n.children, /*useMax=*/false, [](const Node &c) { return c.outcome == Outcome::LOSS; });
        } else {
            n.outcome = Outcome::DRAW;
            n.depth =
                depthFrom(n.children, /*useMax=*/false, [](const Node &c) { return c.outcome == Outcome::DRAW; });
        }
    }
}

void PNSRocks::selectChildOr(const Node &n, int &bestIdx, Number &secondPn) const {
    Number best = INF, second = INF;
    int bi = -1;
    for (size_t i = 0; i < n.children.size(); ++i) {
        Number p = get(n.children[i].key).pn;
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

void PNSRocks::selectChildAnd(const Node &n, int &bestIdx, Number &secondDn) const {
    Number best = INF, second = INF;
    int bi = -1;
    for (size_t i = 0; i < n.children.size(); ++i) {
        Number d = get(n.children[i].key).dn;
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

void PNSRocks::mid(const PositionKey &key, PenteGame game, Number thpn, Number thdn, int depth) {
    if (stopRequested_) return;
    stats_.midCalls++;

    if (depth >= config_.maxRecursionDepth) {
        stopRequested_ = true;
        return;
    }
    if ((stats_.midCalls & 0xFFF) == 0 && config_.maxSeconds > 0) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime_).count();
        if (elapsed >= config_.maxSeconds) {
            stopRequested_ = true;
            return;
        }
    }

    Node n = get(key);
    if (!n.expanded) {
        if (config_.maxNodes > 0 && stats_.nodesCreated >= config_.maxNodes) {
            stopRequested_ = true;
            return;
        }
        expandNode(n, game);
        stats_.nodesCreated++;
        put(key, n);
        return;
    }
    if (n.outcome != Outcome::UNKNOWN) return; // already resolved, nothing to do

    const bool isOr = (game.getCurrentPlayer() == rootPlayer_);
    int currentSym = -1;
    PositionKey::canonical(game, currentSym);

    while (true) {
        if (stopRequested_) return;

        updatePnDn(n, isOr);
        if (n.pn == 0 || n.dn == 0) {
            resolveOutcome(n, isOr);
            put(key, n);
            return;
        }
        if (n.pn >= thpn || n.dn >= thdn) {
            put(key, n);
            return;
        }

        int bestIdx = -1;
        Number secondVal = INF;
        if (isOr) selectChildOr(n, bestIdx, secondVal);
        else selectChildAnd(n, bestIdx, secondVal);
        assert(bestIdx >= 0);
        const Child &bc = n.children[static_cast<size_t>(bestIdx)];

        int physX, physY;
        PositionKey::applyInverseSymToPhysical(game, currentSym, bc.move.x, bc.move.y, physX, physY);
        PenteGame childGame = game;
        childGame.makeMove(physX, physY);

        Node childNode = get(bc.key);
        Number childThPn, childThDn;
        if (isOr) {
            childThPn = (secondVal >= INF) ? thpn : std::min(thpn, secondVal + 1);
            childThDn = (thdn >= INF) ? INF : std::min(INF, thdn - n.dn + childNode.dn);
        } else {
            childThDn = (secondVal >= INF) ? thdn : std::min(thdn, secondVal + 1);
            childThPn = (thpn >= INF) ? INF : std::min(INF, thpn - n.pn + childNode.pn);
        }

        mid(bc.key, std::move(childGame), childThPn, childThDn, depth + 1);
        // Loop back: re-derive n's pn/dn from fresh Get()s of every child
        // (including whichever one we just recursed into) on the next
        // iteration - see updatePnDn()'s comment for why this can't cache.
    }
}

bool PNSRocks::solve(const PenteGame &rootGame) {
    assert(!rootGame.getConfig().tournamentRule &&
           "PNSRocks requires tournamentRule disabled - see PNS.hpp's identical assert for why");
    assert(!rootGame.getConfig().renjuForbiddenMoves && "PNSRocks does not support Renju forbidden-move rules");
    assert(rootGame.getConfig().boardSize <= PositionKey::kMaxBoardSize &&
           "PositionKey packing only supports boardSize <= 5");

    rootPlayer_ = rootGame.getCurrentPlayer();
    int sym = -1;
    rootKey_ = PositionKey::canonical(rootGame, sym);
    stats_ = Stats{};
    stopRequested_ = false;
    startTime_ = std::chrono::steady_clock::now();

    Node root = get(rootKey_);
    while (root.outcome == Outcome::UNKNOWN && !stopRequested_) {
        mid(rootKey_, rootGame, INF, INF, 0);
        root = get(rootKey_);
    }
    return root.outcome != Outcome::UNKNOWN;
}

PNSRocks::Outcome PNSRocks::getRootOutcome() const { return get(rootKey_).outcome; }

int PNSRocks::getRootDepth() const { return static_cast<int>(get(rootKey_).depth); }

PNSRocks::Outcome PNSRocks::getOutcome(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    return get(key).outcome;
}

int PNSRocks::getDepth(const PenteGame &game) const {
    int sym = -1;
    PositionKey key = PositionKey::canonical(game, sym);
    return static_cast<int>(get(key).depth);
}

uint64_t PNSRocks::getApproxNodeCount() const {
    std::string value;
    if (db_->GetProperty("rocksdb.estimate-num-keys", &value)) {
        return std::strtoull(value.c_str(), nullptr, 10);
    }
    return 0;
}

void PNSRocks::printProofStats() const {
    Node root = get(rootKey_);
    std::cout << "PNSRocks: approxNodes=" << getApproxNodeCount() << " midCalls=" << stats_.midCalls
              << " nodesCreated=" << stats_.nodesCreated << " dbGets=" << stats_.dbGets
              << " dbPuts=" << stats_.dbPuts << " root(pn=" << root.pn << ", dn=" << root.dn
              << ", outcome=" << outcomeToString(root.outcome) << ", depth=" << root.depth << ")\n";
}

#endif // WITH_ROCKSDB
