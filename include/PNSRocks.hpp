#ifndef PNSROCKS_HPP
#define PNSROCKS_HPP

// Disk-backed variant of PNS (see PNS.hpp for the df-pn algorithm itself -
// OR/AND pn/dn convention, 3-outcome WIN/LOSS/DRAW resolution, GHI stance -
// all identical here, just copied rather than shared, see below for why).
//
// PNS's own nodeArena_/table_ must fit entirely in RAM - on this project's
// ~29GB machine that caps out around 120M nodes (see
// issues/solve-5x5-pente.md's "Node memory footprint reduction" section),
// and real 5x5 runs show the resolved-position fraction still climbing with
// no sign of closing out by then. PNSRocks stores every node in a RocksDB
// database instead - one Get()/Put() per node access rather than a pointer
// dereference into an in-RAM arena - so the live working set is bounded by
// disk (hundreds of GB), not RAM. Reopening the same dbPath resumes a prior
// run automatically; there's no separate checkpoint format to manage (that
// was PNS's own answer to "how do I persist an in-RAM DAG" - here the DAG
// *is* the persistent store, so the question doesn't arise).
//
// Kept as a wholly separate class from PNS, deliberately duplicating the
// df-pn logic, rather than templatizing/abstracting PNS's storage layer:
//   - PNS.cpp/PNS.hpp are compiled directly into the WASM build
//     (scripts/build_wasm.sh lists sources explicitly, bypassing CMake/
//     pente_core) for the browser's live-search fallback, which must never
//     link RocksDB. A shared storage interface would need PNS's own
//     Child{move,idx} (4-byte arena index, the ~40% memory win from the
//     footprint-reduction work) to grow to Child{move,PositionKey} (8-byte
//     key, since RocksDB has no arena to index into) for BOTH backends,
//     regressing the in-RAM/WASM path's memory for a WASM build that will
//     never use the disk backend anyway.
//   - This file is only ever added to the build when RocksDB is actually
//     found (see CMakeLists.txt's `if(RocksDB_FOUND)` block), so it's
//     impossible for it to be an accidental link dependency of anything
//     that doesn't explicitly want it.
//
// PERFORMANCE, stated plainly rather than assumed away: updatePnDn() and
// selectChildOr()/selectChildAnd() run on every mid() iteration and each
// read EVERY child of a node fresh from RocksDB (no in-RAM caching of
// sibling values across calls - see updatePnDn()'s own comment for why that
// would be unsound for a shared DAG, not just slow). If a real run performs
// hundreds of millions of these, and every child read is a full Get(), this
// could be far slower than PNS's in-RAM pointer chase. Deliberately built as
// the simplest-correct version first: rely on RocksDB's own configurable
// block cache (Config::blockCacheBytes) for hot-data caching rather than a
// custom write-back layer, measure real throughput (see
// issues/solve-5x5-pente.md's RocksDB calibration section once it exists),
// and only add custom caching if that measurement shows it's needed.
#ifdef WITH_ROCKSDB

#include "PenteGame.hpp"
#include "PositionKey.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocksdb {
class DB;
} // namespace rocksdb

class PNSRocks {
  public:
    enum class Outcome : uint8_t { UNKNOWN = 0, WIN, LOSS, DRAW };
    using Number = uint32_t;
    static constexpr Number INF = 1u << 28;

    struct Config {
        // Required: directory RocksDB stores its data in. Reusing the same
        // path across runs resumes automatically - see class comment.
        std::string dbPath;

        // Soft cap on how many NEW nodes this run is allowed to create
        // (0 = unlimited) - not a cap on the database's total prior size,
        // since that's unknown without a full scan. solve() stops (DB left
        // in a valid, resumable state) once reached.
        uint64_t maxNodes = 0;

        double maxSeconds = 0; // 0 = unlimited
        int maxRecursionDepth = 300; // see PNS::Config::maxRecursionDepth's comment - identical reasoning

        // RocksDB's own block cache size - the in-RAM caching layer this
        // design deliberately relies on instead of a custom one (see class
        // comment). Larger = more hot nodes served from RAM instead of
        // disk I/O, at the cost of RAM the rest of the machine can't use.
        uint64_t blockCacheBytes = 4ULL << 30; // 4GB

        Config() {}
    };

    struct Stats {
        uint64_t nodesCreated = 0;
        uint64_t midCalls = 0;
        uint64_t dbGets = 0;
        uint64_t dbPuts = 0;
    };

    explicit PNSRocks(const Config &config);
    ~PNSRocks();
    PNSRocks(const PNSRocks &) = delete;
    PNSRocks &operator=(const PNSRocks &) = delete;

    // Same preconditions/semantics as PNS::solve() (tournamentRule and
    // renjuForbiddenMoves both disabled, boardSize <= PositionKey::kMaxBoardSize).
    bool solve(const PenteGame &rootGame);

    Outcome getRootOutcome() const;
    int getRootDepth() const;
    Outcome getOutcome(const PenteGame &game) const;
    int getDepth(const PenteGame &game) const;

    // Approximate - RocksDB doesn't track an exact live key count cheaply;
    // backed by the "rocksdb.estimate-num-keys" property.
    uint64_t getApproxNodeCount() const;

    const Stats &getStats() const { return stats_; }
    void printProofStats() const;

  private:
    struct Child {
        PenteGame::Move move;
        PositionKey key;
    };

    struct Node {
        Number pn = 1;
        Number dn = 1;
        Outcome outcome = Outcome::UNKNOWN;
        uint16_t depth = 0;
        bool expanded = false;
        // Unlike PNS::Node::children, each child's identity (canonical
        // PositionKey) is computed once at expand time (simulating every
        // candidate move - see expandNode()) rather than lazily on first
        // descent, so updatePnDn()/selectChild* never need to materialize a
        // child before reading its pn/dn - a missing key just means
        // "untried" (get() below returns a default Node{} for it, exactly
        // like PNS's kInvalidIdx nodeAt() convention).
        std::vector<Child> children;
    };

    Config config_;
    // mutable: get()/put() are const (read-only from the algorithm's point of
    // view - they don't change *this's own Config or player/root fields) but
    // still need to bump db access counters for throughput reporting.
    mutable Stats stats_;
    mutable std::unique_ptr<rocksdb::DB> db_;
    PenteGame::Player rootPlayer_ = PenteGame::NONE;
    PositionKey rootKey_;
    bool stopRequested_ = false;
    std::chrono::steady_clock::time_point startTime_;

    Node get(const PositionKey &key) const;
    void put(const PositionKey &key, const Node &n) const;
    static std::string serialize(const Node &n);
    static Node deserialize(const std::string &value);

    void expandNode(Node &n, const PenteGame &game);
    void updatePnDn(Node &n, bool isOrNode) const;
    void resolveOutcome(Node &n, bool isOrNode) const;
    void selectChildOr(const Node &n, int &bestIdx, Number &secondPn) const;
    void selectChildAnd(const Node &n, int &bestIdx, Number &secondDn) const;
    void mid(const PositionKey &key, PenteGame game, Number thpn, Number thdn, int depth);

    template <typename Predicate>
    uint16_t depthFrom(const std::vector<Child> &children, bool useMax, Predicate want) const;

    static std::vector<PenteGame::Move> enumerateLegalMoves(const PenteGame &game);
};

#endif // WITH_ROCKSDB
#endif // PNSROCKS_HPP
