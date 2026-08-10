#ifndef PNS_HPP
#define PNS_HPP

/*
Proof Number Search (PNS) - df-pn variant.
Each iteration starts at the root until the proof number or disproof number reaches 0

"Win" is relative to the player who made the first move (the Root player).

Open Q's:
- If a node has two parents, and those parents share the same parent, when we backprop the node, wouldnt it get counted
twice in the grandparent? Yes. Ignore it or use a solution such as "PNS^2" or "GHI".
- Transposition table stored in RAM or disk or hybrid?
A high-speed LRU (Least Recently Used) Cache in RAM and a Persistent Key-Value Store on disk.
Use a packed struct to save bytes. #pragma pack(push, 1) to prevent compiler from adding padding.


Stages:
1.) Selection
- traverse to an unexpanded leaf node.
- At an OR Node (your turn), selects the child with the smallest Proof Number. We want to prove a win?
- At an AND Node (opps turn), selects the child with the smallest Disproof number. Opp wants a win?

2.) Expansion
- Generate all possible legal moves from that state
- Check if any are terminal **
- Nodes get pn=1 dn=1. Wins pn=0 dn=inf. Loss/Draw pn=inf dn=0

3.) Backpropagation
- OR Nodes (your turn): pn = min(pn of children) dn = sum(dn of children)
- AND nodes (opps turn): pn = sum(pn of children) dn = min(dn of children)

4.) Termination Check
- if root pn or dn = 0, end. pn=0, first player wins. dn = 0, lose.
*/

// Plan. We can use a MCTS solver policy value NN to efficiently do PNS. Policy for proofs and Value for disproof.

// ============================================================================
// Implementation notes (df-pn, see AlphaPente's issues/mcts-draw-not-proven-
// through-tree.md for why MCTS/PUCT alone couldn't close out 5x5 Pente).
//
// - df-pn (depth-first, iterative-deepening with (pn,dn) thresholds), not
//   naive best-first PNS: bounds working memory to the transposition table +
//   current recursion path, no separate global priority queue over the whole
//   tree. See PNS::mid() for the threshold formulas (Nagai's df-pn).
//
// - 3-outcome (WIN/LOSS/DRAW) resolution, generalizing classic binary
//   proven/disproven df-pn, mirrors MCTS::backpropagate's already-verified
//   convention: an OR node (root's own turn) becomes WIN as soon as any child
//   is WIN (pn=0 short-circuit); once dn=0 (no WIN child; for an OR node this
//   requires literally every child resolved, since dn=sum), it's DRAW if any
//   resolved child is DRAW, else LOSS - root just avoids the losing replies by
//   playing whichever move preserves at least a draw. An AND node (opponent's
//   turn) becomes LOSS as soon as any single child is LOSS (opponent takes the
//   escape that's worst for root - this falls out of dn=min needing only one
//   term at 0, no extra bookkeeping needed); once dn=0 without any LOSS child
//   present yet, it's DRAW. See PNS::resolveOutcome().
//
// - GHI / double-counting (the "Open Q's" note above): accepted as a known,
//   documented simplification rather than solved. Shared DAG nodes reached via
//   multiple parents can make proof numbers overcount slightly, which affects
//   search *efficiency* only, never *soundness* - a WIN/LOSS/DRAW status is
//   only ever assigned when pn or dn actually reaches 0 through a fully
//   descended, fully verified line (see resolveOutcome()). No PNS^2 / GHI-
//   aware counting is implemented.
//
// - True cycles (a position recurring along a single line of play) cannot
//   happen here, so the position graph is a genuine DAG, never worse:
//   PositionKey packs (stones, side-to-move, captures), and moveCount is
//   always exactly recoverable from those contents alone (occupied-cell count
//   + total captured-stone count); every move strictly increments moveCount,
//   so no two positions on one root-to-leaf path can ever share a key. This
//   also means two different paths that reach an identical key always agree
//   on moveCount too, so - unlike the original plan assumed - no moveCount
//   gating is needed before transposition-sharing/canonicalizing: legality
//   only ever depends on the *current* position, never on how it was reached.
//   (This does still require the tournamentRule move-count-gated restriction
//   to be off - see the assert in solve() - since that rule's own legal-move
//   set isn't reproduced by this engine's exhaustive move enumeration below;
//   5x5 already runs with tournamentRule disabled by existing convention,
//   see apps/Pente.cpp's boardSize<7 auto-disable.)
//
// - Move enumeration is intentionally NOT PenteGame::getLegalMoves() (which
//   only returns cells within a small neighborhood of existing stones, an
//   approximation acceptable for MCTS's heuristic play but NOT complete - it
//   can omit genuinely legal, if unusual, replies). A sound proof requires an
//   exhaustive legal-move set. Since the board is tiny (<=25 cells), PNS
//   enumerates every empty cell in the logical window directly instead - see
//   enumerateLegalMoves() in PNS.cpp.
// ============================================================================

#include "PenteGame.hpp"
#include "PositionKey.hpp"
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

class PNS {
  public:
    enum class Outcome : uint8_t { UNKNOWN = 0, WIN, LOSS, DRAW };

    using Number = uint64_t;
    // Sentinel for "infinite" proof/disproof number. Deliberately far below
    // Number's actual range so sums across a modest branching factor can
    // never wrap; comparisons should test `>= INF`, not `== INF`, since
    // intermediate threshold arithmetic can legitimately produce values
    // slightly above INF without that meaning anything different.
    static constexpr Number INF = 1ULL << 40;

    struct Config {
        // Hard cap on transposition-table size (one entry per distinct
        // canonical position). solve() stops (leaving the DAG in a valid,
        // inspectable but incomplete state) once this is reached, rather than
        // growing unbounded.
        uint64_t maxNodes = 20'000'000;

        // Wall-clock budget in seconds; 0 = unlimited. Checked periodically
        // (not every mid() call, to keep the clock query off the hot path),
        // so the actual stop can overshoot slightly. Like maxNodes, hitting
        // this leaves the DAG in a valid, resumable-in-spirit but incomplete
        // state - solve() returns false rather than crashing or looping.
        double maxSeconds = 0;

        // Hard cap on mid()'s recursion depth (df-pn's threshold-driven
        // descent can legitimately commit very deep into one narrow line -
        // observed in practice on a long 4x4 run). mid() recurses once per
        // ply and takes a full PenteGame by value each level (>8KB, mostly
        // its embedded mt19937), so stack use per level is real; a crash
        // here is a silent stack overflow (SIGSEGV), not a catchable
        // exception. This default is sized to stay safe on an UNRAISED
        // default ~8MB thread stack (empirically ~8KB/level observed =>
        // headroom well under ~1000); callers who raise their own stack
        // size (see apps/Solve5x5.cpp) should raise this correspondingly.
        // Hitting it stops the whole search (like maxNodes/maxSeconds) -
        // not just the offending line - since a capped-but-still-selected
        // child would otherwise make the parent spin forever reselecting a
        // child whose pn/dn can never change.
        int maxRecursionDepth = 300;

        Config() {}
    };

    struct Stats {
        uint64_t nodesCreated = 0;
        uint64_t midCalls = 0;
        uint64_t transpositionHits = 0; // getOrCreateNode() found an existing entry
    };

    explicit PNS(const Config &config = Config());

    // Runs df-pn from rootGame until the root resolves to WIN/LOSS/DRAW, or
    // until config.maxNodes is reached (returns false in that case; the DAG
    // built so far remains valid and query-able via getOutcome()).
    // Requires rootGame's config to have tournamentRule and
    // renjuForbiddenMoves both disabled, and boardSize <= PositionKey::kMaxBoardSize -
    // see the class-level comment above for why.
    //
    // NOTE: df-pn is deliberately efficient - it stops exploring a branch
    // the instant it's no longer needed to prove/disprove the root, so most
    // reachable positions are left UNKNOWN even after a full proof (e.g. a
    // 3x3 solve touches 264 nodes but resolves only 86). Use
    // solveExhaustive() instead if you want every reachable position's
    // outcome, e.g. to build a complete book rather than just prove the root.
    bool solve(const PenteGame &rootGame);

    // Same preconditions as solve(), but resolves EVERY position reachable
    // via legal play from rootGame, not just the minimal subset needed to
    // prove the root - a full memoized postorder traversal (visit every
    // child, no proof-number-driven pruning) rather than df-pn's
    // threshold-guided descent. Reuses expandNode()/updatePnDn()/
    // resolveOutcome() unchanged: once every child of a node is fully
    // resolved, that node's pn/dn always end up exactly 0 in the direction
    // resolveOutcome() expects (provable from the OR/AND aggregation rules),
    // so no separate combine logic was needed for this mode.
    // Can visit substantially more nodes than solve() on the same game.
    // Still respects maxNodes/maxSeconds/maxRecursionDepth as safety nets;
    // returns false if any of them cut the traversal short (root and
    // whatever got fully resolved along the way remain valid/query-able,
    // but incomplete - not a full book in that case).
    bool solveExhaustive(const PenteGame &rootGame);

    Outcome getRootOutcome() const;

    // Plies from the root until the game genuinely ends under the specific
    // adversarial line implied by getRootOutcome() (0 if the root is itself
    // already game-over). Meaningless while getRootOutcome() == UNKNOWN.
    int getRootDepth() const;

    // Looks up an already-visited position's resolved outcome (UNKNOWN if
    // never visited or not yet resolved). Position is packed/canonicalized
    // exactly like solve()'s own nodes, so this works for any position
    // reachable from the last solve() call's root.
    Outcome getOutcome(const PenteGame &game) const;
    int getDepth(const PenteGame &game) const;

    uint64_t getNodeCount() const;
    const Stats &getStats() const { return stats_; }
    void printProofStats() const;

    // One resolved (outcome != UNKNOWN) position, for handing off to
    // PositionBook to persist. Deliberately excludes still-UNKNOWN nodes -
    // there's nothing useful to checkpoint about a position solve() hasn't
    // finished with yet.
    struct Record {
        PositionKey key;
        Outcome outcome;
        uint16_t depth;
    };
    std::vector<Record> exportResolved() const;

  private:
    struct Node {
        Number pn = 1;
        Number dn = 1;
        Outcome outcome = Outcome::UNKNOWN;
        // Plies from this node until game-over under the adversarial line
        // that produced `outcome` (see resolveOutcome()). 0 for a node that's
        // already terminal itself (set directly in expandNode) and for any
        // node still UNKNOWN (meaningless until resolved).
        uint16_t depth = 0;
        bool expanded = false;
        // Canonical coordinates, evaluateMove()-ordered best-first - NOT
        // physical, since this node may be shared (via the transposition
        // table) with parents reached through different physical
        // orientations. mid() re-derives the current orientation and
        // un-rotates before calling makeMove() - see the comment there.
        std::vector<PenteGame::Move> childMoves;
        std::vector<Node *> childPtr; // lazily materialized; nullptr = untried (default pn=dn=1)
    };

    Config config_;
    Stats stats_;
    std::unordered_map<PositionKey, Node> table_;
    Node *rootNode_ = nullptr;
    PenteGame::Player rootPlayer_ = PenteGame::NONE;
    bool stopRequested_ = false;
    std::chrono::steady_clock::time_point startTime_;

    static Number pnOf(const Node *n) { return n ? n->pn : 1; }
    static Number dnOf(const Node *n) { return n ? n->dn : 1; }

    // Depth helper: min/max over resolved children matching `want`, +1 for
    // this ply. Mirrors standard chess-engine mate-distance convention: the
    // side steering TOWARD an outcome takes the fastest line (min), the side
    // forced INTO it delays as long as possible (max).
    template <typename Predicate>
    static uint16_t depthFrom(const std::vector<Node *> &children, bool useMax, Predicate want);

    Node *getOrCreateNode(const PenteGame &game);
    void expandNode(Node *n, const PenteGame &game);
    void updatePnDn(Node *n, bool isOrNode) const;
    void resolveOutcome(Node *n, bool isOrNode) const;
    void selectChildOr(const Node *n, int &bestIdx, Number &secondPn) const;
    void selectChildAnd(const Node *n, int &bestIdx, Number &secondDn) const;
    void mid(Node *n, PenteGame game, Number thpn, Number thdn, int depth);
    void dfsExhaustive(Node *n, PenteGame game, int depth);

    static std::vector<PenteGame::Move> enumerateLegalMoves(const PenteGame &game);
};

#endif // PNS_HPP
