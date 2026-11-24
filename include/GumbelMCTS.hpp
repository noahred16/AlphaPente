#ifndef GUMBEL_MCTS_HPP
#define GUMBEL_MCTS_HPP

#include "Evaluator.hpp"
#include "PenteGame.hpp"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// ============================================================================
// Gumbel MCTS (Danihelka et al. 2022, "Policy improvement by planning with Gumbel")
//
// Modelled on DeepMind's mctx `gumbel_muzero_policy`:
//   * root: Gumbel-Top-k + Sequential Halving, expressed as a stateless
//     "considered visit count" table (no explicit phases / barriers)
//   * interior: deterministic argmax of  pi'(a) - N(a)/(1+sum N)  where
//     pi' = softmax(logits + sigma(completedQ))
//   * output: improved policy softmax(logits + sigma(completedQ)) at the root
//     (training target) and the Sequential Halving winner (move to play)
//
// A GumbelTree is a single-threaded, self-contained search for ONE position.
// It never calls the evaluator itself: the caller drives it with
//     while (tree.nextLeaf(leaf)) tree.finishLeaf(evaluate(leaf));
// so many trees can be stepped in lockstep and their leaves evaluated in one
// batched forward pass (see GumbelSearch below).
//
// Value convention (same as Evaluator / ParallelMCTS): a value is from the
// perspective of the player who made the move INTO the position
// (+1 = that player is winning).
// ============================================================================
class GumbelTree {
  public:
    using Policy = std::vector<std::pair<PenteGame::Move, float>>;

    struct Config {
        int numSimulations = 64;       // n: simulations after the root evaluation
        int maxConsideredActions = 16; // m: actions considered at the root
        float valueScale = 0.1f;       // mctx qtransform value_scale
        float maxVisitInit = 50.0f;    // mctx qtransform maxvisit_init
        bool gumbelNoise = true;       // false: g = 0 (deterministic top-m by prior)
        uint32_t seed = 0;             // 0 = non-deterministic
    };

    enum class Status : uint8_t { UNSOLVED = 0, SOLVED_WIN, SOLVED_LOSS, SOLVED_DRAW };

    struct Result {
        PenteGame::Move move;                // Move::INVALID if the root has no legal move
        std::vector<PenteGame::Move> moves;  // root legal moves
        std::vector<float> improvedPolicy;   // aligned with moves; sums to 1
        std::vector<int> visits;             // aligned with moves
        float rootValue = 0.0f;              // raw network value, root mover's perspective
        bool solved = false;                 // result proven by minimax
        int simulations = 0;                 // simulations actually run
    };

    GumbelTree() : GumbelTree(Config{}) {}
    explicit GumbelTree(const Config &config);

    // Begin a new search of `game` (discards any previous tree).
    void start(const PenteGame &game);

    // Advance until a position needs evaluation. Returns true and fills `leaf`
    // (caller must evaluate it and call finishLeaf), or false when the search is
    // finished. Terminal / already-proven leaves are resolved internally.
    bool nextLeaf(PenteGame &leaf);

    // Supply the evaluation for the leaf returned by the last nextLeaf().
    // policy: one entry per legal move; value: previous-mover perspective.
    void finishLeaf(const Policy &policy, float value);

    bool done() const { return phase_ == Phase::DONE; }
    Result result() const;

    // ---- pure helpers (public so they can be tested against mctx) ----------

    // mctx seq_halving.get_sequence_of_considered_visits.
    static std::vector<int> consideredVisitSequence(int maxConsideredActions, int numSimulations);

    // mctx qtransform_completed_by_mix_value. q[i] is only read where visits[i] > 0.
    // rawValue is from the perspective of the player to move at the node.
    static void completedQValues(double rawValue, const float *logits, const double *q, const int *visits, int k,
                                 float maxVisitInit, float valueScale, float *out);

  private:
    enum class Phase : uint8_t { IDLE, AWAIT_ROOT, SEARCHING, DONE };

    struct Node {
        int32_t parent = -1;
        int32_t begin = 0; // slice [begin, begin+count) into moves_/logits_/child_
        int32_t count = 0;
        int32_t visits = 0;
        double valueSum = 0.0; // sum of backed-up values, mover-into-node perspective
        float value = 0.0f;    // raw evaluator value, mover-into-node perspective
        Status solved = Status::UNSOLVED;
        bool expanded = false;
    };

    int createNode(int parent);
    void expandNode(int node, const Policy &policy, float value);
    void backup(double leafValue);
    void propagateSolved(int parent, int child);
    bool rootFinished() const;

    // Q of child c from its parent's mover perspective (c must have visits > 0).
    double childQ(int child) const;
    // Fills q/visits for node's children and returns completed transformed Q in out.
    void nodeCompletedQ(int node, std::vector<float> &out) const;
    // Argmax of g + logit + sigma(q) over root actions whose visit count == considered
    // (uses the buffers filled by nodeCompletedQ(0, compBuf_)).
    int argmaxConsidered(int considered) const;
    int selectRoot();
    int selectInterior(int node);
    int selectFinal() const;
    int rootWinningChild() const;

    Config config_;
    std::mt19937 rng_;
    Phase phase_ = Phase::IDLE;

    PenteGame rootGame_;
    PenteGame scratch_;

    std::vector<Node> nodes_;
    std::vector<PenteGame::Move> moves_;
    std::vector<float> logits_;
    std::vector<int32_t> child_;

    std::vector<float> gumbel_;           // per root action
    std::vector<int> consideredSeq_;      // sequence for current root
    std::vector<int32_t> path_;           // root..pending leaf
    int pending_ = -1;
    int simsDone_ = 0;

    // scratch buffers (avoid per-select allocation)
    mutable std::vector<double> qBuf_;
    mutable std::vector<int> visitBuf_;
    mutable std::vector<float> compBuf_;
};

// ============================================================================
// GumbelSearch: runs B trees in lockstep with one batched evaluator call per
// tick (the mctx model, on CPU). Tree selection / expansion / backup is spread
// over a small thread pool; trees share nothing so no locking is needed.
// ============================================================================
class GumbelSearch {
  public:
    struct Config {
        GumbelTree::Config tree;
        int numThreads = 1; // threads for per-tree work (selection, expand, backup)
    };

    GumbelSearch(const Config &config, Evaluator *evaluator);
    ~GumbelSearch();

    // Tree i is seeded with tree.seed + i (when tree.seed != 0), so results do
    // not depend on batch composition or thread count.
    std::vector<GumbelTree::Result> searchBatch(const std::vector<PenteGame> &games);
    GumbelTree::Result search(const PenteGame &game);

    // Number of evaluateBatch calls and total leaves evaluated in the last search.
    int lastBatchCalls() const { return lastBatchCalls_; }
    int lastLeavesEvaluated() const { return lastLeaves_; }

  private:
    class Pool;
    Config config_;
    Evaluator *evaluator_;
    std::unique_ptr<Pool> pool_;
    std::vector<GumbelTree> trees_;
    int lastBatchCalls_ = 0;
    int lastLeaves_ = 0;
};

#endif // GUMBEL_MCTS_HPP
