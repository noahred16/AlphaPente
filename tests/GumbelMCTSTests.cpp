#include "Evaluator.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <algorithm>
#include <cmath>

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

// doctest must come AFTER the torch-dependent headers: the c10 logging header
// (pulled in via Evaluator.hpp when WITH_TORCH is set) defines its own CHECK
// macro, which silently shadows doctest's CHECK and turns assertions into no-ops.
#ifdef CHECK
#undef CHECK
#endif
#include "doctest.h"

#ifdef WITH_TORCH
#include "SelfPlay.hpp"
#endif

// ============================================================================
// gumbelPolicyTarget (pure function; only built when SelfPlay.cpp is, i.e.
// WITH_TORCH)
// ============================================================================

#ifdef WITH_TORCH

TEST_CASE("gumbelPolicyTarget - sums to 1 and is non-negative") {
    std::vector<float> logits{0.1f, -0.3f, 0.5f, 0.0f};
    std::vector<float> completedQ{0.2f, 0.0f, -0.1f, 0.3f};
    auto target = gumbelPolicyTarget(logits, completedQ, 1.0f, -1);

    REQUIRE(target.size() == logits.size());
    float sum = 0.0f;
    for (float p : target) {
        CHECK(p >= 0.0f);
        sum += p;
    }
    CHECK(sum == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("gumbelPolicyTarget - proven win takes all the mass") {
    std::vector<float> logits{0.1f, -0.3f, 0.5f};
    std::vector<float> completedQ{-0.9f, 0.9f, 0.0f};
    auto target = gumbelPolicyTarget(logits, completedQ, 1.0f, 1);

    std::vector<float> expected{0.0f, 1.0f, 0.0f};
    CHECK(target == expected);
}

TEST_CASE("gumbelPolicyTarget - invariant under a uniform logit shift") {
    // root->priors are already-softmaxed, legal-move-renormalized probabilities,
    // not raw logits: log(prob) differs from the true logit only by a per-node
    // additive constant (the renormalization divides every legal prob by the
    // same total). The policy target must be unaffected by that constant.
    std::vector<float> logits{0.1f, -0.3f, 0.5f, 0.0f};
    std::vector<float> completedQ{0.2f, 0.0f, -0.1f, 0.3f};
    auto base = gumbelPolicyTarget(logits, completedQ, 1.0f, -1);

    std::vector<float> shifted(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) shifted[i] = logits[i] + 7.0f;
    auto target = gumbelPolicyTarget(shifted, completedQ, 1.0f, -1);

    REQUIRE(target.size() == base.size());
    for (size_t i = 0; i < target.size(); ++i)
        CHECK(target[i] == doctest::Approx(base[i]).epsilon(1e-4));
}

TEST_CASE("SelfPlay GUMBEL mode - bootstrap examples form valid training targets") {
    // Mirrors SelfPlayPolicyTests.cpp's PUCT bootstrap test, but through the
    // SearchMode::GUMBEL path: gumbelPolicyTarget instead of policyTargetVisits,
    // and the move actually played comes straight from search() rather than
    // being sampled from visit counts (see runGame's `gumbel` branch).
    HeuristicEvaluator eval;
    SelfPlayConfig cfg;
    cfg.simulations      = 200;
    cfg.numWorkerThreads = 2;
    cfg.numEvalThreads   = 0;
    cfg.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    cfg.gumbelM          = 16;
    std::mt19937 rng(7);
    auto examples = runGame(eval, PenteGame::Config::pente(), cfg, rng);

    REQUIRE(!examples.empty());
    constexpr int B = PenteGame::BOARD_SIZE;
    for (const auto &ex : examples) {
        float psum = ex.policy.sum().item<float>();
        CHECK(std::abs(psum - 1.0f) < 1e-4f);
        CHECK(ex.policy.min().item<float>() >= 0.0f);

        // Mass only on empty cells
        auto emptyPlane = ex.planes[2].reshape({B * B});
        CHECK((((ex.policy > 0.0f) & (emptyPlane < 0.5f))).sum().item<int64_t>() == 0);

        CHECK((ex.outcome == 1.0f || ex.outcome == 0.0f || ex.outcome == -1.0f));
        CHECK(ex.rootValue >= -1.0f);
        CHECK(ex.rootValue <= 1.0f);
    }
}

#endif // WITH_TORCH

// ============================================================================
// Gumbel root selection (ParallelMCTS::SearchMode::GUMBEL) -- synthetic
// evaluator with a hand-baked best move, so Sequential Halving's convergence
// can be checked deterministically rather than against real playing strength.
// ============================================================================

namespace {

// Uniform policy over legal moves; value rewards whichever player holds
// bestMove_'s cell, at any depth -- a persistent (not one-shot-and-decaying)
// +0.9/-0.9 signal, so completedQ keeps reflecting it however many
// simulations deepen past the first ply. canonicalHashDepth must be disabled
// by the caller (Config::canonicalHashDepth = 0): symmetric root siblings
// would otherwise be folded onto the same shared tree node (one of this
// synthetic position's two D4 symmetries), corrupting the per-candidate
// stats this test inspects.
class BestMoveEvaluator : public Evaluator {
  public:
    explicit BestMoveEvaluator(PenteGame::Move bestMove) : bestMove_(bestMove) {}

    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) override {
        return {evaluatePolicy(game), evaluateValue(game)};
    }

    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) override {
        const auto &legal = game.getLegalMoves();
        if (legal.empty()) return {};
        float p = 1.0f / static_cast<float>(legal.size());
        std::vector<std::pair<PenteGame::Move, float>> result;
        result.reserve(legal.size());
        for (const auto &m : legal) result.emplace_back(m, p);
        return result;
    }

    float evaluateValue(const PenteGame &game) override {
        PenteGame::Player owner = game.getStoneAt(bestMove_.x, bestMove_.y);
        if (owner == PenteGame::NONE) return 0.0f;
        // Despite Evaluator.hpp's doc comment, the engine's real convention
        // (see HeuristicEvaluator::evaluateValue's "invert for opponent's
        // perspective" and Evaluator::rollout, which returns -1 when the
        // *current* player wins) is previous-mover-perspective: +1 means the
        // player who just moved into this position is winning. game.getCurrentPlayer()
        // here is the player about to move NEXT, i.e. the opponent of the
        // previous mover.
        PenteGame::Player previousMover =
            (game.getCurrentPlayer() == PenteGame::BLACK) ? PenteGame::WHITE : PenteGame::BLACK;
        return (previousMover == owner) ? 0.9f : -0.9f;
    }

  private:
    PenteGame::Move bestMove_;
};

// One move in from a fresh reset (the very first move is forced to the
// board center), so the root position has several legal candidates. Returns
// the game and one of those candidates to designate as "best".
std::pair<PenteGame, PenteGame::Move> makeTestPosition() {
    PenteGame game(PenteGame::Config::pente());
    game.reset();
    const auto &opening = game.getLegalMoves();
    REQUIRE(opening.size() == 1); // forced center opening
    game.makeMove(opening[0].x, opening[0].y);

    const auto &candidates = game.getLegalMoves();
    REQUIRE(candidates.size() > 1);
    return {game, candidates[0]};
}

} // namespace

TEST_CASE("runGumbelRootSearch converges to the known best move (inline, single-threaded)") {
    auto [game, bestMove] = makeTestPosition();
    BestMoveEvaluator evaluator(bestMove);

    ParallelMCTS::Config config;
    config.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    config.evaluator        = &evaluator;
    config.numWorkerThreads = 1;
    config.numEvalThreads   = 0; // inline mode -- no threading nondeterminism
    config.maxIterations    = 600;
    config.gumbelM          = 64; // >= legal move count: guarantees bestMove survives Top-k
    config.gumbelCScale      = 100.0f; // dominate Gumbel(0,1) noise so completedQ reliably decides
    config.seed              = 42;
    config.canonicalHashDepth = 0; // keep each symmetric root sibling a distinct, independently-tracked node

    ParallelMCTS mcts(config);
    PenteGame::Move chosen = mcts.search(game);

    CHECK(chosen.x == bestMove.x);
    CHECK(chosen.y == bestMove.y);
}

TEST_CASE("runGumbelRootSearch converges to the known best move (multi-threaded worker/eval pools)") {
    auto [game, bestMove] = makeTestPosition();
    BestMoveEvaluator evaluator(bestMove);

    ParallelMCTS::Config config;
    config.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    config.evaluator        = &evaluator;
    config.numWorkerThreads = 4;
    config.numEvalThreads   = 2;
    config.maxIterations    = 1200;
    config.gumbelM          = 64;
    config.gumbelCScale      = 100.0f; // dominate Gumbel(0,1) noise so completedQ reliably decides
    config.seed              = 7;
    config.canonicalHashDepth = 0; // keep each symmetric root sibling a distinct, independently-tracked node

    ParallelMCTS mcts(config);
    PenteGame::Move chosen = mcts.search(game);

    CHECK(chosen.x == bestMove.x);
    CHECK(chosen.y == bestMove.y);
}

TEST_CASE("GUMBEL mode degrades gracefully when gumbelM collapses to a single candidate") {
    auto [game, bestMove] = makeTestPosition();
    BestMoveEvaluator evaluator(bestMove);

    ParallelMCTS::Config config;
    config.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    config.evaluator        = &evaluator;
    config.numWorkerThreads = 1;
    config.numEvalThreads   = 0;
    config.maxIterations    = 50;
    config.gumbelM          = 1; // collapses to a single plain-PUCT pass
    config.seed              = 1;
    config.canonicalHashDepth = 0; // keep each symmetric root sibling a distinct, independently-tracked node

    ParallelMCTS mcts(config);
    PenteGame::Move chosen = mcts.search(game);

    CHECK(chosen.x >= 0);
    CHECK(chosen.y >= 0);
    CHECK(mcts.getRoot()->visits.load() > 0);
}

TEST_CASE("GUMBEL mode ignores dirichletAlpha/minRootVisits instead of compounding them") {
    auto [game, bestMove] = makeTestPosition();
    BestMoveEvaluator evaluator(bestMove);

    ParallelMCTS::Config config;
    config.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    config.evaluator        = &evaluator;
    config.numWorkerThreads = 1;
    config.numEvalThreads   = 0;
    config.maxIterations    = 600;
    config.gumbelM          = 64;
    config.gumbelCScale      = 100.0f; // dominate Gumbel(0,1) noise so completedQ reliably decides
    config.seed              = 42;
    config.canonicalHashDepth = 0; // keep each symmetric root sibling a distinct, independently-tracked node
    // Both PUCT-only knobs set alongside GUMBEL -- must not crash, double-apply
    // the root floor, or otherwise corrupt the search; Gumbel-Top-k + Sequential
    // Halving is the only exploration mechanism that actually runs.
    config.dirichletAlpha   = 0.3f;
    config.minRootVisits    = 10;

    ParallelMCTS mcts(config);
    PenteGame::Move chosen = mcts.search(game);

    CHECK(chosen.x == bestMove.x);
    CHECK(chosen.y == bestMove.y);
}

TEST_CASE("Sequential Halving phase boundaries don't leak virtual loss") {
    auto [game, bestMove] = makeTestPosition();
    BestMoveEvaluator evaluator(bestMove);

    ParallelMCTS::Config config;
    config.searchMode       = ParallelMCTS::SearchMode::GUMBEL;
    config.evaluator        = &evaluator;
    config.numWorkerThreads = 4;
    config.numEvalThreads   = 2;
    config.maxIterations    = 1200;
    config.gumbelM          = 64;
    config.gumbelCScale      = 100.0f; // dominate Gumbel(0,1) noise so completedQ reliably decides
    config.seed              = 7;
    config.canonicalHashDepth = 0; // keep each symmetric root sibling a distinct, independently-tracked node

    ParallelMCTS mcts(config);
    mcts.search(game);

    // Every node's virtual loss must have been fully returned to zero by the
    // end of search() -- the phase-boundary drain (drainQueuesRemovingVirtualLoss,
    // called from runWorkerPhase) is what's responsible for this across the
    // several stop/drain cycles Sequential Halving runs per move.
    const auto *root = mcts.getRoot();
    REQUIRE(root != nullptr);
    CHECK(root->virtualLosses.load() == 0);
    for (int i = 0; i < root->childCapacity; ++i) {
        const auto *child = root->children[i].load(std::memory_order_acquire);
        if (child) CHECK(child->virtualLosses.load() == 0);
    }
}
