#include "GumbelMCTS.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

// ============================================================================
// Pure helpers (ports of mctx seq_halving.py / qtransforms.py)
// ============================================================================

std::vector<int> GumbelTree::consideredVisitSequence(int maxConsideredActions, int numSimulations) {
    std::vector<int> sequence;
    if (numSimulations <= 0) return sequence;
    if (maxConsideredActions <= 1) {
        sequence.resize(numSimulations);
        std::iota(sequence.begin(), sequence.end(), 0);
        return sequence;
    }
    const int log2max = static_cast<int>(std::ceil(std::log2(static_cast<double>(maxConsideredActions))));
    std::vector<int> visits(maxConsideredActions, 0);
    int numConsidered = maxConsideredActions;
    while (static_cast<int>(sequence.size()) < numSimulations) {
        const int numExtraVisits = std::max(1, numSimulations / (log2max * numConsidered));
        for (int e = 0; e < numExtraVisits; ++e) {
            sequence.insert(sequence.end(), visits.begin(), visits.begin() + numConsidered);
            for (int i = 0; i < numConsidered; ++i) ++visits[i];
        }
        numConsidered = std::max(2, numConsidered / 2);
    }
    sequence.resize(numSimulations);
    return sequence;
}

void GumbelTree::completedQValues(double rawValue, const float *logits, const double *q, const int *visits, int k,
                                  float maxVisitInit, float valueScale, float *out, double *mixedValueOut) {
    if (k <= 0) {
        if (mixedValueOut) *mixedValueOut = rawValue;
        return;
    }

    // softmax prior, floored at the smallest normal float so visited zero-prior
    // actions can't produce NaN (mctx: jnp.finfo.tiny)
    float maxLogit = logits[0];
    for (int i = 1; i < k; ++i) maxLogit = std::max(maxLogit, logits[i]);
    std::vector<double> prior(k);
    double z = 0.0;
    for (int i = 0; i < k; ++i) {
        prior[i] = std::exp(static_cast<double>(logits[i] - maxLogit));
        z += prior[i];
    }
    const double tiny = std::numeric_limits<float>::min();
    for (int i = 0; i < k; ++i) prior[i] = std::max(tiny, prior[i] / z);

    // mixed value (mctx _compute_mixed_value)
    long sumVisits = 0;
    double sumProbs = 0.0;
    int maxVisit = 0;
    for (int i = 0; i < k; ++i) {
        sumVisits += visits[i];
        maxVisit = std::max(maxVisit, visits[i]);
        if (visits[i] > 0) sumProbs += prior[i];
    }
    double weightedQ = 0.0;
    if (sumProbs > 0.0) {
        for (int i = 0; i < k; ++i)
            if (visits[i] > 0) weightedQ += prior[i] * q[i] / sumProbs;
    }
    const double mixed = (rawValue + static_cast<double>(sumVisits) * weightedQ) / (static_cast<double>(sumVisits) + 1.0);
    if (mixedValueOut) *mixedValueOut = mixed;

    // complete, rescale to [0,1], scale by visit count (mctx qtransform_completed_by_mix_value)
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    std::vector<double> completed(k);
    for (int i = 0; i < k; ++i) {
        completed[i] = visits[i] > 0 ? q[i] : mixed;
        lo = std::min(lo, completed[i]);
        hi = std::max(hi, completed[i]);
    }
    const double denom = std::max(hi - lo, 1e-8);
    const double scale = (static_cast<double>(maxVisitInit) + maxVisit) * static_cast<double>(valueScale);
    for (int i = 0; i < k; ++i) out[i] = static_cast<float>(scale * (completed[i] - lo) / denom);
}

// ============================================================================
// GumbelTree
// ============================================================================

GumbelTree::GumbelTree(const Config &config) : config_(config) {
    if (config_.seed != 0) {
        rng_.seed(config_.seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

int GumbelTree::createNode(int parent) {
    Node n;
    n.parent = parent;
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size()) - 1;
}

void GumbelTree::start(const PenteGame &game) {
    rootGame_.syncFrom(game);
    nodes_.clear();
    moves_.clear();
    logits_.clear();
    child_.clear();
    gumbel_.clear();
    consideredSeq_.clear();
    path_.clear();
    pending_ = -1;
    simsDone_ = 0;
    nodes_.reserve(static_cast<size_t>(std::max(0, config_.numSimulations)) + 2);

    if (game.isGameOver() || game.getLegalMoves().empty()) {
        phase_ = Phase::DONE;
        return;
    }
    createNode(-1);
    phase_ = Phase::AWAIT_ROOT;
}

bool GumbelTree::rootFinished() const {
    return simsDone_ >= config_.numSimulations || nodes_[0].solved != Status::UNSOLVED;
}

static double solvedValue(GumbelTree::Status s) {
    switch (s) {
    case GumbelTree::Status::SOLVED_WIN: return 1.0;
    case GumbelTree::Status::SOLVED_LOSS: return -1.0;
    default: return 0.0;
    }
}

bool GumbelTree::nextLeaf(PenteGame &leaf) {
    if (phase_ == Phase::AWAIT_ROOT) {
        pending_ = 0;
        path_.assign(1, 0);
        leaf = rootGame_;
        return true;
    }
    if (phase_ != Phase::SEARCHING) return false;

    while (!rootFinished()) {
        scratch_.syncFrom(rootGame_);
        path_.clear();
        int node = 0;
        path_.push_back(0);

        for (;;) {
            // A node proven by minimax needs no further search below it.
            if (nodes_[node].solved != Status::UNSOLVED) {
                backup(solvedValue(nodes_[node].solved));
                ++simsDone_;
                break;
            }

            const int slot = (node == 0) ? selectRoot() : selectInterior(node);
            const int idx = nodes_[node].begin + slot;
            const PenteGame::Move mv = moves_[idx];
            scratch_.makeMove(mv.x, mv.y);

            int ch = child_[idx];
            if (ch < 0) {
                ch = createNode(node);
                child_[idx] = ch;
                path_.push_back(ch);

                if (scratch_.isGameOver()) {
                    // The player who just moved won.
                    nodes_[ch].solved = Status::SOLVED_WIN;
                    nodes_[ch].expanded = true;
                    nodes_[ch].value = 1.0f;
                    backup(1.0);
                    ++simsDone_;
                    break;
                }
                if (scratch_.getLegalMoves().empty()) {
                    nodes_[ch].solved = Status::SOLVED_DRAW;
                    nodes_[ch].expanded = true;
                    nodes_[ch].value = 0.0f;
                    backup(0.0);
                    ++simsDone_;
                    break;
                }
                pending_ = ch;
                leaf = scratch_;
                return true;
            }
            node = ch;
            path_.push_back(ch);
        }
    }
    phase_ = Phase::DONE;
    return false;
}

void GumbelTree::finishLeaf(const Policy &policy, float value) {
    if (pending_ < 0) return;
    const int node = pending_;
    pending_ = -1;

    if (node == 0) {
        if (policy.empty()) {
            phase_ = Phase::DONE;
            return;
        }
        expandNode(0, policy, value);
        const int k = nodes_[0].count;
        gumbel_.assign(k, 0.0f);
        if (config_.gumbelNoise) {
            std::uniform_real_distribution<double> uni(1e-12, 1.0);
            for (int i = 0; i < k; ++i) gumbel_[i] = static_cast<float>(-std::log(-std::log(uni(rng_))));
        }
        consideredSeq_ = consideredVisitSequence(std::min(config_.maxConsideredActions, k), config_.numSimulations);
        phase_ = Phase::SEARCHING;
        return;
    }

    if (policy.empty()) {
        // No legal moves and not a win: proven draw.
        nodes_[node].solved = Status::SOLVED_DRAW;
        nodes_[node].expanded = true;
        nodes_[node].value = 0.0f;
        backup(0.0);
    } else {
        expandNode(node, policy, value);
        backup(value);
    }
    ++simsDone_;
}

void GumbelTree::expandNode(int node, const Policy &policy, float value) {
    Node &nd = nodes_[node];
    nd.begin = static_cast<int32_t>(moves_.size());
    nd.count = static_cast<int32_t>(policy.size());
    for (const auto &[move, prob] : policy) {
        moves_.push_back(move);
        logits_.push_back(std::log(std::max(prob, 1e-12f)));
        child_.push_back(-1);
    }
    nd.value = value;
    nd.expanded = true;
}

void GumbelTree::backup(double v) {
    for (int i = static_cast<int>(path_.size()) - 1; i >= 0; --i) {
        Node &nd = nodes_[path_[i]];
        ++nd.visits;
        nd.valueSum += v;
        if (i > 0) propagateSolved(path_[i - 1], path_[i]);
        v = -v;
    }
}

void GumbelTree::propagateSolved(int parent, int child) {
    Node &p = nodes_[parent];
    if (p.solved != Status::UNSOLVED) return;
    const Status st = nodes_[child].solved;
    if (st == Status::SOLVED_WIN) {
        // The player to move at `parent` has a winning move: whoever moved into parent loses.
        p.solved = Status::SOLVED_LOSS;
    } else if (st == Status::SOLVED_LOSS || st == Status::SOLVED_DRAW) {
        bool anyDraw = false;
        for (int i = 0; i < p.count; ++i) {
            const int ch = child_[p.begin + i];
            if (ch < 0) return;
            const Status cs = nodes_[ch].solved;
            if (cs == Status::SOLVED_DRAW) anyDraw = true;
            else if (cs != Status::SOLVED_LOSS) return;
        }
        p.solved = anyDraw ? Status::SOLVED_DRAW : Status::SOLVED_WIN;
    }
}

double GumbelTree::childQ(int child) const {
    const Node &c = nodes_[child];
    if (c.solved != Status::UNSOLVED) return solvedValue(c.solved);
    return c.valueSum / c.visits;
}

void GumbelTree::nodeCompletedQ(int node, std::vector<float> &out, double *mixedValueOut) const {
    const Node &nd = nodes_[node];
    const int k = nd.count;
    qBuf_.assign(k, 0.0);
    visitBuf_.assign(k, 0);
    for (int i = 0; i < k; ++i) {
        const int ch = child_[nd.begin + i];
        if (ch >= 0 && nodes_[ch].visits > 0) {
            visitBuf_[i] = nodes_[ch].visits;
            qBuf_[i] = childQ(ch);
        }
    }
    out.resize(k);
    // nd.value is from the mover-into-node's perspective; the mixed value needs the mover-at-node's.
    completedQValues(-static_cast<double>(nd.value), &logits_[nd.begin], qBuf_.data(), visitBuf_.data(), k,
                     config_.maxVisitInit, config_.valueScale, out.data(), mixedValueOut);
}

int GumbelTree::argmaxConsidered(int considered) const {
    const Node &r = nodes_[0];
    const int k = r.count;
    const float *logits = &logits_[r.begin];
    float maxLogit = logits[0];
    for (int i = 1; i < k; ++i) maxLogit = std::max(maxLogit, logits[i]);

    auto score = [&](int i) { return std::max(-1e9f, gumbel_[i] + (logits[i] - maxLogit) + compBuf_[i]); };

    int best = -1;
    float bestScore = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        if (visitBuf_[i] != considered) continue;
        const float s = score(i);
        if (best < 0 || s > bestScore) {
            best = i;
            bestScore = s;
        }
    }
    if (best >= 0) return best;
    // No action at the considered visit count (can't happen with a valid table); fall back to best overall.
    for (int i = 0; i < k; ++i) {
        const float s = score(i);
        if (best < 0 || s > bestScore) {
            best = i;
            bestScore = s;
        }
    }
    return best;
}

int GumbelTree::selectRoot() {
    nodeCompletedQ(0, compBuf_);
    const int simIndex = std::min(nodes_[0].visits, static_cast<int>(consideredSeq_.size()) - 1);
    return argmaxConsidered(consideredSeq_[std::max(0, simIndex)]);
}

int GumbelTree::selectFinal() const {
    nodeCompletedQ(0, compBuf_);
    const int considered = *std::max_element(visitBuf_.begin(), visitBuf_.end());
    return argmaxConsidered(considered);
}

int GumbelTree::selectInterior(int node) {
    nodeCompletedQ(node, compBuf_);
    const Node &nd = nodes_[node];
    const int k = nd.count;
    const float *logits = &logits_[nd.begin];

    float zmax = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) zmax = std::max(zmax, logits[i] + compBuf_[i]);
    double z = 0.0;
    long sumN = 0;
    for (int i = 0; i < k; ++i) {
        z += std::exp(static_cast<double>(logits[i] + compBuf_[i] - zmax));
        sumN += visitBuf_[i];
    }
    int best = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < k; ++i) {
        const double p = std::exp(static_cast<double>(logits[i] + compBuf_[i] - zmax)) / z;
        const double s = p - static_cast<double>(visitBuf_[i]) / (1.0 + static_cast<double>(sumN));
        if (s > bestScore) {
            bestScore = s;
            best = i;
        }
    }
    return best;
}

int GumbelTree::rootWinningChild() const {
    const Node &r = nodes_[0];
    for (int i = 0; i < r.count; ++i) {
        const int ch = child_[r.begin + i];
        if (ch >= 0 && nodes_[ch].solved == Status::SOLVED_WIN) return i;
    }
    return -1;
}

GumbelTree::Result GumbelTree::result() const {
    Result res;
    if (nodes_.empty() || !nodes_[0].expanded) return res;

    const Node &r = nodes_[0];
    const int k = r.count;
    res.moves.assign(moves_.begin() + r.begin, moves_.begin() + r.begin + k);
    res.visits.resize(k);
    for (int i = 0; i < k; ++i) {
        const int ch = child_[r.begin + i];
        res.visits[i] = ch >= 0 ? nodes_[ch].visits : 0;
    }
    res.rootValue = -r.value;
    res.simulations = simsDone_;

    const float *logits = &logits_[r.begin];

    // Root prior (softmax of the raw logits) and each child's own backed-up Q,
    // independent of win/solved status -- both describe what the evaluator and
    // the search itself observed, not the final move-selection outcome.
    {
        float maxLogit = logits[0];
        for (int i = 1; i < k; ++i) maxLogit = std::max(maxLogit, logits[i]);
        double z = 0.0;
        std::vector<double> ex(k);
        for (int i = 0; i < k; ++i) {
            ex[i] = std::exp(static_cast<double>(logits[i] - maxLogit));
            z += ex[i];
        }
        res.prior.resize(k);
        for (int i = 0; i < k; ++i) res.prior[i] = static_cast<float>(ex[i] / z);
    }
    res.q.assign(k, 0.0f);
    for (int i = 0; i < k; ++i) {
        const int ch = child_[r.begin + i];
        if (ch >= 0 && nodes_[ch].visits > 0) res.q[i] = static_cast<float>(childQ(ch));
    }

    const int win = rootWinningChild();
    if (win >= 0) {
        // Proven forced win: play it and train on a one-hot target.
        res.solved = true;
        res.improvedPolicy.assign(k, 0.0f);
        res.improvedPolicy[win] = 1.0f;
        res.move = res.moves[win];
        res.improvedValue = 1.0f;
        return res;
    }
    res.solved = r.solved != Status::UNSOLVED;

    double mixedValue = 0.0;
    nodeCompletedQ(0, compBuf_, &mixedValue);
    res.improvedValue = static_cast<float>(mixedValue);
    float zmax = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) zmax = std::max(zmax, logits[i] + compBuf_[i]);
    double z = 0.0;
    res.improvedPolicy.resize(k);
    for (int i = 0; i < k; ++i) {
        res.improvedPolicy[i] = std::exp(logits[i] + compBuf_[i] - zmax);
        z += res.improvedPolicy[i];
    }
    for (int i = 0; i < k; ++i) res.improvedPolicy[i] = static_cast<float>(res.improvedPolicy[i] / z);

    res.move = res.moves[selectFinal()];
    return res;
}

// ============================================================================
// GumbelSearch
// ============================================================================

// Minimal fork-join pool: parallelFor(n, f) runs f(0..n-1) across the pool plus the caller.
class GumbelSearch::Pool {
  public:
    explicit Pool(int numThreads) {
        for (int i = 1; i < numThreads; ++i) threads_.emplace_back([this] { workerMain(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        startCv_.notify_all();
        for (auto &t : threads_) t.join();
    }

    void parallelFor(int n, const std::function<void(int)> &f) {
        if (threads_.empty() || n <= 1) {
            for (int i = 0; i < n; ++i) f(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            job_ = &f;
            total_ = n;
            next_.store(0);
            pending_ = static_cast<int>(threads_.size());
            ++generation_;
        }
        startCv_.notify_all();
        drain();
        std::unique_lock<std::mutex> lock(m_);
        doneCv_.wait(lock, [this] { return pending_ == 0; });
    }

  private:
    void drain() {
        for (;;) {
            const int i = next_.fetch_add(1);
            if (i >= total_) return;
            (*job_)(i);
        }
    }

    void workerMain() {
        int seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(m_);
                startCv_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            drain();
            {
                std::lock_guard<std::mutex> lock(m_);
                if (--pending_ == 0) doneCv_.notify_one();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable startCv_, doneCv_;
    const std::function<void(int)> *job_ = nullptr;
    std::atomic<int> next_{0};
    int total_ = 0;
    int pending_ = 0;
    int generation_ = 0;
    bool stop_ = false;
};

GumbelSearch::GumbelSearch(const Config &config, Evaluator *evaluator)
    : config_(config), evaluator_(evaluator), pool_(std::make_unique<Pool>(std::max(1, config.numThreads))) {}

GumbelSearch::~GumbelSearch() = default;

GumbelTree::Result GumbelSearch::search(const PenteGame &game) {
    return searchBatch({game})[0];
}

std::vector<GumbelTree::Result> GumbelSearch::searchBatch(const std::vector<PenteGame> &games) {
    const int B = static_cast<int>(games.size());
    lastBatchCalls_ = 0;
    lastLeaves_ = 0;

    trees_.clear();
    trees_.reserve(B);
    for (int i = 0; i < B; ++i) {
        GumbelTree::Config tc = config_.tree;
        if (tc.seed != 0) tc.seed += static_cast<uint32_t>(i);
        trees_.emplace_back(tc);
    }
    pool_->parallelFor(B, [&](int i) { trees_[i].start(games[i]); });

    std::vector<PenteGame> leaves(B);
    std::vector<uint8_t> need(B);
    std::vector<int> active(B);
    std::iota(active.begin(), active.end(), 0);
    std::vector<PenteGame> batch;
    std::vector<int> batchTree;

    while (!active.empty()) {
        // 1. every active tree advances to its next leaf (independent trees -> no locking)
        pool_->parallelFor(static_cast<int>(active.size()), [&](int j) {
            need[j] = trees_[active[j]].nextLeaf(leaves[j]) ? 1 : 0;
        });

        // 2. one batched evaluation for all leaves that need it
        batch.clear();
        batchTree.clear();
        for (size_t j = 0; j < active.size(); ++j) {
            if (!need[j]) continue;
            batch.push_back(std::move(leaves[j]));
            batchTree.push_back(active[j]);
        }
        if (batch.empty()) break;
        auto results = evaluator_->evaluateBatch(batch);
        ++lastBatchCalls_;
        lastLeaves_ += static_cast<int>(batch.size());

        // 3. expand + backup
        pool_->parallelFor(static_cast<int>(batch.size()), [&](int j) {
            trees_[batchTree[j]].finishLeaf(results[j].first, results[j].second);
        });
        active = batchTree;
    }

    std::vector<GumbelTree::Result> out(B);
    for (int i = 0; i < B; ++i) out[i] = trees_[i].result();
    return out;
}
