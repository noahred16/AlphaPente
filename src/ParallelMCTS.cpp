#include "ParallelMCTS.hpp"
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <chrono>

thread_local ParallelMCTS::SlabView *ParallelMCTS::tl_slab = nullptr;

// ============================================================================
// VirtualLossManager Implementation
// ============================================================================

void ParallelMCTS::VirtualLossManager::addVirtualLoss(ThreadSafeNode *node, int32_t amount) {
    if (!node) return;
    node->virtualLosses.fetch_add(amount, std::memory_order_relaxed);
}

void ParallelMCTS::VirtualLossManager::removeVirtualLoss(ThreadSafeNode *node, int32_t amount) {
    if (!node) return;
    node->virtualLosses.fetch_sub(amount, std::memory_order_relaxed);
}

int32_t ParallelMCTS::VirtualLossManager::getEffectiveVisits(ThreadSafeNode *node) const {
    if (!node) return 0;
    // Effective visits accounts for virtual losses to reduce PUCT value during search
    int32_t actual = node->visits.load(std::memory_order_relaxed);
    int32_t virtual_losses = node->virtualLosses.load(std::memory_order_relaxed);
    return actual + virtual_losses;
}

// ============================================================================
// EvaluationQueue Implementation
// ============================================================================

ParallelMCTS::EvaluationQueue::EvaluationQueue(size_t capacity) : capacity(capacity) {}

bool ParallelMCTS::EvaluationQueue::tryPush(const EvaluationRequest &request) {
    std::lock_guard<std::mutex> lock(queueLock);
    if (queue.size() >= capacity) {
        return false;  // Queue is full
    }
    queue.push_back(request);
    return true;
}

std::vector<ParallelMCTS::EvaluationRequest> ParallelMCTS::EvaluationQueue::popBatch(size_t batchSize) {
    std::lock_guard<std::mutex> lock(queueLock);
    std::vector<EvaluationRequest> batch;
    batch.reserve(std::min(batchSize, queue.size()));

    while (!queue.empty() && batch.size() < batchSize) {
        batch.push_back(queue.front());
        queue.pop_front();
    }
    return batch;
}

bool ParallelMCTS::EvaluationQueue::empty() const {
    std::lock_guard<std::mutex> lock(queueLock);
    return queue.empty();
}

size_t ParallelMCTS::EvaluationQueue::size() const {
    std::lock_guard<std::mutex> lock(queueLock);
    return queue.size();
}

// ============================================================================
// BackpropagationQueue Implementation
// ============================================================================

void ParallelMCTS::BackpropagationQueue::push(const EvaluationResult &result) {
    std::lock_guard<std::mutex> lock(queueLock);
    queue.push_back(result);
}

std::vector<ParallelMCTS::EvaluationResult> ParallelMCTS::BackpropagationQueue::popAll() {
    std::lock_guard<std::mutex> lock(queueLock);
    std::vector<EvaluationResult> results(queue.begin(), queue.end());
    queue.clear();
    return results;
}

std::optional<ParallelMCTS::EvaluationResult> ParallelMCTS::BackpropagationQueue::tryPop() {
    std::lock_guard<std::mutex> lock(queueLock);
    if (queue.empty()) return std::nullopt;
    EvaluationResult result = std::move(queue.front());
    queue.pop_front();
    return result;
}

bool ParallelMCTS::BackpropagationQueue::empty() const {
    std::lock_guard<std::mutex> lock(queueLock);
    return queue.empty();
}

// ============================================================================
// Per-Thread Slab Allocator
// ============================================================================

void ParallelMCTS::setupSlabs() {
    if (!workerSlabs_.empty()) return;

    int numSlabs = config_.numWorkerThreads + 1;  // +1 for main thread (index 0)
    // Divide the arena into (numSlabs + 1) equal shares: numSlabs initial slabs +
    // one share left as the fallback/refill pool at the tail of the arena.
    size_t slabBytes = (arena_->totalSize() / static_cast<size_t>(numSlabs + 1)) & ~size_t(63);
    if (slabBytes == 0) throw std::bad_alloc();

    workerSlabs_.resize(numSlabs);
    for (int i = 0; i < numSlabs; ++i) {
        void *mem = arena_->allocateBytes(slabBytes, 64);
        if (!mem) throw std::bad_alloc();
        char *ptr          = reinterpret_cast<char *>(mem);
        workerSlabs_[i].current = ptr;
        workerSlabs_[i].end     = ptr + slabBytes;
    }
    // The remaining ~1/(numSlabs+1) of the arena is the fallback pool consumed by
    // refillSlab() and direct-fallback allocations, accessed only under arenaMutex_.
}

void ParallelMCTS::refillSlab(SlabView &slab) {
    std::lock_guard<std::mutex> lock(arenaMutex_);
    void *chunk = arena_->allocateBytes(kSlabRefillBytes, 64);
    if (chunk) {
        char *ptr   = reinterpret_cast<char *>(chunk);
        slab.current = ptr;
        slab.end     = ptr + kSlabRefillBytes;
    }
    // If chunk is nullptr the arena is fully exhausted; slab stays empty and the
    // caller falls through to the direct-fallback path (which will also return nullptr).
}

void *ParallelMCTS::allocateFromSlab(size_t bytes, size_t alignment) {
    if (tl_slab) {
        void *ptr = tl_slab->allocateBytes(bytes, alignment);
        if (ptr) return ptr;
        // Hot-path slab exhausted — carve a fresh chunk from the fallback pool.
        refillSlab(*tl_slab);
        ptr = tl_slab->allocateBytes(bytes, alignment);
        if (ptr) return ptr;
    }
    // No slab registered (main thread without search() setup, or arena fully exhausted).
    std::lock_guard<std::mutex> lock(arenaMutex_);
    return arena_->allocateBytes(bytes, alignment);
}

// ============================================================================
// WorkerPool Implementation
// ============================================================================

ParallelMCTS::WorkerPool::WorkerPool(int numWorkerThreads, ParallelMCTS *parent)
    : numWorkerThreads(numWorkerThreads), parent(parent) {}

ParallelMCTS::WorkerPool::~WorkerPool() {
    stop();
}

void ParallelMCTS::WorkerPool::start() {
    std::lock_guard<std::mutex> lock(poolLock);
    if (running) return;

    completedWorkers = 0;
    running = true;
    threads.clear();
    for (int i = 0; i < numWorkerThreads; ++i) {
        threads.emplace_back(&WorkerPool::workerThreadMain, this, i);
    }
}

bool ParallelMCTS::WorkerPool::waitForCompletion(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(completionMutex);
    return completionCv.wait_for(lock, timeout, [this] {
        return completedWorkers.load() == numWorkerThreads;
    });
}

void ParallelMCTS::WorkerPool::stop() {
    {
        std::lock_guard<std::mutex> lock(poolLock);
        running = false;
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads.clear();
}

bool ParallelMCTS::WorkerPool::isRunning() const {
    std::lock_guard<std::mutex> lock(poolLock);
    return running;
}

void ParallelMCTS::WorkerPool::workerThreadMain(int workerId) {
    // Register this thread's slab (index 0 is reserved for the main thread).
    // Guard against tests that start workers without calling search() first.
    int slabIndex = workerId + 1;
    if (slabIndex < static_cast<int>(parent->workerSlabs_.size())) {
        ParallelMCTS::tl_slab = &parent->workerSlabs_[slabIndex];
    }

    // Cache root and initial game state once — both are written in prepareRoot()
    // before any worker starts and are never modified during search, so reading
    // them here without a lock is safe (happens-before via thread start).
    ThreadSafeNode *const root     = parent->root_;
    const PenteGame        rootGame = parent->initialGame_;

    if (!root) {
        // Should never happen, but guard defensively.
        if (completedWorkers.fetch_add(1) + 1 == numWorkerThreads)
            completionCv.notify_one();
        return;
    }

    PenteGame workerGame;
    int maxIterations = parent->config_.maxIterations;

    try {
    while (running) {
        // PHASE 1: Atomically claim an iteration slot — prevents over-selection
        int slot = parent->totalInProgress.fetch_add(1, std::memory_order_relaxed);
        if (slot < maxIterations) {
            std::vector<ThreadSafeNode *> searchPath;

            workerGame.syncFrom(rootGame);
            searchPath.push_back(root);
            ThreadSafeNode *leaf = parent->select(root, workerGame, searchPath);

            if (parent->config_.numEvalThreads == 0) {
                // Inline mode: evaluate, expand, and backprop in this worker.
                // No queue round-trip — eliminates pipeline overhead when the
                // evaluator is cheap (CPU heuristic).

                // If the game is already over, the previous mover won.
                // Use +1.0f directly rather than calling the evaluator, which reads
                // board features (e.g. open fours that no longer matter) and returns
                // the wrong sign for terminal positions.
                float value;
                if (workerGame.isGameOver()) {
                    value = 1.0f;
                } else {
                    value = parent->config_.evaluator->evaluateValue(workerGame);
                }

                auto  policy = parent->config_.evaluator->evaluatePolicy(workerGame);
                parent->expand(leaf, workerGame, policy);
                parent->backpropagate(leaf, value, searchPath);
                parent->totalIterations.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Queue mode: hand off to eval thread; backprop happens in Phase 2.
                EvaluationRequest request;
                request.node       = leaf;
                request.gameState  = workerGame;
                request.searchPath = searchPath;
                if (!parent->evaluationQueue_->tryPush(request)) {
                    // Queue full — return the slot so it can be retried
                    parent->totalInProgress.fetch_sub(1, std::memory_order_relaxed);
                }
            }
        } else {
            parent->totalInProgress.fetch_sub(1, std::memory_order_relaxed);
        }

        // PHASE 2 (queue mode only): pull one backprop result per loop iteration
        // so all workers share the backprop work rather than one worker draining all.
        if (parent->config_.numEvalThreads > 0) {
            auto result = parent->backpropagationQueue_->tryPop();
            if (result.has_value()) {
                parent->expand(result->node, result->gameState, result->policy);
                parent->backpropagate(result->node, result->value, result->searchPath);
                parent->totalIterations.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (parent->totalIterations.load(std::memory_order_relaxed) >= maxIterations)
            break;

        std::this_thread::yield();
    }
    } catch (const std::exception &e) {
        std::cerr << "Worker " << workerId << " caught exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Worker " << workerId << " caught unknown exception\n";
    }

    ParallelMCTS::tl_slab = nullptr;  // unregister before signaling completion

    if (completedWorkers.fetch_add(1) + 1 == numWorkerThreads) {
        completionCv.notify_one();
    }
}

// ============================================================================
// EvalPool Implementation
// ============================================================================

ParallelMCTS::EvalPool::EvalPool(int numEvalThreads, ParallelMCTS *parent)
    : numEvalThreads(numEvalThreads), parent(parent) {}

ParallelMCTS::EvalPool::~EvalPool() {
    stop();
}

void ParallelMCTS::EvalPool::start() {
    std::lock_guard<std::mutex> lock(poolLock);
    if (running) return;

    running = true;
    threads.clear();
    for (int i = 0; i < numEvalThreads; ++i) {
        threads.emplace_back(&EvalPool::evalThreadMain, this, i);
    }
}

void ParallelMCTS::EvalPool::stop() {
    {
        std::lock_guard<std::mutex> lock(poolLock);
        running = false;
    }
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }
    threads.clear();
}

bool ParallelMCTS::EvalPool::isRunning() const {
    std::lock_guard<std::mutex> lock(poolLock);
    return running;
}

void ParallelMCTS::EvalPool::evalThreadMain(int /*evalId*/) {
    while (running) {
        auto batch = parent->evaluationQueue_->popBatch(parent->config_.evaluationBatchSize);

        for (const auto &req : batch) {
            EvaluationResult result;
            result.node = req.node;
            result.gameState = req.gameState;
            result.value = parent->config_.evaluator->evaluateValue(req.gameState);
            result.policy = parent->config_.evaluator->evaluatePolicy(req.gameState);
            result.searchPath = req.searchPath;

            parent->backpropagationQueue_->push(result);
        }

        if (batch.empty()) {
            std::this_thread::yield();
        }
    }
}

// ============================================================================
// ParallelMCTS Main Implementation
// ============================================================================

ParallelMCTS::ParallelMCTS(const Config &config) : config_(config) {
    // Initialize arena
    arena_ = std::make_unique<Arena>(config_.arenaSize);

    // Initialize queues and managers
    virtualLossManager_ = std::make_unique<VirtualLossManager>();
    evaluationQueue_ = std::make_unique<EvaluationQueue>(config_.queueCapacity);
    backpropagationQueue_ = std::make_unique<BackpropagationQueue>();

    // Initialize worker and eval pools
    workerPool_ = std::make_unique<WorkerPool>(config_.numWorkerThreads, this);
    evalPool_ = std::make_unique<EvalPool>(config_.numEvalThreads, this);

    // Seed RNG
    if (config_.seed != 0) {
        rng_.seed(config_.seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

ParallelMCTS::~ParallelMCTS() {
    if (evalPool_) evalPool_->stop();
    if (workerPool_) workerPool_->stop();
}

PenteGame::Move ParallelMCTS::search(const PenteGame &game) {
    totalIterations = 0;
    totalInProgress = 0;
    setupSlabs();
    tl_slab = &workerSlabs_[0];  // main thread uses slab 0 during prepareRoot
    prepareRoot(game);
    tl_slab = nullptr;

    if (config_.numEvalThreads > 0) evalPool_->start();
    workerPool_->start();

    // Scale timeout with iteration count: assume at least 20k iters/sec floor,
    // with a 60s minimum. For 9M iterations this gives ~450s vs the 136s actually
    // needed at typical throughput — avoids the 60s hardcoded cutoff.
    auto timeoutMs = std::chrono::milliseconds(
        std::max(60000LL, static_cast<long long>(config_.maxIterations) / 20LL));
    bool completed = workerPool_->waitForCompletion(timeoutMs);
    if (!completed)
        std::cerr << "Warning: parallel MCTS search timed out\n";

    workerPool_->stop();
    if (config_.numEvalThreads > 0) evalPool_->stop();

    return getBestMove();
}

void ParallelMCTS::prepareRoot(const PenteGame &game) {
    initialGame_ = game;

    std::lock_guard<std::mutex> lock(treeLock);

    root_ = allocateNode();
    root_->player = game.getCurrentPlayer();
    root_->positionHash = game.getHash();

    auto policy = config_.evaluator->evaluatePolicy(game);
    root_->value = config_.evaluator->evaluateValue(game);

    int capacity = static_cast<int>(policy.size());
    initNodeChildren(root_, capacity);

    root_->moves  = static_cast<PenteGame::Move *>(
        allocateFromSlab(sizeof(PenteGame::Move) * capacity, alignof(PenteGame::Move)));
    root_->priors = static_cast<float *>(
        allocateFromSlab(sizeof(float) * capacity, alignof(float)));

    for (int i = 0; i < capacity; ++i) {
        root_->moves[i] = policy[i].first;
        root_->priors[i] = policy[i].second;
    }

    root_->childCount = static_cast<uint16_t>(capacity);
    root_->expanded = true;
    root_->evaluated = true;
}

const ParallelMCTS::ThreadSafeNode *ParallelMCTS::getRoot() const {
    return root_;
}

void ParallelMCTS::startWorkerThreads() {
    workerPool_->start();
}

void ParallelMCTS::stopWorkerThreads() {
    workerPool_->stop();
}

std::vector<ParallelMCTS::EvaluationRequest> ParallelMCTS::drainEvalQueue() {
    return evaluationQueue_->popBatch(evaluationQueue_->size());
}

void ParallelMCTS::startEvalThreads() {
    evalPool_->start();
}

void ParallelMCTS::stopEvalThreads() {
    evalPool_->stop();
}

void ParallelMCTS::pushEvalRequest(const EvaluationRequest &request) {
    evaluationQueue_->tryPush(request);
}

std::vector<ParallelMCTS::EvaluationResult> ParallelMCTS::drainBackpropQueue() {
    return backpropagationQueue_->popAll();
}

PenteGame::Move ParallelMCTS::getBestMove() const {
    if (!root_ || root_->childCapacity == 0) return PenteGame::Move{-1, -1};

    int bestIndex = -1;
    int32_t bestVisits = -1;

    for (int i = 0; i < root_->childCapacity; ++i) {
        ThreadSafeNode *child = root_->children[i];
        if (child == nullptr) continue;

        if (child->solvedStatus == SolvedStatus::SOLVED_WIN)
            return root_->moves[i];

        int32_t visits = child->visits.load(std::memory_order_relaxed);
        if (visits > bestVisits) {
            bestVisits = visits;
            bestIndex = i;
        }
    }

    return bestIndex >= 0 ? root_->moves[bestIndex] : PenteGame::Move{-1, -1};
}

void ParallelMCTS::reset() {
    std::lock_guard<std::mutex> lock(treeLock);
    if (arena_) arena_->reset();
    workerSlabs_.clear();  // slabs are views into arena_; cleared with it
    root_ = nullptr;
    nodeCount = 0;
    totalIterations = 0;
}

void ParallelMCTS::clearTree() {
    reset();
}

int ParallelMCTS::getTotalVisits() const {
    if (!root_) return 0;
    return root_->visits.load(std::memory_order_relaxed);
}

int ParallelMCTS::getTreeSize() const {
    return nodeCount.load(std::memory_order_relaxed);
}

void ParallelMCTS::printStats(double wallTime) const {
    int iters = totalIterations.load();
    double itersPerSec = wallTime > 0.0 ? iters / wallTime : 0.0;
    std::cout << "=== Parallel MCTS Stats ===" << std::endl;
    std::cout << "Total Iterations: " << iters << std::endl;
    std::cout << "Root Visits: " << getTotalVisits() << std::endl;
    std::cout << "Wall Time: " << wallTime << "s" << std::endl;
    std::cout << "Throughput (Wall): " << static_cast<int>(itersPerSec) << " iters/sec" << std::endl;
    std::cout << "Evaluation Queue Size: " << evaluationQueue_->size() << std::endl;
    if (arena_) {
        double usedMB  = arena_->bytesUsed()  / (1024.0 * 1024.0);
        double totalGB = arena_->totalSize()  / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Arena: " << std::fixed << std::setprecision(1)
                  << usedMB << " MB / " << totalGB << " GB ("
                  << std::setprecision(1) << arena_->utilizationPercent() << "%)\n";
    }
}

void ParallelMCTS::printBestMoves(int n) const {
    if (!root_ || root_->childCapacity == 0) return;

    // Collect and sort children by visit count descending
    struct Entry {
        int index;
        int32_t visits;
    };
    std::vector<Entry> entries;
    entries.reserve(root_->childCapacity);
    for (int i = 0; i < root_->childCapacity; ++i) {
        const ThreadSafeNode *child = root_->children[i];
        if (!child) continue;
        entries.push_back({i, child->visits.load(std::memory_order_relaxed)});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) { return a.visits > b.visits; });

    int show = std::min(n, static_cast<int>(entries.size()));

    // Move label: skip 'I' to match board notation
    auto moveLabel = [](int x, int y) -> std::string {
        char col = static_cast<char>('A' + x);
        if (col >= 'I') col++;
        return std::string(1, col) + std::to_string(y + 1);
    };

    std::cout << "\n=== Top " << show << " Moves of " << entries.size() << " Considered ===\n";
    std::cout << std::setw(6)  << "Move"
              << std::setw(10) << "Visits"
              << std::setw(10) << "Prior"
              << std::setw(10) << "Avg Val"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << std::string(46, '-') << "\n";

    std::cout << std::fixed;
    for (int k = 0; k < show; ++k) {
        int i = entries[k].index;
        const ThreadSafeNode *child = root_->children[i];
        int32_t visits = child->visits.load(std::memory_order_relaxed);
        double  avgVal = visits > 0
            ? child->totalValue.load(std::memory_order_relaxed) / visits
            : 0.0;
        const char *status =
            child->solvedStatus == SolvedStatus::SOLVED_WIN  ? "WIN"  :
            child->solvedStatus == SolvedStatus::SOLVED_LOSS ? "LOSS" : "-";

        std::cout << std::setw(6)  << moveLabel(root_->moves[i].x, root_->moves[i].y)
                  << std::setw(10) << visits
                  << std::setw(10) << std::setprecision(3) << root_->priors[i]
                  << std::setw(10) << std::setprecision(3) << avgVal
                  << std::setw(10) << status
                  << "\n";
    }
    std::cout << std::string(46, '=') << "\n";
}

void ParallelMCTS::setConfig(const Config &config) {
    config_ = config;
}

const ParallelMCTS::Config &ParallelMCTS::getConfig() const {
    return config_;
}

// ============================================================================
// Private Methods
// ============================================================================

ParallelMCTS::ThreadSafeNode *ParallelMCTS::select(ThreadSafeNode *node, PenteGame &game, std::vector<ThreadSafeNode *> &searchPath) {
    // Traverse tree using PUCT selection until we find a leaf node
    while (node->expanded.load(std::memory_order_acquire) && !node->isTerminal()) {
        // Select best child index using PUCT
        int bestIndex = selectBestMoveIndex(node, game);
        if (bestIndex < 0) return node;  // no children (e.g. zero-capacity terminal)

        assert(bestIndex < node->childCapacity && "bestIndex out of range");
        assert(node->moves != nullptr && "moves array must be allocated before selection");
        PenteGame::Move move = node->moves[bestIndex];
        game.makeMove(move.x, move.y);

        // Double-checked locking: fast path avoids lock when child already exists
        ThreadSafeNode *child = node->children[bestIndex];
        if (child == nullptr) {
            std::lock_guard<std::mutex> lock(node->nodeSubtreeLock);
            child = node->children[bestIndex];  // re-read under lock
            if (child == nullptr) {
                child = allocateNode();
                if (!child) return node;  // arena full, treat as leaf
                child->move = move;
                child->player = game.getCurrentPlayer();
                node->children[bestIndex] = child;
            }
        }

        virtualLossManager_->addVirtualLoss(child);
        node = child;
        searchPath.push_back(node);
    }

    return node;
}

void ParallelMCTS::expand(ThreadSafeNode *node, const PenteGame &game,
                          const std::vector<std::pair<PenteGame::Move, float>> &policy) {
    if (node->expanded.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lock(node->nodeSubtreeLock);
    if (node->expanded.load(std::memory_order_relaxed)) return;  // double-check

    int capacity = static_cast<int>(policy.size());

    initNodeChildren(node, capacity);
    // initNodeChildren may zero childCapacity if the arena was exhausted;
    // use the actual post-allocation capacity for all subsequent work.
    int actualCap = static_cast<int>(node->childCapacity);
    if (actualCap > 0) {
        node->moves  = static_cast<PenteGame::Move *>(
            allocateFromSlab(sizeof(PenteGame::Move) * actualCap, alignof(PenteGame::Move)));
        node->priors = static_cast<float *>(
            allocateFromSlab(sizeof(float) * actualCap, alignof(float)));

        if (node->moves && node->priors) {
            for (int i = 0; i < actualCap; ++i) {
                node->moves[i]  = policy[i].first;
                node->priors[i] = policy[i].second;
            }
        } else {
            // moves/priors allocation failed — collapse to unexpandable leaf
            node->children     = nullptr;
            node->childCapacity = 0;
            actualCap = 0;
        }
    }

    node->childCount = static_cast<uint16_t>(actualCap);
    if (game.isGameOver()) {
        // After a winning move, currentPlayer has already toggled to the NEXT player.
        // The winner is always the previous mover (opponent of currentPlayer).
        // Convention matches single-threaded MCTS: SOLVED_WIN means the last mover won,
        // which is a win from the parent's perspective (the parent made this winning move).
        node->value = 1.0f;
        node->solvedStatus = SolvedStatus::SOLVED_WIN;
    }
    node->evaluated.store(true, std::memory_order_relaxed);
    node->expanded.store(true, std::memory_order_release);
}

void ParallelMCTS::backpropagate(ThreadSafeNode *node, float value,
                                 std::vector<ThreadSafeNode *> &searchPath) {
    float currentValue = value;

    while (!searchPath.empty()) {
        ThreadSafeNode *current = searchPath.back();
        searchPath.pop_back();

        current->visits.fetch_add(1, std::memory_order_relaxed);

        // Atomic double add for totalValue (fetch_add not available for double pre-C++20)
        double expected = current->totalValue.load(std::memory_order_relaxed);
        while (!current->totalValue.compare_exchange_weak(
                   expected, expected + currentValue, std::memory_order_relaxed));

        // Remove virtual loss if one was applied (root has none)
        if (current->virtualLosses.load(std::memory_order_relaxed) > 0)
            virtualLossManager_->removeVirtualLoss(current);

        currentValue = -currentValue;  // flip for parent's perspective
    }
}

int ParallelMCTS::selectBestMoveIndex(ThreadSafeNode *node, const PenteGame &game) const {
    // Simplified PUCT selection - assumes priors are already set
    int cap = node->childCapacity;
    double explorationFactor = config_.explorationConstant;
    int32_t parentVisits = node->visits.load(std::memory_order_relaxed);
    double sqrtParentVisits = std::sqrt(static_cast<double>(parentVisits));

    int bestIndex = -1;
    double bestValue = -std::numeric_limits<double>::infinity();

    // Score all children (visited and unvisited) in one pass.
    // Unvisited (null) children contribute exploitation=0 so their exploration
    // term drives selection until they are visited — correct AlphaZero PUCT.
    for (int i = 0; i < cap; ++i) {
        ThreadSafeNode *child = node->children[i];

        int32_t effectiveVisits;
        double  exploitation;
        if (child == nullptr) {
            effectiveVisits = 0;
            exploitation    = 0.0;
        } else {
            effectiveVisits = virtualLossManager_->getEffectiveVisits(child);
            if (child->solvedStatus == SolvedStatus::SOLVED_WIN) {
                exploitation = std::numeric_limits<double>::infinity();
            } else if (child->solvedStatus == SolvedStatus::SOLVED_LOSS) {
                exploitation = -std::numeric_limits<double>::infinity();
            } else {
                exploitation = (effectiveVisits == 0) ? 0.0 :
                    static_cast<double>(child->totalValue.load(std::memory_order_relaxed)) / effectiveVisits;
            }
        }

        double exploration = explorationFactor * node->priors[i] * sqrtParentVisits / (1.0 + effectiveVisits);
        double score       = exploitation + exploration;

        if (score > bestValue) {
            bestValue = score;
            bestIndex = i;
        }
    }

    return bestIndex;
}

ParallelMCTS::ThreadSafeNode *ParallelMCTS::allocateNode() {
    void *mem = allocateFromSlab(sizeof(ThreadSafeNode), alignof(ThreadSafeNode));
    if (!mem) return nullptr;
    auto *node = new (mem) ThreadSafeNode();
    nodeCount.fetch_add(1, std::memory_order_relaxed);
    return node;
}

void ParallelMCTS::initNodeChildren(ThreadSafeNode *node, int capacity) {
    if (capacity <= 0) {
        node->children = nullptr;
        node->childCapacity = 0;
        return;
    }

    auto *children = static_cast<ThreadSafeNode **>(
        allocateFromSlab(sizeof(ThreadSafeNode *) * capacity, alignof(ThreadSafeNode *)));
    if (!children) {
        node->children = nullptr;
        node->childCapacity = 0;
        return;
    }
    std::memset(children, 0, sizeof(ThreadSafeNode *) * capacity);
    node->children    = children;
    node->childCapacity = static_cast<uint16_t>(capacity);
    node->childCount  = 0;
}
