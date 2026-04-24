#include "ParallelMCTS.hpp"
#include <cassert>
#include <iostream>
#include <chrono>

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
    PenteGame workerGame;
    int maxIterations = parent->config_.maxIterations;

    while (running) {
        // PHASE 1: Atomically claim an iteration slot — prevents over-selection
        int slot = parent->totalInProgress.fetch_add(1, std::memory_order_relaxed);
        if (slot < maxIterations) {
            std::vector<ThreadSafeNode *> searchPath;

            ThreadSafeNode *root = nullptr;
            {
                std::lock_guard<std::mutex> lock(parent->treeLock);
                root = parent->root_;
                if (root == nullptr) {
                    parent->totalInProgress.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                workerGame.syncFrom(parent->initialGame_);
            }

            searchPath.push_back(root);
            ThreadSafeNode *leaf = parent->select(root, workerGame, searchPath);

            EvaluationRequest request;
            request.node = leaf;
            request.gameState = workerGame;
            request.searchPath = searchPath;
            if (!parent->evaluationQueue_->tryPush(request)) {
                // Queue full — return the slot so it can be retried
                parent->totalInProgress.fetch_sub(1, std::memory_order_relaxed);
            }
        } else {
            parent->totalInProgress.fetch_sub(1, std::memory_order_relaxed);
        }

        // PHASE 2: Pull one backprop result and process it.
        // Each worker grabs a single item so all workers can backprop concurrently
        // instead of one worker draining the whole queue (convoy effect).
        auto result = parent->backpropagationQueue_->tryPop();
        if (result.has_value()) {
            parent->expand(result->node, result->gameState, result->policy);
            parent->backpropagate(result->node, result->value, result->searchPath);
            parent->totalIterations.fetch_add(1, std::memory_order_relaxed);
        }

        if (parent->totalIterations.load(std::memory_order_relaxed) >= maxIterations)
            break;

        std::this_thread::yield();
    }

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
    prepareRoot(game);

    evalPool_->start();
    workerPool_->start();

    bool completed = workerPool_->waitForCompletion();
    if (!completed)
        std::cerr << "Warning: parallel MCTS search timed out\n";

    workerPool_->stop();
    evalPool_->stop();

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

    root_->moves = arena_->allocate<PenteGame::Move>(capacity);
    root_->priors = arena_->allocate<float>(capacity);

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

    {
        std::lock_guard<std::mutex> arenaLock(arenaMutex_);
        initNodeChildren(node, capacity);
        node->moves  = arena_->allocate<PenteGame::Move>(capacity);
        node->priors = arena_->allocate<float>(capacity);
    }

    for (int i = 0; i < capacity; ++i) {
        node->moves[i]  = policy[i].first;
        node->priors[i] = policy[i].second;
    }

    assert(capacity >= 0 && "policy capacity must be non-negative");
    node->childCount = static_cast<uint16_t>(capacity);
    if (game.isGameOver()) {
        node->value = (game.getWinner() == game.getCurrentPlayer()) ? 1.0f : -1.0f;
        node->solvedStatus = (game.getWinner() == game.getCurrentPlayer())
                                 ? SolvedStatus::SOLVED_WIN : SolvedStatus::SOLVED_LOSS;
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

    // Evaluate expanded children using PUCT with virtual loss
    for (int i = 0; i < cap; ++i) {
        if (node->children[i] == nullptr) continue;

        ThreadSafeNode *child = node->children[i];
        int32_t effectiveVisits = virtualLossManager_->getEffectiveVisits(child);

        // Exploitation term: average value
        double exploitation = (effectiveVisits == 0) ? 0.0 : 
            static_cast<double>(child->totalValue.load(std::memory_order_relaxed)) / effectiveVisits;

        // Exploration term: prior * sqrt(parent_visits) / (1 + effective_visits)
        double exploration = explorationFactor * node->priors[i] * sqrtParentVisits / (1.0 + effectiveVisits);

        double score = exploitation + exploration;

        if (score > bestValue) {
            bestValue = score;
            bestIndex = i;
        }
    }

    // If no expanded children, select first unexpanded child
    if (bestIndex == -1) {
        for (int i = 0; i < cap; ++i) {
            if (node->children[i] == nullptr) {
                bestIndex = i;
                break;
            }
        }
    }

    return bestIndex;
}

ParallelMCTS::ThreadSafeNode *ParallelMCTS::allocateNode() {
    std::lock_guard<std::mutex> lock(arenaMutex_);
    void *mem = arena_->allocate<ThreadSafeNode>();
    if (!mem) return nullptr;
    auto *node = new (mem) ThreadSafeNode();  // placement new: properly constructs mutex and atomics
    nodeCount.fetch_add(1, std::memory_order_relaxed);
    return node;
}

void ParallelMCTS::initNodeChildren(ThreadSafeNode *node, int capacity) {
    if (capacity <= 0) {
        node->children = nullptr;
        node->childCapacity = 0;
        return;
    }

    node->children = arena_->allocate<ThreadSafeNode *>(capacity);
    if (!node->children) throw std::bad_alloc();
    std::memset(node->children, 0, sizeof(ThreadSafeNode *) * capacity);
    node->childCapacity = static_cast<uint16_t>(capacity);
    node->childCount = 0;
}
