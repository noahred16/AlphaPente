#include "ParallelMCTS.hpp"
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

bool ParallelMCTS::BackpropagationQueue::empty() const {
    std::lock_guard<std::mutex> lock(queueLock);
    return queue.empty();
}

// ============================================================================
// WorkerPool Implementation
// ============================================================================

ParallelMCTS::WorkerPool::WorkerPool(int numWorkers, ParallelMCTS *parent)
    : numWorkers(numWorkers), parent(parent) {}

ParallelMCTS::WorkerPool::~WorkerPool() {
    stop();
}

void ParallelMCTS::WorkerPool::start() {
    std::lock_guard<std::mutex> lock(poolLock);
    if (running) return;

    running = true;
    threads.clear();
    for (int i = 0; i < numWorkers; ++i) {
        threads.emplace_back(&WorkerPool::workerThreadMain, this, i);
    }
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
    // Each worker has its own game state for thread-safe move simulation
    // All workers share the same arena and tree for collaborative building
    PenteGame workerGame;
    std::mt19937 workerRng(parent->config_.seed + workerId);

    int n_in_progress = 0;
    int n_complete = 0;
    int numIterations = parent->config_.maxIterations;

    while (running && (n_in_progress < numIterations || n_complete < numIterations)) {
        // PHASE 1: Selection and Enqueueing
        if (n_in_progress < numIterations) {
            // Select a node from root down the tree
            std::vector<ThreadSafeNode *> searchPath;
            
            {
                std::lock_guard<std::mutex> treeLock(parent->treeLock);
                if (parent->root_ == nullptr) {
                    continue;  // Tree not yet initialized
                }

                // Reset worker game to initial state
                workerGame.syncFrom(parent->initialGame_);

                // Traverse tree to find leaf node
                ThreadSafeNode *leaf = parent->select(parent->root_, workerGame, searchPath);

                // Push leaf node to evaluation queue
                EvaluationRequest request;
                request.node = leaf;
                request.gameState = workerGame;
                request.searchPath = searchPath;
                parent->evaluationQueue_->tryPush(request);
            }

            n_in_progress++;
        }

        // PHASE 2: Backpropagation
        if (!parent->backpropagationQueue_->empty()) {
            auto results = parent->backpropagationQueue_->popAll();
            for (const auto &result : results) {
                // TODO: Implement expand phase with policy from evaluation
                // parent->expand(result.node, result.policy);

                // TODO: Implement backpropagate phase with value
                // parent->backpropagate(result.node, result.value, result.searchPath);

                // TODO: Remove virtual loss
                // parent->virtualLossManager_->removeVirtualLoss(result.node);

                n_complete++;
            }
        }

        // Small yield to prevent busy-waiting
        std::this_thread::yield();
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

    // Initialize worker pool
    workerPool_ = std::make_unique<WorkerPool>(config_.numWorkers, this);

    // Seed RNG
    if (config_.seed != 0) {
        rng_.seed(config_.seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

ParallelMCTS::~ParallelMCTS() {
    if (workerPool_) {
        workerPool_->stop();
    }
}

PenteGame::Move ParallelMCTS::search(const PenteGame &game) {
    std::cout << "Starting Parallel MCTS search with " << config_.numWorkers << " workers..." << std::endl;

    // Store initial game state for workers
    initialGame_ = game;

    // Initialize tree with root node
    {
        std::lock_guard<std::mutex> lock(treeLock);
        if (root_ == nullptr) {
            root_ = allocateNode();
            root_->move = {-1, -1};  // Invalid move for root
            root_->player = PenteGame::Player::NONE;
        }
    }

    // Start worker pool
    workerPool_->start();

    // TODO: Implement parallel search orchestration
    // For now, let workers run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Stop workers
    workerPool_->stop();

    return PenteGame::Move{-1, -1};
}

PenteGame::Move ParallelMCTS::getBestMove() const {
    // TODO: Implement best move selection from root's children
    return PenteGame::Move{-1, -1};
}

void ParallelMCTS::reset() {
    std::lock_guard<std::mutex> lock(treeLock);
    if (arena_) {
        arena_->reset();
    }
    root_ = nullptr;
    nodeTranspositionTable.clear();
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
    // TODO: Implement tree size counting
    return 0;
}

void ParallelMCTS::printStats(double wallTime) const {
    std::cout << "=== Parallel MCTS Stats ===" << std::endl;
    std::cout << "Total Iterations: " << totalIterations.load() << std::endl;
    std::cout << "Root Visits: " << getTotalVisits() << std::endl;
    std::cout << "Wall Time: " << wallTime << "s" << std::endl;
    std::cout << "Evaluation Queue Size: " << evaluationQueue_->size() << std::endl;
}

void ParallelMCTS::setConfig(const Config &config) {
    config_ = config;
}

const ParallelMCTS::Config &ParallelMCTS::getConfig() const {
    return config_;
}

// ============================================================================
// Private Methods (Placeholder implementations)
// ============================================================================

ParallelMCTS::ThreadSafeNode *ParallelMCTS::select(ThreadSafeNode *node, PenteGame &game, std::vector<ThreadSafeNode *> &searchPath) {
    // Traverse tree using PUCT selection until we find a leaf node
    while (node->expanded.load(std::memory_order_acquire) && !node->isTerminal()) {
        // Select best child index using PUCT
        int bestIndex = selectBestMoveIndex(node, game);

        // Get the move
        PenteGame::Move move = node->moves[bestIndex];

        // Make move on game state
        game.makeMove(move.x, move.y);

        // Get child node
        ThreadSafeNode *child = node->children[bestIndex];

        // Add virtual loss to discourage other workers from exploring this path
        virtualLossManager_->addVirtualLoss(child);

        // Move to child
        node = child;
        searchPath.push_back(node);
    }

    return node;
}

ParallelMCTS::ThreadSafeNode *ParallelMCTS::expand(ThreadSafeNode *node, PenteGame &game) {
    // TODO: Implement node expansion with policy
    return node;
}

void ParallelMCTS::backpropagate(ThreadSafeNode *node, float result,
                                const std::vector<ThreadSafeNode *> &searchPath) {
    // TODO: Implement value backpropagation up the tree
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

void ParallelMCTS::updateChildrenPriors(ThreadSafeNode *node, const PenteGame &game) {
    // TODO: Implement prior update from evaluator
}

ParallelMCTS::ThreadSafeNode *ParallelMCTS::allocateNode() {
    return arena_->allocate<ThreadSafeNode>();
}

void ParallelMCTS::initNodeChildren(ThreadSafeNode *node, int capacity) {
    // TODO: Implement child array initialization
}
