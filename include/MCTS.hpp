#ifndef MCTS_HPP
#define MCTS_HPP

#include "PenteGame.hpp"
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <atomic>

// ============================================================================
// Arena Allocator for O(1) Tree Destruction
// ============================================================================

class MCTSArena {
public:
    static constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024; // 256 MB

    explicit MCTSArena(size_t size = DEFAULT_SIZE)
        : size_(size)
        , offset_(0)
        , memory_(nullptr) {
        memory_ = static_cast<char*>(std::aligned_alloc(64, size_)); // 64-byte alignment for cache lines
        if (!memory_) {
            throw std::bad_alloc();
        }
    }

    ~MCTSArena() {
        std::free(memory_);
    }

    // Non-copyable, non-movable
    MCTSArena(const MCTSArena&) = delete;
    MCTSArena& operator=(const MCTSArena&) = delete;

    // Allocate memory for type T with proper alignment
    template<typename T>
    T* allocate(size_t count = 1) {
        // Align to T's alignment requirement
        size_t alignment = alignof(T);
        size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        size_t totalBytes = sizeof(T) * count;

        if (alignedOffset + totalBytes > size_) {
            // Out of arena memory
            return nullptr;
        }

        T* ptr = reinterpret_cast<T*>(memory_ + alignedOffset);
        offset_ = alignedOffset + totalBytes;
        return ptr;
    }

    // O(1) tree destruction - just reset the offset
    void reset() {
        offset_ = 0;
    }

    // Swap internals with another arena (for subtree reuse)
    void swap(MCTSArena& other) {
        std::swap(size_, other.size_);
        std::swap(offset_, other.offset_);
        std::swap(memory_, other.memory_);
    }

    // Statistics
    size_t bytesUsed() const { return offset_; }
    size_t bytesRemaining() const { return size_ - offset_; }
    size_t totalSize() const { return size_; }
    double utilizationPercent() const { return 100.0 * offset_ / size_; }

private:
    size_t size_;
    size_t offset_;
    char* memory_;
};

// ============================================================================
// Transposition Table for MCTS
// ============================================================================

class TranspositionTable {
public:
    // Entry stores aggregated statistics for a board position
    struct Entry {
        uint64_t hash;           // Full hash for verification (handles collisions)
        std::atomic<int32_t> visits;
        std::atomic<int32_t> wins;       // Win count (for tracking)
        std::atomic<int64_t> totalValueFixed;  // Fixed-point total value (scaled by 1000)

        Entry() : hash(0), visits(0), wins(0), totalValueFixed(0) {}

        // Non-atomic getters for reading (caller should handle races)
        double getTotalValue() const {
            return static_cast<double>(totalValueFixed.load(std::memory_order_relaxed)) / 1000.0;
        }
    };

    static constexpr size_t DEFAULT_SIZE = 1 << 20;  // ~1M entries (~32 MB)

    explicit TranspositionTable(size_t numEntries = DEFAULT_SIZE)
        : size_(numEntries)
        , mask_(numEntries - 1)
        , table_(nullptr)
        , hits_(0)
        , misses_(0)
        , stores_(0) {
        // Ensure size is power of 2 for fast modulo
        if ((numEntries & (numEntries - 1)) != 0) {
            // Round up to next power of 2
            size_t n = 1;
            while (n < numEntries) n <<= 1;
            size_ = n;
            mask_ = n - 1;
        }
        table_ = new Entry[size_]();
    }

    ~TranspositionTable() {
        delete[] table_;
    }

    // Non-copyable
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    // Probe the table for a position. Returns nullptr if not found or hash mismatch.
    Entry* probe(uint64_t hash) {
        size_t index = hash & mask_;
        Entry& entry = table_[index];

        // Check if this slot contains our position (compare full hash)
        if (entry.hash == hash && entry.visits.load(std::memory_order_relaxed) > 0) {
            hits_++;
            return &entry;
        }
        misses_++;
        return nullptr;
    }

    // Get or create entry for a hash (always returns valid pointer)
    // Note: In case of collision, we replace the existing entry
    Entry* getOrCreate(uint64_t hash) {
        size_t index = hash & mask_;
        Entry& entry = table_[index];

        // If empty or different position, initialize for new hash
        if (entry.hash != hash) {
            // Replace existing entry (replacement scheme: always replace)
            entry.hash = hash;
            entry.visits.store(0, std::memory_order_relaxed);
            entry.wins.store(0, std::memory_order_relaxed);
            entry.totalValueFixed.store(0, std::memory_order_relaxed);
        }

        return &entry;
    }

    // Update entry with new statistics (thread-safe via atomics)
    void update(uint64_t hash, double value, bool isWin) {
        Entry* entry = getOrCreate(hash);

        // Atomic increments - relaxed ordering is fine for statistics
        entry->visits.fetch_add(1, std::memory_order_relaxed);
        if (isWin) {
            entry->wins.fetch_add(1, std::memory_order_relaxed);
        }
        // Convert value to fixed-point and add
        int64_t valueFixed = static_cast<int64_t>(value * 1000.0);
        entry->totalValueFixed.fetch_add(valueFixed, std::memory_order_relaxed);

        stores_++;
    }

    // Clear all entries
    void clear() {
        for (size_t i = 0; i < size_; i++) {
            table_[i].hash = 0;
            table_[i].visits.store(0, std::memory_order_relaxed);
            table_[i].wins.store(0, std::memory_order_relaxed);
            table_[i].totalValueFixed.store(0, std::memory_order_relaxed);
        }
        hits_ = 0;
        misses_ = 0;
        stores_ = 0;
    }

    // Statistics
    size_t getHits() const { return hits_; }
    size_t getMisses() const { return misses_; }
    size_t getStores() const { return stores_; }
    size_t getSize() const { return size_; }
    double getHitRate() const {
        size_t total = hits_ + misses_;
        return total > 0 ? static_cast<double>(hits_) / total : 0.0;
    }
    size_t getMemoryUsage() const { return size_ * sizeof(Entry); }

private:
    size_t size_;
    size_t mask_;
    Entry* table_;

    // Statistics (not thread-safe, but acceptable for monitoring)
    mutable size_t hits_;
    mutable size_t misses_;
    mutable size_t stores_;
};

// ============================================================================
// MCTS Class with Arena-Allocated Nodes
// ============================================================================

class MCTS {
public:
    // Solved status for minimax backpropagation (1 byte)
    enum class SolvedStatus : uint8_t {
        UNSOLVED = 0,  // Not proven yet
        SOLVED_WIN,    // Proven win for the player who made the move
        SOLVED_LOSS    // Proven loss for the player who made the move
    };

    // Configuration parameters
    struct Config {
        double explorationConstant;    // UCB1 exploration parameter
        int maxIterations = 10000;     // Number of MCTS iterations
        int maxSimulationDepth = 200;  // Max playout depth
        size_t arenaSize = MCTSArena::DEFAULT_SIZE; // Arena size in bytes
        size_t ttSize = TranspositionTable::DEFAULT_SIZE; // TT entries
        bool useTT = true;             // Enable transposition table

        Config() : explorationConstant(std::sqrt(2.0)) {}
    };

    // Node in the MCTS tree - trivially destructible, ~72 bytes
    // All dynamic arrays are arena-allocated via raw pointers
    struct Node {
        // Zobrist hash of position after this move (8 bytes)
        uint64_t positionHash = 0;

        // Move that led to this node (4 bytes)
        PenteGame::Move move;

        // Player who made the move (1 byte, uint8_t-backed enum)
        PenteGame::Player player;

        // Minimax proof status (1 byte)
        SolvedStatus solvedStatus = SolvedStatus::UNSOLVED;

        // Child array metadata (2 bytes each = 4 bytes)
        uint16_t childCount = 0;
        uint16_t childCapacity = 0;

        // Untried moves metadata (2 bytes each = 4 bytes)
        uint16_t untriedMoveCount = 0;
        int16_t unprovenCount = 0;

        // Statistics (16 bytes)
        int32_t visits = 0;
        int32_t wins = 0;
        double totalValue = 0.0;

        // Pointers (24 bytes)
        Node* parent = nullptr;
        Node** children = nullptr;            // Arena-allocated array of child pointers
        PenteGame::Move* untriedMoves = nullptr; // Arena-allocated array of untried moves

        // Total: 8 + 4 + 1 + 1 + 4 + 4 + 16 + 24 = 62 bytes + padding = ~72 bytes

        bool isFullyExpanded() const { return untriedMoveCount == 0; }
        bool isTerminal() const { return childCount == 0 && untriedMoveCount == 0; }
        double getUCB1Value(double explorationFactor) const;
    };

    // Constructor
    explicit MCTS(const Config& config = Config());
    ~MCTS();

    // Main search interface
    PenteGame::Move search(const PenteGame& game);

    // Get best move from current tree (no additional search)
    PenteGame::Move getBestMove() const;

    // Tree management
    void reset();
    void clearTree();
    void reuseSubtree(const PenteGame::Move& move);

    // Statistics and debugging
    int getTotalVisits() const;
    int getTreeSize() const;
    void printStats() const;
    void printBestMoves(int topN = 5) const;
    void printBranch(const char* moveStr, int topN = 5) const;
    void printBranch(int x, int y, int topN = 5) const;

    // Arena statistics
    size_t getArenaUsedBytes() const { return arena_.bytesUsed(); }
    double getArenaUtilization() const { return arena_.utilizationPercent(); }

    // Transposition table statistics
    size_t getTTHits() const { return tt_.getHits(); }
    size_t getTTMisses() const { return tt_.getMisses(); }
    double getTTHitRate() const { return tt_.getHitRate(); }
    void clearTT() { tt_.clear(); }

    // Configuration
    void setConfig(const Config& config);
    const Config& getConfig() const;

private:
    // MCTS phases
    Node* select(Node* node, PenteGame& game);
    Node* expand(Node* node, PenteGame& game);
    double simulate(PenteGame& game);
    void backpropagate(Node* node, double result);

    // Helper methods
    Node* selectBestChild(Node* node) const;
    double evaluateTerminalState(const PenteGame& game, int depth = 0) const;
    PenteGame::Move selectSimulationMove(const PenteGame& game) const;

    // Arena allocation helpers
    Node* allocateNode();
    void initNodeChildren(Node* node, int capacity);
    void initNodeUntriedMoves(Node* node, const std::vector<PenteGame::Move>& moves);

    // Tree reuse helpers (copies subtree to fresh arena)
    Node* copySubtree(Node* source, MCTSArena& destArena);
    void pruneTree(Node* keepNode);

    // Print helpers
    Node* findChildNode(Node* parent, int x, int y) const;
    void printMovesFromNode(Node* node, int topN) const;
    int countNodes(Node* node) const;

    // Member variables
    Config config_;
    MCTSArena arena_;
    TranspositionTable tt_;
    Node* root_ = nullptr;  // Raw pointer into arena
    mutable std::mt19937 rng_;

    // Statistics
    int totalSimulations_ = 0;
    double totalSearchTime_ = 0.0;
};

#endif // MCTS_HPP
