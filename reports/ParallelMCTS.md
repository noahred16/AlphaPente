# Parallel MCTS — Design Report

## Overview

This document describes the design and implementation of `ParallelMCTS`, a multi-threaded Monte Carlo Tree Search built for AlphaPente. The architecture is intended to support batched neural network evaluation (GPU/NN), with a CPU heuristic evaluator used in the interim.

---

## Architecture

The system is a producer/consumer pipeline with three distinct thread roles:

```
Worker Threads          Eval Threads          Worker Threads
(select → leaf)  ──→  (evaluate leaf)  ──→  (expand + backprop)
                   eval queue           backprop queue
```

**Worker threads** traverse the shared tree using PUCT selection, apply a virtual loss to the selected leaf, and push an evaluation request to the eval queue. After pushing, they poll the backprop queue and process any available results (expand + backprop).

**Eval threads** pop batches from the eval queue, call the evaluator (heuristic or NN) to get a policy and value, and push results to the backprop queue.

This design mirrors the AlphaGo/AlphaZero pipeline where GPU evaluation is the bottleneck and multiple CPU workers keep it fed.

---

## Key Data Structures

### `ThreadSafeNode`

Each tree node stores:
- **Immutable after creation**: move, player, position hash, child capacity, moves array, priors array, children pointer array
- **Mutable statistics**: `atomic<int32_t> visits`, `atomic<double> totalValue`, `atomic<int32_t> virtualLosses`
- **Expansion state**: `atomic<bool> expanded`, `atomic<bool> evaluated`, `SolvedStatus solvedStatus`
- **Synchronization**: `std::mutex nodeSubtreeLock` — protects child slot allocation and expansion

Policy (moves + priors) is stored on the **parent** node as edges, not on child nodes. Child `ThreadSafeNode` objects are allocated lazily on first visit.

### `EvaluationQueue` / `BackpropagationQueue`

Two bounded `std::deque`-backed queues protected by mutexes:
- `EvaluationQueue`: worker → eval thread. Bounded capacity; `tryPush` returns false when full (worker returns the iteration slot to prevent livelock).
- `BackpropagationQueue`: eval thread → worker. Unbounded; workers drain with `popAll()`.

### Arena Allocator

All tree nodes and their arrays (children, moves, priors) are allocated from a single pre-allocated `Arena`. Allocation is O(1) bump-pointer. Destruction is O(1) reset. A global `arenaMutex_` serializes all allocations across threads.

---

## Thread Safety Design

### Virtual Loss

Virtual loss deters concurrent workers from selecting the same path. When a worker selects a node, it increments `virtualLosses` atomically. This inflates the effective visit count used in PUCT scoring, making the path look less attractive to other workers. Virtual loss is removed during backpropagation.

PUCT effective visits: `effectiveVisits = visits + virtualLosses`

### Double-Checked Locking for Child Allocation

Child node slots are allocated lazily in `select()` using double-checked locking:

```cpp
ThreadSafeNode *child = node->children[bestIndex];  // fast path (no lock)
if (child == nullptr) {
    std::lock_guard lock(node->nodeSubtreeLock);
    child = node->children[bestIndex];              // re-read under lock
    if (child == nullptr) {
        child = allocateNode();
        node->children[bestIndex] = child;
    }
}
```

### Node Expansion

`expand()` uses the same per-node lock with a double-check on `expanded` to prevent redundant expansion by concurrent workers:

```cpp
if (node->expanded.load(acquire)) return;           // fast path
std::lock_guard lock(node->nodeSubtreeLock);
if (node->expanded.load(relaxed)) return;           // double-check
// ... allocate and write policy ...
node->expanded.store(true, release);
```

### Lock Hierarchy

To prevent deadlock, lock acquisition always follows this order:
1. `nodeSubtreeLock` (per-node)
2. `arenaMutex_` (global)

---

## Iteration Accounting

Shared atomic counters replace per-worker counters to correctly track completion across threads:

- `totalInProgress`: atomically claimed iteration slots. Workers claim a slot before selecting; if `tryPush` fails (queue full), the slot is returned.
- `totalIterations`: incremented once per completed backpropagation. Workers exit when this reaches `maxIterations`.

Workers use `completedWorkers` + a `std::condition_variable` to signal the main thread when all workers have exited naturally, avoiding a busy-wait in `search()`.

---

## Root Preparation

`prepareRoot()` runs synchronously before any threads start. It evaluates the root position and allocates its policy (moves + priors) eagerly. This follows the AlphaGo pattern: workers always start with an expanded root, avoiding a race on first-visit expansion.

Child `ThreadSafeNode` objects under the root are still allocated lazily during selection.

---

## Terminal Node Handling

When a node is expanded with an empty policy (game over), `solvedStatus` is set to `SOLVED_WIN` or `SOLVED_LOSS`. This causes `isTerminal()` to return true, halting `select()` before it attempts to descend into a zero-capacity node — which would otherwise produce an out-of-bounds read on the moves array.

---

## Bugs Found and Fixed

### 1. `initNodeChildren` arena race (data corruption)
`initNodeChildren` called `arena_->allocate` without holding `arenaMutex_`, while `moves`/`priors` were allocated with it. Concurrent expansions on different nodes could corrupt the arena offset. Fixed by moving `initNodeChildren` inside the `arenaMutex_` scope in `expand`.

### 2. Terminal node crash (heap buffer overflow)
Nodes with no legal moves were marked `expanded = true` but `solvedStatus` was left as `UNSOLVED`. `select()` continued descending into them, `selectBestMoveIndex` returned -1 (no children), and `node->moves[-1]` was read — 2 bytes before the moves array. Detected with AddressSanitizer. Fixed by setting `solvedStatus` in `expand` for game-over positions.

### 3. Iteration slot livelock
When `tryPush` failed (eval queue full), `totalInProgress` was left incremented. With enough workers and a slow eval thread, all slots could be claimed but never completed, hanging indefinitely. Fixed by decrementing `totalInProgress` on push failure so the slot can be retried.

### 4. Placement new for arena nodes
`Arena::allocate<T>` casts raw memory without calling constructors. `std::mutex` inside `ThreadSafeNode` requires proper construction. Fixed by using placement new: `new (mem) ThreadSafeNode()`.

---

## Benchmark Results (Release build, blank board, 10k iterations)

| Config | iters/sec | vs single-thread |
|--------|-----------|-----------------|
| Single-thread MCTS | ~12–14k | 1.00x |
| 1w / 1e | ~8–9k | ~0.7x |
| 2w / 1e | ~11–13k | ~1.0x |
| 4w / 2e | ~14–16k | ~1.2x |
| 5w / 3e | ~25–26k | ~2.0x |
| 6w / 2e | ~22–33k | ~1.8–2.5x |

Results are variable at 10k iterations due to OS scheduling noise. At 100k+ iterations throughput stabilizes but decreases as tree depth grows (longer selection paths per iteration).

**Key finding**: 1 eval thread is always the bottleneck when paired with 2+ workers. The sweet spot with the CPU heuristic evaluator is roughly 5 workers / 3 eval threads.

---

## Integration

`ParallelMCTS` is available in the `pente` app via the `-p` flag (requires `-n`):

```bash
./pente -n -p "" 100000          # blank board, 100k iterations
./pente -n -p "1. K10 L9" 50000  # mid-game position
./pente -p ""                    # error: -p requires -n
```

Default thread config: 5 workers, 3 eval threads, 1 GB arena.

---

## Future Work

The current implementation has two hidden serialization points that prevent the parallel version from outperforming single-threaded MCTS at typical iteration counts with a CPU evaluator:

**1. Global arena mutex (`arenaMutex_`) serializes all expansion**

Every call to `expand()` — which writes children pointers, moves, and priors — acquires a single global mutex and holds it for the full allocation. With 5 workers, expansions queue up and execute one at a time. This negates the parallelism benefit of running multiple workers.

*Fix: per-thread arena slabs.* Pre-divide the arena into N regions at startup, one per worker thread. Each worker allocates from its own region with no locking. Pointer bumps within a slab are thread-local and require zero synchronization.

**2. `popAll()` serializes backpropagation**

One worker drains the entire backprop queue and processes every result sequentially. Other workers spin-yield until that worker finishes. Backprop is effectively single-threaded even with multiple workers running.

*Fix: pop one result per worker per loop iteration.* Each worker should pop a single result, expand and backprop it, then loop. This distributes backprop work across all workers proportionally.

These two fixes together would allow tree traversal, expansion, and backpropagation to all run in true parallel — which should yield meaningful CPU speedup independent of the evaluator. The current architecture is already well-suited for GPU/NN evaluation (where eval latency is long enough to amortize all other overheads), and these fixes would make it competitive for CPU-heuristic search as well.
