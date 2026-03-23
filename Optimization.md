 I've made some significant improvements in the mcts implementation. A big one was converting it to a graph from a tree so that nodes can be reused if they get reached from difference paths. But I'm seeing that the sims/sec have dropped. At the GraphMCTS tag I'm seeing
around 20k per sec whereas in my previous checkpoint, I was seeing over 40k per sec. Please compare these two tags and do an analysis of the changes as well as what the major slow downs may be. Primarily, I remember there was a big slowdown when I started considering all
 moves. I did this so that the index management could be a bit easier. I was seeing that the way I was tracking promising/legal moves previously was flawed for a graph implementation because of the way that I was handling captures. In some cases the captures were causing
 the number of legal moves to be different because of how I was doing the undos for diagonal captures. See the "canon moves" seciont of the README for more details on that.


● Now I have a complete picture. Here's the full analysis:

  ---
  Performance Analysis: ClassicMCTS → GraphMCTS

  Summary of the 2× slowdown

  The ~20k→40k sims/sec regression comes from 4 compounding changes, with one dominant culprit:

  ---
  1. Selection branching factor: O(visited) → O(all_moves) [biggest impact]

  ClassicMCTS selectBestChild only scans childCount — the number of children already visited. Early in search that's a handful; it grows slowly as the tree fills.

  GraphMCTS selectBestMoveIndex scans every slot from 0..nextPriorIdx plus a frontier probe. As a node gets more visits, nextPriorIdx approaches childCapacity — which is 200–300 moves on an open board. Every select() call at every node in the path does this.

  For a tree depth of 5, that's 5 × O(200) = ~1000 comparisons per sim just for selection, vs ~5 × O(5–20) in ClassicMCTS.

  This is the change you identified: switching to all legal moves for the graph correctness fix.

  ---
  2. Expand cost: 1 child → allocate all ~200+ slots

  ClassicMCTS expand: create one Node*, store one move. Done.

  GraphMCTS expand per node (happens once per unique node):
  int childCapacity = game.getLegalMoves().size();  // e.g. 280
  initNodeChildren(node, childCapacity);             // allocate Node**[280]
  node->moves = arena_.allocate<Move>(childCapacity);   // 280 Moves
  node->priors = arena_.allocate<float>(childCapacity); // 280 floats
  std::fill(node->priors, node->priors + childCapacity, -1.0f); // zero 280 floats
  Three arena allocations + a memset-equivalent over ~280 floats, every time a node is first visited.

  ---
  3. Lazy policy load in selectBestMoveIndex [hidden cost]

  When node->priors[0] < 0.0f (the -1 sentinel, set in expand), the first call to selectBestMoveIndex triggers:
  evaluatePolicy(game)  // heuristic-scores ALL legal moves → nth_element + partial sort
  This is a full evaluation over ~280 moves involving game.evaluateMove() per move, happening the first time each node is selected after being expanded. This overlaps with the per-sim rollout cost, so it's not additive — but it means the first selection through a node is
  significantly heavier than ClassicMCTS's equivalent.

  ---
  4. Canonical hash overhead in the hot path

  Inside the select() loop, at every already-expanded canonical node:
  game.getCanonicalHash(currentSym);  // 8× Zobrist hash computations
  This runs at every node in the path per iteration. Plus the unordered_map::find() lookup via nodeTranspositionTable whenever needsTTLookup is true. Both are cache-unfriendly.

  ---
  What did NOT meaningfully change

  - syncFrom cost is the same in both (full board copy per iteration)
  - Rollouts still happen in GraphMCTS — evaluateValue calls rollout() when evaluatePosition() returns 0
  - getLegalMoves() itself is O(1) in both (it just returns legalMovesVector)

  ---
  The capture/graph index problem you described

  The root cause was that diagonal capture undos could restore cells that weren't in the promising-moves list (or were at wrong indices), so the untriedMoves pool would be corrupted when the same position was reached via two different paths with different capture
  sequences. This forced switching to all-legal-moves as the canonical move set, which is correct but expensive.

  ---
  Potential fixes to recover throughput

  Option A — Cap branching factor in selection (easiest win)
  Don't scan all childCapacity slots. Track the actual number of "considered" moves separately from childCapacity. Sort by prior once during expand (instead of lazy nth_element on every lazy load), and only consider the top-K in PUCT. K=50 would cut selection cost by ~5×.

  Option B — Don't pre-allocate all slots upfront
  Instead of initNodeChildren(node, childCapacity) allocating Node**[280], keep the full moves[]+priors[] for policy ordering, but allocate children lazily with a flat childCount like ClassicMCTS did. This saves the Node** array pressure and turns selectBestMoveIndex's
  scan into just the "next to try" frontier logic.

  Option C — Sort priors at expand time, not lazy
  Currently policy is loaded lazily in selectBestMoveIndex. If you sort moves[]/priors[] by prior descending once in expand(), the frontier logic becomes trivially O(1): the best unexplored move is always moves[nextPriorIdx], and the PUCT loop only needs to scan
  nextPriorIdx visited children. This doesn't reduce worst-case scan but removes the nth_element + partial sort from the hot path.

  Option D — Re-examine the capture undo issue
  The real fix would be re-examining whether promising moves can be made stable across different graph paths. The issue is that captures restore cells, but the promising-moves set is path-dependent. If captures always restore cells that are then added to the promising set
   (rather than removed), you might be able to maintain a graph-safe promising set. But this is more invasive to the game logic.

  The lowest-effort, highest-impact fix is Option C + a branching factor cap — sort priors once on expand, scan only visited children + one frontier candidate in PUCT, and cap K to something like 30–50 moves.