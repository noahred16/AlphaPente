# Pente AlphaZero Implementation
C++ implementation of AlphaZero for the game of Pente.

## TODOS
- Review Virtual loss/mean for parallezation of search
[x] Add logic for solved wins/losses, Minimax Backprop
- Enable other game modes, gomoku...
- Refactor that time constraint concept
- Review early solved stopping
- store first as canonical instead of min as canonical


```
/*
WORKER THREAD PSEUDO-CODE
# we'll need to keep track of the ones in progress as well so that we don't over shoot the number of iterations
evaluationQueue = []
backpropagationQueue = []

numOfIterations = 1000
n_in_progress = 0
n_complete = 0
while n_in_progress < numOfIterations or n_complete < numOfIterations:
    if n_in_progress < numOfIterations: 
        n_in_progress += 1
        v = select(root)
        evaluationQueue.push(v)
    
    
    if backpropagationQueue.size() > 0
        v, p = backpropagationQueue.pop()
        value = v
        policy = p
        expand(v, policy)
        backpropagate(value)
        n_complete += 1


EVALUATION THREAD PSEUDO-CODE
batch_size = 1
while true:
    batch = []
    for i in range(batch_size):
        if evaluationQueue.size() > 0:
            batch.push(evaluationQueue.pop())
    
    if batch.size() > 0:
        values, policies = evaluate(batch)
        for v, p in zip(values, policies):
            backpropagationQueue.push((v, p))
*/
```


```
./compete "1  K10     J8
2       K7      L11
3       K11     K9
4       H7      K12
5       J7      G7
6       L10     J13
7       M10     H14
8       G15     K10
9       N10     M12
10      J9      K11
11      K9      J8
12      L7      M7
13      K8      K13
14      K14     N13
15      O14     M13"

10:38:33 noah-local build (cpp-simple *)$ ./compete "1  K10     L9
2       K14     N11
3       K12     M10
4       K11     K13
5       K8      K9
6       M9      J9
7       H13     J12
8       H11     L14
9       H12     H14
10      H9" 300000
Playing Pente...
Parsed moves:
1 K10 L9 2 K14 N11 3 K12 M10 4 K11 K13 5 K8 K9 6 M9 J9 7 H13 J12 8 H11 L14 9 H12 H14 10 H9
   A B C D E F G H J K L M N O P Q R S T
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · ·             · · · · · · · 15
14 · · · · · ·   ●   ○ ●   · · · · · · · 14
13 · · · · · ·   ○   ●     · · · · · · · 13
12 · · · · · ·   ○ ● ○         · · · · · 12
11 · · · · · ·   ○   ○     ●   · · · · · 11
10 · · · · · ·       ○   ●     · · · · · 10
 9 · · · · · ·   ○ ● ● ● ○   · · · · · · 9
 8 · · · · · ·       ○       · · · · · · 8
 7 · · · · · · · ·       · · · · · · · · 7
 6 · · · · · · · · · · · · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T
0 Black ○, 0 White ●
Current player: White
Segmentation fault (core dumped)


./compete "1. K10 L9 2. K14 N11 3. K12 M10 4. K11 K13 5. K8 K9 6. M9 J9 7. H13 J12 8. H11 L14 9. H12 H14 10. H9"

```

## Project Structure
```
AlphaPente/
├── include/           # Header files (.h or .hpp)
│   ├── BitBoard.hpp
│   ├── PenteGame.hpp
│   └── MCTS.hpp
├── src/              # Implementation files (.cpp)
│   ├── BitBoard.cpp
│   ├── PenteGame.cpp
│   └── MCTS.cpp
├── apps/             # Main programs/executables
│   ├── Compete.cpp
│   ├── Train.cpp
│   ├── Play.cpp
│   └── Test.cpp
├── build/            # Build output (created by build system)
├── CMakeLists.txt    # Build configuration
└── README.md
```

## Build Instructions

### Initial Setup
```bash
mkdir build && cd build
cmake ..
make
```

### Rebuild After Changes
```bash
cd build
make
```

### Clean Rebuild
```bash
cd build
rm -rf *
cmake ..
make
```

## Running the Executables

From the `build/` directory:

```bash
./train   # Training AlphaPente
./play    # Playing Pente
./test    # Testing AlphaPente

gdb -ex run -ex bt -ex quit --args ./pente "1  K10 ..."
```

## Unit Tests

Run all unit tests:
```bash
cd build
make unit_tests && ./unit_tests
```

Useful flags:
```bash
./unit_tests -s              # Show successful assertions (verbose)
./unit_tests -tc="BitBoard*" # Run only tests matching pattern
./unit_tests -sf="*PenteGame*" # Run only tests from matching files
./unit_tests -ltc            # List all test cases
```

See `tests/README.md` for more details.

## PenteGame
Pente is an amusing two-player strategy board game where the objective is to get five of your pieces in a row or capture five pairs of your opponent's pieces. The game is played on a 19x19 grid.




## canon moves
Good news — the existing code is well-set-up for this. computeCanonicalHash already computes all 8 hashes in a single pass using precomputed symStoneKeys.
  It just doesn't return which sym won. And symMap[sym][cell] is already there for coordinate transformation.

  Here's the complete plan:

  The core idea: store all moves in nodes in canonical coordinates. In expand() apply the transform before storing. In select() apply the inverse before
  playing. Since transform=0 (identity) when moveCount > 10, it degrades gracefully with no-ops.

  ---
  1. Zobrist — minor additions

  - computeCanonicalHashWithTransform — same loop as computeCanonicalHash, also tracks which s produced the min. Returns pair<uint64_t, int>.
  - applySymToCell(sym, x, y) — public, uses existing symMap[sym][cell] to transform coordinates.
  - static constexpr int symInverse[8] = {0, 3, 2, 1, 4, 5, 6, 7} — precomputed inverse table.

  2. PenteGame — one new method

  - getHashAndTransform() -> pair<uint64_t, int> — for moveCount <= 10 calls computeCanonicalHashWithTransform, else returns {hash_, 0}.
  - getHash() becomes return getHashAndTransform().first.

  3. MCTS::expand() — transform moves before storing

  auto [hash, transform] = game.getHashAndTransform();
  // ...
  for (int i = 0; i < policyCount; i++) {
      auto [cx, cy] = zob.applySymToCell(transform, policy[i].first.x, policy[i].first.y);
      node->moves[i] = Move(cx, cy);  // canonical coords
  }

  4. MCTS::select() — inverse-transform before makeMove

  auto [hash, transform] = game.getHashAndTransform();
  int best = selectBestMoveIndex(node, game, transform);
  auto [ax, ay] = zob.applySymToCell(Zobrist::symInverse[transform], node->moves[best].x, node->moves[best].y);
  game.makeMove(ax, ay);

  5. MCTS::selectBestMoveIndex() — add transform param for lazy policy load

  Signature becomes selectBestMoveIndex(Node*, const PenteGame&, int transform). When lazy-loading priors, apply transform before storing moves.

  6. MCTS display (printBestMoves etc.) — store rootTransform_

  Add int rootTransform_ to MCTS, set it in search(). Display functions inverse-transform moves from canonical back to actual board coordinates before
  printing.

  ---
  The only real risk: symInverse needs to be verified against the exact applySym definitions in Zobrist. Based on what's there:
  - sym1 (x,y)→(y, n-x) and sym3 (x,y)→(n-y, x) are inverses ✓
  - sym0, sym2, sym4, sym5, sym6, sym7 are all self-inverse ✓

  Want me to go ahead and implement this?

## the problem
     A B C D E F G H J K L M N O P Q R S T 
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · · · · · · · · · · · · · · · 15
14 · · · · · · · · · · · · · · · · · · · 14
13 · · · · · · · · · · · · · · · · · · · 13
12 · · · · · · · · · · · · · · · · · · · 12
11 · · · · · · · ·       · · · · · · · · 11
10 · · · · · ·       ○   · · · · · · · · 10
 9 · · · · · ·   ●   ●   · · · · · · · · 9
 8 · · · · · ·     ● ○   · · · · · · · · 8
 7 · · · · · · ·     ○   · · · · · · · · 7
 6 · · · · · · · ·       · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T 
0/10 Black ○, 0/10 White ●, Current player: Black
Selected move: (H9)  Game hash: 18359040013536140658

   A B C D E F G H J K L M N O P Q R S T 
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · · · · · · · · · · · · · · · 15
14 · · · · · · · · · · · · · · · · · · · 14
13 · · · · · · · · · · · · · · · · · · · 13
12 · · · · · · · · · · · · · · · · · · · 12
11 · · · · ·             · · · · · · · · 11
10 · · · · ·   ○     ○   · · · · · · · · 10
 9 · · · · ·         ●   · · · · · · · · 9
 8 · · · · · ·       ○   · · · · · · · · 8
 7 · · · · · · ·     ○   · · · · · · · · 7
 6 · · · · · · · ·       · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T 
2/10 Black ○, 0/10 White ●, Current player: White
Selected move: (G10)  Game hash: 5452371368974874235


SCENARIO TWO:

   A B C D E F G H J K L M N O P Q R S T 
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · · · · · · · · · · · · · · · 15
14 · · · · · · · · · · · · · · · · · · · 14
13 · · · · · · · · · · · · · · · · · · · 13
12 · · · · · · · · · · · · · · · · · · · 12
11 · · · · · ·           · · · · · · · · 11
10 · · · · · ·   ● ● ○   · · · · · · · · 10
 9 · · · · · ·       ●   · · · · · · · · 9
 8 · · · · · · · ·   ○   · · · · · · · · 8
 7 · · · · · · · ·   ○   · · · · · · · · 7
 6 · · · · · · · ·       · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T 
2/10 Black ○, 0/10 White ●, Current player: Black


   A B C D E F G H J K L M N O P Q R S T 
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · · · · · · · · · · · · · · · 15
14 · · · · · · · · · · · · · · · · · · · 14
13 · · · · · · · · · · · · · · · · · · · 13
12 · · · · · · · · · · · · · · · · · · · 12
11 · · · · ·             · · · · · · · · 11
10 · · · · ·   ○     ○   · · · · · · · · 10
 9 · · · · ·         ●   · · · · · · · · 9
 8 · · · · · · · ·   ○   · · · · · · · · 8
 7 · · · · · · · ·   ○   · · · · · · · · 7
 6 · · · · · · · ·       · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T 
2/10 Black ○, 0/10 White ●, Current player: White
Selected move: (G10)  Game hash: 5452371368974874235

FATAL ERROR: Policy size does not match node's child capacity during lazy policy load.
Policy size: 22, Child capacity: 25
   A B C D E F G H J K L M N O P Q R S T 
19 · · · · · · · · · · · · · · · · · · · 19
18 · · · · · · · · · · · · · · · · · · · 18
17 · · · · · · · · · · · · · · · · · · · 17
16 · · · · · · · · · · · · · · · · · · · 16
15 · · · · · · · · · · · · · · · · · · · 15
14 · · · · · · · · · · · · · · · · · · · 14
13 · · · · · · · · · · · · · · · · · · · 13
12 · · · · · · · · · · · · · · · · · · · 12
11 · · · · ·             · · · · · · · · 11
10 · · · · ·   ○     ○   · · · · · · · · 10
 9 · · · · ·         ●   · · · · · · · · 9
 8 · · · · · · · ·   ○   · · · · · · · · 8
 7 · · · · · · · ·   ○   · · · · · · · · 7
 6 · · · · · · · ·       · · · · · · · · 6
 5 · · · · · · · · · · · · · · · · · · · 5
 4 · · · · · · · · · · · · · · · · · · · 4
 3 · · · · · · · · · · · · · · · · · · · 3
 2 · · · · · · · · · · · · · · · · · · · 2
 1 · · · · · · · · · · · · · · · · · · · 1
   A B C D E F G H J K L M N O P Q R S T 
2/10 Black ○, 0/10 White ●, Current player: White
test: /home/noah-cabem/repos/AlphaPente/src/MCTS.cpp:522: int MCTS::selectBestMoveIndex(Node*, const PenteGame&) const: Assertion `priorsSize == childCapSize' failed.



SOLUTION: https://chatgpt.com/c/699a7288-6f8c-8332-9f4f-0e9ff9acb3d2
1️⃣ Lazy expansion
2️⃣ Implicit progressive widening
3️⃣ PUCT-driven widening

use all legal moves along with a sort policy prior. 
only have to check up to the first unexpanded node since any further would have a lower PUCT score. 
but I guess that policy prior would need to be calculated for all moves. 
in heuristic, we can just default all to 0. 
and even faster, we can use a kmax for selection sort to speed up the sorting. 
(std::partial_sort) where kmax can be the policy_size! neat.

need vec for expanded_children an done vec for unexpanded_children