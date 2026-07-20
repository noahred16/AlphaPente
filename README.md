# Pente AlphaZero Implementation
C++ implementation of AlphaZero for the game of Pente. Pente is an amusing two-player strategy board game where the objective is to get five of your pieces in a row or capture five pairs of your opponent's pieces. The game is played on a 19x19 grid.







## Setup
The setup script installs torch dependency and detects your system gpu and install the corresponding torch version in the lib dir. 
It also detects the number of threads and free RAM and creates a .env file. You can adjust this to use more or less RAM or CPU threads. 
```bash
cd ~/repos/AlphaPente
bash setup.sh
cat .env  # to see the specs used

# to load model weight from HuggingFace (requires hf auth login)
hf auth login
bash init_models.sh

```




## Python scripts setup (temp)
for `scripts/generate_tests.py` - TODO convert to cpp app
same for `scripts/review.py` - TODO convert
```bash
python -m venv ~/.venvs/py312
source ~/.venvs/py312/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```



## Usage
bootstrap heuristc generation. 
```
./generate -b [-n games] [-s sims] [-g game]
./generate -b -n 4 -s 25000 -g pente
./generate -b -t 10 -n 220 -s 25000 -g pente
./generate -b -t 10 -n 1 -s 250 -g pente 
./generate -b -t 10 -n 4 -s 25000 -g pente -a
```
train on bootstrap data
```
./train -b — train on bootstrap data, all positions
./train -b -p 15 — train on bootstrap data, only positions with 15+ pieces on board
./train — current behavior, trains on buffer.pt
```

## TODOS
- Review Virtual loss/mean for parallezation of search
[x] Add logic for solved wins/losses, Minimax Backprop
- Enable other game modes, gomoku...
- Refactor that time constraint concept
- Review early solved stopping
- store first as canonical instead of min as canonical
- initialize a checkpoints/best_model in order to validate the NNEvaluator
- simple benchmark script to validate training progress. 
- training schedule
- use ethernet cord. 

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
cp .env.example .env   # configure arena memory (edit ARENA_SIZE_GB as needed)
cd build
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

## Training

Training is split into two steps: `generate` (self-play → buffer) and `train` (buffer → model checkpoint).

### Bootstrap (first run)

Fill the replay buffer with heuristic-guided games before the network has learned anything:

```bash
cd build

# Repeat until buffer is large enough (need 5000+ positions to start training)
./generate -n 100 -s 1000 -e heuristic

# Train on the buffer
./train
```

### Normal training loop

Once a checkpoint exists, `generate` automatically switches to NN self-play:

```bash
./generate -n 100 -s 400   # -e auto: uses nn after first checkpoint
./train
```

Or run the loop script to iterate continuously:

```bash
./scripts/train_loop.sh -n 100 -s 400 -i 20   # 20 iterations
```

### Flags

```
generate [-g pente|gomoku|keryopente] [-n games] [-s sims] [-e auto|heuristic|nn]
train    [-g pente|gomoku|keryopente] [-t steps]   # -t overrides gradient steps
```

### Benchmark

Check model quality against the open-3 test suite (results appended to `reports/pente/benchmark.csv`):

```bash
./benchmark         # open-three suite only (uses latest model_iterXXXX.pt)
./benchmark -a      # also runs arena: NN vs Heuristic (20 games, 1000 sims/move)
./benchmark -a -G 40 -S 2000   # more games / higher sims for a stronger signal
```

The arena result is appended to the same CSV with suite name `arena`.
Win rate > 50% (decisive games) means the NN has surpassed the heuristic.

### Reset

Wipe checkpoints and buffer to start fresh:

```bash
cd /path/to/AlphaPente
./scripts/reset_training.sh [-g pente]
```

### Moving checkpoints between machines

`checkpoints/` is gitignored (files are multi-GB, too big for GitHub). When switching to a
new machine, the files worth carrying over are:

- `checkpoints/pente/bootstrap.pt` — the bootstrap replay buffer (multi-GB)
- `checkpoints/pente/best_model.pt` — the current best model checkpoint
- `checkpoints/pente/roster/*.pt`  — promoted historical models used for arena comparisons (if any)

Use `rsync` (resumable, unlike `scp`, which matters for the multi-GB bootstrap file) with your
usual SSH alias (e.g. `vast-rtx3060` from `~/.ssh/config`). Whichever path has the `hostalias:`
prefix is remote; the other is local to wherever you run the command.

**Starting a new session — push saved files onto the fresh instance:**

```bash
ssh vast-rtx3060 "mkdir -p /root/repos/AlphaPente/checkpoints/pente"

rsync -avP --stats -e ssh \
  ./checkpoints/pente/best_model.pt \
  ./checkpoints/pente/bootstrap.pt \
  ./checkpoints/pente/roster \
  vast-rtx3060:/root/repos/AlphaPente/checkpoints/pente/
```

**Retiring an old instance — pull files down before it disappears:**

```bash
rsync -avP --stats -e ssh \
  vast-rtx3060:'/root/repos/AlphaPente/checkpoints/pente/best_model.pt /root/repos/AlphaPente/checkpoints/pente/bootstrap.pt /root/repos/AlphaPente/checkpoints/pente/roster' \
  ./checkpoints/pente/
```

Update the alias's `HostName`/`Port` in `~/.ssh/config` before tearing down the old instance —
once it's gone, so is the SSH endpoint.

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
