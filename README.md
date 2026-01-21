# Pente AlphaZero Implementation
C++ implementation of AlphaZero for the game of Pente.

## TODOS
- Review Virtual loss/mean for parallezation of search
[x] Add logic for solved wins/losses, Minimax Backprop
- Enable other game modes, gomoku...
- Refactor that time constraint concept
- Review early solved stopping

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
```

## PenteGame
Pente is an amusing two-player strategy board game where the objective is to get five of your pieces in a row or capture five pairs of your opponent's pieces. The game is played on a 19x19 grid.

