# Pente AlphaZero Implementation
C++ implementation of AlphaZero for the game of Pente.



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

