# Pente AlphaZero Implementation
C++ implementation of AlphaZero for the game of Pente.

## TODOS
- Review Virtual loss/mean for parallezation of search
[x] Add logic for solved wins/losses, Minimax Backprop
- Enable other game modes, gomoku...
- Refactor that time constraint concept
- Review early solved stopping
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

