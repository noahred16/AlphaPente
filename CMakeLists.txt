cmake_minimum_required(VERSION 3.10)
project(AlphaPente)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(include)

# Library: Core game logic
add_library(pente_core
    src/BitBoard.cpp
    src/PenteGame.cpp
    src/MCTS.cpp
    src/GameUtils.cpp
)

# Executable: Train
add_executable(train apps/Train.cpp)
target_link_libraries(train pente_core)

# Executable: Play
add_executable(play apps/Play.cpp)
target_link_libraries(play pente_core)

# Executable: Test
add_executable(test apps/Test.cpp)
target_link_libraries(test pente_core)

# Executable: Pente
add_executable(pente apps/Pente.cpp)
target_link_libraries(pente pente_core)

# Executable: Gomoku
add_executable(gomoku apps/Gomoku.cpp)
target_link_libraries(gomoku pente_core)

# Executable: Keryo-Pente
add_executable(keryopente apps/KeryoPente.cpp)
target_link_libraries(keryopente pente_core)

# Unit Tests
add_executable(unit_tests
    tests/BitBoardTests.cpp
    tests/PenteGameTests.cpp
)
target_link_libraries(unit_tests pente_core)
