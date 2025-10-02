cmake_minimum_required(VERSION 3.15)
project(AlphaPente VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Include directories
include_directories(include)

# Find dependencies
find_package(GTest REQUIRED)

# Core library
add_library(alphapente_core 
    src/core/bitboard.cpp
    src/core/game_state.cpp
    src/core/distance_rings.cpp
    src/core/move_generator.cpp
    src/mcts/mcts_node.cpp
    src/mcts/rollout.cpp
    src/mcts/mcts_engine.cpp
)

# Main executable (start simple)
add_executable(alphapente src/main.cpp)
target_link_libraries(alphapente alphapente_core)

# Debug executable for MCTS tree analysis
add_executable(debug_mcts_tree debug_mcts_tree.cpp)
target_link_libraries(debug_mcts_tree alphapente_core)

# Enable testing
enable_testing()
add_subdirectory(tests)