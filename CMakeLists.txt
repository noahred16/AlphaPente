cmake_minimum_required(VERSION 3.15)
project(AlphaPente-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Core library
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
add_library(alphapente_core ${CORE_SOURCES})

# MCTS library
file(GLOB_RECURSE MCTS_SOURCES "src/mcts/*.cpp")
add_library(alphapente_mcts ${MCTS_SOURCES})
target_link_libraries(alphapente_mcts alphapente_core)

# Main executable
add_executable(alphapente src/main.cpp)
target_link_libraries(alphapente alphapente_core alphapente_mcts)

# Tests
file(GLOB_RECURSE TEST_SOURCES "tests/unit/*.cpp" "tests/integration/*.cpp")
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests alphapente_core alphapente_mcts GTest::gtest GTest::gtest_main)

# Benchmarks
file(GLOB_RECURSE BENCHMARK_SOURCES "tests/benchmarks/*.cpp")
add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks alphapente_core alphapente_mcts)

# Compiler flags
target_compile_options(alphapente_core PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(alphapente_mcts PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(alphapente PRIVATE -O3 -march=native -Wall -Wextra)