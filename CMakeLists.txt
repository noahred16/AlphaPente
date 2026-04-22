cmake_minimum_required(VERSION 3.10)
project(AlphaPente)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
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
    src/Evaluator.cpp
    src/Zobrist.cpp
    src/NNInputEncoder.cpp
)

# Optional: LibTorch neural network support.
# To enable, download LibTorch from https://pytorch.org and configure with:
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "LibTorch found — building with neural network support")
    target_sources(pente_core PRIVATE src/NNModel.cpp src/NeuralNetEvaluator.cpp)
    target_link_libraries(pente_core "${TORCH_LIBRARIES}")
    target_include_directories(pente_core PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(pente_core PUBLIC WITH_TORCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "LibTorch not found — neural network support disabled")
endif()

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
    tests/NNInputEncoderTests.cpp
)
target_link_libraries(unit_tests pente_core)
