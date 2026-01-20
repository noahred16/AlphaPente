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

# Executable: Compete
add_executable(compete apps/Compete.cpp)
target_link_libraries(compete pente_core)
