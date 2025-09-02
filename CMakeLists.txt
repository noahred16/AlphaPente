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

# Main executable (start simple)
add_executable(alphapente src/main.cpp)

# Enable testing
enable_testing()
add_subdirectory(tests)