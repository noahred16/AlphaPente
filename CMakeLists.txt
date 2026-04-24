cmake_minimum_required(VERSION 3.15)
project(AlphaPente)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Compiler flags ───────────────────────────────────────────────────────────

if(MSVC)
  # Windows / Visual Studio
  set(CMAKE_CXX_FLAGS_DEBUG   "/Od /Zi /W4 /RTC1")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /GL")  # /GL enables link-time codegen
  add_link_options($<$<CONFIG:Release>:/LTCG>)      # /LTCG pairs with /GL

else()
  # GCC / Clang (macOS + Linux)
  set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

  add_compile_options(
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-mtune=native>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Release>:-funroll-loops>
    $<$<CONFIG:Release>:-flto=auto>
  )
  add_link_options($<$<CONFIG:Release>:-flto=auto>)

  # Apple-specific: explicit libc++ to avoid ABI issues
  if(APPLE)
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
  endif()
endif()

# ─── Optional: OpenMP (MCTS parallelism) ──────────────────────────────────────

find_package(OpenMP)

# ─── Includes ─────────────────────────────────────────────────────────────────

include_directories(include)

# ─── Core library ─────────────────────────────────────────────────────────────

add_library(pente_core
    src/BitBoard.cpp
    src/PenteGame.cpp
    src/MCTS.cpp
    src/ParallelMCTS.cpp
    src/GameUtils.cpp
    src/Evaluator.cpp
    src/Zobrist.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(pente_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─── Executables ──────────────────────────────────────────────────────────────

macro(add_pente_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE pente_core)
endmacro()

add_pente_executable(train      apps/Train.cpp)
add_pente_executable(play       apps/Play.cpp)
add_pente_executable(test       apps/Test.cpp)
add_pente_executable(pente      apps/Pente.cpp)
add_pente_executable(gomoku     apps/Gomoku.cpp)
add_pente_executable(keryopente apps/KeryoPente.cpp)

# ─── Unit tests ───────────────────────────────────────────────────────────────

add_executable(unit_tests
    tests/BitBoardTests.cpp
    tests/PenteGameTests.cpp
)
target_link_libraries(unit_tests PRIVATE pente_core)