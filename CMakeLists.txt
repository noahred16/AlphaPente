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

# ─── Optional: LibTorch (NNEvaluator + AlphaNet) ──────────────────────────────
# Find Torch before OpenMP: LibTorch bundles its own libgomp, so when Torch is
# present we skip the separate OpenMP link to avoid loading two OpenMP runtimes.

if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "/opt/libtorch")
  list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
endif()

# Auto-detect LibTorch from the active Python environment if not already set.
# On clusters without a libtorch module, `pip install torch` inside a venv is
# the easiest path. This lets `cmake ..` work without a manual -DCMAKE_PREFIX_PATH.
if(NOT TORCH_FOUND)
  find_program(PYTHON_EXE NAMES python3 python)
  if(PYTHON_EXE)
    execute_process(
      COMMAND ${PYTHON_EXE} -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(TORCH_CMAKE_PREFIX)
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    endif()
  endif()
endif()

# Stub out CUDA::nvToolsExt before find_package(Torch) — newer CUDA toolkits
# (13.x) don't ship this target, but PyTorch's cmake scripts still reference it.
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

set(USE_KINETO OFF CACHE BOOL "" FORCE)
find_package(Torch QUIET)
if(TORCH_FOUND)
  message(STATUS "LibTorch found: ${TORCH_VERSION}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# ─── Optional: OpenMP (MCTS parallelism) ──────────────────────────────────────
# Skip when Torch is present — Torch's bundled libgomp provides OpenMP support.

if(NOT TORCH_FOUND)
  find_package(OpenMP)
endif()

# ─── Includes ─────────────────────────────────────────────────────────────────

include_directories(include)
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

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

if(TORCH_FOUND)
  target_compile_definitions(pente_core PUBLIC WITH_TORCH)
  target_sources(pente_core PRIVATE src/NNModel.cpp src/NNEvaluator.cpp src/SelfPlay.cpp)
  target_link_libraries(pente_core PUBLIC "${TORCH_LIBRARIES}")
  target_include_directories(pente_core SYSTEM PUBLIC "${TORCH_INCLUDE_DIRS}")
else()
  message(STATUS "LibTorch not found — NNEvaluator disabled")
  if(OpenMP_CXX_FOUND)
    target_link_libraries(pente_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# ─── Executables ──────────────────────────────────────────────────────────────

macro(add_pente_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE pente_core)
endmacro()

add_pente_executable(generate   apps/Generate.cpp)
add_pente_executable(train      apps/Train.cpp)
add_pente_executable(play       apps/Play.cpp)
add_pente_executable(test       apps/Test.cpp)
add_pente_executable(pente      apps/Pente.cpp)
add_pente_executable(gomoku     apps/Gomoku.cpp)
add_pente_executable(keryopente apps/KeryoPente.cpp)
add_pente_executable(benchmark  apps/Benchmark.cpp)
add_pente_executable(inspect    apps/Inspect.cpp)

# ─── Unit tests ───────────────────────────────────────────────────────────────

add_executable(unit_tests
    tests/BitBoardTests.cpp
    tests/PenteGameTests.cpp
    tests/ParallelMCTSTests.cpp
    tests/AugmentTests.cpp
)
target_link_libraries(unit_tests PRIVATE pente_core)

# Slow: runs full self-play games through a real NNEvaluator. Separate binary
# so `./unit_tests` stays fast; run these explicitly with `./selfplay_tests`.
add_executable(selfplay_tests tests/SelfPlayTests.cpp)
target_link_libraries(selfplay_tests PRIVATE pente_core)