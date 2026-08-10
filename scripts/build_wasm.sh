#!/bin/bash
# Builds the browser AI (PenteGame + single-threaded MCTS + HeuristicEvaluator,
# plus solved-book lookup support) to WebAssembly via the emscripten/emsdk
# Docker image, no local toolchain needed. Output: docs/wasm/pente.js + .wasm.
#
# The solved 4x4 book (docs/data/book4x4.bin) is NOT embedded in this build
# (no --preload-file) - it's fetched lazily by JS only when the user actually
# picks the 4x4 board size, and passed into WasmGame::loadBookFromBytes() at
# runtime. Embedding it here would force every page load to download the
# whole ~67MB book up front regardless of board size chosen. Regenerate the
# book itself (from the repo root, native build already configured, ~82s):
#   ./build/solve5x5 -B 4 -x -o docs/data/book4x4.bin
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCES="src/PenteGame.cpp src/BitBoard.cpp src/Zobrist.cpp src/GameUtils.cpp src/Evaluator.cpp src/MCTS.cpp src/RenjuRules.cpp src/RenjuForbiddenPointFinder.cpp src/PositionKey.cpp src/PositionBook.cpp wasm/PenteWasm.cpp"

sudo docker run --rm -v "$PWD":/src -u "$(id -u)":"$(id -g)" -w /src emscripten/emsdk \
  em++ -std=c++20 -O2 -I include \
  $SOURCES \
  --bind \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=PenteModule \
  -s ENVIRONMENT=web,node \
  -o docs/wasm/pente.js

echo "Built docs/wasm/pente.js + docs/wasm/pente.wasm"
