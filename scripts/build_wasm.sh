#!/bin/bash
# Builds the browser AI (PenteGame + single-threaded MCTS + HeuristicEvaluator)
# to WebAssembly via the emscripten/emsdk Docker image, no local toolchain
# needed. Output: web/wasm/pente.js + web/wasm/pente.wasm
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCES="src/PenteGame.cpp src/BitBoard.cpp src/Zobrist.cpp src/GameUtils.cpp src/Evaluator.cpp src/MCTS.cpp wasm/PenteWasm.cpp"

sudo docker run --rm -v "$PWD":/src -u "$(id -u)":"$(id -g)" -w /src emscripten/emsdk \
  em++ -std=c++20 -O2 -I include \
  $SOURCES \
  --bind \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=PenteModule \
  -s ENVIRONMENT=web,node \
  -o web/wasm/pente.js

echo "Built web/wasm/pente.js + web/wasm/pente.wasm"
