#!/bin/bash
# Builds the browser AI (PenteGame + single-threaded MCTS + HeuristicEvaluator,
# plus a solved 4x4 book) to WebAssembly via the emscripten/emsdk Docker image,
# no local toolchain needed. Output: docs/wasm/pente.js + .wasm + .data.
#
# Needs docs/data/book4x4.bin to exist first (gitignored - not worth
# duplicating docs/wasm/pente.data's 67MB in git alongside it). Regenerate it
# with (from the repo root, native build already configured, ~82s):
#   ./build/solve5x5 -B 4 -x -o docs/data/book4x4.bin
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f docs/data/book4x4.bin ]; then
  echo "Missing docs/data/book4x4.bin - regenerate it first with:" >&2
  echo "  ./build/solve5x5 -B 4 -x -o docs/data/book4x4.bin" >&2
  exit 1
fi

SOURCES="src/PenteGame.cpp src/BitBoard.cpp src/Zobrist.cpp src/GameUtils.cpp src/Evaluator.cpp src/MCTS.cpp src/RenjuRules.cpp src/RenjuForbiddenPointFinder.cpp src/PositionKey.cpp src/PositionBook.cpp wasm/PenteWasm.cpp"

# The 4x4 book is preloaded into the WASM virtual filesystem at the same path
# WasmGame::WasmGame() opens (PositionBook::load()) - see wasm/PenteWasm.cpp.
# --preload-file's second path is where it lands in the virtual FS; pente.data
# (written alongside pente.js/.wasm) carries the actual bytes, fetched by the
# generated loader.
sudo docker run --rm -v "$PWD":/src -u "$(id -u)":"$(id -g)" -w /src emscripten/emsdk \
  em++ -std=c++20 -O2 -I include \
  $SOURCES \
  --bind \
  --preload-file docs/data/book4x4.bin@/book4x4.bin \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=PenteModule \
  -s ENVIRONMENT=web,node \
  -o docs/wasm/pente.js

echo "Built docs/wasm/pente.js + docs/wasm/pente.wasm"
