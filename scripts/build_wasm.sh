#!/bin/bash
# Builds the browser AI (PenteGame + single-threaded MCTS + HeuristicEvaluator,
# plus solved-book lookup with a live PNS::solve() fallback past the book's
# coverage - see wasm/PenteWasm.cpp's rankedMoves()) to WebAssembly via the
# emscripten/emsdk Docker image, no local toolchain needed. Output:
# docs/wasm/pente.js + .wasm.
#
# The solved 4x4 book is NOT embedded in this build (no --preload-file) -
# it's fetched lazily by JS (gzipped - see docs/js/app.js's fetchBookBytes)
# only when the user actually picks the 4x4 board size, and passed into
# WasmGame::loadBookFromBytes() at runtime. Embedding it here would force
# every page load to download the whole book up front regardless of board
# size chosen - the actual bug behind an earlier "gets stuck on loading"
# report. Regenerate the book itself (from the repo root, native build
# already configured, ~82s to solve + regzip; -m 10 trims it to only the
# first 10 plies - see PNS.hpp's class comment and the project's solve-5x5
# issue doc for why a live-solve fallback covers the rest cheaply):
#   ./build/solve5x5 -B 4 -x -m 10 -o docs/data/book4x4.bin
#   gzip -kf docs/data/book4x4.bin   # -> docs/data/book4x4.bin.gz, the served asset
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCES="src/PenteGame.cpp src/BitBoard.cpp src/Zobrist.cpp src/GameUtils.cpp src/Evaluator.cpp src/MCTS.cpp src/RenjuRules.cpp src/RenjuForbiddenPointFinder.cpp src/PositionKey.cpp src/PositionBook.cpp src/PNS.cpp wasm/PenteWasm.cpp"

# STACK_SIZE raised from Emscripten's small default: PNS::mid()'s recursion
# takes a full PenteGame (8KB+, mostly its embedded mt19937) by value per
# level - see PNS::Config::maxRecursionDepth's comment. WasmGame's live-solve
# fallback config caps recursion at 60 levels; 8MB gives generous headroom
# (~130KB/level) without needing to plumb a runtime-adjustable stack size
# through to WASM the way the native build does via setrlimit
# (apps/Solve5x5.cpp) - WASM's stack size is fixed at compile time.
sudo docker run --rm -v "$PWD":/src -u "$(id -u)":"$(id -g)" -w /src emscripten/emsdk \
  em++ -std=c++20 -O2 -I include \
  $SOURCES \
  --bind \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s STACK_SIZE=8388608 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=PenteModule \
  -s ENVIRONMENT=web,node \
  -o docs/wasm/pente.js

echo "Built docs/wasm/pente.js + docs/wasm/pente.wasm"
