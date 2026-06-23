#!/bin/bash

REPO_ROOT="$HOME/repos/AlphaPente"

# Computing cluster: load modules if on known host
if [[ "$(hostname)" == explorer* ]]; then
    echo "Host $(hostname) detected. Loading CUDA, cuDNN, and CMake..."
    module load cuda/12.3.0
    module load cuDNN/9.10.2
    module load cmake/4.2.3
fi

# Auto-detect system specs if .env doesn't exist yet
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "No .env found. Detecting system specs..."
    bash "$REPO_ROOT/scripts/detect_system_specs.sh"
    echo ""
fi

# Detect GPU and pick matching libtorch variant
if nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    VARIANT="cu121"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip"
    echo "GPU detected (sm_${CUDA_VER}) — using CUDA libtorch (cu121)"
else
    VARIANT="cpu"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
    echo "No GPU detected — using CPU-only libtorch"
fi

# Download libtorch if missing or wrong variant
mkdir -p "$REPO_ROOT/libs" && cd "$REPO_ROOT/libs"

LIBTORCH_CHANGED=false
if [ -d libtorch ] && grep -q "$VARIANT" libtorch/build-version 2>/dev/null; then
    echo "✅ libtorch ($VARIANT) already installed. Skipping download."
else
    LIBTORCH_CHANGED=true
    if [ -d libtorch ]; then
        echo "Removing mismatched libtorch ($(cat libtorch/build-version 2>/dev/null || echo unknown))..."
        rm -rf libtorch
    fi
    echo "Downloading libtorch ($VARIANT)..."
    wget "$LIBTORCH_URL" -O libtorch.zip
    unzip libtorch.zip
    rm libtorch.zip
    echo "✅ libtorch installed."
fi

# Configure and build
cd "$REPO_ROOT"
if [ "$LIBTORCH_CHANGED" = true ]; then
    echo "libtorch changed — clearing build dir."
    rm -rf build
fi
mkdir -p build && cd build

if cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$REPO_ROOT/libs/libtorch" -Wno-dev; then
    echo "cmake configured successfully."
    make -j"$(nproc)"
    echo "✅ Build completed."
else
    echo "cmake failed — see output above."
    exit 1
fi
