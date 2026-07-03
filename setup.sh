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
# On macOS, PyTorch doesn't ship a cxx11-abi libtorch; use the macOS-native zip instead.
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    VARIANT="cpu"
    LIBTORCH_URL="__wheel__"
    echo "macOS detected — will extract libtorch from PyTorch wheel"
elif nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    # Blackwell (sm_120+) requires CUDA 12.8+ wheels; older GPUs use cu121
    if [ "${CUDA_VER}" -ge 120 ] 2>/dev/null; then
        VARIANT="cu128"
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip"
        echo "GPU detected (sm_${CUDA_VER}, Blackwell) — using CUDA libtorch (cu128)"
    else
        VARIANT="cu121"
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip"
        echo "GPU detected (sm_${CUDA_VER}) — using CUDA libtorch (cu121)"
    fi
else
    VARIANT="cpu"
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
    echo "No GPU detected — using CPU-only libtorch"
fi

# Download libtorch if missing or wrong variant
mkdir -p "$REPO_ROOT/libs" && cd "$REPO_ROOT/libs"

LIBTORCH_CHANGED=false
if [ -d libtorch ] && [ "$(cat libtorch/.variant 2>/dev/null)" = "$VARIANT" ]; then
    echo "✅ libtorch ($VARIANT) already installed. Skipping download."
else
    LIBTORCH_CHANGED=true
    if [ -d libtorch ]; then
        echo "Removing mismatched libtorch..."
        rm -rf libtorch
    fi

    if [[ "$LIBTORCH_URL" == "__wheel__" ]]; then
        PYTHON=$(command -v python3 || command -v python)
        echo "Downloading PyTorch wheel via pip..."
        "$PYTHON" -m pip download torch --no-deps -d tmp_whl_dl -q
        WHL=$(ls tmp_whl_dl/torch-*.whl 2>/dev/null | head -1)
        if [[ -z "$WHL" ]]; then
            echo "ERROR: pip download failed — no wheel found"
            exit 1
        fi
        unzip -q "$WHL" -d tmp_torch
        mv tmp_torch/torch ./libtorch
        rm -rf tmp_torch tmp_whl_dl
    else
        echo "Downloading libtorch ($VARIANT)..."
        wget "$LIBTORCH_URL" -O libtorch.zip
        unzip libtorch.zip
        rm libtorch.zip
    fi
    echo "$VARIANT" > libtorch/.variant
    echo "✅ libtorch installed."
fi

# Configure and build
cd "$REPO_ROOT"
if [ "$LIBTORCH_CHANGED" = true ]; then
    echo "libtorch changed — clearing build dir."
    rm -rf build
fi
mkdir -p build && cd build

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -Wno-dev)
if [ -d "$REPO_ROOT/libs/libtorch" ]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$REPO_ROOT/libs/libtorch")
fi

# Pass compute capability explicitly so PyTorch's cmake scripts don't need to
# auto-detect it (auto-detection compiles a CUDA 20 test that fails on CMake < 3.30).
if nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$COMPUTE_CAP" ]; then
        CMAKE_ARGS+=("-DTORCH_CUDA_ARCH_LIST=${COMPUTE_CAP}")
    fi
fi

MAKE_JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)

if cmake .. "${CMAKE_ARGS[@]}"; then
    echo "cmake configured successfully."
    make -j"$MAKE_JOBS"
    echo "✅ Build completed."
else
    echo "cmake failed — see output above."
    exit 1
fi
