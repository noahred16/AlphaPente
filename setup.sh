#!/bin/bash

# Computing cluster steps, optional
if [[ "$(hostname)" == "explorer-02" ]]; then
    echo "Host explorer-02 detected. Loading CUDA, cuDNN, and CMake..."
    module load cuda/12.3.0
    module load cuDNN/9.10.2
    module load cmake/4.2.3
fi

# Auto-detect system specs if .env doesn't exist yet
cd ~/repos/AlphaPente
if [ ! -f .env ]; then
    echo "No .env found. Detecting system specs..."
    bash scripts/detect_system_specs.sh
    echo ""
fi

# Setup libs (just libtorch)
cd ~/repos/AlphaPente
mkdir -p libs && cd libs

# Check if the libtorch directory does not exist before installing
if [ ! -d "libtorch" ]; then
    echo "libtorch not found. Downloading and installing..."
    # CPU-only, cxx11 ABI (matches GCC 11.5 fine)
    wget "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip" \
        -O libtorch-2.5.1-cu121.zip
    unzip libtorch-2.5.1-cu121.zip
    # Optional: Clean up the zip file after extraction
    rm libtorch-2.5.1-cu121.zip
    echo "✅ libtorch installation completed."
else
    echo "✅ libtorch directory already exists. Skipping installation."
fi

# Build
cd ~/repos/AlphaPente
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/repos/AlphaPente/libs/libtorch
# make -j$(nproc)
make
echo "✅ Build completed."

# mkdir build
# cd build
# cmake ..
# make
# cd ../
