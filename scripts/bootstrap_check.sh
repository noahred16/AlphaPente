#!/usr/bin/env bash
# Print info about checkpoints/<game>/bootstrap.pt (sample count, tensor shapes, file size).

set -euo pipefail

GAME="${1:-pente}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PT="$REPO/checkpoints/$GAME/bootstrap.pt"

if [ ! -f "$PT" ]; then
    echo "No bootstrap file: $PT"
    exit 1
fi

LIBTORCH="$REPO/libs/libtorch"
TMP="$(mktemp /tmp/check_bootstrap_XXXXXX.cpp)"
BIN="$(mktemp /tmp/check_bootstrap_XXXXXX)"

trap 'rm -f "$TMP" "$BIN"' EXIT

cat > "$TMP" << 'EOF'
#include <torch/torch.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string path = argv[1];
    torch::serialize::InputArchive ar;
    ar.load_from(path);
    torch::Tensor states, captures, policies, values;
    ar.read("states",   states);
    ar.read("captures", captures);
    ar.read("policies", policies);
    ar.read("values",   values);
    int64_t n = states.size(0);
    std::cout << "samples  : " << n << "\n";
    std::cout << "states   : " << states.sizes()   << "\n";
    std::cout << "captures : " << captures.sizes() << "\n";
    std::cout << "policies : " << policies.sizes() << "\n";
    std::cout << "values   : " << values.sizes()   << "\n";
}
EOF

/usr/bin/c++ \
    -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++20 -O0 \
    -isystem "$LIBTORCH/include" \
    -isystem "$LIBTORCH/include/torch/csrc/api/include" \
    -L"$LIBTORCH/lib" \
    -Wl,-rpath,"$LIBTORCH/lib" \
    "$TMP" -o "$BIN" \
    -ltorch -ltorch_cpu -lc10 2>&1

echo "file     : $PT"
echo "size     : $(du -h "$PT" | cut -f1)"
"$BIN" "$PT"
