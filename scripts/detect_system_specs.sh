#!/usr/bin/env bash
# Detects system CPU and memory and writes suggested .env values (80% of available).
# Run from anywhere in the repo; writes to <repo-root>/.env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

TOTAL_THREADS=$(nproc)
NUM_THREADS=$(( TOTAL_THREADS * 8 / 10 ))
[ "$NUM_THREADS" -lt 1 ] && NUM_THREADS=1

TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(( TOTAL_MEM_KB / 1024 / 1024 ))
ARENA_SIZE_GB=$(( TOTAL_MEM_GB * 8 / 10 ))
[ "$ARENA_SIZE_GB" -lt 1 ] && ARENA_SIZE_GB=1

echo "Detected system specs:"
echo "  CPU threads : $TOTAL_THREADS total → NUM_THREADS=$NUM_THREADS (80%)"
echo "  RAM         : ${TOTAL_MEM_GB} GB total → ARENA_SIZE_GB=${ARENA_SIZE_GB} GB (80%)"
echo ""

cat > "$ENV_FILE" <<EOF
ARENA_SIZE_GB=$ARENA_SIZE_GB
NUM_THREADS=$NUM_THREADS
EOF

echo "Written to $ENV_FILE"
