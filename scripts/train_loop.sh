#!/usr/bin/env bash
# Self-play training loop — generates games, trains, and benchmarks each iteration.
# Run from anywhere in the repo.
# Usage: ./scripts/train_loop.sh [-n games] [-s sims] [-i max_iters] [-g game]

set -euo pipefail

GAMES=50
SIMS=400
MAX_ITERS=0          # 0 = run forever
GAME="pente"

while getopts "n:s:i:g:" opt; do
    case $opt in
        n) GAMES=$OPTARG ;;
        s) SIMS=$OPTARG ;;
        i) MAX_ITERS=$OPTARG ;;
        g) GAME=$OPTARG ;;
        *) echo "Usage: $0 [-n games] [-s sims] [-i max_iters] [-g game]" >&2; exit 1 ;;
    esac
done

BUILD_DIR="$(cd "$(dirname "$0")/../build" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

cd "$BUILD_DIR"

LIMIT_STR=$([ "$MAX_ITERS" -eq 0 ] && echo "unlimited" || echo "$MAX_ITERS")

echo "Self-play training loop"   | tee "$LOG"
echo "  game  : $GAME"           | tee -a "$LOG"
echo "  games : $GAMES"          | tee -a "$LOG"
echo "  sims  : $SIMS"           | tee -a "$LOG"
echo "  limit : $LIMIT_STR"      | tee -a "$LOG"
echo "  log   : $LOG"            | tee -a "$LOG"
echo "  start : $(date)"         | tee -a "$LOG"
echo ""                          | tee -a "$LOG"

ITER=0
START=$(date +%s)

cleanup() {
    ELAPSED=$(( $(date +%s) - START ))
    echo "" | tee -a "$LOG"
    echo "Stopped after $ITER iteration(s), ${ELAPSED}s elapsed." | tee -a "$LOG"
    exit 0
}
trap cleanup INT TERM

while true; do
    ITER=$(( ITER + 1 ))
    ITER_START=$(date +%s)

    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "Iteration $ITER  —  $(date)"                                  | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    ./generate -g "$GAME" -n "$GAMES" -s "$SIMS" -a 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    ./train -g "$GAME" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    echo "── Benchmark ────────────────────────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" 2>&1 | tee -a "$LOG"

    ITER_ELAPSED=$(( $(date +%s) - ITER_START ))
    echo "" | tee -a "$LOG"
    echo "Iteration $ITER done in ${ITER_ELAPSED}s" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    if [[ $MAX_ITERS -gt 0 && $ITER -ge $MAX_ITERS ]]; then
        echo "Reached max iterations ($MAX_ITERS)." | tee -a "$LOG"
        break
    fi
done

cleanup
