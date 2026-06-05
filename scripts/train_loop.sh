#!/usr/bin/env bash
# Usage: ./scripts/train_loop.sh [-n games] [-s sims] [-i max_iters] [-g game]
# Run from the build/ directory.

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
LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

cd "$BUILD_DIR"

echo "AlphaPente training loop" | tee "$LOG"
echo "  game : $GAME"           | tee -a "$LOG"
echo "  games: $GAMES"          | tee -a "$LOG"
echo "  sims : $SIMS"           | tee -a "$LOG"
echo "  limit: ${MAX_ITERS:-unlimited}" | tee -a "$LOG"
echo "  log  : $LOG"            | tee -a "$LOG"
echo ""                         | tee -a "$LOG"

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
    echo "── Iteration $ITER  $(date '+%H:%M:%S') ────────────────────────────────" | tee -a "$LOG"

    ./generate -g "$GAME" -n "$GAMES" -s "$SIMS" 2>&1 | tee -a "$LOG"
    ./train    -g "$GAME" 2>&1 | tee -a "$LOG"

    # Find the checkpoint this iteration produced
    LATEST=$(ls -t "../checkpoints/$GAME"/model_iter*.pt 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        echo "" | tee -a "$LOG"
        echo "Benchmark: $LATEST" | tee -a "$LOG"
        ./benchmark -p "$LATEST" 2>&1 | grep -E "Result:|model_iter" | tee -a "$LOG"
    fi

    ITER_ELAPSED=$(( $(date +%s) - ITER_START ))
    echo "Iteration $ITER done in ${ITER_ELAPSED}s" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    if [[ $MAX_ITERS -gt 0 && $ITER -ge $MAX_ITERS ]]; then
        echo "Reached max iterations ($MAX_ITERS)." | tee -a "$LOG"
        break
    fi
done

cleanup
