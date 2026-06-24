#!/usr/bin/env bash
# Train on pre-generated bootstrap data, benchmarking after each step.
# Run from anywhere in the repo.
# Usage: ./scripts/bootstrap_train.sh [-g game] [-r rounds] [-n trains_per_round]

set -euo pipefail

GAME="pente"
ROUNDS=6
TRAINS_PER_ROUND=5
ARENA_SIMS=400

while getopts "g:r:n:s:" opt; do
    case $opt in
        g) GAME=$OPTARG ;;
        r) ROUNDS=$OPTARG ;;
        n) TRAINS_PER_ROUND=$OPTARG ;;
        s) ARENA_SIMS=$OPTARG ;;
        *) echo "Usage: $0 [-g game] [-r rounds] [-n trains_per_round] [-s arena_sims]" >&2; exit 1 ;;
    esac
done

BUILD_DIR="$(cd "$(dirname "$0")/../build" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bootstrap_train_$(date +%Y%m%d_%H%M%S).log"

cd "$BUILD_DIR"

echo "Bootstrap training"      | tee "$LOG"
echo "  game  : $GAME"         | tee -a "$LOG"
echo "  rounds: $ROUNDS x $TRAINS_PER_ROUND trains" | tee -a "$LOG"
echo "  arena sims: $ARENA_SIMS"                  | tee -a "$LOG"
echo "  log   : $LOG"          | tee -a "$LOG"
echo "  start : $(date)"       | tee -a "$LOG"
echo ""                        | tee -a "$LOG"

START=$(date +%s)

cleanup() {
    ELAPSED=$(( $(date +%s) - START ))
    echo "" | tee -a "$LOG"
    echo "Stopped after ${ELAPSED}s elapsed." | tee -a "$LOG"
}
trap cleanup INT TERM

for round in $(seq 1 "$ROUNDS"); do
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "Round $round / $ROUNDS  —  $(date)"                          | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    for step in $(seq 1 "$TRAINS_PER_ROUND"); do
        echo "── Train step $step / $TRAINS_PER_ROUND ──────────────────────────────" | tee -a "$LOG"
        ./train -g "$GAME" -b 2>&1 | tee -a "$LOG"
        echo "" | tee -a "$LOG"

        echo "── Benchmark ────────────────────────────────────────────────────" | tee -a "$LOG"
        ./benchmark -g "$GAME" 2>&1 | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    done
done

echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "Final arena: NN vs heuristic" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
./benchmark -g "$GAME" -a -S "$ARENA_SIMS" 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "All done!  $(date)" | tee -a "$LOG"
ELAPSED=$(( $(date +%s) - START ))
echo "Total time: ${ELAPSED}s" | tee -a "$LOG"
