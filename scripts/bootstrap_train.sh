#!/usr/bin/env bash
# Train on pre-generated bootstrap data, benchmarking after each cycle.
# Run from anywhere in the repo.
# Usage: ./scripts/bootstrap_train.sh [-g game] [-n cycles] [-a arena_every] [-s benchmark_arena_sims]

set -euo pipefail

GAME="pente"
CYCLES=10
ARENA_EVERY=5          # run the full battery (incl. 3-tier heuristic arena) every N cycles; 0 = never mid-loop
BENCHMARK_ARENA_SIMS=800

while getopts "g:n:a:s:" opt; do
    case $opt in
        g) GAME=$OPTARG ;;
        n) CYCLES=$OPTARG ;;
        a) ARENA_EVERY=$OPTARG ;;
        s) BENCHMARK_ARENA_SIMS=$OPTARG ;;
        *) echo "Usage: $0 [-g game] [-n cycles] [-a arena_every] [-s benchmark_arena_sims]" >&2; exit 1 ;;
    esac
done

BUILD_DIR="$(cd "$(dirname "$0")/../build" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bootstrap_train_$(date +%Y%m%d_%H%M%S).log"

cd "$BUILD_DIR"

echo "Bootstrap training"      | tee "$LOG"
echo "  game  : $GAME"         | tee -a "$LOG"
echo "  cycles: $CYCLES (full battery every $ARENA_EVERY)" | tee -a "$LOG"
echo "  arena sims: $BENCHMARK_ARENA_SIMS"                  | tee -a "$LOG"
echo "  log   : $LOG"          | tee -a "$LOG"
echo "  start : $(date)"       | tee -a "$LOG"
echo ""                        | tee -a "$LOG"

START=$(date +%s)
LAST_CYCLE_WAS_FULL=false

cleanup() {
    ELAPSED=$(( $(date +%s) - START ))
    echo "" | tee -a "$LOG"
    echo "Stopped after ${ELAPSED}s elapsed." | tee -a "$LOG"
}
trap cleanup INT TERM

for cycle in $(seq 1 "$CYCLES"); do
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "Cycle $cycle / $CYCLES  —  $(date)"                          | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    echo "── Train ────────────────────────────────────────────────────────" | tee -a "$LOG"
    ./train -g "$GAME" -b 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    if (( ARENA_EVERY > 0 && cycle % ARENA_EVERY == 0 )); then
        echo "── Benchmark (full battery) ────────────────────────────────────────" | tee -a "$LOG"
        ./benchmark -g "$GAME" 2>&1 | tee -a "$LOG"
        LAST_CYCLE_WAS_FULL=true
    else
        echo "── Benchmark (quick: policy + value sign) ──────────────────────────" | tee -a "$LOG"
        ./benchmark -g "$GAME" -s 0 2>&1 | tee -a "$LOG"
        ./benchmark -g "$GAME" -V 2>&1 | tee -a "$LOG"
        LAST_CYCLE_WAS_FULL=false
    fi
    echo "" | tee -a "$LOG"
done

if [[ "$LAST_CYCLE_WAS_FULL" == true ]]; then
    echo "Last cycle already ran the full battery (3-tier heuristic arena) — skipping redundant final arena." | tee -a "$LOG"
else
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "Final arena: NN vs heuristic" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
    ./benchmark -g "$GAME" -a -S "$BENCHMARK_ARENA_SIMS" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
fi

echo "All done!  $(date)" | tee -a "$LOG"
ELAPSED=$(( $(date +%s) - START ))
echo "Total time: ${ELAPSED}s" | tee -a "$LOG"
