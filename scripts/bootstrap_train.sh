#!/usr/bin/env bash
# Train on pre-generated bootstrap data, benchmarking after each cycle.
# Run from anywhere in the repo.
# Usage: ./scripts/bootstrap_train.sh [-g game] [-n cycles] [-a arena_every] [-s benchmark_arena_sims]

set -euo pipefail

GAME="pente"
CYCLES=10
ARENA_EVERY=5          # run the full battery (incl. 3-tier heuristic arena) every N cycles; 0 = never mid-loop
BENCHMARK_ARENA_SIMS=800
PROMOTION_ARENA_GAMES=40  # games played vs previous best_model.pt to confirm each val-gate promotion
PROMOTION_ARENA_SIMS=400  # lower than BENCHMARK_ARENA_SIMS to afford more games at similar total compute

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
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
CKPT_DIR="$ROOT_DIR/checkpoints/$GAME"
PREV_BEST="$CKPT_DIR/best_model.pt.prev"
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

    had_baseline=false
    if [[ -f "$CKPT_DIR/best_model.pt" ]]; then
        had_baseline=true
        cp "$CKPT_DIR/best_model.pt" "$PREV_BEST"
    fi

    echo "── Train ────────────────────────────────────────────────────────" | tee -a "$LOG"
    train_out=$(./train -g "$GAME" -b 2>&1)
    echo "$train_out" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    promoted=true
    echo "$train_out" | grep -q "PROMOTED" || promoted=false

    # Benchmark the model this cycle just trained, not necessarily
    # best_model.pt (which now defaults for ./benchmark, but may still be a
    # previous cycle's model if this candidate isn't promoted below).
    CANDIDATE=$(ls -t "$CKPT_DIR/model_iter"*.pt 2>/dev/null | head -1)

    # ── Promotion arena: the val-loss gate is measured against a held-out
    # slice of this same static bootstrap buffer, so it can't tell memorizing
    # that buffer apart from actually playing better. Require an arena win
    # over the previous best_model.pt too before keeping the promotion.
    if [[ "$promoted" == true && "$had_baseline" == true ]]; then
        echo "── Promotion arena: candidate vs previous best_model.pt ────────────" | tee -a "$LOG"
        arena_out=$(./benchmark -g "$GAME" -a -G "$PROMOTION_ARENA_GAMES" -S "$PROMOTION_ARENA_SIMS" -P "$PREV_BEST" -p "$CANDIDATE" 2>&1)
        echo "$arena_out" | tee -a "$LOG"
        nn_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $4}')
        opp_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $6}')
        if [[ -z "$nn_wins" || "$nn_wins" -lt "$opp_wins" ]]; then
            cp "$PREV_BEST" "$CKPT_DIR/best_model.pt"
            promoted=false
            echo "  → candidate did not beat previous best_model.pt ($nn_wins-$opp_wins) — reverted, best_model.pt unchanged" | tee -a "$LOG"
        else
            echo "  → candidate beat previous best_model.pt ($nn_wins-$opp_wins) — promotion confirmed" | tee -a "$LOG"
        fi
        echo "" | tee -a "$LOG"
    fi

    if (( ARENA_EVERY > 0 && cycle % ARENA_EVERY == 0 )); then
        echo "── Benchmark (full battery) ────────────────────────────────────────" | tee -a "$LOG"
        ./benchmark -g "$GAME" -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
        LAST_CYCLE_WAS_FULL=true
    else
        echo "── Benchmark (quick: policy + value sign) ──────────────────────────" | tee -a "$LOG"
        ./benchmark -g "$GAME" -s 0 -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
        ./benchmark -g "$GAME" -V -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
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
