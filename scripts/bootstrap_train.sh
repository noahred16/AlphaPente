#!/usr/bin/env bash
# Train on pre-generated bootstrap data with validation-loss early stopping.
# Every cycle re-trains on the same bootstrap.pt (more cycles = more passes
# over the same data). train -b holds out the tail 10% as validation; training
# stops once val loss hasn't improved for PATIENCE cycles. The best-val
# checkpoint is then promoted to best_model.pt and the full benchmark battery
# runs on it.
# Run from anywhere in the repo.
# Usage: ./scripts/bootstrap_train.sh [-g game] [-n max_cycles] [-p patience] [-a arena_every]

set -euo pipefail

GAME="pente"
CYCLES=20
PATIENCE=3
ARENA_EVERY=5          # run the full battery (incl. 3-tier heuristic arena) every N cycles; 0 = never mid-loop
PROMOTION_ARENA_GAMES=40  # games played vs previous best_model.pt to confirm each val-gate promotion
PROMOTION_ARENA_SIMS=400  # lower than the full battery's sims to afford more games at similar total compute

while getopts "g:n:p:a:" opt; do
    case $opt in
        g) GAME=$OPTARG ;;
        n) CYCLES=$OPTARG ;;
        p) PATIENCE=$OPTARG ;;
        a) ARENA_EVERY=$OPTARG ;;
        *) echo "Usage: $0 [-g game] [-n max_cycles] [-p patience] [-a arena_every]" >&2; exit 1 ;;
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
echo "  cycles: max $CYCLES (early stop after $PATIENCE stale, full battery every $ARENA_EVERY)" | tee -a "$LOG"
echo "  log   : $LOG"          | tee -a "$LOG"
echo "  start : $(date)"       | tee -a "$LOG"
echo ""                        | tee -a "$LOG"

START=$(date +%s)
BEST_VAL=""
BEST_CKPT=""
LAST_CKPT=""
STALE=0
LAST_CYCLE_WAS_FULL=false

cleanup() {
    ELAPSED=$(( $(date +%s) - START ))
    echo "" | tee -a "$LOG"
    echo "Stopped after ${ELAPSED}s elapsed." | tee -a "$LOG"
    if [[ -n "$BEST_CKPT" ]]; then
        echo "Best so far: $BEST_CKPT (val $BEST_VAL). best_model.pt still holds the LAST" | tee -a "$LOG"
        echo "cycle's weights — promote manually: cp checkpoints/$GAME/$BEST_CKPT checkpoints/$GAME/best_model.pt" | tee -a "$LOG"
    fi
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

    VAL=$(grep -oP 'val\s+policy:.*total:\s*\K[0-9.]+' "$LOG" | tail -1 || true)
    LAST_CKPT=$(grep -oP 'model_iter[0-9]+\.pt' "$LOG" | tail -1 || true)
    if [[ -z "$VAL" || -z "$LAST_CKPT" ]]; then
        echo "Could not parse val loss / saved checkpoint from train output — aborting." | tee -a "$LOG"
        exit 1
    fi

    if [[ -z "$BEST_VAL" ]] || awk -v a="$VAL" -v b="$BEST_VAL" 'BEGIN { exit !(a < b) }'; then
        BEST_VAL=$VAL
        BEST_CKPT=$LAST_CKPT
        STALE=0
        echo ">>> val $VAL — new best ($BEST_CKPT)" | tee -a "$LOG"
    else
        STALE=$((STALE + 1))
        echo ">>> val $VAL — no improvement on $BEST_VAL ($BEST_CKPT), stale $STALE/$PATIENCE" | tee -a "$LOG"
        if (( STALE >= PATIENCE )); then
            echo ">>> early stop after cycle $cycle" | tee -a "$LOG"
            echo "" | tee -a "$LOG"
            break
        fi
    fi
    echo "" | tee -a "$LOG"

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

echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "Promoting best checkpoint: $BEST_CKPT (val $BEST_VAL)"        | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
cp "$CKPT_DIR/$BEST_CKPT" "$CKPT_DIR/best_model.pt"

if [[ "$LAST_CYCLE_WAS_FULL" == true && "$BEST_CKPT" == "$LAST_CKPT" ]]; then
    echo "Best checkpoint already ran the full battery last cycle — skipping redundant final battery." | tee -a "$LOG"
else
    echo "── Benchmark (full battery on best) ────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" -p "$CKPT_DIR/$BEST_CKPT" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
fi

echo "All done!  $(date)" | tee -a "$LOG"
ELAPSED=$(( $(date +%s) - START ))
echo "Total time: ${ELAPSED}s" | tee -a "$LOG"
