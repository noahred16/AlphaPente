#!/usr/bin/env bash
# Self-play training loop — generates games, trains, and benchmarks each iteration.
# Run from anywhere in the repo.
# Usage: ./scripts/train_loop.sh [-n games] [-s sims] [-i max_iters] [-g game]

set -euo pipefail

GAMES=200
SIMS=800
MAX_ITERS=0          # 0 = run forever
GAME="pente"
ARENA=false
ARENA_GAMES=10

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n games] [-s sims] [-i max_iters] [-g game] [-a]

Options:
  -n  self-play games per iteration        (default: $GAMES)
  -s  MCTS simulations per move            (default: $SIMS)
  -i  max iterations, 0 = run forever      (default: $MAX_ITERS)
  -g  game: pente | gomoku | keryopente    (default: $GAME)
  -a  run arena evaluation after each iteration

Examples:
  # standard self-play training
  ./scripts/train_loop.sh -n 50 -s 800 -g pente

  # with arena enabled (NN vs heuristic at 3 sim tiers + any roster models)
  ./scripts/train_loop.sh -n 50 -s 800 -g pente -a

  # quick smoke-test: 3 iterations, small game count
  ./scripts/train_loop.sh -n 10 -s 100 -i 3 -g pente
EOF
}

for arg in "$@"; do [[ "$arg" == "--help" ]] && { usage; exit 0; }; done

while getopts "n:s:i:g:ah" opt; do
    case $opt in
        n) GAMES=$OPTARG ;;
        s) SIMS=$OPTARG ;;
        i) MAX_ITERS=$OPTARG ;;
        g) GAME=$OPTARG ;;
        a) ARENA=true ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

BUILD_DIR="$(cd "$(dirname "$0")/../build" && pwd)"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
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

    ./generate -g "$GAME" -n "$GAMES" -s "$SIMS" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    train_out=$(./train -g "$GAME" 2>&1)
    echo "$train_out" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    promoted=true
    echo "$train_out" | grep -q "PROMOTED" || promoted=false

    echo "── Benchmark (raw policy) ───────────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" -s 0 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "── Benchmark (MCTS $SIMS sims) ──────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" -s "$SIMS" 2>&1 | tee -a "$LOG"

    if [[ "$ARENA" == true ]]; then
        echo "" | tee -a "$LOG"
        echo "── Arena ────────────────────────────────────────────────────────" | tee -a "$LOG"
        for tier_sims in 100 400 1600; do
            echo "  vs heuristic @ ${tier_sims} sims" | tee -a "$LOG"
            ./benchmark -g "$GAME" -a -G "$ARENA_GAMES" -S "$SIMS" -T "$tier_sims" 2>&1 | tee -a "$LOG"
            echo "" | tee -a "$LOG"
        done
        ROSTER_DIR="$ROOT_DIR/checkpoints/$GAME/roster"
        if [[ -d "$ROSTER_DIR" ]]; then
            shopt -s nullglob
            for model in "$ROSTER_DIR"/*.pt; do
                name=$(basename "$model" .pt)
                echo "  vs roster: $name" | tee -a "$LOG"
                ./benchmark -g "$GAME" -a -P "$model" -G "$ARENA_GAMES" -S "$SIMS" 2>&1 | tee -a "$LOG"
                echo "" | tee -a "$LOG"
            done
            shopt -u nullglob
        fi
    fi

    # ── Roster promotion ─────────────────────────────────────────────────
    if [[ "$promoted" == false ]]; then
        echo "" | tee -a "$LOG"
        echo "── Roster promotion skipped — validation gate rejected this iteration (best_model.pt unchanged) ──" | tee -a "$LOG"
    else
        ROSTER_DIR="$ROOT_DIR/checkpoints/$GAME/roster"
        CURRENT_MODEL=$(ls -t "$ROOT_DIR/checkpoints/$GAME/model_iter"*.pt 2>/dev/null | head -1)

        if [[ -n "$CURRENT_MODEL" ]]; then
            mkdir -p "$ROSTER_DIR"
            shopt -s nullglob
            roster_models=("$ROSTER_DIR"/*.pt)
            shopt -u nullglob

            promote=true
            for model in "${roster_models[@]}"; do
                [[ "$(basename "$model")" == "$(basename "$CURRENT_MODEL")" ]] && continue
                name=$(basename "$model" .pt)
                echo "" | tee -a "$LOG"
                echo "── Roster check: $name ──────────────────────────────────────────" | tee -a "$LOG"
                arena_out=$(./benchmark -g "$GAME" -a -G "$ARENA_GAMES" -S "$SIMS" -P "$model" 2>&1)
                echo "$arena_out" | tee -a "$LOG"
                nn_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $4}')
                opp_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $6}')
                if [[ -z "$nn_wins" || "$nn_wins" -le "$opp_wins" ]]; then
                    promote=false
                    echo "  → did not beat $name — not added to roster (training continues unaffected)" | tee -a "$LOG"
                    break
                fi
            done

            if [[ "$promote" == true ]]; then
                model_name=$(basename "$CURRENT_MODEL")
                cp "$CURRENT_MODEL" "$ROSTER_DIR/$model_name"
                echo "" | tee -a "$LOG"
                echo "★ New roster member: $model_name" | tee -a "$LOG"
            fi
        fi
    fi

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
