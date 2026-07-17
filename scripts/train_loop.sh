#!/usr/bin/env bash
# Self-play training loop — generates games, trains, and benchmarks each iteration.
# Run from anywhere in the repo.
# Usage: ./scripts/train_loop.sh [-n games] [-s sims] [-i max_iters] [-g game]

set -euo pipefail

GAMES=200
SIMS=1600
MAX_ITERS=0          # 0 = run forever
GAME="pente"
ARENA=false  # informational only (3-tier heuristic + full roster sweep every iteration);
             # the promotion arena and roster-check gates below always run regardless
ARENA_GAMES=30
ARENA_SIMS=400  # promotion/roster arenas use this for both sides instead of $SIMS

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n games] [-s sims] [-i max_iters] [-g game]

Options:
  -n  self-play games per iteration        (default: $GAMES)
  -s  MCTS simulations per move            (default: $SIMS)
  -i  max iterations, 0 = run forever      (default: $MAX_ITERS)
  -g  game: pente | gomoku | keryopente    (default: $GAME)

Each iteration runs a promotion arena (candidate vs previous best_model.pt)
after training; roster growth/checks are not part of the self-play loop.

Examples:
  # standard self-play training
  ./scripts/train_loop.sh -n 50 -s 800 -g pente

  # quick smoke-test: 3 iterations, small game count
  ./scripts/train_loop.sh -n 10 -s 100 -i 3 -g pente
EOF
}

for arg in "$@"; do [[ "$arg" == "--help" ]] && { usage; exit 0; }; done

while getopts "n:s:i:g:h" opt; do
    case $opt in
        n) GAMES=$OPTARG ;;
        s) SIMS=$OPTARG ;;
        i) MAX_ITERS=$OPTARG ;;
        g) GAME=$OPTARG ;;
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

    CKPT_DIR="$ROOT_DIR/checkpoints/$GAME"
    PREV_BEST="$CKPT_DIR/best_model.pt.prev"
    cp "$CKPT_DIR/best_model.pt" "$PREV_BEST"

    train_out=$(./train -g "$GAME" 2>&1)
    echo "$train_out" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    promoted=true
    echo "$train_out" | grep -q "PROMOTED" || promoted=false

    # Benchmark the model this iteration just trained, not necessarily
    # best_model.pt (which now defaults for ./benchmark, but may still be the
    # previous iteration's model if this candidate isn't promoted below).
    CANDIDATE=$(ls -t "$CKPT_DIR/model_iter"*.pt 2>/dev/null | head -1)

    echo "── Benchmark (raw policy) ───────────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" -s 0 -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "── Benchmark (MCTS $SIMS sims) ──────────────────────────────────" | tee -a "$LOG"
    ./benchmark -g "$GAME" -s "$SIMS" -p "$CANDIDATE" 2>&1 | tee -a "$LOG"

    if [[ "$ARENA" == true ]]; then
        echo "" | tee -a "$LOG"
        echo "── Arena ────────────────────────────────────────────────────────" | tee -a "$LOG"
        for tier_sims in 100 400 1600; do
            echo "  vs heuristic @ ${tier_sims} sims" | tee -a "$LOG"
            ./benchmark -g "$GAME" -a -G "$ARENA_GAMES" -S "$SIMS" -T "$tier_sims" -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
            echo "" | tee -a "$LOG"
        done
        ROSTER_DIR="$ROOT_DIR/checkpoints/$GAME/roster"
        if [[ -d "$ROSTER_DIR" ]]; then
            shopt -s nullglob
            for model in "$ROSTER_DIR"/*.pt; do
                name=$(basename "$model" .pt)
                echo "  vs roster: $name" | tee -a "$LOG"
                ./benchmark -g "$GAME" -a -P "$model" -G "$ARENA_GAMES" -S "$SIMS" -p "$CANDIDATE" 2>&1 | tee -a "$LOG"
                echo "" | tee -a "$LOG"
            done
            shopt -u nullglob
        fi
    fi

    # ── Promotion arena: val-loss gate passing isn't enough on its own — it's
    # measured against a held-out slice of the model's own tiny recent self-play
    # buffer, which can keep looking "improved" even while real playing strength
    # regresses. Require an actual arena win over the previous best_model.pt too.
    if [[ "$promoted" == true ]]; then
        echo "" | tee -a "$LOG"
        echo "── Promotion arena: candidate vs previous best_model.pt ────────────" | tee -a "$LOG"
        arena_out=$(./benchmark -g "$GAME" -a -G "$ARENA_GAMES" -S "$ARENA_SIMS" -P "$PREV_BEST" -p "$CANDIDATE" 2>&1)
        echo "$arena_out" | tee -a "$LOG"
        nn_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $4}')
        opp_wins=$(echo "$arena_out" | awk '/Arena result:/ {print $6}')
        decisive=$(( ${nn_wins:-0} + ${opp_wins:-0} ))
        # Promote at >=45% win rate rather than requiring an outright win/tie —
        # a slightly-behind candidate on a 30-40 game sample is still within
        # noise of "roughly as strong," and self-play iterates fast enough
        # that a genuinely weaker model gets overwritten again soon anyway.
        if [[ -z "$nn_wins" || "$decisive" -eq 0 || $(( nn_wins * 100 )) -lt $(( 45 * decisive )) ]]; then
            cp "$PREV_BEST" "$CKPT_DIR/best_model.pt"
            promoted=false
            echo "  → candidate did not reach 45% vs previous best_model.pt ($nn_wins-$opp_wins) — reverted, best_model.pt unchanged" | tee -a "$LOG"
        else
            echo "  → candidate reached $(( decisive > 0 ? nn_wins * 100 / decisive : 0 ))% vs previous best_model.pt ($nn_wins-$opp_wins) — promotion confirmed" | tee -a "$LOG"
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
