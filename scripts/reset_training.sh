#!/usr/bin/env bash
# Wipe checkpoints and benchmark data for a game, resetting to iteration 0.
# Usage: ./scripts/reset_training.sh [-g game]
# Run from anywhere inside the repo.

set -euo pipefail

GAME="pente"

while getopts "g:" opt; do
    case $opt in
        g) GAME=$OPTARG ;;
        *) echo "Usage: $0 [-g game]" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints/$GAME"
REPORT_DIR="$REPO_ROOT/reports/$GAME"

echo "Resetting training for game: $GAME"
echo ""
echo "Will delete:"
[[ -d "$CKPT_DIR" ]] && ls "$CKPT_DIR" | sed "s|^|  $CKPT_DIR/|"
[[ -d "$REPORT_DIR" ]] && ls "$REPORT_DIR" | sed "s|^|  $REPORT_DIR/|"
echo ""

read -r -p "Confirm? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

rm -f "$CKPT_DIR"/model_iter*.pt
rm -f "$CKPT_DIR"/best_model.pt
rm -f "$CKPT_DIR"/buffer.pt
rm -f "$REPORT_DIR"/benchmark.csv

echo "Done. Run ./train (or train_loop.sh) to start fresh."
