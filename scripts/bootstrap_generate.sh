#!/usr/bin/env bash
# Bootstrap data generation — run from project root.
# ~1.2s/game @ 25k sims. Positions ≈ games x moves kept; the 8 board
# symmetries are applied at train time (augmentBatch), not stored.

set -euo pipefail

GAME="${1:-pente}"
BATCHES=50
GAMES_PER_BATCH=50
SIMS=50000
TAIL=999

echo "Bootstrap generation"
echo "  game   : $GAME"
echo "  batches: $BATCHES x $GAMES_PER_BATCH games"
echo "  sims   : $SIMS"
echo "  tail   : $TAIL moves"
echo "  start  : $(date)"
echo ""

for i in $(seq 1 $BATCHES); do
    echo "════════════════════════════════════════════════════════════"
    echo "Batch $i / $BATCHES  —  $(date)"
    echo "════════════════════════════════════════════════════════════"
    build/generate -b -t $TAIL -n $GAMES_PER_BATCH -s $SIMS -g $GAME
    echo ""
done

echo "Done — $(date)"
