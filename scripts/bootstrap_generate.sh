#!/usr/bin/env bash
# Bootstrap data generation — run from project root.
# ~1.2s/game @ 25k sims. 16 batches x 100 games = 1,600 games, ~32 minutes.
# Produces ~256,000 positions (1,600 games x 20 tail moves x 8 symmetries).

set -euo pipefail

GAME="${1:-pente}"
BATCHES=7
GAMES_PER_BATCH=100
SIMS=25000
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
    build/generate -b -t $TAIL -n $GAMES_PER_BATCH -s $SIMS -g $GAME -a
    echo ""
done

echo "Done — $(date)"
