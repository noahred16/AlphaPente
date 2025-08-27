# Move-Count Based Valuation System

## Requirements
- Wins: Scale from 1.0 (fast win) to 0.2 (slow win)
- Losses: Scale from -1.0 (fast loss) to -0.2 (slow loss)
- Ties: Always 0.0
- Scale based on number of moves played

## Formula Design

### Scaling Function
For move count `m`, we want to map:
- Low move count → High absolute value (±1.0)
- High move count → Low absolute value (±0.2)

**Scaling Formula:**
```
scale_factor = 0.2 + 0.8 * exp(-m / max_moves_threshold)
```

Where:
- `max_moves_threshold` ≈ 50 (moves where scaling approaches minimum)
- For m=5: scale_factor ≈ 0.96 (fast game)
- For m=25: scale_factor ≈ 0.68 (medium game)  
- For m=50: scale_factor ≈ 0.49 (long game)
- For m=100: scale_factor ≈ 0.31 (very long game)

### Final Valuation
```python
def calculate_valuation(winner, original_player, move_count):
    if winner is None:
        return 0.0  # Tie
    
    # Calculate scaling factor (decreases with more moves)
    scale_factor = 0.2 + 0.8 * math.exp(-move_count / 50.0)
    
    if winner == original_player:
        return scale_factor  # Win (positive, scaled)
    else:
        return -scale_factor  # Loss (negative, scaled)
```

## Expected Behavior
- 10-move win: +0.85 (fast aggressive win)
- 50-move win: +0.49 (slow strategic win)  
- 100-move win: +0.31 (very slow win)
- 10-move loss: -0.85 (fast devastating loss)
- 50-move loss: -0.49 (slow defensive loss)
- Any tie: 0.0

This encourages:
1. **Aggressive play**: Seeking quick wins
2. **Defensive play**: Avoiding quick losses
3. **Risk management**: Balancing speed vs safety