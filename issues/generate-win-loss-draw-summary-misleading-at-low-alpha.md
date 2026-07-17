# Generate B/W/D summary is misleading at low valueBlendAlpha

## Where
`apps/Generate.cpp`, the per-game outcome tally used in the progress log:

```cpp
if (!examples.empty()) {
    float v0 = examples[0].value;
    if      (v0 >  0.5f) bWins++;
    else if (v0 < -0.5f) wWins++;
    else                 draws++;
}
```

## Problem
This buckets games by `examples[0].value` — the **blended** training target
(`valueBlendAlpha * z + (1 - valueBlendAlpha) * rootQ`) for the first move of
the game, not the actual game outcome `z`.

At low `valueBlendAlpha` (e.g. 0.2, used for the 2026-07-16 bootstrap
regeneration), the value is dominated by `rootQ`. The heuristic evaluator's
root Q at move 1 is essentially never above 0.5 in magnitude, so nearly every
game is bucketed as a "draw" in the log regardless of how the game actually
ended. Observed during that run: every batch reported `B/W/D: 0/0/100`, which
does not reflect the real game outcomes.

## Impact
Cosmetic only — the progress log is misleading but the stored training data
(`ex.value` for every example, derived from the same blend) is unaffected and
correct. No corruption of the buffer.

## Possible fix
Track the true outcome `z` separately from the blended value for display
purposes, e.g. have `runGame` also return/expose `winner` so `Generate.cpp`
can tally real W/L/D independent of `valueBlendAlpha`.

## Status
Not fixed — noise-only, deferred.
