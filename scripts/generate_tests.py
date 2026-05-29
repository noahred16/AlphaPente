"""
generate_tests.py — one-shot test suite generator for AlphaPente benchmarks.

Usage:
    python scripts/generate_tests.py

Outputs:
    tests/open-three-suite.json

Each entry:
    {
        "state":    "1. K10 C7 2. H7 ...",   # game string up to the position
        "expected": ["G7", "K7"]               # both open ends of the open-3
    }

Board: 19x19. Columns A–S skipping I (so A–H, J–S). Rows 1–19.
Center: K10 = (9, 9) in 0-indexed (x, y).

Tournament rule: Black's second stone (move 3 overall) must be on the perimeter
of the 7x7 square centred on K10, i.e. Chebyshev distance exactly 3.

Game structure (7 moves played, White to move):
    B1 = K10   (required first move)
    W1 = random
    B2 = one open-3 stone on the tournament perimeter
    W2 = random
    B3 = second open-3 stone
    W3 = random
    B4 = third open-3 stone
"""

import json
import os
import random

BOARD_SIZE = 19
CENTER = (9, 9)
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


# ── Coordinate helpers ────────────────────────────────────────────────────────

def display_move(x: int, y: int) -> str:
    col = ord("A") + x
    if col >= ord("I"):
        col += 1  # skip 'I'
    return chr(col) + str(y + 1)


def chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def on_board(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def on_tournament_perimeter(pos) -> bool:
    return chebyshev(pos, CENTER) == 3


# ── Capture detection ─────────────────────────────────────────────────────────

def would_capture(placing, own: set, opp: set) -> bool:
    """True if placing a stone at `placing` (same colour as `own`) would capture
    two opponent stones. Pattern in any direction: placing, opp, opp, own."""
    px, py = placing
    for dx, dy in DIRECTIONS:
        for s in (1, -1):
            a1 = (px + s * dx,     py + s * dy)
            a2 = (px + 2 * s * dx, py + 2 * s * dy)
            a3 = (px + 3 * s * dx, py + 3 * s * dy)
            if a1 in opp and a2 in opp and a3 in own:
                return True
    return False


# ── Open-3 suite ──────────────────────────────────────────────────────────────

def _try_generate(rng: random.Random):
    """Attempt to build one open-3 test case. Returns a dict or None."""
    direction = rng.choice(DIRECTIONS)
    dx, dy = direction

    for _ in range(300):
        px = rng.randint(0, BOARD_SIZE - 1)
        py = rng.randint(0, BOARD_SIZE - 1)

        stones = [(px + i * dx, py + i * dy) for i in range(3)]
        end1   = (px - dx,       py - dy)
        end2   = (px + 3 * dx,   py + 3 * dy)

        if not all(on_board(x, y) for x, y in stones + [end1, end2]):
            continue

        # K10 must stay clear — it holds Black's first stone and cannot be an open end
        if CENTER in stones or CENTER in (end1, end2):
            continue

        # One open-3 stone must satisfy the tournament perimeter rule for B2
        perimeter = [s for s in stones if on_tournament_perimeter(s)]
        if not perimeter:
            continue

        b2 = rng.choice(perimeter)
        rest = [s for s in stones if s != b2]
        rng.shuffle(rest)
        b3, b4 = rest

        all_black       = {CENTER, b2, b3, b4}
        must_stay_empty = {end1, end2}
        forbidden       = all_black | must_stay_empty

        pool = [(x, y)
                for x in range(BOARD_SIZE)
                for y in range(BOARD_SIZE)
                if (x, y) not in forbidden]
        rng.shuffle(pool)

        # Pick W1, W2, W3 in game order, checking captures at each step
        b_seq = [b2, b3, b4]
        white = []
        for w_idx, candidate in enumerate(pool):
            if len(white) == 3:
                break
            black_before = {CENTER} | set(b_seq[:w_idx])
            white_before = set(white)

            if would_capture(candidate, white_before, black_before):
                continue
            if w_idx < len(b_seq):
                if would_capture(b_seq[w_idx], black_before, white_before | {candidate}):
                    continue

            white.append(candidate)

        if len(white) < 3:
            continue

        w1, w2, w3 = white
        seq = [CENTER, w1, b2, w2, b3, w3, b4]
        parts = []
        for i in range(0, len(seq), 2):
            num = i // 2 + 1
            if i + 1 < len(seq):
                parts.append(f"{num}. {display_move(*seq[i])} {display_move(*seq[i+1])}")
            else:
                parts.append(f"{num}. {display_move(*seq[i])}")

        return {
            "state":    " ".join(parts),
            "expected": [display_move(*end1), display_move(*end2)],
        }

    return None


def generate_open_three_suite(n: int = 250, seed: int = 42) -> list:
    rng = random.Random(seed)
    results = []
    attempts = 0

    while len(results) < n:
        attempts += 1
        if attempts > n * 2000:
            print(f"Warning: hit attempt limit, only generated {len(results)}/{n}")
            break
        tc = _try_generate(rng)
        if tc:
            results.append(tc)

    print(f"Generated {len(results)}/{n} cases in {attempts} attempts")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite = generate_open_three_suite(250)

    out_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "tests", "open-three-suite.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(suite, f, indent=2)

    print(f"Written to {out_path}")
