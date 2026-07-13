# purpose of this script is to do a quick review of the shapes of these pente training sets: bootstrap and buffer.

import torch

# Files are written by LibTorch's OutputArchive (see apps/TrainCommon.hpp),
# so load them as jit modules and read the named tensors.
def load_buffer(path):
    ar = torch.jit.load(path)
    return {name: getattr(ar, name) for name in ("states", "captures", "policies", "values")}

for name in ("bootstrap", "buffer"):
    buf = load_buffer(f"checkpoints/pente/{name}.pt")
    print(f"{name}:")
    for key, t in buf.items():
        print(f"  {key:<8} {tuple(t.shape)}  {t.dtype}")

# ── Metadata: game count and prefix diversity ───────────────────────────────
# Where the randomness comes from (scripts/bootstrap_generate.sh →
# apps/Generate.cpp → src/SelfPlay.cpp): for the first 15 moves
# (explorationDropoff) the move is SAMPLED proportional to MCTS visit counts
# (temperature 1) and Dirichlet(alpha=0.3) noise is mixed into the root prior
# at epsilon=0.25; from move 15 on, noise is off and play is argmax (unless a
# proven win exists, which is always played). So all game diversity comes from
# the first 15 plies.

def stone_counts(states):
    ident = states[::8]  # identity augmentation is every 8th sample
    return (ident[:, 0].sum((1, 2)) + ident[:, 1].sum((1, 2))).int().tolist()

# Within a game a move changes the stone count by 1 - 2*captured_stones
# (+1, -1, -3, ...); any other delta, or an empty board, starts a new game
# record. Records can be tail-trimmed (generate -t), so some start mid-game.
# (Approximate: a boundary between two trimmed records can look like a legal
# delta by coincidence.)
def segment_games(stones):
    games, start = [], 0
    for j in range(1, len(stones)):
        d = stones[j] - stones[j - 1]
        if stones[j] == 0 or d > 1 or (d - 1) % 2 != 0:
            games.append((start, j))
            start = j
    games.append((start, len(stones)))
    return games  # (begin, end) position-index ranges

def analyze(name):
    buf = load_buffer(f"checkpoints/pente/{name}.pt")
    states, values = buf["states"], buf["values"]
    stones = stone_counts(states)
    games = segment_games(stones)
    full = [(b, e) for b, e in games if stones[b] == 0]
    lengths = sorted(e - b for b, e in games)
    print(f"\n{name}: {states.size(0)} samples = {len(stones)} positions"
          f" (x8 symmetries) = {len(games)} games"
          f" ({len(full)} full, {len(games) - len(full)} tail-trimmed)")
    print(f"  record lengths: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")

    if not full:
        print("  (no full games — skipping outcome and diversity stats)")
        return

    # Outcome from the empty-board value of each full game. Value convention
    # (SelfPlay.cpp): +1 = the player who moved INTO the position wins, so at
    # the empty board -1 = first player won, +1 = second player won.
    w = [values[b * 8].item() for b, _ in full]
    p1 = sum(v < -0.5 for v in w)
    p2 = sum(v > 0.5 for v in w)
    print(f"  outcomes over {len(w)} full games  P1/P2/draw: {p1}/{p2}/{len(w) - p1 - p2}")

    # Prefix diversity: distinct positions after N moves, over full games.
    # raw = exact board+captures; mod-sym = up to the 8 board symmetries (min
    # hash over the stored augmentations) — mirrored/rotated games produce the
    # same 8 augmented training samples, so mod-sym is the honest count.
    print(f"  distinct positions after N moves:")
    print(f"    {'move':>4} {'games':>6} {'raw':>6} {'mod-sym':>8}")
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40):
        reach = [b + d for b, e in full if b + d < e]
        if not reach:
            break
        raw = {states[p * 8].numpy().tobytes() for p in reach}
        canon = {min(states[p * 8 + k].numpy().tobytes() for k in range(8)) for p in reach}
        print(f"    {d:>4} {len(reach):>6} {len(raw):>6} {len(canon):>8}")

for name in ("bootstrap", "buffer"):
    analyze(name)

# Peek at the first few bootstrap samples.
# states[i] planes: 0=my stones, 1=opp stones, 2=empty, 3=my caps/max, 4=opp caps/max
# planes and policy are indexed [y][x]; policy flat index = y*19 + x (see NNEvaluator.cpp)
COLS = "ABCDEFGHJKLMNOPQRST"  # Pente notation skips 'I'

def move_name(idx):
    y, x = divmod(idx, 19)
    return f"{COLS[x]}{y + 1}"

bootstrap = load_buffer("checkpoints/pente/bootstrap.pt")
# each position is stored as 8 consecutive symmetry augmentations, so sample
# index = position * 8 walks the buffer one position at a time. Note: games are
# stored as tails only (generate -t 20 keeps the last 20 moves per game), so a
# record can start mid-game. A legal move adds 1 stone (minus 2 per capture);
# any other stone-count delta means we've crossed into the next game's record.
prev_stones = None
num_of_games = 80 
for pos in range(num_of_games):
    i = pos * 8
    state = bootstrap["states"][i]
    caps = bootstrap["captures"][i]
    value = bootstrap["values"][i].item()
    policy = bootstrap["policies"][i]

    # # for the first one, lets print the raw values just to catch a feel. 
    # if pos == 10 or pos == 14:
    #     print(f"Raw state:\n{state}")
    #     print(f"Raw captures: {caps}")
    #     print(f"Raw value: {value}")
    #     print(f"Raw policy: {policy}")

    stones = int(state[0].sum() + state[1].sum())
    if prev_stones is not None:
        diff = stones - prev_stones
        if diff > 1 or (diff - 1) % 2 != 0:
            print(f"\n(end of first game record after {pos} positions — next record starts mid-game)")
            break
    prev_stones = stones

    print(f"\n=== position {pos + 1} ({stones} stones) ===  (X = to-move, O = opponent)")
    for y in range(18, -1, -1):
        row = "".join("X" if state[0, y, x] else "O" if state[1, y, x] else "." for x in range(19))
        print(f"{y + 1:>2} {row}")
    print("   " + COLS)
    print(f"captures (my, opp): {caps.tolist()}   value: {value:+.3f}")

    top = policy.topk(5)
    moves = ", ".join(f"{move_name(idx)}={p:.3f}" for p, idx in zip(top.values.tolist(), top.indices.tolist()))
    print(f"top policy moves: {moves}")

# Verify the all-zero policy rows: a valid policy target should sum to 1,
# but some positions appear to have no visit distribution at all.
for name in ("bootstrap", "buffer"):
    pol = load_buffer(f"checkpoints/pente/{name}.pt")["policies"]
    sums = pol.sum(1)
    zero_rows = (sums == 0).nonzero().flatten()
    ok_rows = ((sums - 1).abs() < 1e-4).sum().item()
    n = pol.size(0)
    print(f"\n{name}: {n} rows — sum≈1: {ok_rows}, all-zero: {zero_rows.numel()} "
          f"({100.0 * zero_rows.numel() / n:.1f}%), other: {n - ok_rows - zero_rows.numel()}")
    if zero_rows.numel():
        first = zero_rows[0].item()
        row = pol[first]
        assert row.count_nonzero() == 0, "expected exactly zero everywhere"
        print(f"first zero row is sample {first} (position {first // 8}): "
              f"min={row.min().item()}, max={row.max().item()}, nonzero={row.count_nonzero().item()}")
        print("raw 19x19 policy of that sample:")
        print(row.view(19, 19))

# bootstrap:
#   states   (614744, 5, 19, 19)  torch.float32
#   captures (614744, 2)  torch.float32
#   policies (614744, 361)  torch.float32
#   values   (614744, 1)  torch.float32
# buffer:
#   states   (123280, 5, 19, 19)  torch.float32
#   captures (123280, 2)  torch.float32
#   policies (123280, 361)  torch.float32
#   values   (123280, 1)  torch.float32
