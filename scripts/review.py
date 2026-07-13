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
