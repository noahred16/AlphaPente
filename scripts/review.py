# purpose of this script is to do a quick review of the shapes of these pente training sets: bootstrap and buffer.
#
# usage: review.py [bu|bo] [-i]
#   (no arg)  review both bootstrap and buffer
#   bu        review buffer only
#   bo        review bootstrap only
#   -i        interactive game review: one position per screen, n=next b=back q=quit

import sys

import torch

_MODES = {"bu": ("buffer",), "bo": ("bootstrap",)}
args = sys.argv[1:]
INTERACTIVE = "-i" in args and sys.stdin.isatty()
args = [a for a in args if a != "-i"]
if args:
    if args[0] not in _MODES:
        sys.exit(f"usage: {sys.argv[0]} [bu|bo] [-i]")
    BUFFERS = _MODES[args[0]]
else:
    BUFFERS = ("bootstrap", "buffer")

# Files are written by LibTorch's OutputArchive (see apps/TrainCommon.hpp),
# so load them as jit modules and read the named tensors.
def load_buffer(path):
    ar = torch.jit.load(path)
    return {name: getattr(ar, name) for name in ("states", "captures", "policies", "values")}

for name in BUFFERS:
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
    return (states[:, 0].sum((1, 2)) + states[:, 1].sum((1, 2))).int().tolist()

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

# The 8 board symmetries of one position (they're applied at train time now,
# not stored, so diversity-mod-symmetry has to compute them here).
def sym8(t):  # [C, 19, 19] tensor
    for flip in (False, True):
        base = t.flip(2) if flip else t
        for rot in range(4):
            yield torch.rot90(base, rot, dims=(1, 2)) if rot else base

def analyze(name):
    buf = load_buffer(f"checkpoints/pente/{name}.pt")
    states, captures, values = buf["states"], buf["captures"], buf["values"]
    stones = stone_counts(states)
    games = segment_games(stones)
    full = [(b, e) for b, e in games if stones[b] == 0]
    lengths = sorted(e - b for b, e in games)
    print(f"\n{name}: {len(stones)} positions = {len(games)} games"
          f" ({len(full)} full, {len(games) - len(full)} tail-trimmed)")
    print(f"  record lengths: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")

    if not full:
        print("  (no full games — skipping outcome and diversity stats)")
        return

    # Outcome from the empty-board z of each full game (values are stored as
    # [N, 2] = (z, rootQ), unblended). Convention (SelfPlay.cpp): +1 = the
    # player who moved INTO the position wins, so at the empty board -1 =
    # first player won, +1 = second player won.
    w = [values[b, 0].item() for b, _ in full]
    p1 = sum(v < -0.5 for v in w)
    p2 = sum(v > 0.5 for v in w)
    print(f"  outcomes over {len(w)} full games  P1/P2/draw: {p1}/{p2}/{len(w) - p1 - p2}")

    # Prefix diversity: distinct positions after N moves, over full games.
    # raw = exact board+captures; mod-sym = up to the 8 board symmetries (min
    # hash over the computed symmetries) — mirrored/rotated games train the
    # same after augmentation, so mod-sym is the honest count.
    print(f"  distinct positions after N moves:")
    print(f"    {'move':>4} {'games':>6} {'raw':>6} {'mod-sym':>8}")
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40):
        reach = [b + d for b, e in full if b + d < e]
        if not reach:
            break
        raw = {states[p].numpy().tobytes() + captures[p].numpy().tobytes()
               for p in reach}
        canon = {min(s.contiguous().numpy().tobytes() for s in sym8(states[p]))
                 + captures[p].numpy().tobytes()
                 for p in reach}
        print(f"    {d:>4} {len(reach):>6} {len(raw):>6} {len(canon):>8}")

for name in BUFFERS:
    analyze(name)

# Peek at the last game record in each buffer.
# states[i] planes (compact uint8 format): 0=my stones, 1=opp stones
# (empty + capture planes are reconstructed at train time, see TrainCommon.hpp)
# planes and policy are indexed [y][x]; policy flat index = y*19 + x (see NNEvaluator.cpp)
COLS = "ABCDEFGHJKLMNOPQRST"  # Pente notation skips 'I'

def move_name(idx):
    y, x = divmod(idx, 19)
    return f"{COLS[x]}{y + 1}"

# No move history is stored, but it can be recovered: a state's plane 0 is
# always "to-move"'s stones and plane 1 "opponent"'s, so identity flips every
# ply. The stone just placed is exactly the one in plane 1 of pos+1 that
# wasn't already in plane 0 of pos (everything else there is the mover's own
# older stones, carried over minus whatever this move captured).
def infer_moves(states, begin, end):
    moves = []
    for pos in range(begin, end - 1):
        diff = (states[pos + 1, 1].int() - states[pos, 0].int()).flatten()
        idx = (diff == 1).nonzero()
        moves.append(move_name(idx[0].item()) if idx.numel() else "?")
    return moves

# Round-numbered notation, e.g. "1. K10 L9 2. K12 M10" (see CLAUDE.md). Pente's
# first move is always the center (K10), so for full games moves[0] is
# reliably player 1's opener and the alternation below lines up correctly.
def format_moves(moves):
    pairs = (f"{i // 2 + 1}. " + " ".join(moves[i:i + 2]) for i in range(0, len(moves), 2))
    return " ".join(pairs)

def print_position(buf, stones_all, move_history, begin, pos):
    state = buf["states"][pos]
    caps = buf["captures"][pos]
    z, q = buf["values"][pos].tolist()
    policy = buf["policies"][pos]
    stones = stones_all[pos]

    # Planes/captures are (my, opp) from the to-move player's view; "my" is
    # Black iff the stone count is even (Black moves first and parity flips
    # every ply — captures remove stones in pairs). Stone glyphs match the C++
    # app (GameUtils::printBoard): ○ = Black, ● = White.
    black_to_move = stones % 2 == 0
    mine, theirs = ("○", "●") if black_to_move else ("●", "○")
    print(f"\n=== position {pos - begin + 1} ({stones} stones, {'Black ○' if black_to_move else 'White ●'} to move) ===")
    for y in range(18, -1, -1):
        row = " ".join(mine if state[0, y, x] else theirs if state[1, y, x] else "·" for x in range(19))
        print(f"{y + 1:>2} {row}")
    print("   " + " ".join(COLS))
    print(f"moves: {format_moves(move_history[:pos - begin]) or '(none yet)'}")
    # caps normalized by capturesToWin (10 for Pente)
    my, opp = (round(c * 10) for c in caps.tolist())
    black, white = (my, opp) if black_to_move else (opp, my)
    print(f"captures: {black}/10 Black ○, {white}/10 White ●   z: {z:+.0f}   rootQ: {q:+.3f}")
    # Train-time value target a*z + (1-a)*rootQ across the alpha sweep
    # (a=0 is pure search estimate, a=1 pure game outcome; train -a picks one).
    blends = "  ".join(f"{a / 10:.1f}:{(a / 10) * z + (1 - a / 10) * q:+.2f}"
                       for a in range(11))
    print(f"target by alpha: {blends}")

    top = policy.topk(5)
    moves = ", ".join(f"{move_name(idx)}={p:.3f}" for p, idx in zip(top.values.tolist(), top.indices.tolist()))
    print(f"top policy moves: {moves}")

def read_key():
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, termios.TCSADRAIN)  # DRAIN keeps typed-ahead keys (default FLUSH drops them)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# Positions are stored once, in game order (symmetries applied at train time).
# Reuses stone_counts/segment_games (defined above) to find the last game's
# record rather than re-deriving boundaries here.
def print_game(name):
    buf = load_buffer(f"checkpoints/pente/{name}.pt")
    stones_all = stone_counts(buf["states"])
    begin, end = segment_games(stones_all)[-1]
    full = stones_all[begin] == 0
    tag = "" if full else ", tail-trimmed — starts mid-game"
    header = f"\n── Last game record: {name} ({end - begin} positions{tag}) ──────────────────────────"

    move_history = infer_moves(buf["states"], begin, end)

    if not INTERACTIVE:
        print(header)
        for pos in range(begin, end):
            print_position(buf, stones_all, move_history, begin, pos)
        return

    pos = begin
    while True:
        print("\033[2J\033[H", end="")  # clear screen, cursor home
        print(header)
        print_position(buf, stones_all, move_history, begin, pos)
        hint = "   (end of game)" if pos == end - 1 else ""
        print(f"\n[n] next   [b] back   [q] quit{hint}", end="", flush=True)
        key = read_key()
        if key == "n":
            pos = min(end - 1, pos + 1)
        elif key == "b":
            pos = max(begin, pos - 1)
        elif key == "q":  # done with this game; next buffer's game, if any
            print()
            return
        elif key == "\x03":  # Ctrl-C exits entirely
            print()
            sys.exit(0)

for name in BUFFERS:
    print_game(name)

# Verify the all-zero policy rows: a valid policy target should sum to 1,
# but some positions appear to have no visit distribution at all.
for name in BUFFERS:
    pol = load_buffer(f"checkpoints/pente/{name}.pt")["policies"].float()
    sums = pol.sum(1)
    zero_rows = (sums == 0).nonzero().flatten()
    ok_rows = ((sums - 1).abs() < 1e-2).sum().item()  # fp16 storage rounds each entry
    n = pol.size(0)
    print(f"\n{name}: {n} rows — sum≈1: {ok_rows}, all-zero: {zero_rows.numel()} "
          f"({100.0 * zero_rows.numel() / n:.1f}%), other: {n - ok_rows - zero_rows.numel()}")
    if zero_rows.numel():
        first = zero_rows[0].item()
        row = pol[first]
        assert row.count_nonzero() == 0, "expected exactly zero everywhere"
        print(f"first zero row is position {first}: "
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
