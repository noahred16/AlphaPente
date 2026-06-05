#!/usr/bin/env python3
"""Pente opening book decision tree visualizer.

Dependencies:
    pip install graphviz matplotlib pandas
    apt install graphviz   (or brew install graphviz on Mac)

Usage:
    python opening_book.py opening_moves_L9.csv
    python opening_book.py opening_moves_L9.csv -o my_tree
"""

import argparse
import os
import tempfile

import graphviz
import matplotlib.pyplot as plt
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CROP_CENTER = "K10"   # board square to center the crop window on
CROP_RADIUS = 5       # squares in each direction → (2*CROP_RADIUS+1)² board shown
MIN_GAMES   = 0       # prune edges with fewer games than this (0 = show all)
CSV_PATH    = "opening_moves.csv"
OUTPUT_FILE = "opening_tree"
OUTPUT_FMT  = "pdf"
NODE_DPI    = 120
NODE_W_IN   = 2.8    # graphviz node width  (inches)
NODE_H_IN   = 3.5    # graphviz node height (inches)
# ──────────────────────────────────────────────────────────────────────────────


def parse_move(s: str) -> tuple[int, int]:
    """'K10' → (col, row) both 0-based."""
    s = s.strip()
    return ord(s[0].upper()) - ord('A'), int(s[1:]) - 1


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['game_state'] = df['game_state'].str.strip()
    df['move'] = df['move'].astype(str).str.strip()
    return df


def build_tree(df: pd.DataFrame) -> dict:
    """Returns {game_state: {move: row_dict}, ...}"""
    tree: dict = {}
    for _, row in df.iterrows():
        if MIN_GAMES and row['number_of_games'] < MIN_GAMES:
            continue
        tree.setdefault(row['game_state'], {})[row['move']] = row.to_dict()
    return tree


def _fmt(val, fmt: str, fallback: str = 'N/A') -> str:
    try:
        if pd.isna(val):
            return fallback
        return format(val, fmt)
    except Exception:
        return fallback


def render_node(game_state: str, highlight: str, stats: dict, path: str) -> None:
    cc, cr  = parse_move(CROP_CENTER)
    col_min = cc - CROP_RADIUS
    row_min = cr - CROP_RADIUS
    win_sz  = 2 * CROP_RADIUS + 1

    fig = plt.figure(figsize=(NODE_W_IN, NODE_H_IN), facecolor='white')
    ax_b = fig.add_axes([0.08, 0.26, 0.86, 0.70])  # board: upper 70%
    ax_s = fig.add_axes([0.00, 0.00, 1.00, 0.23])  # stats: lower 23%

    # ── Board ─────────────────────────────────────────────────────────────
    ax_b.set_facecolor('#DCBA6A')
    ax_b.set_xlim(-0.5, win_sz - 0.5)
    ax_b.set_ylim(-0.5, win_sz - 0.5)
    ax_b.set_aspect('equal')
    for i in range(win_sz):
        ax_b.axhline(i, color='#3a2a00', lw=0.5, zorder=1)
        ax_b.axvline(i, color='#3a2a00', lw=0.5, zorder=1)

    ax_b.set_xticks(range(win_sz))
    ax_b.set_xticklabels([chr(ord('A') + col_min + i) for i in range(win_sz)], fontsize=5)
    ax_b.set_yticks(range(win_sz))
    ax_b.set_yticklabels([str(row_min + 1 + i) for i in range(win_sz)], fontsize=5)
    ax_b.tick_params(length=0)

    moves = [m.strip() for m in game_state.split(',') if m.strip()]
    for i, mv in enumerate(moves):
        col, row = parse_move(mv)
        x, y = col - col_min, row - row_min
        if not (0 <= x < win_sz and 0 <= y < win_sz):
            continue
        stone = plt.Circle(
            (x, y), 0.38,
            fc='white' if i % 2 == 0 else 'black',
            ec='black', lw=1.0, zorder=3,
        )
        ax_b.add_patch(stone)
        if mv == highlight:
            ax_b.add_patch(plt.Circle((x, y), 0.44, fill=False, ec='red', lw=2.0, zorder=4))

    # ── Stats ─────────────────────────────────────────────────────────────
    ax_s.axis('off')

    mn    = stats.get('move_number')
    mn_s  = _fmt(mn, '.0f') if mn is not None else 'N/A'
    g_s   = _fmt(stats.get('number_of_games'), ',.0f')
    w_s   = _fmt(stats.get('win_percentage'), '.1%')
    v_s   = _fmt(stats.get('value'), '.4f')

    ax_s.text(
        0.5, 0.65,
        f"{highlight}\nMove #{mn_s}  Games: {g_s}\nWin%: {w_s}   Value: {v_s}",
        ha='center', va='center', fontsize=6.5,
        family='monospace', transform=ax_s.transAxes,
    )

    fig.savefig(path, dpi=NODE_DPI, facecolor='white')
    plt.close(fig)


def nid(gs: str) -> str:
    """Stable graphviz node ID from a game state string."""
    return gs.replace(',', '_').replace(' ', '')


def _best_child_state(gs: str, moves_dict: dict) -> str:
    """Return the child game_state with the highest value; fall back to first child."""
    best_gs, best_val = None, None
    for mv, row in moves_dict.items():
        child = gs + ',' + mv
        try:
            val = float(row.get('value'))
            if not pd.isna(val) and (best_val is None or val > best_val):
                best_val, best_gs = val, child
        except (TypeError, ValueError):
            pass
    return best_gs if best_gs is not None else gs + ',' + next(iter(moves_dict))


def build_graph(tree: dict, tmp: str) -> graphviz.Digraph:
    all_children = {gs + ',' + mv for gs, moves in tree.items() for mv in moves}
    roots = [gs for gs in tree if gs not in all_children]

    best_nodes = {_best_child_state(gs, moves) for gs, moves in tree.items() if moves}

    dot = graphviz.Digraph(
        'opening_book',
        graph_attr={'rankdir': 'TB', 'ranksep': '0.5', 'nodesep': '0.3', 'splines': 'false'},
    )

    visited: set = set()

    def add_node(gs: str, highlight: str, stats: dict) -> None:
        if gs in visited:
            return
        visited.add(gs)
        img = os.path.join(tmp, nid(gs) + '.png')
        render_node(gs, highlight, stats, img)
        is_best = gs in best_nodes
        dot.node(
            nid(gs), label='', image=img,
            shape='box', imagescale='true',
            fixedsize='true', width=str(NODE_W_IN), height=str(NODE_H_IN),
            style='filled', fillcolor='lightyellow' if is_best else 'white',
            color='goldenrod' if is_best else 'black',
            penwidth='3' if is_best else '1',
        )

    for root in roots:
        moves = [m.strip() for m in root.split(',') if m.strip()]
        add_node(root, moves[-1] if moves else '', {})

    queue = list(roots)
    while queue:
        gs = queue.pop(0)
        for mv, row in tree.get(gs, {}).items():
            child = gs + ',' + mv
            add_node(child, mv, row)
            dot.edge(nid(gs), nid(child))
            queue.append(child)

    return dot


def _derive_output(csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return stem.replace('opening_moves', 'opening_tree', 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Pente opening book visualizer')
    parser.add_argument('input', nargs='?', default=CSV_PATH, help='input CSV file')
    parser.add_argument('-o', '--output', default=None, help='output file stem (no extension)')
    args = parser.parse_args()

    csv_path   = args.input
    output     = args.output or _derive_output(csv_path)

    df   = load_csv(csv_path)
    tree = build_tree(df)
    print(f"Loaded {len(df)} edges, {len(tree)} parent nodes.")
    with tempfile.TemporaryDirectory() as tmp:
        g = build_graph(tree, tmp)
        out = g.render(output, format=OUTPUT_FMT, cleanup=True)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
