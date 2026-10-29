#!/usr/bin/env python3
"""Benchmark a model run's saved checkpoints: two fast deterministic suites
(open-three tactical suite + value-sign suite — raw policy, no MCTS, no real
games) plus a small arena vs the heuristic (real games, real search), all
idempotent per (checkpoint, suite) pair.

Not run for every checkpoint live during training — model.py saves every
model_iterNNNN.pt regardless, so this walks whatever's already on disk and
only benchmarks pairs it hasn't recorded yet. Safe to re-run any time; by
default model.py calls this automatically once training finishes.

Also supports a one-off, heavier arena spot-check (--strong) against a single
checkpoint (default: best_model.pt) — more games and sims for a tighter
confidence interval, used to confirm a trend the fast per-checkpoint arena
suggested rather than to track every checkpoint's lifecycle (at 40 games/1600
sims it's much too slow to backfill across dozens of checkpoints).

Usage:
    python training/benchmark.py [config_name]
    python training/benchmark.py [config_name] --strong [--games N] [--sims N] [--checkpoint path]
"""

import argparse
import csv
import subprocess
import sys

from tqdm import tqdm

import model

BENCHMARK_BIN = model.BUILD_DIR / "benchmark"
OPEN_THREE_SUITE = model.REPO_ROOT / "tests" / "open-three-suite.json"

# Matches the codebase's existing "fast confirmation" arena tier (train_loop.sh's
# ARENA_SIMS=400, bootstrap_train.sh's PROMOTION_ARENA_SIMS=400) rather than the
# full battery's 800-100000 sim tiers, which are far too slow to run every
# checkpoint. 20 games is noisy as a single point (~±20% at 95% CI) but this is
# a trend tracked over many checkpoints, not a one-off promotion decision.
ARENA_GAMES = 20
ARENA_SIMS = 400

# --strong: a slower, tighter-confidence spot-check (2x games, 4x sims ~= 8x
# the per-game cost of the fast tier) run against one checkpoint on demand.
STRONG_ARENA_GAMES = 40
STRONG_ARENA_SIMS = 1600

# (suite label as Benchmark.cpp records it, extra CLI args to produce it)
SUITES = [
    ("open-three-suite", ["-t", str(OPEN_THREE_SUITE)]),
    ("value-suite-value-sign", ["-V"]),
    ("arena", ["-a", "-G", str(ARENA_GAMES), "-S", str(ARENA_SIMS), "-T", str(ARENA_SIMS)]),
]


def check_binary():
    if not BENCHMARK_BIN.exists():
        sys.exit(f"{BENCHMARK_BIN} not found — build it first: cd build && make benchmark")


def csv_path(name: str):
    return model.run_dir(name) / "benchmark.csv"


def list_checkpoints(name: str) -> list:
    return sorted(model.run_dir(name).glob("model_iter*.pt"))


def read_results(name: str) -> list:
    """Returns every recorded row for this run, as raw CSV dicts."""
    path = csv_path(name)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def already_done(name: str) -> set:
    """Returns {(checkpoint_relpath, suite), ...} already recorded."""
    return {(row["checkpoint"], row["suite"]) for row in read_results(name)}


def run_one(config: dict, game_type: str, checkpoint, extra_args: list) -> subprocess.CompletedProcess:
    m = config["model"]
    args = [
        str(BENCHMARK_BIN),
        "-g", game_type,
        "-p", str(checkpoint),
        "-o", str(csv_path(config["name"])),
        "-c", str(m["channels"]),
        "-r", str(m["res_blocks"]),
        *extra_args,
    ]
    return subprocess.run(args, cwd=model.REPO_ROOT, capture_output=True, text=True)


def benchmark_run(config: dict, bootstrap_manifest: dict):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    checkpoints = list_checkpoints(name)
    if not checkpoints:
        print(f"'{name}': no checkpoints to benchmark yet.")
        return

    done = already_done(name)
    pending = [
        (ckpt, suite, args)
        for ckpt in checkpoints
        for suite, args in SUITES
        if (str(ckpt.relative_to(model.REPO_ROOT)), suite) not in done
    ]

    if not pending:
        print(f"'{name}': all {len(checkpoints)} checkpoints already benchmarked ({len(SUITES)} suites each).")
        return

    print(f"Benchmarking '{name}': {len(pending)} (checkpoint, suite) pairs pending...")
    for ckpt, suite, extra_args in tqdm(pending, desc=name, unit="run"):
        result = run_one(config, game_type, ckpt, extra_args)
        if result.returncode != 0:
            tqdm.write("")
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            sys.exit(f"Benchmark failed on {ckpt.name} ({suite}).")
    print(f"Done. Results -> {csv_path(name)}")


def strong_arena(config: dict, bootstrap_manifest: dict, checkpoint=None,
                 games: int = STRONG_ARENA_GAMES, sims: int = STRONG_ARENA_SIMS):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    checkpoint = checkpoint or model.best_model_path(name)
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    # Benchmark.cpp always records the suite column as plain "arena" — the
    # sims/games detail lives in the evaluator column instead, e.g.
    # "nn@1600-vs-heuristic@1600" (see its arenaLabel construction).
    evaluator_label = f"nn@{sims}-vs-heuristic@{sims}"
    rel = str(checkpoint.relative_to(model.REPO_ROOT))
    already = {(row["checkpoint"], row["evaluator"]) for row in read_results(name)}
    if (rel, evaluator_label) in already:
        print(f"'{name}': {checkpoint.name} already has a {evaluator_label} result — skipping.")
        return

    print(f"Strong arena: '{name}' {checkpoint.name} vs heuristic, {games} games @ {sims} sims/side...")
    result = run_one(config, game_type, checkpoint,
                     ["-a", "-G", str(games), "-S", str(sims), "-T", str(sims)])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"Strong arena failed on {checkpoint.name}.")
    print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "")
    print(f"Done. Results -> {csv_path(name)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="model config name (prompts if omitted and >1 exists)")
    parser.add_argument("--strong", action="store_true", help="run a single heavier arena spot-check instead")
    parser.add_argument("--games", type=int, default=STRONG_ARENA_GAMES)
    parser.add_argument("--sims", type=int, default=STRONG_ARENA_SIMS)
    parser.add_argument("--checkpoint", type=str, default=None, help="default: best_model.pt")
    args = parser.parse_args()

    check_binary()
    config = model.pick_config(args.name)
    bootstrap_manifest = model.check_bootstrap_ready(config)

    if args.strong:
        checkpoint = (model.REPO_ROOT / args.checkpoint) if args.checkpoint else None
        strong_arena(config, bootstrap_manifest, checkpoint, args.games, args.sims)
    else:
        benchmark_run(config, bootstrap_manifest)


if __name__ == "__main__":
    main()
