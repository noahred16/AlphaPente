#!/usr/bin/env python3
"""Benchmark a model run's saved checkpoints: two fast deterministic suites
(open-three tactical suite + value-sign suite — raw policy, no MCTS, no real
games) plus a small arena vs the heuristic (real games, real search), all
idempotent per (checkpoint, suite) pair.

Not run for every checkpoint live during training — model.py saves every
model_iterNNNN.pt regardless, so this walks whatever's already on disk and
only benchmarks pairs it hasn't recorded yet. Safe to re-run any time; by
default model.py calls this automatically once training finishes.

Usage:
    python training/benchmark.py [config_name]
"""

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


def main():
    check_binary()
    arg_name = sys.argv[1] if len(sys.argv) > 1 else None
    config = model.pick_config(arg_name)
    bootstrap_manifest = model.check_bootstrap_ready(config)
    benchmark_run(config, bootstrap_manifest)


if __name__ == "__main__":
    main()
