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

A checkpoint that clears 90% on the fast arena tier also gets escalated
automatically: 20 games at 1600, then (if that also clears 90%) 3200, 6400,
then 12800 sims/side, stopping at the first tier that misses 90%. Each tier is
its own idempotent (checkpoint, evaluator) pair, so re-running just
resumes/backfills the ladder rather than repeating it.

Every arena above (fast tier, escalation, --strong) is paired from 10 fixed
opening positions rather than independent random games — see Benchmark.cpp's
kArenaOpenings — and scored as expected score (wins + 0.5*draws)/total rather
than raw win rate.

Also provides promotion_arena(), called by selfplay.py after each round's
train call: confirms a val-loss-gated promotion with real games against the
pre-promotion best_model.pt before letting it stick.

Also provides elo_match()/elo_from_score(), used by selfplay.py to maintain
a rolling Elo ladder (elo.json per run): every 10 self-play rounds, the
current best_model.pt is rated against the most recent roster entry.

Usage:
    python training/benchmark.py [config_name]
    python training/benchmark.py [config_name] --strong [--games N] [--sims N] [--checkpoint path]
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

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

# Auto-escalation ladder for a checkpoint that clears 90% on the fast tier:
# 20 games each, sims/side climbing until one tier misses 90%.
ESCALATION_GAMES = 20
ESCALATION_SIMS_TIERS = [1600, 3200, 6400, 12800]
ESCALATION_THRESHOLD = 90.0

# Promotion-confirmation arena (selfplay.py): candidate vs. the pre-promotion
# best_model.pt, every round the val-loss gate (Train.cpp) would otherwise
# promote on its own. Val loss is measured against a held-out slice of that
# round's own self-play buffer, which can't tell "actually plays better" apart
# from "fit this round's buffer slightly differently" — so require a real-game
# edge too before the promotion sticks. 400 sims/side matches self-play's own
# sims (fast enough not to dominate round time); threshold is on *expected*
# score so a draw-heavy "basically even" result correctly fails to promote
# rather than needing an outright plurality of wins.
PROMOTION_SNAPSHOT_NAME = "prev_best_snapshot.pt"
PROMOTION_ARENA_GAMES = 20
PROMOTION_ARENA_SIMS = 400
PROMOTION_GATE_THRESHOLD = 52.5  # expected score %, same units as score_pct

# Elo ladder (selfplay.py): every 10 self-play rounds, rate the current
# best_model.pt against the most recent roster entry and derive an Elo via
# the standard logistic formula. Base entry (the last bootstrapped model) is
# hardcoded to ELO_BASE_RATING; everything else is relative to it.
ELO_ARENA_GAMES = 20
ELO_ARENA_SIMS = 400
ELO_BASE_RATING = 1000.0
_ELO_SCORE_CLAMP = 0.025  # avoid -inf/+inf on an exact 0/20 or 20/20 sweep

# (suite label as Benchmark.cpp records it, extra CLI args to produce it)
# Bootstrap checkpoints get the full battery. Self-play checkpoints skip the
# two heuristic suites (tactical pattern-matching, value-head sign) -- by the
# time self-play starts those are already near-saturated and stop telling
# you anything the arena doesn't cover better; self-play tracks arena only.
ARENA_SUITE = ("arena", ["-a", "-G", str(ARENA_GAMES), "-S", str(ARENA_SIMS), "-T", str(ARENA_SIMS)])
BOOTSTRAP_SUITES = [
    ("open-three-suite", ["-t", str(OPEN_THREE_SUITE)]),
    ("value-suite-value-sign", ["-V"]),
    ARENA_SUITE,
]
SELFPLAY_SUITES = [ARENA_SUITE]


def selfplay_cycles(name: str) -> set:
    """Cycle numbers trained during self-play (as opposed to bootstrap),
    read from metrics.jsonl's phase field."""
    path = model.metrics_path(name)
    if not path.exists():
        return set()
    cycles = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("phase") == "selfplay":
            cycles.add(entry["cycle"])
    return cycles


def arena_label(sims: int) -> str:
    # Benchmark.cpp always records the suite column as plain "arena" — the
    # sims/games detail lives in the evaluator column instead, e.g.
    # "nn@1600-vs-heuristic@1600" (see its arenaLabel construction).
    return f"nn@{sims}-vs-heuristic@{sims}"


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

    sp_cycles = selfplay_cycles(name)
    done = already_done(name)
    pending = [
        (ckpt, suite, args)
        for ckpt in checkpoints
        for suite, args in (SELFPLAY_SUITES if int(ckpt.stem.removeprefix("model_iter")) in sp_cycles else BOOTSTRAP_SUITES)
        if (str(ckpt.relative_to(model.REPO_ROOT)), suite) not in done
    ]

    if not pending:
        print(f"'{name}': all {len(checkpoints)} checkpoints already benchmarked.")
    else:
        print(f"Benchmarking '{name}': {len(pending)} (checkpoint, suite) pairs pending...")
        for ckpt, suite, extra_args in tqdm(pending, desc=name, unit="run"):
            result = run_one(config, game_type, ckpt, extra_args)
            if result.returncode != 0:
                tqdm.write("")
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                sys.exit(f"Benchmark failed on {ckpt.name} ({suite}).")
        print(f"Done. Results -> {csv_path(name)}")

    for ckpt in checkpoints:
        escalate_checkpoint(config, bootstrap_manifest, ckpt)


def escalate_checkpoint(config: dict, bootstrap_manifest: dict, checkpoint) -> None:
    """If `checkpoint` cleared ESCALATION_THRESHOLD on the fast arena tier,
    keep re-confirming it at climbing sims/side (ESCALATION_SIMS_TIERS),
    stopping at the first tier that misses the threshold. Safe to call on
    every checkpoint on every benchmark_run() — idempotent per (checkpoint,
    evaluator), so it just resumes/backfills the ladder rather than
    repeating finished tiers.
    """
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    rel = str(checkpoint.relative_to(model.REPO_ROOT))
    by_evaluator = {row["evaluator"]: row["score_pct"] for row in read_results(name) if row["checkpoint"] == rel}

    score = by_evaluator.get(arena_label(ARENA_SIMS))
    if score is None or float(score) < ESCALATION_THRESHOLD:
        return

    score = float(score)
    for sims in ESCALATION_SIMS_TIERS:
        if score < ESCALATION_THRESHOLD:
            break
        label = arena_label(sims)
        existing = by_evaluator.get(label)
        if existing is not None:
            score = float(existing)
            continue

        print(f"'{name}': {checkpoint.name} hit {score:.1f}% — escalating to "
              f"{ESCALATION_GAMES} games @ {sims} sims/side...")
        result = run_one(config, game_type, checkpoint,
                         ["-a", "-G", str(ESCALATION_GAMES), "-S", str(sims), "-T", str(sims)])
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            sys.exit(f"Escalation arena failed on {checkpoint.name} @ {sims} sims.")

        score = float(next(r["score_pct"] for r in reversed(read_results(name))
                           if r["checkpoint"] == rel and r["evaluator"] == label))
        print(f"  -> {score:.1f}%")


def resolve_best_iter_path(name: str) -> Path:
    """best_model.pt is a fixed filename that gets overwritten on every
    promotion, so (best_model.pt, suite/evaluator) is a meaningless
    idempotency key across runs: it collides with whatever used to be at
    that path even after the weights underneath it have moved on. Resolve
    it to the concrete model_iterNNNN.pt it currently matches instead —
    byte-identical, since Train.cpp/model.py only ever copy an
    already-promoted checkpoint onto best_model.pt — by finding the last
    promoted cycle recorded in metrics.jsonl.
    """
    path = model.metrics_path(name)
    if not path.exists():
        sys.exit(f"No metrics.jsonl for '{name}' — can't resolve best_model.pt to a cycle.")
    best_iter_path = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("promoted"):
            best_iter_path = entry["iter_path"]
    if best_iter_path is None:
        sys.exit(f"No promoted cycle found in {path} — can't resolve best_model.pt.")
    return Path(best_iter_path)


def strong_arena(config: dict, bootstrap_manifest: dict, checkpoint=None,
                 games: int = STRONG_ARENA_GAMES, sims: int = STRONG_ARENA_SIMS):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    checkpoint = checkpoint or model.best_model_path(name)
    if checkpoint.name == "best_model.pt":
        checkpoint = resolve_best_iter_path(name)
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    evaluator_label = arena_label(sims)
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


def promotion_evaluator_label() -> str:
    return arena_label(PROMOTION_ARENA_SIMS).replace("heuristic", Path(PROMOTION_SNAPSHOT_NAME).stem)


def promotion_arena(config: dict, game_type: str, candidate: Path, prev_best_snapshot: Path) -> float:
    """Candidate vs. prev_best_snapshot, paired fixed-opening arena. Returns
    the expected score (0-100) and records the result in benchmark.csv like
    any other arena run (checkpoint = candidate, so it's always a fresh row —
    no idempotency check needed, every round has a distinct candidate)."""
    result = run_one(config, game_type, candidate,
                     ["-a", "-G", str(PROMOTION_ARENA_GAMES),
                      "-S", str(PROMOTION_ARENA_SIMS), "-T", str(PROMOTION_ARENA_SIMS),
                      "-P", str(prev_best_snapshot)])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit("Promotion arena failed.")

    name = config["name"]
    rel = str(candidate.relative_to(model.REPO_ROOT))
    label = promotion_evaluator_label()
    return float(next(r["score_pct"] for r in reversed(read_results(name))
                      if r["checkpoint"] == rel and r["evaluator"] == label))


def elo_match(config: dict, game_type: str, candidate: Path, anchor: Path) -> float:
    """Candidate vs. anchor (a roster entry's iter_path), paired fixed-opening
    arena. Returns the raw expected score (0-100); recorded in benchmark.csv
    like any other arena run."""
    result = run_one(config, game_type, candidate,
                     ["-a", "-G", str(ELO_ARENA_GAMES),
                      "-S", str(ELO_ARENA_SIMS), "-T", str(ELO_ARENA_SIMS),
                      "-P", str(anchor)])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit("Elo arena failed.")

    name = config["name"]
    rel = str(candidate.relative_to(model.REPO_ROOT))
    label = arena_label(ELO_ARENA_SIMS).replace("heuristic", anchor.stem)
    return float(next(r["score_pct"] for r in reversed(read_results(name))
                      if r["checkpoint"] == rel and r["evaluator"] == label))


def elo_from_score(score_pct: float, anchor_elo: float) -> float:
    """Standard logistic Elo: given a candidate's expected score against a
    fixed-rating anchor, solve for the candidate's rating. score_pct is
    clamped away from the exact 0/100 edges, where the log blows up."""
    score = min(max(score_pct / 100.0, _ELO_SCORE_CLAMP), 1 - _ELO_SCORE_CLAMP)
    return anchor_elo - 400.0 * math.log10(1.0 / score - 1.0)


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
