#!/usr/bin/env python3
"""Self-play training: extend an already-trained model config with rounds of
self-play (generate games with the current best model -> train on the
accumulated buffer -> repeat).

Deliberately simpler than model.py's bootstrap-train loop, and different in
one important way: self-play's buffer shifts round to round (each round adds
fresh, model-generated games), so val loss isn't comparable *across* rounds
the way it is in bootstrap-training (one fixed dataset, one fixed val split).
There's no "track the global best round and force-promote it" step here —
promotion is left to Train.cpp's own per-round val-loss gate (candidate vs.
that round's immediate baseline), confirmed by a real-game arena against the
pre-promotion best_model.pt (benchmark.promotion_arena) before it's allowed
to stick — same two-step gate the old scripts/train_loop.sh used, just with
the new paired-opening/expected-score arena methodology.

Also not idempotent the way bootstrap.py/model.py are: each invocation just
runs `rounds` more rounds on top of whatever's already in the run. Requires
a model config that's already been bootstrap-trained (training/model.py) —
self-play continues from an existing best_model.pt, it doesn't start cold.

Usage:
    python training/selfplay.py [config_name] [-n rounds]
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

import benchmark
import model

GENERATE_BIN = model.BUILD_DIR / "generate"


def check_binaries():
    if not GENERATE_BIN.exists():
        sys.exit(f"{GENERATE_BIN} not found — build it first: cd build && make generate")
    model.check_binary()  # train
    benchmark.check_binary()


def check_trained(config: dict):
    name = config["name"]
    if not model.best_model_path(name).exists():
        sys.exit(
            f"'{name}' has no trained model yet — self-play continues from an existing "
            f"model, it doesn't start cold. Run bootstrap training first:\n"
            f"  python training/model.py {name}"
        )


def _parse_json_line(result: subprocess.CompletedProcess):
    if result.returncode != 0:
        return None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None


def run_generate(config: dict, game_type: str):
    sp, m = config["self_play"], config["model"]
    run_dir = Path("training") / "models" / "runs" / config["name"]
    args = [
        str(GENERATE_BIN),
        "-g", game_type,
        "-n", str(sp["games_per_round"]),
        "-s", str(sp["simulations"]),
        "-e", "nn",
        "-d", str(run_dir),
        "-c", str(m["channels"]),
        "-r", str(m["res_blocks"]),
        "-a", str(sp["dirichlet_alpha"]),
        "-E", str(sp["dirichlet_epsilon"]),
        "-o", str(sp["exploration_dropoff"]),
        "-j",
    ]
    result = subprocess.run(args, cwd=model.REPO_ROOT, capture_output=True, text=True)
    return result, _parse_json_line(result)


def run_train(config: dict, game_type: str):
    m, t = config["model"], config["train"]
    run_dir = Path("training") / "models" / "runs" / config["name"]
    args = [
        str(model.TRAIN_BIN),
        "-g", game_type,
        "-i", str(run_dir),
        "-d", str(run_dir),
        "-a", str(t["value_alpha"]),
        "-B", str(t["batch_size"]),
        "-l", str(t["learning_rate"]),
        "-W", str(t["weight_decay"]),
        "-v", str(t["value_loss_weight"]),
        "-f", str(t["val_fraction"]),
        "-G", str(t["val_gate_tolerance"]),
        "-m", str(t["min_grad_steps"]),
        "-c", str(m["channels"]),
        "-r", str(m["res_blocks"]),
        "-j",
    ]
    result = subprocess.run(args, cwd=model.REPO_ROOT, capture_output=True, text=True)
    return result, _parse_json_line(result)


def fail(result: subprocess.CompletedProcess, message: str):
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    sys.exit(message)


# ── Elo ladder ────────────────────────────────────────────────────────────────
# elo.json: one roster entry per rated checkpoint. Base entry (the last
# bootstrapped model, i.e. whatever best_model.pt is when self-play first
# starts) is hardcoded to benchmark.ELO_BASE_RATING; every 10 self-play
# rounds after that, the current best_model.pt is rated against the most
# recent roster entry.

def elo_path(name: str) -> Path:
    return model.run_dir(name) / "elo.json"


def load_roster(name: str) -> list:
    path = elo_path(name)
    return json.loads(path.read_text()) if path.exists() else []


def save_roster(name: str, roster: list):
    elo_path(name).write_text(json.dumps(roster, indent=2))


def count_selfplay_rounds(name: str) -> int:
    path = model.metrics_path(name)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and json.loads(line).get("phase") == "selfplay")


def ensure_base_elo_entry(name: str) -> list:
    roster = load_roster(name)
    if roster:
        return roster
    base_path = benchmark.resolve_best_iter_path(name)
    roster = [{
        "cycle": int(base_path.stem.removeprefix("model_iter")),
        "iter_path": str(base_path),
        "elo": benchmark.ELO_BASE_RATING,
        "anchor_cycle": None,
        "score_pct": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }]
    save_roster(name, roster)
    return roster


def update_elo_roster(config: dict, game_type: str, name: str):
    roster = ensure_base_elo_entry(name)
    candidate_path = benchmark.resolve_best_iter_path(name)
    cycle = int(candidate_path.stem.removeprefix("model_iter"))
    if roster[-1]["cycle"] == cycle:
        return  # best_model.pt hasn't moved since the last check -- nothing new to rate

    anchor = roster[-1]
    score = benchmark.elo_match(config, game_type, candidate_path, Path(anchor["iter_path"]))
    if score in (0.0, 100.0) and len(roster) >= 2:
        # Shutout against the closest anchor -- the log-odds formula blows up
        # at the exact edges, so fall back one step to a (presumably) less
        # lopsided comparison instead. elo_from_score's own clamp is the
        # final safety net if even this saturates.
        anchor = roster[-2]
        score = benchmark.elo_match(config, game_type, candidate_path, Path(anchor["iter_path"]))

    elo = benchmark.elo_from_score(score, anchor["elo"])
    roster.append({
        "cycle": cycle,
        "iter_path": str(candidate_path),
        "elo": elo,
        "anchor_cycle": anchor["cycle"],
        "score_pct": score,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    save_roster(name, roster)
    tqdm.write(f"  elo: {candidate_path.stem} rated {elo:.0f} "
               f"({score:.1f}% vs {Path(anchor['iter_path']).stem} @ {anchor['elo']:.0f})")


def selfplay(config: dict, bootstrap_manifest: dict, rounds: int):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    run_dir = model.run_dir(name)
    snapshot_path = run_dir / benchmark.PROMOTION_SNAPSHOT_NAME

    ensure_base_elo_entry(name)

    start = time.perf_counter()
    for i in tqdm(range(rounds), desc=name, unit="round"):
        gen_result, gen_payload = run_generate(config, game_type)
        if gen_payload is None or gen_payload.get("status") != "ok":
            tqdm.write("")
            fail(gen_result, f"Self-play generation failed on round {i + 1}/{rounds}.")

        # Snapshot before training overwrites best_model.pt in place — this is
        # what the promotion arena below plays the candidate against.
        shutil.copy(model.best_model_path(name), snapshot_path)

        train_result, train_payload = run_train(config, game_type)
        if train_payload is None:
            tqdm.write("")
            fail(train_result, f"Training failed on round {i + 1}/{rounds}.")
        if train_payload["status"] == "skipped":
            tqdm.write("")
            sys.exit(f"Buffer too small to train on ({train_payload['buffer_size']} positions) "
                      f"after round {i + 1} — need more games/round or more rounds.")

        promoted = train_payload["promoted"]
        arena_score = None
        if promoted:
            arena_score = benchmark.promotion_arena(config, game_type, Path(train_payload["iter_path"]), snapshot_path)
            if arena_score <= benchmark.PROMOTION_GATE_THRESHOLD:
                shutil.copy(snapshot_path, model.best_model_path(name))
                promoted = False
                tqdm.write(f"  round {i + 1}: promotion arena {arena_score:.1f}% — "
                           f"reverted, best_model.pt unchanged")
            else:
                tqdm.write(f"  round {i + 1}: promotion arena {arena_score:.1f}% — confirmed")

        cycle = int(Path(train_payload["iter_path"]).stem.removeprefix("model_iter"))
        with model.metrics_path(name).open("a") as f:
            f.write(json.dumps({
                "cycle": cycle,
                "phase": "selfplay",
                "val_loss": train_payload["val_loss"],
                "promoted": promoted,
                "arena_score_pct": arena_score,
                "iter_path": train_payload["iter_path"],
                "positions_generated": gen_payload["positions"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }) + "\n")

        status = "promoted" if promoted else "not promoted"
        tqdm.write(f"  round {i + 1}: +{gen_payload['positions']} positions, "
                   f"val {train_payload['val_loss']:.4f} ({status})")

        if count_selfplay_rounds(name) % 10 == 0:
            update_elo_roster(config, game_type, name)
    elapsed = time.perf_counter() - start

    print(f"Done in {model.bootstrap.format_eta(elapsed)}, {rounds} rounds.")

    print()
    benchmark.benchmark_run(config, bootstrap_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="model config name (prompts if omitted and >1 exists)")
    parser.add_argument("-n", "--rounds", type=int, default=None, help="override self_play.rounds from the config")
    args = parser.parse_args()

    check_binaries()
    config = model.pick_config(args.name)
    check_trained(config)
    bootstrap_manifest = model.check_bootstrap_ready(config)

    rounds = args.rounds if args.rounds is not None else config["self_play"]["rounds"]
    selfplay(config, bootstrap_manifest, rounds)


if __name__ == "__main__":
    main()
