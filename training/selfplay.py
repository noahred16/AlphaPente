#!/usr/bin/env python3
"""Self-play training: extend an already-trained model config with rounds of
self-play (generate games with the current best model -> train on the
accumulated buffer -> repeat).

Deliberately simpler than model.py's bootstrap-train loop, and different in
one important way: self-play's buffer shifts round to round (each round adds
fresh, model-generated games), so val loss isn't comparable *across* rounds
the way it is in bootstrap-training (one fixed dataset, one fixed val split).
There's no "track the global best round and force-promote it" step here —
promotion is left entirely to Train.cpp's own per-round gate (candidate vs.
that round's immediate baseline), same as the old scripts/train_loop.sh did.

Also not idempotent the way bootstrap.py/model.py are: each invocation just
runs `rounds` more rounds on top of whatever's already in the run. Requires
a model config that's already been bootstrap-trained (training/model.py) —
self-play continues from an existing best_model.pt, it doesn't start cold.

Usage:
    python training/selfplay.py [config_name] [-n rounds]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

import model

GENERATE_BIN = model.BUILD_DIR / "generate"


def check_binaries():
    if not GENERATE_BIN.exists():
        sys.exit(f"{GENERATE_BIN} not found — build it first: cd build && make generate")
    model.check_binary()  # train
    import benchmark
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


def selfplay(config: dict, bootstrap_manifest: dict, rounds: int):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]

    start = time.perf_counter()
    for i in tqdm(range(rounds), desc=name, unit="round"):
        gen_result, gen_payload = run_generate(config, game_type)
        if gen_payload is None or gen_payload.get("status") != "ok":
            tqdm.write("")
            fail(gen_result, f"Self-play generation failed on round {i + 1}/{rounds}.")

        train_result, train_payload = run_train(config, game_type)
        if train_payload is None:
            tqdm.write("")
            fail(train_result, f"Training failed on round {i + 1}/{rounds}.")
        if train_payload["status"] == "skipped":
            tqdm.write("")
            sys.exit(f"Buffer too small to train on ({train_payload['buffer_size']} positions) "
                      f"after round {i + 1} — need more games/round or more rounds.")

        cycle = int(Path(train_payload["iter_path"]).stem.removeprefix("model_iter"))
        with model.metrics_path(name).open("a") as f:
            f.write(json.dumps({
                "cycle": cycle,
                "phase": "selfplay",
                "val_loss": train_payload["val_loss"],
                "promoted": train_payload["promoted"],
                "iter_path": train_payload["iter_path"],
                "positions_generated": gen_payload["positions"],
            }) + "\n")

        status = "promoted" if train_payload["promoted"] else "not promoted"
        tqdm.write(f"  round {i + 1}: +{gen_payload['positions']} positions, "
                   f"val {train_payload['val_loss']:.4f} ({status})")
    elapsed = time.perf_counter() - start

    print(f"Done in {model.bootstrap.format_eta(elapsed)}, {rounds} rounds.")

    print()
    import benchmark
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
