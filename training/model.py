#!/usr/bin/env python3
"""Interactive harness for bootstrap-training a model config.

Picks a model config from training/models/configs/, checks that its
referenced bootstrap dataset (training/bootstrap/configs/<name>.yaml, see
bootstrap.py) has actually been generated, and runs repeated `train -b`
cycles against it with early stopping on validation loss — same idea as
scripts/bootstrap_train.sh, minus its promotion-arena confirmation step
(dropped for this first pass).

Usage:
    python training/model.py [config_name]
"""

import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

import bootstrap

REPO_ROOT = bootstrap.REPO_ROOT
MODEL_CONFIGS_DIR = REPO_ROOT / "training" / "models" / "configs"
MODEL_RUNS_DIR = REPO_ROOT / "training" / "models" / "runs"
BUILD_DIR = bootstrap.BUILD_DIR
TRAIN_BIN = BUILD_DIR / "train"


def list_model_configs() -> list[Path]:
    return sorted(MODEL_CONFIGS_DIR.glob("*.yaml"))


def pick_config(arg_name: str | None) -> dict:
    configs = list_model_configs()
    if not configs:
        sys.exit(f"No model configs found in {MODEL_CONFIGS_DIR}")

    if arg_name:
        for path in configs:
            if path.stem == arg_name:
                return bootstrap.load_yaml(path)
        sys.exit(f"No model config named '{arg_name}' in {MODEL_CONFIGS_DIR}")

    if len(configs) == 1:
        return bootstrap.load_yaml(configs[0])

    print("Model configs:")
    for i, path in enumerate(configs):
        print(f"  [{i}] {path.stem}")
    choice = input(f"Select [0-{len(configs) - 1}]: ").strip()
    try:
        return bootstrap.load_yaml(configs[int(choice)])
    except (ValueError, IndexError):
        sys.exit("Invalid selection.")


def run_dir(name: str) -> Path:
    return MODEL_RUNS_DIR / name


def manifest_path(name: str) -> Path:
    return run_dir(name) / "manifest.json"


def best_model_path(name: str) -> Path:
    return run_dir(name) / "best_model.pt"


def metrics_path(name: str) -> Path:
    return run_dir(name) / "metrics.jsonl"


def load_manifest(name: str) -> dict | None:
    path = manifest_path(name)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def resolved_sections(config: dict) -> dict:
    return {
        "bootstrap": config["bootstrap"],
        "model": config["model"],
        "train": config["train"],
        "bootstrap_training": config["bootstrap_training"],
    }


def check_binary():
    if not TRAIN_BIN.exists():
        sys.exit(f"{TRAIN_BIN} not found — build it first: cd build && make train")


def check_bootstrap_ready(config: dict) -> dict:
    """Returns the referenced bootstrap dataset's manifest, or exits with
    instructions if it hasn't been generated yet."""
    bootstrap_name = config["bootstrap"]
    manifest = bootstrap.load_manifest(bootstrap_name)
    if manifest is None:
        sys.exit(
            f"Bootstrap data for '{bootstrap_name}' doesn't exist yet.\n"
            f"Run: python training/bootstrap.py {bootstrap_name}"
        )
    return manifest


def get_status(config: dict) -> tuple[str, dict | None]:
    manifest = load_manifest(config["name"])
    if manifest is None:
        return "not_trained", None

    current = resolved_sections(config)
    stored = {k: manifest[k] for k in current}
    if stored != current:
        return "stale", manifest

    bootstrap_manifest = bootstrap.load_manifest(config["bootstrap"])
    if bootstrap_manifest is None or bootstrap_manifest["generated_at"] != manifest.get("bootstrap_generated_at"):
        return "stale", manifest

    return "up_to_date", manifest


def print_diff(old: dict, new: dict):
    for section in ("bootstrap", "model", "train", "bootstrap_training"):
        old_val, new_val = old.get(section), new.get(section)
        if section == "bootstrap":
            if old_val != new_val:
                print(f"    bootstrap: {old_val!r} -> {new_val!r}")
            continue
        for key in sorted(set((old_val or {})) | set(new_val or {})):
            ov, nv = (old_val or {}).get(key), (new_val or {}).get(key)
            if ov != nv:
                print(f"    {section}.{key}: {ov!r} -> {nv!r}")


def run_train(config: dict, game_type: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    """Returns (result, payload), payload being the parsed final JSON status
    line from `train -j` (status "ok" or "skipped"), or None on failure."""
    name = config["name"]
    m, t = config["model"], config["train"]
    args = [
        str(TRAIN_BIN),
        "-g", game_type,
        "-b",
        "-i", str(Path("training") / "bootstrap" / "data" / config["bootstrap"]),
        "-d", str(Path("training") / "models" / "runs" / name),
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

    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)

    payload = None
    if result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            break

    return result, payload


def fail(result: subprocess.CompletedProcess, message: str):
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    sys.exit(message)


def train(config: dict, bootstrap_manifest: dict):
    name = config["name"]
    game_type = bootstrap_manifest["game"]["type"]
    bt = config["bootstrap_training"]

    run_dir(name).mkdir(parents=True, exist_ok=True)
    metrics_path(name).write_text("")  # fresh run starts a fresh curve

    best_val = None
    best_iter_path = None
    stale = 0
    cycles_run = 0

    start = time.perf_counter()
    for cycle in tqdm(range(bt["max_cycles"]), desc=name, unit="cycle"):
        result, payload = run_train(config, game_type)
        if payload is None:
            tqdm.write("")
            fail(result, f"Cycle {cycle + 1}/{bt['max_cycles']} failed.")
        if payload["status"] == "skipped":
            tqdm.write("")
            sys.exit(f"Buffer too small to train on ({payload['buffer_size']} positions) — check the bootstrap data.")

        cycles_run += 1
        val = payload["val_loss"]
        if best_val is None or val < best_val:
            best_val, best_iter_path, stale = val, payload["iter_path"], 0
            promoted = True
            tqdm.write(f"  cycle {cycle + 1}: val {val:.4f} — new best")
        else:
            stale += 1
            promoted = False
            tqdm.write(f"  cycle {cycle + 1}: val {val:.4f} — no improvement, stale {stale}/{bt['patience']}")

        with metrics_path(name).open("a") as f:
            f.write(json.dumps({"cycle": cycle + 1, "val_loss": val, "promoted": promoted,
                                "iter_path": payload["iter_path"]}) + "\n")

        if not promoted and stale >= bt["patience"]:
            tqdm.write(f"  early stop after cycle {cycle + 1}")
            break
    elapsed = time.perf_counter() - start

    # Train.cpp's own val-gate promotion can leave best_model.pt pointing at
    # a later, worse cycle than the best one seen across this whole run —
    # make sure it ends up holding the actual best.
    shutil.copy(best_iter_path, best_model_path(name))

    manifest = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "cycles_run": cycles_run,
        "best_val_loss": best_val,
        "bootstrap_generated_at": bootstrap_manifest["generated_at"],
        **resolved_sections(config),
    }
    manifest_path(name).write_text(json.dumps(manifest, indent=2))
    print(f"Done in {bootstrap.format_eta(elapsed)}, {cycles_run} cycles, best val {best_val:.4f}. "
          f"Saved -> {best_model_path(name)}")


def format_config_summary(config: dict) -> str:
    m, t, bt = config["model"], config["train"], config["bootstrap_training"]
    return (
        f"  bootstrap: {config['bootstrap']}\n"
        f"  model    : {m['channels']} channels, {m['res_blocks']} res blocks\n"
        f"  train    : batch={t['batch_size']} lr={t['learning_rate']} alpha={t['value_alpha']} "
        f"val_fraction={t['val_fraction']}\n"
        f"  cycles   : max {bt['max_cycles']}, patience {bt['patience']}"
    )


def print_status(config: dict, state: str, manifest: dict | None):
    print(f"\nModel config: {config['name']}")
    print(format_config_summary(config))
    if state == "not_trained":
        print("  status: not trained")
    elif state == "up_to_date":
        print(f"  status: up to date (best val {manifest['best_val_loss']:.4f}, "
              f"trained {manifest['trained_at']}, took {bootstrap.format_eta(manifest['elapsed_s'])})")
    else:
        print("  status: STALE — config or bootstrap data has changed since this model was trained:")
        print_diff(manifest, resolved_sections(config))


def wipe(name: str):
    for p in run_dir(name).glob("model_iter*.pt"):
        p.unlink()
    best_model_path(name).unlink(missing_ok=True)
    manifest_path(name).unlink(missing_ok=True)
    metrics_path(name).unlink(missing_ok=True)


def main():
    check_binary()
    arg_name = sys.argv[1] if len(sys.argv) > 1 else None
    config = pick_config(arg_name)

    state, manifest = get_status(config)
    print_status(config, state, manifest)

    options = {"1": "Train" if state == "not_trained" else "Retrain"}
    print()
    for key, label in options.items():
        print(f"  [{key}] {label}")
    print("  [q] Quit")
    choice = input("> ").strip().lower()

    if choice == "1":
        if state != "not_trained":
            if input(f"This will wipe the existing training run for '{config['name']}'. Continue? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return
            wipe(config["name"])
        bootstrap_manifest = check_bootstrap_ready(config)
        train(config, bootstrap_manifest)
    elif choice == "q":
        return
    else:
        print("Unknown option.")


if __name__ == "__main__":
    main()
