#!/usr/bin/env python3
"""Interactive harness for generating bootstrap self-play data.

Picks a bootstrap config from training/bootstrap/configs/, reports whether
data already exists for it (and whether that data still matches the config),
and offers to (re)generate it — shelling out to the compiled `generate`
binary, same as the old scripts/bootstrap_generate.sh did, just config-driven
and with an ETA.

Usage:
    python training/bootstrap.py [config_name]
"""

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_CONFIGS_DIR = REPO_ROOT / "training" / "bootstrap" / "configs"
BOOTSTRAP_DATA_DIR = REPO_ROOT / "training" / "bootstrap" / "data"
MODELS_CONFIGS_DIR = REPO_ROOT / "training" / "models" / "configs"
BUILD_DIR = REPO_ROOT / "build"
GENERATE_BIN = BUILD_DIR / "generate"

CALIBRATION_GAMES = 3


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def list_bootstrap_configs() -> list[Path]:
    return sorted(BOOTSTRAP_CONFIGS_DIR.glob("*.yaml"))


def pick_config(arg_name: str | None) -> dict:
    configs = list_bootstrap_configs()
    if not configs:
        sys.exit(f"No bootstrap configs found in {BOOTSTRAP_CONFIGS_DIR}")

    if arg_name:
        for path in configs:
            if path.stem == arg_name:
                return load_yaml(path)
        sys.exit(f"No bootstrap config named '{arg_name}' in {BOOTSTRAP_CONFIGS_DIR}")

    if len(configs) == 1:
        return load_yaml(configs[0])

    print("Bootstrap configs:")
    for i, path in enumerate(configs):
        print(f"  [{i}] {path.stem}")
    choice = input(f"Select [0-{len(configs) - 1}]: ").strip()
    try:
        return load_yaml(configs[int(choice)])
    except (ValueError, IndexError):
        sys.exit("Invalid selection.")


def data_dir(name: str) -> Path:
    return BOOTSTRAP_DATA_DIR / name


def manifest_path(name: str) -> Path:
    return data_dir(name) / "manifest.json"


def bootstrap_pt_path(name: str) -> Path:
    return data_dir(name) / "bootstrap.pt"


def load_manifest(name: str) -> dict | None:
    path = manifest_path(name)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def resolved_sections(config: dict) -> dict:
    return {"game": config["game"], "bootstrap": config["bootstrap"]}


def check_binary():
    if not GENERATE_BIN.exists():
        sys.exit(f"{GENERATE_BIN} not found — build it first: cd build && make generate")


def get_status(config: dict) -> tuple[str, dict | None]:
    manifest = load_manifest(config["name"])
    if manifest is None:
        return "not_generated", None
    current = resolved_sections(config)
    stored = {"game": manifest["game"], "bootstrap": manifest["bootstrap"]}
    return ("up_to_date" if stored == current else "stale"), manifest


def print_diff(old: dict, new: dict):
    for section in ("game", "bootstrap"):
        for key in sorted(set(old[section]) | set(new[section])):
            old_val, new_val = old[section].get(key), new[section].get(key)
            if old_val != new_val:
                print(f"    {section}.{key}: {old_val!r} -> {new_val!r}")


def run_generate(config: dict, *, n: int, sims: int, dry_run: bool, extra_args=None) -> tuple[subprocess.CompletedProcess, dict | None]:
    """Returns (result, payload). payload is the parsed final {"status": "ok", ...}
    line (e.g. {"status": "ok", "positions": 123)), or None on failure/parse error."""
    game = config["game"]["type"]
    args = [str(GENERATE_BIN), "-g", game, "-n", str(n), "-s", str(sims), "-b", "-j"]
    if dry_run:
        args.append("-x")
    else:
        args += ["-d", str(Path("training") / "bootstrap" / "data" / config["name"])]
    if extra_args:
        args.extend(extra_args)

    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)

    payload = None
    if result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                break
            if parsed.get("status") == "ok":
                payload = parsed
            break

    return result, payload


def fail(result: subprocess.CompletedProcess, message: str):
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    sys.exit(message)


def calibrate(config: dict) -> tuple[float, float]:
    """Returns (avg_seconds_per_game, avg_positions_per_game)."""
    sims = config["bootstrap"]["simulations"]
    print(f"Calibrating ({CALIBRATION_GAMES} games @ {sims} sims)...")
    start = time.perf_counter()
    result, payload = run_generate(config, n=CALIBRATION_GAMES, sims=sims, dry_run=True)
    elapsed = time.perf_counter() - start
    if payload is None:
        fail(result, "Calibration run failed.")
    return elapsed / CALIBRATION_GAMES, payload["positions"] / CALIBRATION_GAMES


def format_eta(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


def generate(config: dict):
    name = config["name"]
    b = config["bootstrap"]
    total_games = b["batches"] * b["games_per_batch"]

    per_game_s, per_game_positions = calibrate(config)
    eta = per_game_s * total_games
    est_positions = round(per_game_positions * total_games)
    print(f"Estimated: {total_games} games @ ~{per_game_s:.2f}s/game ≈ {format_eta(eta)}, "
          f"~{est_positions:,} positions")

    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return

    target_dir = data_dir(name)
    target_dir.mkdir(parents=True, exist_ok=True)

    total_positions = 0
    start = time.perf_counter()
    for i in tqdm(range(b["batches"]), desc=name, unit="batch"):
        result, payload = run_generate(
            config,
            n=b["games_per_batch"],
            sims=b["simulations"],
            dry_run=False,
            extra_args=["-t", str(b["tail_moves"])],
        )
        if payload is None:
            tqdm.write("")
            fail(result, f"Batch {i + 1}/{b['batches']} failed.")
        total_positions += payload["positions"]
    elapsed = time.perf_counter() - start

    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "positions": total_positions,
        **resolved_sections(config),
    }
    manifest_path(name).write_text(json.dumps(manifest, indent=2))
    print(f"Done in {format_eta(elapsed)}, {total_positions:,} positions. Saved -> {bootstrap_pt_path(name)}")


def wipe(name: str):
    pt = bootstrap_pt_path(name)
    pt.unlink(missing_ok=True)
    manifest_path(name).unlink(missing_ok=True)


def format_config_summary(config: dict) -> str:
    g, b = config["game"], config["bootstrap"]
    tail = f"last {b['tail_moves']}" if b["tail_moves"] > 0 else "all"
    return (
        f"  game    : {g['type']} (board {g['board_size']}x{g['board_size']})\n"
        f"  settings: {b['evaluator']} evaluator, {b['batches']}x{b['games_per_batch']} games, "
        f"{b['simulations']} sims/move, tail={tail}, "
        f"dirichlet(α={b['dirichlet_alpha']}, ε={b['dirichlet_epsilon']}), dropoff={b['exploration_dropoff']}"
    )


def print_status(config: dict, state: str, manifest: dict | None):
    print(f"\nBootstrap config: {config['name']}")
    print(format_config_summary(config))
    if state == "not_generated":
        print("  status: not generated")
    elif state == "up_to_date":
        print(f"  status: up to date ({manifest['positions']:,} positions, generated {manifest['generated_at']}, took {format_eta(manifest['elapsed_s'])})")
    else:
        print("  status: STALE — config has changed since this data was generated:")
        print_diff(manifest, resolved_sections(config))


def main():
    check_binary()
    arg_name = sys.argv[1] if len(sys.argv) > 1 else None
    config = pick_config(arg_name)

    state, manifest = get_status(config)
    print_status(config, state, manifest)

    if state == "not_generated":
        options = {"1": "Generate"}
    else:
        options = {"1": "Regenerate"}

    print()
    for key, label in options.items():
        print(f"  [{key}] {label}")
    print("  [q] Quit")
    choice = input("> ").strip().lower()

    if choice == "1":
        if state != "not_generated":
            if input(f"This will wipe the existing data for '{config['name']}'. Continue? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return
            wipe(config["name"])
        generate(config)
    elif choice == "q":
        return
    else:
        print("Unknown option.")


if __name__ == "__main__":
    main()
