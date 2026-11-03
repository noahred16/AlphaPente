#!/usr/bin/env python3
"""Run build/pente against a trained model config, without typing -c/-r by
hand — looks up channels/res_blocks (and the checkpoint path) from the
config so they can't drift out of sync with how it was trained.

Usage:
    python training/play.py <config_name> ["move string"] [pente options...]
    python training/play.py pente_v1_small "1. K10 L9 2. K12 M10" -n -j
"""

import subprocess
import sys

import model

PENTE_BIN = model.BUILD_DIR / "pente"


def main():
    if len(sys.argv) < 2:
        names = [p.stem for p in model.list_model_configs()]
        sys.exit(f"Usage: python training/play.py <config_name> [pente options...]\n"
                  f"Available configs: {', '.join(names)}")

    name, rest = sys.argv[1], sys.argv[2:]
    config = model.pick_config(name)
    checkpoint = model.best_model_path(config["name"])
    if not checkpoint.exists():
        sys.exit(f"'{config['name']}' has no trained model yet — run training/model.py first.")
    if not PENTE_BIN.exists():
        sys.exit(f"{PENTE_BIN} not found — build it first: cd build && make pente")

    m = config["model"]
    args = [str(PENTE_BIN), "-p", str(checkpoint), "-c", str(m["channels"]), "-r", str(m["res_blocks"]), *rest]
    sys.exit(subprocess.run(args, cwd=model.REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
