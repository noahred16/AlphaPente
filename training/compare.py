#!/usr/bin/env python3
"""Compare model training runs: a summary table plus a val-loss-vs-cycle chart.

Usage:
    python training/compare.py                  # compare every run under training/models/runs/
    python training/compare.py pente_v1 pente_v1_small
    python training/compare.py -o my_chart.png pente_v1 pente_v1_small
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
from imgcat import imgcat

import model

DEFAULT_OUTPUT = model.MODEL_RUNS_DIR / "comparison.png"


def list_run_names() -> list[str]:
    return sorted(p.name for p in model.MODEL_RUNS_DIR.iterdir() if p.is_dir())


def load_run(name: str) -> tuple[dict, list[dict]]:
    """Returns (manifest, metrics). metrics is [] if the run predates
    per-cycle metrics tracking or hasn't been trained yet."""
    manifest = model.load_manifest(name)
    if manifest is None:
        sys.exit(f"'{name}' hasn't been trained yet — no manifest at {model.manifest_path(name)}")

    metrics = []
    path = model.metrics_path(name)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                metrics.append(json.loads(line))
    return manifest, metrics


def print_table(runs: dict[str, tuple[dict, list[dict]]]):
    rows = []
    for name, (manifest, _) in runs.items():
        m = manifest["model"]
        rows.append((
            name,
            f"{m['channels']}ch/{m['res_blocks']}rb",
            str(manifest["cycles_run"]),
            f"{manifest['best_val_loss']:.4f}",
            model.bootstrap.format_eta(manifest["elapsed_s"]),
        ))

    headers = ("name", "architecture", "cycles", "best val loss", "elapsed")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def plot_curves(runs: dict[str, tuple[dict, list[dict]]], output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for name, (_, metrics) in runs.items():
        if not metrics:
            print(f"  (skipping '{name}' in the chart — no metrics.jsonl, trained before this was tracked)")
            continue
        cycles = [m["cycle"] for m in metrics]
        losses = [m["val_loss"] for m in metrics]
        ax.plot(cycles, losses, marker="o", label=name)
        plotted = True

    if not plotted:
        print("No runs have per-cycle metrics to plot — skipping chart.")
        return

    ax.set_xlabel("cycle")
    ax.set_ylabel("validation loss")
    ax.set_title("Bootstrap-training validation loss by run")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved chart -> {output_path}")

    try:
        with open(output_path, "rb") as f:
            imgcat(f)
    except Exception as e:
        print(f"  (couldn't preview inline — {e}; open {output_path} directly)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="model config names to compare (default: all runs)")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="PNG output path")
    args = parser.parse_args()

    names = args.names or list_run_names()
    if not names:
        sys.exit(f"No runs found under {model.MODEL_RUNS_DIR}")

    runs = {name: load_run(name) for name in names}

    print_table(runs)
    print()
    plot_curves(runs, args.output)


if __name__ == "__main__":
    main()
