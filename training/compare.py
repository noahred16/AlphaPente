#!/usr/bin/env python3
"""Compare model training runs: a summary table plus charts — val loss and
benchmark suite scores, each vs. cycle.

Usage:
    python training/compare.py                  # compare every run under training/models/runs/
    python training/compare.py pente_v1 pente_v1_small
    python training/compare.py -o my_chart.png pente_v1 pente_v1_small
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from imgcat import imgcat

import benchmark
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


def cycle_from_checkpoint(checkpoint: str) -> int | None:
    """'.../model_iter0007.pt' -> 7. Valid because model.py wipes a run before
    retraining, so iteration number and cycle number are the same sequence.
    Returns None for anything else (e.g. best_model.pt, used by benchmark.py's
    --strong spot-check) — not part of the per-cycle curve."""
    stem = Path(checkpoint).stem
    if not stem.startswith("model_iter"):
        return None
    try:
        return int(stem.removeprefix("model_iter"))
    except ValueError:
        return None


def load_benchmark_curves(name: str) -> dict:
    """Returns {suite: [(cycle, score_pct), ...]}, sorted by cycle."""
    curves: dict = {}
    for row in benchmark.read_results(name):
        cycle = cycle_from_checkpoint(row["checkpoint"])
        if cycle is None:
            continue
        curves.setdefault(row["suite"], []).append((cycle, float(row["score_pct"])))
    for points in curves.values():
        points.sort(key=lambda p: p[0])
    return curves


def print_table(runs: dict, bench: dict):
    suite_names = sorted({suite for curves in bench.values() for suite in curves})

    headers = ["name", "architecture", "cycles", "best val loss", "elapsed", *suite_names]
    rows = []
    for name, (manifest, _) in runs.items():
        m = manifest["model"]
        row = [
            name,
            f"{m['channels']}ch/{m['res_blocks']}rb",
            str(manifest["cycles_run"]),
            f"{manifest['best_val_loss']:.4f}",
            model.bootstrap.format_eta(manifest["elapsed_s"]),
        ]
        for suite in suite_names:
            points = bench.get(name, {}).get(suite)
            row.append(f"{points[-1][1]:.1f}%" if points else "-")
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    if suite_names:
        print("(suite columns show the latest/highest-cycle score)")


def plot_charts(runs: dict, bench: dict, output_path):
    suite_names = sorted({suite for curves in bench.values() for suite in curves})
    panels = ["validation loss", *suite_names]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    run_color = {name: colors[i % len(colors)] for i, name in enumerate(runs)}

    ax = axes[0]
    plotted = False
    for name, (_, metrics) in runs.items():
        if not metrics:
            print(f"  (skipping '{name}' in the loss chart — no metrics.jsonl, trained before this was tracked)")
            continue
        ax.plot([m["cycle"] for m in metrics], [m["val_loss"] for m in metrics],
                marker="o", label=name, color=run_color[name])
        plotted = True

        # Self-play validates against a different (shifting) buffer slice each
        # round, so its val loss isn't comparable to bootstrap-phase val loss
        # or across self-play rounds themselves — mark where it starts rather
        # than let the line imply one continuous, comparable trend.
        selfplay_start = next((m["cycle"] for m in metrics if m.get("phase") == "selfplay"), None)
        if selfplay_start is not None:
            ax.axvline(selfplay_start - 0.5, color=run_color[name], linestyle=":", alpha=0.5)

    ax.set_xlabel("cycle")
    ax.set_ylabel("validation loss")
    ax.set_title("validation loss (dotted line: self-play starts — not comparable across it)")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend()

    for i, suite in enumerate(suite_names, start=1):
        ax = axes[i]
        for name in runs:
            points = bench.get(name, {}).get(suite)
            if not points:
                continue
            ax.plot([c for c, _ in points], [s for _, s in points],
                    marker="o", label=name, color=run_color[name])
        ax.set_xlabel("cycle")
        ax.set_ylabel("score %")
        ax.set_title(suite)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend()

    if len(panels) == 1:
        print("No runs have benchmark.csv results yet — showing validation loss only.")

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
    bench = {name: load_benchmark_curves(name) for name in names}

    print_table(runs, bench)
    print()
    plot_charts(runs, bench, args.output)


if __name__ == "__main__":
    main()
