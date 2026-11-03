#!/usr/bin/env python3
"""Compare model training runs: a summary table plus charts — val loss and
benchmark suite scores, each vs. cycle.

Usage:
    python training/compare.py                  # compare every run under training/models/runs/
    python training/compare.py pente_v1 pente_v1_small
    python training/compare.py -o my_chart.png pente_v1 pente_v1_small
"""

import argparse
import datetime
import json
import re
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
    """Returns {column: [(cycle, score_pct), ...]}, sorted by cycle. Arena
    rows are split by column the same way print_cycle_table does (sims
    tier / opponent via arena_column(), elo-ladder matches excluded via
    is_roster_match()) -- otherwise different sims tiers and opponents
    collapse into one meaningless "latest" value under a shared "arena" key."""
    curves: dict = {}
    for row in benchmark.read_results(name):
        cycle = cycle_from_checkpoint(row["checkpoint"])
        if cycle is None:
            continue
        if row["suite"] == "arena":
            if is_roster_match(row["evaluator"]):
                continue
            col = arena_column(row["evaluator"])
        else:
            col = row["suite"]
        curves.setdefault(col, []).append((cycle, float(row["score_pct"])))
    for points in curves.values():
        points.sort(key=lambda p: p[0])
    return curves


def load_elo(name: str) -> dict[int, float]:
    """Returns {cycle: elo}."""
    path = model.run_dir(name) / "elo.json"
    if not path.exists():
        return {}
    return {entry["cycle"]: entry["elo"] for entry in json.loads(path.read_text())}


def split_phase(metrics: list) -> tuple[list, list]:
    """Bootstrap trains repeated epochs over one static dataset; self-play's
    buffer shifts round to round, so they're measured differently (see
    benchmark.py's BOOTSTRAP_SUITES/SELFPLAY_SUITES) and shown separately."""
    bootstrap = [m for m in metrics if m.get("phase") != "selfplay"]
    selfplay = [m for m in metrics if m.get("phase") == "selfplay"]
    return bootstrap, selfplay


def print_table(label: str, runs: dict, bench: dict, elo: dict, cycles_of_interest: dict,
                 extra_cols: list, include_suites: bool = True):
    """One row per run. extra_cols: [(header, fn(name) -> str), ...], for
    columns that aren't a plain suite lookup (e.g. architecture, elo)."""
    suite_names = sorted({suite for curves in bench.values() for suite in curves})
    if not include_suites:
        suite_names = [s for s in suite_names if s not in ("open-three-suite", "value-suite-value-sign")]
    headers = [h for h, _ in extra_cols] + suite_names

    rows = []
    row_names = []
    for name in runs:
        cycles = cycles_of_interest.get(name, set())
        if not cycles:
            continue
        row_names.append(name)
        row = [fn(name) for _, fn in extra_cols]
        for suite in suite_names:
            points = [(c, s) for c, s in bench.get(name, {}).get(suite, []) if c in cycles]
            row.append(f"{points[-1][1]:.1f}%" if points else "-")
        rows.append(row)

    if not rows:
        return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{label}")
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    if suite_names:
        print("(suite columns show the latest/highest-cycle score within this phase)")


def arena_column(evaluator: str) -> str:
    """'nn@1600-vs-heuristic@1600' -> 'arena@1600'. 'nn@400-vs-prev_best_snapshot@400'
    -> 'promotion@400'. Falls back to the raw evaluator string for anything
    else, rather than hiding it."""
    m = re.match(r"nn@(\d+)-vs-heuristic@\d+", evaluator)
    if m:
        return f"arena@{m.group(1)}"
    m = re.match(rf"nn@(\d+)-vs-{re.escape(Path(benchmark.PROMOTION_SNAPSHOT_NAME).stem)}@\d+", evaluator)
    if m:
        return f"promotion@{m.group(1)}"
    return evaluator


def is_roster_match(evaluator: str) -> bool:
    """Elo-ladder matches (vs. whichever roster checkpoint was the most
    recent entry at the time — a different opponent name every match, so
    not a stable table column). Represented via the elo column instead."""
    return re.match(r"nn@\d+-vs-model_iter\d+@\d+$", evaluator) is not None


def print_cycle_table(name: str, phase_label: str, metrics: list, bench_rows: list,
                       elo: dict, include_suites: bool):
    """Full per-cycle history for one run's bootstrap or self-play phase:
    val loss / promotion from metrics.jsonl, joined with every suite and
    arena sims-tier recorded for that cycle's checkpoint, plus the elo rating
    for cycles that have one. A tier that was never run for a given cycle
    (e.g. arena@1600+ only exists past the 90% escalation threshold) prints
    as N/A rather than being silently omitted."""
    if not metrics:
        return

    cycle_set = {m["cycle"] for m in metrics}
    by_cycle: dict = {}
    columns: list = []
    for row in bench_rows:
        cycle = cycle_from_checkpoint(row["checkpoint"])
        if cycle is None or cycle not in cycle_set:
            continue
        if row["suite"] == "arena":
            if is_roster_match(row["evaluator"]):
                continue
            col = arena_column(row["evaluator"])
        else:
            if not include_suites:
                continue
            col = row["suite"]
        if col not in columns:
            columns.append(col)
        by_cycle.setdefault(cycle, {})[col] = f"{float(row['score_pct']):.1f}"

    def sort_key(col):
        m = re.match(r"(arena|promotion)@(\d+)", col)
        return (0, m.group(1), int(m.group(2))) if m else (1, col, 0)
    columns.sort(key=sort_key)

    # Older cycles were recorded before metrics.jsonl tracked a timestamp —
    # fall back to the checkpoint file's own mtime, which is exactly when
    # that cycle finished regardless.
    def completed_at(m: dict) -> datetime.datetime | None:
        if "timestamp" in m:
            return datetime.datetime.fromisoformat(m["timestamp"])
        path = Path(m["iter_path"])
        return datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone() if path.exists() else None

    times = [completed_at(m) for m in metrics]

    headers = ["cycle", "val_loss", "promoted", "completed", "duration", "elo", *columns]
    rows = []
    for i, m in enumerate(metrics):
        t = times[i]
        completed_str = t.strftime("%Y-%m-%d %H:%M:%S") if t else "N/A"
        duration_str = "-"
        if t and i > 0 and times[i - 1]:
            duration_str = model.bootstrap.format_eta((t - times[i - 1]).total_seconds())
        rows.append([
            str(m["cycle"]),
            f"{m['val_loss']:.4f}",
            "yes" if m.get("promoted") else "no",
            completed_str,
            duration_str,
            f"{elo[m['cycle']]:.0f}" if m["cycle"] in elo else "-",
            *[by_cycle.get(m["cycle"], {}).get(c, "N/A") for c in columns],
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{name}: {phase_label} per-cycle history")
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


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
    elo = {name: load_elo(name) for name in names}

    bootstrap_metrics = {}
    selfplay_metrics = {}
    for name, (_, metrics) in runs.items():
        bootstrap_metrics[name], selfplay_metrics[name] = split_phase(metrics)
    bootstrap_cycles = {n: {m["cycle"] for m in ms} for n, ms in bootstrap_metrics.items()}
    selfplay_cycles = {n: {m["cycle"] for m in ms} for n, ms in selfplay_metrics.items()}

    print_table(
        "Bootstrap", runs, bench, elo, bootstrap_cycles,
        extra_cols=[
            ("name", lambda n: n),
            ("architecture", lambda n: f"{runs[n][0]['model']['channels']}ch/{runs[n][0]['model']['res_blocks']}rb"),
            ("cycles", lambda n: str(runs[n][0]["cycles_run"])),
            ("best val loss", lambda n: f"{runs[n][0]['best_val_loss']:.4f}"),
            ("elapsed", lambda n: model.bootstrap.format_eta(runs[n][0]["elapsed_s"])),
        ],
    )
    print_table(
        "Self-play", runs, bench, elo, selfplay_cycles,
        extra_cols=[
            ("name", lambda n: n),
            ("rounds", lambda n: str(len(selfplay_metrics[n]))),
            ("elo", lambda n: f"{elo[n][max(elo[n])]:.0f}" if elo.get(n) else "-"),
        ],
        include_suites=False,
    )

    for name in runs:
        bench_rows = benchmark.read_results(name)
        print_cycle_table(name, "bootstrap", bootstrap_metrics[name], bench_rows, elo[name], include_suites=True)
        print_cycle_table(name, "self-play", selfplay_metrics[name], bench_rows, elo[name], include_suites=False)
    print()
    plot_charts(runs, bench, args.output)


if __name__ == "__main__":
    main()
