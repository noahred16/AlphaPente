#!/usr/bin/env python3
"""Print a human-readable summary of reports/pente/benchmark.csv, grouped by checkpoint."""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "reports", "pente", "benchmark.csv")


def date_arg(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD")


def benchmark_label(suite, evaluator):
    if suite == "arena":
        return evaluator
    if evaluator == "nn":
        return suite
    return f"{suite} [{evaluator}]"


def checkpoint_label(checkpoint):
    return os.path.basename(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to benchmark.csv")
    parser.add_argument(
        "--start-date",
        type=date_arg,
        default=date_arg("2026-07-04"),
        help="Only include rows on/after this date (YYYY-MM-DD). Default: 2026-07-04",
    )
    args = parser.parse_args()

    start_date = args.start_date

    # checkpoint -> benchmark -> list of (passed, total, score_pct)
    checkpoints = defaultdict(lambda: defaultdict(list))
    checkpoint_order = []

    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%S")
            if ts < start_date:
                continue

            ckpt = checkpoint_label(row["checkpoint"])
            label = benchmark_label(row["suite"], row["evaluator"])
            checkpoints[ckpt][label].append(
                (int(row["passed"]), int(row["total"]), float(row["score_pct"]))
            )
            if ckpt not in checkpoint_order:
                checkpoint_order.append(ckpt)

    if not checkpoint_order:
        print(f"No benchmark results on/after {args.start_date}.")
        return

    for ckpt in checkpoint_order:
        print(f"\n{ckpt}")
        print("-" * len(ckpt))
        benchmarks = checkpoints[ckpt]
        width = max(len(label) for label in benchmarks) + 2
        for label in benchmarks:
            runs = benchmarks[label]
            avg_score = sum(score for _, _, score in runs) / len(runs)
            passed = sum(p for p, _, _ in runs)
            total = sum(t for _, t, _ in runs)
            suffix = f"  (n={len(runs)})" if len(runs) > 1 else ""
            print(f"  {label:<{width}} {avg_score:6.1f}%  ({passed}/{total}){suffix}")


if __name__ == "__main__":
    main()
