"""Summarize the separately tagged arms of a paired Fisher benchmark."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def sample_median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle] + ordered[(len(ordered) - 1) // 2]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    samples = defaultdict(list)
    for row in report["results"]:
        samples[row["capture"]["comparison_arm"]].append(row["seconds"])
    baseline = sample_median(samples["current_gpu"])
    print(f"B={report['batch_size']} T={report['sequence_length']} rows={report['rows']} dtype={report['dtype']}")
    print(f"{'Collector':<20} {'N':>3} {'Median (s)':>12} {'Min (s)':>10} {'Max (s)':>10} {'Speedup':>9}")
    for arm in ("current_gpu", "grouped_gpu", "cpu_finalize"):
        values = samples[arm]
        median = sample_median(values)
        print(
            f"{arm:<20} {len(values):>3} {median:>12.6f} {min(values):>10.6f} {max(values):>10.6f} {baseline / median:>8.3f}x"
        )
    print("Warmed complete-collector timing; inspect paired variation before attributing small CPU-only gains.")


if __name__ == "__main__":
    main()
