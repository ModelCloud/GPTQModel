#!/usr/bin/env python3
"""Compare two Pangolin CPU benchmark Markdown files and emit an A/B table.

Example:
    python scripts/benchmark_pangolin_cpu_compare.py \
        /tmp/pangolin_cpu_prev.md /tmp/pangolin_cpu_curr.md
"""
import argparse
import math
import re
from collections import defaultdict


def parse(path: str):
    text = open(path).read()
    sections = re.split(r"## (\w[\w_]+) \(bits=(\d+)", text)
    rows = []
    for i in range(1, len(sections), 3):
        set_name = sections[i]
        bits = int(sections[i + 1])
        body = sections[i + 2]
        for line in body.splitlines():
            m = re.match(
                r"^(\d+)\s*x\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*$",
                line.strip(),
            )
            if m:
                rows.append(
                    (
                        set_name,
                        bits,
                        int(m.group(1)),
                        int(m.group(2)),
                        int(m.group(3)),
                        float(m.group(4)),
                        float(m.group(5)),
                        float(m.group(6)),
                    )
                )
    return rows


def geomean(values):
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prev")
    parser.add_argument("curr")
    parser.add_argument("--shape", default=None)
    parser.add_argument("--bits", type=int, default=None)
    args = parser.parse_args()

    prev = parse(args.prev)
    curr = parse(args.curr)

    prev_by_key = {(r[0], r[1], r[2], r[3], r[4]): r for r in prev}
    curr_by_key = {(r[0], r[1], r[2], r[3], r[4]): r for r in curr}

    keys = sorted(set(prev_by_key.keys()) & set(curr_by_key.keys()))
    if args.shape:
        keys = [k for k in keys if k[0] == args.shape]
    if args.bits is not None:
        keys = [k for k in keys if k[1] == args.bits]

    print("| set | bits | K x N | M | prev ms | curr ms | kernel ratio | speedup ratio |")
    print("|-----|------|-------|---|--------:|--------:|-------------:|--------------:|")
    by_set_bits = defaultdict(list)
    for k in keys:
        set_name, bits, K, N, M = k
        pr = prev_by_key[k]
        cr = curr_by_key[k]
        kernel_ratio = pr[5] / cr[5]
        speedup_ratio = cr[7] / pr[7]
        by_set_bits[(set_name, bits)].append((kernel_ratio, speedup_ratio))
        print(
            f"| {set_name} | {bits} | {K} x {N} | {M} | "
            f"{pr[5]:.3f} | {cr[5]:.3f} | {kernel_ratio:.2f}x | {speedup_ratio:.2f}x |"
        )

    print()
    print("## Geomean summary")
    print("| set | bits | kernel ratio | speedup ratio |")
    print("|-----|------|-------------:|--------------:|")
    for (set_name, bits), ratios in sorted(by_set_bits.items()):
        kr = geomean([r[0] for r in ratios])
        sr = geomean([r[1] for r in ratios])
        print(f"| {set_name} | {bits} | {kr:.3f}x | {sr:.3f}x |")

    all_kr = [r[0] for ratios in by_set_bits.values() for r in ratios]
    all_sr = [r[1] for ratios in by_set_bits.values() for r in ratios]
    if all_kr:
        print(f"| all | all | {geomean(all_kr):.3f}x | {geomean(all_sr):.3f}x |")


if __name__ == "__main__":
    main()
