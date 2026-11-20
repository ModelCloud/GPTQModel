# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""Benchmark the EXL3 Q4NX small-batch (m <= 8) decode path on Blackwell.

The Ada sweep (``tests/benchmark_exl3_q4nx_gemv_ada.py``) admitted the packed
GEMV on sm_89.  Blackwell (sm_120) was left unmeasured, so
``exl3_gemv_try_launch`` still declines Q4NX there and decode runs the
block-pipelined kernel.  This script runs the same matrix on a Blackwell part
so the decline has data behind it.

It reuses the Ada harness's timing helpers on purpose: identical cold/warm
definitions, identical flush and ramp, identical shape grid, so the two tables
are directly comparable.

    PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \\
        /root/venv/bin/python tests/benchmark_exl3_q4nx_gemv_blackwell.py

On a tree whose gate declines sm_120 every cell reports ``declined`` and only
the block-pipelined kernel is timed -- that is the number Blackwell actually
runs.  With the gate opened for measurement (see ``docs/exl3-native-decode.md``)
the GEMV column is filled too, and ``--check`` reports how far it is from the
reconstructed FP32 weight.  The sm_120 GEMV is **not numerically correct**
today, which is why the gate stays closed; the column is recorded to size the
prize, not to recommend the path.

``cold us`` flushes L2 before every launch, ``warm us`` is the steady state
(trellis resident in the 96 MiB L2).  ``GB/s`` is the packed trellis
(``k * n * bits / 8`` bytes) over the measured time.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from tests.benchmark_exl3_q4nx_gemv_ada import (
    ATOL,
    BATCHES,
    FLUSH_BYTES,
    GEMV_DISPATCH,
    RATES,
    RTOL,
    SHAPES,
    _max_rel,
    _ramp,
    _reference,
    _run,
    _time_us,
    make_input,
    packed_projection,
)


TOLERANCE = 2e-3


def run(args) -> list[dict]:
    if not prewarm_exllamav3_extension():
        raise SystemExit("ExLlamaV3 extension is not available")
    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured
    flush = torch.ones(FLUSH_BYTES // 2, dtype=torch.int16, device="cuda")
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    rows = []
    for bits in args.bits:
        if round(4 * bits) not in (10, 12, 14):
            print(f"Q4NX {bits} bpw: skipped (the codebook supports 10/12/14 step bits, i.e. 2.5/3.0/3.5 bpw)")
            continue
        for k, n in args.shapes:
            weights = packed_projection(k, n, bits)
            packed_bytes = k * n * bits / 8.0
            for m in args.batches:
                x = make_input(m, k)
                y = torch.empty((m, n), device="cuda", dtype=torch.float32)
                _ramp(op, weights, bits, m)
                dispatch = _run(op, weights, bits, x, y, True)
                gemv = dispatch == GEMV_DISPATCH
                if gemv:
                    gemv_cold, gemv_warm = _time_us(op, weights, bits, x, y, True, flush, sink)
                else:
                    gemv_cold = gemv_warm = None
                fallback_cold, fallback_warm = _time_us(op, weights, bits, x, y, False, flush, sink)
                row = {
                    "bits": bits,
                    "k": k,
                    "n": n,
                    "m": m,
                    "packed_bytes": packed_bytes,
                    "dispatch": dispatch,
                    "gemv": gemv,
                    "gemv_cold_us": gemv_cold,
                    "gemv_cold_gbs": packed_bytes / gemv_cold / 1e3 if gemv else None,
                    "gemv_warm_us": gemv_warm,
                    "gemv_warm_gbs": packed_bytes / gemv_warm / 1e3 if gemv else None,
                    "fallback_cold_us": fallback_cold,
                    "fallback_cold_gbs": packed_bytes / fallback_cold / 1e3,
                    "fallback_warm_us": fallback_warm,
                    "fallback_warm_gbs": packed_bytes / fallback_warm / 1e3,
                    "cold_speedup": fallback_cold / gemv_cold if gemv else None,
                    "warm_speedup": fallback_warm / gemv_warm if gemv else None,
                    "gemv_rel_err": None,
                    "packed_rel_err": None,
                }
                if args.check:
                    expected = _reference(weights, x)
                    packed = torch.empty_like(y)
                    _run(op, weights, bits, x, packed, False)
                    row["packed_rel_err"] = _max_rel(packed, expected)
                    torch.testing.assert_close(packed, expected, rtol=RTOL, atol=ATOL)
                    if gemv:
                        _run(op, weights, bits, x, y, True)
                        row["gemv_rel_err"] = _max_rel(y, expected)
                    del expected, packed
                rows.append(row)
                del x, y
            del weights
            torch.cuda.empty_cache()
    del flush, sink
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=float, nargs="+", default=list(RATES))
    parser.add_argument("--shapes", type=int, nargs="+", default=None, help="flattened k n pairs")
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--check", action="store_true", help="compare both paths against the reconstructed FP32 weight")
    parser.add_argument("--json", default=None, help="write the measurements as JSON")
    args = parser.parse_args()
    args.shapes = list(zip(args.shapes[0::2], args.shapes[1::2])) if args.shapes else list(SHAPES)

    device = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{device} (sm_{major}{minor}), {torch.cuda.get_device_properties(0).multi_processor_count} SMs")

    rows = run(args)
    header = (
        f"{'bits':>5} {'k':>5} {'n':>5} {'m':>2} {'path':>9} | {'gemv us':>8} {'GB/s':>7} | "
        f"{'packed us':>9} {'GB/s':>7} | {'speedup':>7} | {'warm us':>8} | {'packed':>8} | {'warm':>6}"
    )
    if args.check:
        header += f" | {'gemv err':>8} {'packed err':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        path = "gemv" if row["gemv"] else ("declined" if row["dispatch"] != GEMV_DISPATCH else f"#{row['dispatch']}")
        gemv_cold = f"{row['gemv_cold_us']:>8.2f}" if row["gemv"] else f"{'n/a':>8}"
        gemv_gbs = f"{row['gemv_cold_gbs']:>7.1f}" if row["gemv"] else f"{'':>7}"
        speedup = f"{row['cold_speedup']:>6.2f}x" if row["gemv"] else f"{'':>7}"
        gemv_warm = f"{row['gemv_warm_us']:>8.2f}" if row["gemv"] else f"{'n/a':>8}"
        warm_speedup = f"{row['warm_speedup']:>5.2f}x" if row["gemv"] else f"{'':>6}"
        line = (
            f"{row['bits']:>5.1f} {row['k']:>5} {row['n']:>5} {row['m']:>2} {path:>9} | "
            f"{gemv_cold} {gemv_gbs} | "
            f"{row['fallback_cold_us']:>9.2f} {row['fallback_cold_gbs']:>7.1f} | {speedup} | "
            f"{gemv_warm} | {row['fallback_warm_us']:>8.2f} | {warm_speedup}"
        )
        if args.check:
            gemv_err = f"{row['gemv_rel_err']:>8.2e}" if row["gemv_rel_err"] is not None else f"{'n/a':>8}"
            line += f" | {gemv_err} {row['packed_rel_err']:>10.2e}"
        print(line)
    launched = [row for row in rows if row["gemv"]]
    declined = [row for row in rows if not row["gemv"]]
    print(f"\nGEMV launched in {len(launched)}/{len(rows)} cells, declined in {len(declined)}")
    if launched:
        cold = statistics.geometric_mean(row["cold_speedup"] for row in launched)
        warm = statistics.geometric_mean(row["warm_speedup"] for row in launched)
        print(f"geometric-mean speedup over the block-pipelined kernel: cold {cold:.2f}x, warm {warm:.2f}x")
        regressions = [row for row in launched if row["cold_speedup"] < 1.0 or row["warm_speedup"] < 1.0]
        for row in regressions:
            print(
                f"regression: {row['bits']} bpw k={row['k']} n={row['n']} m={row['m']} "
                f"cold {row['cold_speedup']:.2f}x warm {row['warm_speedup']:.2f}x"
            )
        if args.check:
            wrong = [row for row in launched if (row["gemv_rel_err"] or 0.0) > TOLERANCE]
            print(f"GEMV outside the {TOLERANCE:g} reference tolerance in {len(wrong)}/{len(launched)} cells")
            for row in wrong[:10]:
                print(
                    f"  wrong: {row['bits']} bpw k={row['k']} n={row['n']} m={row['m']} "
                    f"gemv {row['gemv_rel_err']:.2e} vs packed {row['packed_rel_err']:.2e}"
                )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
