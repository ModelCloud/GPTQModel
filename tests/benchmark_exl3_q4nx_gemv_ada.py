# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""Benchmark the EXL3 Q4NX small-batch (m <= 8) decode path on Ada.

Compares the FP16 GEMV kernel (``fp16_gemv=True``) with the block-pipelined
fused GEMM fallback (``fp16_gemv=False``) for random Q4NX trellis weights at
2.5 / 3.0 / 3.5 bpw.  The Q4NX codebook is defined for 10 / 12 / 14 step bits,
so 4.0 bpw is reported as unsupported rather than benchmarked.

The default matrix is every ``k`` in 2048 / 4096 / 8192 crossed with every
``n`` in 2048 / 4096 / 8192 / 12288 / 16384 and every ``m`` in 1 / 2 / 4 / 8.
The GEMV is only launched for shapes the dispatch admits; the other rows are
timed on the fallback alone and marked ``n/a``, so the table doubles as the
regression report for the enablement.

Two timings are reported per path:

``cold us``
    L2 flushed (160 MiB read) before every launch, CUDA-event timed, median of
    20 launches.  This is the DRAM-bound weight-streaming case of a real model,
    where each layer's trellis is read once per token.
``warm us``
    Steady state: 100 back-to-back launches between one event pair.  The
    trellis stays resident in the 4090's 72 MiB L2, so this isolates the
    kernel's compute/fixed cost from DRAM bandwidth.

``GB/s`` is the effective bandwidth of the packed trellis, i.e.
``k * n * bits / 8`` bytes divided by the measured time.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \\
        /root/venv/bin/python tests/benchmark_exl3_q4nx_gemv_ada.py

Add ``--check`` to compare both paths against the reconstructed FP32 weight
and an FP64 matmul (the existing Q4NX tests' 2e-3 tolerance) and to report the
GEMV-to-fallback difference.  ``--json`` writes the measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear


RATES = (2.5, 3.0, 3.5, 4.0)
KS = (2048, 4096, 8192)
NS = (2048, 4096, 8192, 12288, 16384)
SHAPES = tuple((k, n) for k in KS for n in NS)
BATCHES = (1, 2, 4, 8)
WARMUPS = 5
REPS = 20
LOOP = 100
PASSES = 2
FLUSH_BYTES = 160 * 1024 * 1024
GEMV_DISPATCH = 90
ATOL = 2e-3
RTOL = 2e-3


def packed_projection(k: int, n: int, bits: float, seed: int = 37):
    """Random packed Q4NX trellis plus per-channel scales (as in the decode tests)."""
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, int(16 * bits)), generator=generator, dtype=torch.int16
    ).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def make_input(m: int, k: int, seed: int = 19) -> torch.Tensor:
    return (torch.randn((m, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _launch(op, weights, bits, x, y, gemv: bool) -> int:
    trellis, suh, svh = weights
    return op(trellis, suh, svh, bits, None, False, False, x, y, gemv, 0, False, q4nx=True)


def _run(op, weights, bits, x, y, gemv: bool) -> int:
    path = _launch(op, weights, bits, x, y, gemv)
    torch.cuda.synchronize()
    return path


def _ramp(op, weights, bits, m: int) -> None:
    """Run enough traffic to leave the idle SM clock behind before the first sample."""
    k = weights[1].numel()
    n = weights[2].numel()
    x = make_input(m, k)
    y = torch.empty((m, n), device="cuda", dtype=torch.float32)
    for _ in range(300):
        _launch(op, weights, bits, x, y, True)
    torch.cuda.synchronize()


def _time_us(op, weights, bits, x, y, gemv: bool, flush, sink) -> tuple[float, float]:
    """Return (cold us, warm us); best of two passes."""
    cold_passes, warm_passes = [], []
    for _ in range(PASSES):
        for _ in range(WARMUPS):
            _launch(op, weights, bits, x, y, gemv)
        torch.cuda.synchronize()

        # Cold: the flush kernel runs ahead of each timed launch, so the launch
        # command is already queued when the start event fires (no host gap).
        pairs = []
        for _ in range(REPS):
            torch.sum(flush, 0, out=sink)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _launch(op, weights, bits, x, y, gemv)
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        cold_passes.append(statistics.median([start.elapsed_time(end) * 1_000.0 for start, end in pairs]))

        # Warm: back-to-back launches, one event pair around the whole batch.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(LOOP):
            _launch(op, weights, bits, x, y, gemv)
        end.record()
        end.synchronize()
        warm_passes.append(start.elapsed_time(end) * 1_000.0 / LOOP)
    return min(cold_passes), min(warm_passes)


def _reference(weights, x: torch.Tensor) -> torch.Tensor:
    """Reconstruct the FP32 weight and multiply in FP64."""
    trellis, suh, svh = weights
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=suh.numel(),
        out_features=svh.numel(),
        name="q4nx_reference",
        tensors={
            "trellis": trellis,
            "suh": suh,
            "svh": svh,
            "q4nx": torch.ones((), dtype=torch.int32, device=x.device),
        },
    ).eval()
    dense = module.get_weight_tensor(torch.float32)
    expected = (x.double() @ dense.double()).float()
    del dense
    return expected


def _max_rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-6))


def run(args) -> list[dict]:
    if not prewarm_exllamav3_extension():
        raise SystemExit("ExLlamaV3 extension is not available")
    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured
    flush = torch.ones(FLUSH_BYTES // 2, dtype=torch.int16, device="cuda")
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    rows = []
    for bits in args.bits:
        # Q4NX is defined for 10 / 12 / 14 step bits, i.e. 2.5 / 3.0 / 3.5 bpw.
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
                    "ref_rel_err": None,
                    "fallback_rel_err": None,
                    "gemv_vs_fallback": None,
                }
                if args.check:
                    expected = _reference(weights, x)
                    torch.testing.assert_close(y, expected, rtol=RTOL, atol=ATOL)
                    row["fallback_rel_err"] = _max_rel(y, expected)
                    if gemv:
                        _run(op, weights, bits, x, y, True)
                        torch.testing.assert_close(y, expected, rtol=RTOL, atol=ATOL)
                        row["ref_rel_err"] = _max_rel(y, expected)
                        fallback = torch.empty_like(y)
                        _run(op, weights, bits, x, fallback, False)
                        row["gemv_vs_fallback"] = _max_rel(y, fallback)
                        del fallback
                    else:
                        row["ref_rel_err"] = row["fallback_rel_err"]
                    del expected
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

    rows = run(args)
    header = (
        f"{'bits':>5} {'k':>5} {'n':>5} {'m':>2} {'path':>6} | {'cold us':>8} {'GB/s':>7} | "
        f"{'fallback':>8} {'GB/s':>7} | {'cold':>6} | {'warm us':>8} {'GB/s':>7} | "
        f"{'fallback':>8} {'GB/s':>7} | {'warm':>6}"
    )
    if args.check:
        header += f" | {'ref err':>8} {'vs fb':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        path = "gemv" if row["gemv"] else f"#{row['dispatch']}"
        gemv_cold = f"{row['gemv_cold_us']:>8.2f}" if row["gemv"] else f"{'n/a':>8}"
        gemv_cold_gbs = f"{row['gemv_cold_gbs']:>7.1f}" if row["gemv"] else f"{'':>7}"
        gemv_warm = f"{row['gemv_warm_us']:>8.2f}" if row["gemv"] else f"{'n/a':>8}"
        gemv_warm_gbs = f"{row['gemv_warm_gbs']:>7.1f}" if row["gemv"] else f"{'':>7}"
        cold_speedup = f"{row['cold_speedup']:>5.2f}x" if row["gemv"] else f"{'':>6}"
        warm_speedup = f"{row['warm_speedup']:>5.2f}x" if row["gemv"] else f"{'':>6}"
        line = (
            f"{row['bits']:>5.1f} {row['k']:>5} {row['n']:>5} {row['m']:>2} {path:>6} | "
            f"{gemv_cold} {gemv_cold_gbs} | "
            f"{row['fallback_cold_us']:>8.2f} {row['fallback_cold_gbs']:>7.1f} | {cold_speedup} | "
            f"{gemv_warm} {gemv_warm_gbs} | "
            f"{row['fallback_warm_us']:>8.2f} {row['fallback_warm_gbs']:>7.1f} | {warm_speedup}"
        )
        if args.check:
            line += f" | {row['ref_rel_err']:>8.2e} {row['gemv_vs_fallback'] or 0.0:>8.2e}"
        print(line)
    launched = [row for row in rows if row["gemv"]]
    if launched:
        cold = statistics.geometric_mean(row["cold_speedup"] for row in launched)
        warm = statistics.geometric_mean(row["warm_speedup"] for row in launched)
        print(f"\nGEMV launched in {len(launched)}/{len(rows)} cells")
        print(f"geometric-mean speedup over the block-pipelined kernel: cold {cold:.2f}x, warm {warm:.2f}x")
        regressions = [row for row in launched if row["cold_speedup"] < 1.0 or row["warm_speedup"] < 1.0]
        for row in regressions:
            print(
                f"regression: {row['bits']} bpw k={row['k']} n={row['n']} m={row['m']} "
                f"cold {row['cold_speedup']:.2f}x warm {row['warm_speedup']:.2f}x"
            )
        if not regressions:
            print("no regression cells")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
