# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the EXL3 Q4NX tensor-core GEMM against reconstruct + hgemm on Ada.

The Q4NX prefill path runs the same fused tensor-core GEMM as the
3-instruction codebooks.  On sm_89 the dispatch has to pick that kernel and
size its cooperative grid for a 128-SM part instead of reusing the Ampere
sample of the shared tuning table, so this harness reports both the selection
(shape index, grid) and the cost next to the reconstruct + hgemm path the
module uses when it is asked to reconstruct.

Every shape is checked against the reconstruct path: the packed weights are
decoded by the native ``reconstruct_q4nx`` kernel, rotated with the same
Hadamard the GEMM applies, and multiplied by cuBLAS.  The two paths differ only
in accumulation order, so the recorded absolute error is FP16-level.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. \\
        /root/venv/bin/python tests/benchmark_exl3_q4nx_gemm_ada.py --verify

Timings are the median of ``--iters`` CUDA-event intervals after ``--warmup``
runs.  ``--json`` writes the raw rows for the PR table.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from gptqmodel.exllamav3.ext import exllamav3_ext, prewarm_exllamav3_extension


# 1 / sqrt(42.5): the Q4NX normalisation the reconstruct path applies on output.
Q4NX_ALPHA = 1.0 / 42.5**0.5

DEFAULT_M = (16, 32, 64, 128, 256)
DEFAULT_K = (2048, 4096, 8192)
DEFAULT_N = (2048, 4096, 8192, 16384)
# step bits -> (K, half_k) as the GEMM dispatch sees them
STEP_BITS = {10: (2, True), 12: (3, False), 14: (3, True)}


def words_per_tile(step_bits: int) -> int:
    """uint16 words per 256-weight tile, matching bits_from_K() on the host."""
    return step_bits * 4


def _packed(k: int, n: int, step_bits: int, seed: int = 37):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, words_per_tile(step_bits)), generator=generator, dtype=torch.int16
    ).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def _fused(trellis, suh, svh, x, y, step_bits):
    """The Q4NX GEMM: fused input rotation, tensor-core GEMM and output scale."""
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured(
        trellis,
        suh,
        svh,
        step_bits / 4.0,
        None,
        False,
        False,
        x,
        y,
        False,  # fp16_gemv
        0,  # int8_gemv
        True,  # fused_gemm
        True,  # q4nx
    )


def _reconstruct_weight(trellis, step_bits):
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    K, half_k = STEP_BITS[step_bits]
    weight = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    torch.ops.gptqmodel_exllamav3.reconstruct_q4nx(weight, trellis, K, half_k)
    return weight


def _reconstruct_gemm(weight, suh, svh, x, y, xh):
    exllamav3_ext.had_r_128(x, xh, suh, None, 1.0)
    exllamav3_ext.hgemm(xh, weight, y)
    exllamav3_ext.had_r_128(y, y, None, svh, Q4NX_ALPHA)


def _time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # us
    return statistics.median(samples)


def _probe(m: int, k: int, n: int, step_bits: int, cc: int):
    """Selection record for the same call the module makes (fused, fp32 out)."""
    K, half_k = STEP_BITS[step_bits]
    kernel, shape_idx, num_sms, block_dim, q4nx_shape = torch.ops.gptqmodel_exllamav3.exl3_gemm_kernel_probe(
        cc, k, n, K, True, 3, half_k, True, m
    )
    return {
        "kernel": kernel,
        "shape_idx": shape_idx,
        "num_sms": num_sms,
        "block_dim": block_dim,
        "q4nx_shape": q4nx_shape,
    }


def _case(m, k, n, step_bits, weight, trellis, suh, svh, cc, warmup, iters, verify):
    generator = torch.Generator().manual_seed(19)
    x = (torch.randn((m, k), generator=generator) * 0.1).half().cuda()
    y = torch.empty((m, n), device="cuda", dtype=torch.float32)
    xh = torch.empty_like(x)
    dispatch = _fused(trellis, suh, svh, x, y, step_bits)
    selection = _probe(m, k, n, step_bits, cc)
    if selection["shape_idx"] != dispatch or selection["q4nx_shape"] != dispatch:
        raise AssertionError(
            f"dispatch {dispatch} does not match the probe {selection} for m={m} k={k} n={n} bits={step_bits}"
        )
    errors = None
    if verify:
        reference = torch.empty_like(y)
        _reconstruct_gemm(weight, suh, svh, x, reference, xh)
        diff = (y - reference).abs()
        errors = {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "ref_abs_mean": reference.abs().mean().item(),
        }
        del reference
    fused_us = _time(lambda: _fused(trellis, suh, svh, x, y, step_bits), warmup, iters)
    recon_us = _time(lambda: _reconstruct_gemm(weight, suh, svh, x, y, xh), warmup, iters)
    flops = 2.0 * m * k * n
    return {
        "m": m,
        "k": k,
        "n": n,
        "step_bits": step_bits,
        "dispatch": dispatch,
        "selection": selection,
        "fused_us": fused_us,
        "reconstruct_us": recon_us,
        "speedup": recon_us / fused_us,
        "tflops": flops / (fused_us * 1e6),
        "errors": errors,
    }


def _print_table(rows, verify: bool) -> None:
    header = (
        f"{'bits':>4s} {'m':>4s} {'k':>5s} {'n':>6s} {'shape':>5s} {'sms':>4s} {'fused us':>9s} "
        f"{'recon us':>9s} {'speedup':>7s} {'TFLOP/s':>8s}"
    )
    if verify:
        header += f" {'max_abs':>9s}"
    print(header)
    print("-" * len(header), flush=True)
    for row in rows:
        line = (
            f"{row['step_bits']:4d} {row['m']:4d} {row['k']:5d} {row['n']:6d} {row['dispatch']:5d} "
            f"{row['selection']['num_sms']:4d} {row['fused_us']:9.2f} {row['reconstruct_us']:9.2f} "
            f"{row['speedup']:7.2f} {row['tflops']:8.2f}"
        )
        if verify and row["errors"] is not None:
            line += f" {row['errors']['max_abs']:9.2e}"
        print(line, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M), help="row counts")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K), help="input widths")
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N), help="output widths")
    parser.add_argument(
        "--bits", type=int, nargs="+", default=sorted(STEP_BITS), choices=sorted(STEP_BITS), help="Q4NX step bits"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="compare every shape against reconstruct + hgemm")
    parser.add_argument("--json", default=None, help="write the raw measurements to this path")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required", flush=True)
        return 1
    if torch.version.hip is not None:
        print("the Q4NX GEMM is NVIDIA-only", flush=True)
        return 1
    if not prewarm_exllamav3_extension():
        print("the ExLlamaV3 extension is unavailable", flush=True)
        return 1

    device = torch.cuda.get_device_properties(0)
    # CC_ADA is the internal enum the dispatch uses (see exl3_devctx.cuh).
    cc = 3 if (device.major, device.minor) == (8, 9) else 0
    rows = []
    for step_bits in args.bits:
        for k in args.k:
            for n in args.n:
                trellis, suh, svh = _packed(k, n, step_bits)
                # The reconstruct path is timed for every shape, so its dense
                # weight is needed even when --verify is off.
                weight = _reconstruct_weight(trellis, step_bits)
                for m in args.m:
                    rows.append(_case(m, k, n, step_bits, weight, trellis, suh, svh, cc, args.warmup, args.iters, args.verify))
                    torch.cuda.empty_cache()
                del weight, trellis, suh, svh
                torch.cuda.empty_cache()
                print(f"done bits={step_bits} k={k} n={n}", flush=True)

    print(
        json.dumps(
            {
                "device": device.name,
                "capability": f"{device.major}.{device.minor}",
                "sm_count": device.multi_processor_count,
                "torch": torch.__version__,
                "warmup": args.warmup,
                "iters": args.iters,
            }
        ),
        flush=True,
    )
    _print_table(rows, args.verify)
    if args.verify:
        worst = max(row["errors"]["max_abs"] for row in rows)
        print(f"\nworst |fused - reconstruct| over {len(rows)} shapes: {worst:.3e}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
