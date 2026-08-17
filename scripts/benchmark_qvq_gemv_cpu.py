#!/usr/bin/env python3
"""Benchmark QVQ CPU GEMV against dense reference."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight  # noqa: E402
from gptqmodel.quantization.qvq_rates import qvq_transition_bits  # noqa: E402
from gptqmodel.utils.planar_packing import planar_pack_rows  # noqa: E402
from gptqmodel.utils.qvq_cpu import qvq_cpu_gemv  # noqa: E402


def _timings(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--bits", type=float, default=3.5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    k = args.k
    n = args.n
    m = args.m
    bits = args.bits
    E = qvq_transition_bits(bits)
    tiles = (k // 16) * (n // 16)

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(12345)
    edges = torch.randint(0, 1 << E, (128, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, E).T.contiguous().cpu()
    x = torch.randn((m, k), generator=generator, dtype=torch.float32).cpu()

    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
    )
    reference = x @ inner

    def run_kernel():
        return qvq_cpu_gemv(x, trellis, bits, out_features=n)

    def run_dense():
        return x @ inner

    def run_reference():
        inner_ref = reconstruct_qvq_inner_weight(trellis, bits=bits, in_features=k, out_features=n)
        return x @ inner_ref

    kernel_times, kernel_out = _timings(run_kernel, warmup=args.warmup, iterations=args.iterations)
    dense_times, dense_out = _timings(run_dense, warmup=args.warmup, iterations=args.iterations)
    ref_times, ref_out = _timings(run_reference, warmup=args.warmup, iterations=args.iterations)

    maxdiff = (kernel_out.float() - reference.float()).abs().max().item()
    maxdiff_ref = (ref_out.float() - reference.float()).abs().max().item()

    def _ms(values):
        return statistics.median(values) * 1000

    print(f"shape: x=[{m},{k}] weight=[{k},{n}] bits={bits} E={E} tiles={tiles}")
    print(f"  dense matmul median: {_ms(dense_times):.4f} ms")
    print(f"  python fallback median: {_ms(ref_times):.4f} ms")
    print(f"  qvq_cpu_gemv median: {_ms(kernel_times):.4f} ms")
    print(f"  speedup vs dense: {_ms(dense_times) / _ms(kernel_times):.3f}x")
    print(f"  speedup vs python fallback: {_ms(ref_times) / _ms(kernel_times):.3f}x")
    print(f"  max abs diff vs dense: {maxdiff:.6e} (fallback: {maxdiff_ref:.6e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
