#!/usr/bin/env python3
"""Benchmark QVQ CPU GEMV against dense reference."""

from __future__ import annotations

import argparse
import os
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


def _thread_affinities() -> list[int]:
    affinities = []
    for task in Path("/proc/self/task").iterdir():
        for line in (task / "status").read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                cpu_list = line.split(":", 1)[1].strip()
                if "," in cpu_list or "-" in cpu_list:
                    raise RuntimeError(f"thread {task.name} is not pinned to one CPU: {cpu_list}")
                affinities.append(int(cpu_list))
                break
    return sorted(affinities)


def _timings(fn, warmup=3, iterations=10, expected_cpus=None):
    for _ in range(warmup):
        fn()
    if expected_cpus is not None:
        actual_cpus = _thread_affinities()
        if sorted(set(actual_cpus)) != sorted(expected_cpus):
            raise RuntimeError(
                f"thread affinity mismatch: expected singleton placements {sorted(expected_cpus)}, got {actual_cpus}"
            )
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
    parser.add_argument("--expected-cpus", help="comma-separated singleton OpenMP CPU placements to assert")
    parser.add_argument(
        "--kernel-active-cpus",
        help="comma-separated active placements for a deliberately right-sized direct-kernel team",
    )
    parser.add_argument("--dense-active-cpus", help="comma-separated active placements for the dense series")
    parser.add_argument("--skip-reference", action="store_true", help="skip the reconstruct-per-call fallback timing")
    args = parser.parse_args()
    expected_cpus = None if args.expected_cpus is None else [int(cpu) for cpu in args.expected_cpus.split(",")]
    kernel_active_cpus = (
        expected_cpus if args.kernel_active_cpus is None else [int(cpu) for cpu in args.kernel_active_cpus.split(",")]
    )
    dense_active_cpus = (
        expected_cpus if args.dense_active_cpus is None else [int(cpu) for cpu in args.dense_active_cpus.split(",")]
    )
    if expected_cpus is not None and len(expected_cpus) != int(os.environ.get("OMP_NUM_THREADS", "0")):
        raise ValueError("--expected-cpus count must equal OMP_NUM_THREADS")
    if kernel_active_cpus is not None and (
        expected_cpus is None or not set(kernel_active_cpus).issubset(expected_cpus)
    ):
        raise ValueError("--kernel-active-cpus must be a subset of --expected-cpus")
    if dense_active_cpus is not None and (
        expected_cpus is None or not set(dense_active_cpus).issubset(expected_cpus)
    ):
        raise ValueError("--dense-active-cpus must be a subset of --expected-cpus")
    if expected_cpus is not None:
        # OMP_PROC_BIND pins the importing master before torch initializes and
        # can make its topology probe report one available CPU. Restore the
        # explicitly requested team size; OMP_PLACES still controls placement.
        torch.set_num_threads(len(expected_cpus))

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

    timing_kwargs = {"warmup": args.warmup, "iterations": args.iterations}
    kernel_times, kernel_out = _timings(run_kernel, expected_cpus=kernel_active_cpus, **timing_kwargs)
    dense_times, dense_out = _timings(run_dense, expected_cpus=dense_active_cpus, **timing_kwargs)
    if args.skip_reference:
        ref_times, ref_out = None, reference
    else:
        ref_times, ref_out = _timings(run_reference, expected_cpus=expected_cpus, **timing_kwargs)

    maxdiff = (kernel_out.float() - reference.float()).abs().max().item()
    maxdiff_ref = (ref_out.float() - reference.float()).abs().max().item()

    def _ms(values):
        return statistics.median(values) * 1000

    print(f"shape: x=[{m},{k}] weight=[{k},{n}] bits={bits} E={E} tiles={tiles}")
    print(f"  dense matmul median: {_ms(dense_times):.4f} ms")
    if ref_times is not None:
        print(f"  python fallback median: {_ms(ref_times):.4f} ms")
    print(f"  qvq_cpu_gemv median: {_ms(kernel_times):.4f} ms")
    print(f"  speedup vs dense: {_ms(dense_times) / _ms(kernel_times):.3f}x")
    if ref_times is not None:
        print(f"  speedup vs python fallback: {_ms(ref_times) / _ms(kernel_times):.3f}x")
    print(f"  max abs diff vs dense: {maxdiff:.6e} (fallback: {maxdiff_ref:.6e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
