#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark vs Marlin W8A16 microbenchmark (bits=8, group_size=-1).

This is the benchmark used for the 4090 numbers on PR #3490. It is deliberately
self-contained so anyone can rerun it on their own card:

    python scripts/benchmark_allspark_w8a16.py

Protocol, matching the PR body: random uint8 weights and per-output-channel
fp16 scales, K=4096 N=11008, median of ``--iters`` timed calls after
``--warmup`` warmups. Marlin is called through ``gptq_marlin_gemm`` with the
same GPTQ-format ``qweight``/``scales``.

``--awq`` switches the same protocol to the AWQ 8-bit path (per-channel zero
point, ``group_size=-1``). Marlin has no AWQ entry point, so AWQ reports only
AllSpark timings plus its fp64-oracle error. This is the mode used for the
CUDA and ROCm AWQ numbers on PR #3490; it works on both by selecting the
extension that matches the runtime (``torch.version.hip``).

Three timing modes are reported:

``hot``
    Back-to-back calls, so the weight matrix stays in L2 when the card has
    enough of it. This is the number to quote when comparing against a table
    that also ran hot, and it is *not* a memory-bandwidth measurement.

``flushed``
    A 512 MiB random fp32 buffer is reduced before every timed call so the
    weights are read from device memory (GDDR6X/HBM, whichever the card has).
    This is the memory-roof comparison. The buffer is random rather than zero
    filled: a zero or compressible buffer can be squashed by memory
    compression and fail to evict the weights.

``graph``
    The op is captured into a CUDA graph and replayed, which removes the
    ~8 us of per-call launch overhead that the event-based modes include.
    This is the closest measurement of the kernel itself.

Reported GB/s is ``K * N / time``, i.e. weight bytes per second, the same
convention as the PR body. It counts the weight read only, so values above the
card's memory bandwidth mean the weights came from cache.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


DEFAULT_SHAPES = ((4096, 11008),)
DEFAULT_M = (1, 8, 16, 24, 32, 64, 128, 256)
FLUSH_BYTES = 512 * 2**20


def pack_gptq_8bit(weight: torch.Tensor):
    """Pack a (K, N) weight the way a GPTQ 8-bit checkpoint stores it."""
    k, n = weight.shape
    scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-8)
    codes = (weight / scale).round().add(128).clamp(0, 255).to(torch.int32)
    shifts = (torch.arange(4, device=weight.device, dtype=torch.int32) * 8).view(1, 4, 1)
    qweight = (codes.view(k // 4, 4, n) << shifts).sum(dim=1).to(torch.int32)
    return qweight.contiguous(), scale.view(1, n).to(weight.dtype), codes


def time_per_call(fn, flush_buf=None, warmup=20, iters=100):
    """Median of ``iters`` per-call CUDA-event timings, in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.sum()
            torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def capture_graph(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def replay_ms(graph, warmup=20, iters=100):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def build_marlin(k, n, dtype, device):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.utils.backend import BACKEND

    module = MarlinLinear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=k,
        out_features=n,
        bias=False,
        dtype=dtype,
        backend=BACKEND.MARLIN,
    ).to(device=device, dtype=dtype)
    return module


def marlin_gemm(module, x):
    from gptqmodel.utils.marlin import gptq_marlin_gemm

    return gptq_marlin_gemm(
        x,
        None,
        module.qweight,
        None,
        module.scales,
        None,
        module.qzeros,
        module.g_idx,
        module.g_idx_sort_indices,
        module.workspace,
        module.weight_type,
        size_m=x.shape[0],
        size_n=module.out_features,
        size_k=module.in_features,
        is_k_full=module.is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=module.fp32,
        is_zp_float=False,
    )


def parse_shapes(values):
    shapes = []
    for value in values:
        k_text, n_text = value.lower().split("x")
        shapes.append((int(k_text), int(n_text)))
    return shapes


def benchmark_shape(k, n, dtype, device, m_values, modes, warmup, iters, flush_buf):
    from gptqmodel.utils.allspark import (
        allspark_gemm,
        allspark_repack,
        prewarm_allspark_extension,
    )

    if not prewarm_allspark_extension():
        raise RuntimeError("AllSpark extension failed to build")

    torch.manual_seed(0)
    weight = torch.randn(k, n, dtype=dtype, device=device)
    qweight, scales, codes = pack_gptq_8bit(weight)
    packed, packed_scales, _ = allspark_repack(
        codes.to(torch.uint8).contiguous(),
        scales.reshape(-1).contiguous(),
        n,
        k,
    )

    marlin = build_marlin(k, n, dtype, device)
    marlin.qweight.data.copy_(qweight)
    marlin.scales.data.copy_(scales)
    marlin.post_init()
    if getattr(marlin, "_marlin_tile_padding", None) is not None:
        print(
            "warning: marlin is tile-padding this shape; its numbers include that work",
            file=sys.stderr,
        )

    reference_weight = (codes.float() - 128) * scales.float()
    weight_bytes = k * n
    rows = []

    for m in m_values:
        x = torch.randn(m, k, dtype=dtype, device=device)

        def run_allspark():
            return allspark_gemm(x, packed, packed_scales, n)

        def run_marlin():
            return marlin_gemm(marlin, x)

        reference = (x.float() @ reference_weight).to(dtype)
        allspark_err = (
            (run_allspark().float() - reference.float()).abs().mean()
            / reference.float().abs().mean()
        )
        marlin_err = (
            (run_marlin().float() - reference.float()).abs().mean()
            / reference.float().abs().mean()
        )

        row = {"m": m, "allspark_err": allspark_err.item(), "marlin_err": marlin_err.item()}

        if "hot" in modes:
            row["allspark_hot_ms"] = time_per_call(run_allspark, None, warmup, iters)
            row["marlin_hot_ms"] = time_per_call(run_marlin, None, warmup, iters)
        if "flushed" in modes:
            row["allspark_flushed_ms"] = time_per_call(run_allspark, flush_buf, warmup, iters)
            row["marlin_flushed_ms"] = time_per_call(run_marlin, flush_buf, warmup, iters)
        if "graph" in modes:
            flush_graph = capture_graph(lambda: flush_buf.sum())
            flush_only_ms = replay_ms(flush_graph)
            allspark_graph = capture_graph(run_allspark)
            marlin_graph = capture_graph(run_marlin)
            row["allspark_graph_ms"] = replay_ms(allspark_graph)
            row["marlin_graph_ms"] = replay_ms(marlin_graph)
            # Subtracting the flush replay isolates the kernel from the eviction pass.
            row["allspark_graph_flushed_ms"] = (
                replay_ms(capture_graph(lambda: (flush_buf.sum(), run_allspark()))) - flush_only_ms
            )
            row["marlin_graph_flushed_ms"] = (
                replay_ms(capture_graph(lambda: (flush_buf.sum(), run_marlin()))) - flush_only_ms
            )

        for key, value in list(row.items()):
            if key.endswith("_ms"):
                row[key.replace("_ms", "_gbps")] = weight_bytes / (value * 1e-3) / 1e9
        rows.append(row)

    return {
        "k": k,
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "weight_bytes": weight_bytes,
        "rows": rows,
    }


def _awq_tensors(k, n, dtype, device, seed=0):
    """Random AWQ 8-bit on-disk tensors. Never zero: zero data lets hardware
    skip work and hides layout/dequant bugs."""
    from gptqmodel.quantization.awq.utils.packing_utils import pack_awq

    gen = torch.Generator(device=device).manual_seed(seed)
    codes = torch.randint(0, 256, (k, n), dtype=torch.int32, device=device, generator=gen)
    zeros = torch.randint(0, 256, (1, n), dtype=torch.int32, device=device, generator=gen)
    scales = (
        torch.rand(1, n, dtype=torch.float32, device=device, generator=gen) * 0.01 + 0.005
    ).to(dtype)
    qweight, qzeros = pack_awq(codes, zeros, 8)
    return qweight, qzeros, scales, codes, zeros


def pack_awq_for_runtime(qweight, qzeros, scales, n, k):
    """Repack AWQ tensors with the extension matching this runtime."""
    if torch.version.hip is not None:
        from gptqmodel.utils.allspark_hip import allspark_hip_repack_awq

        return allspark_hip_repack_awq(qweight, qzeros, scales, n, k, -1)
    from gptqmodel.utils.allspark import allspark_repack_awq

    return allspark_repack_awq(qweight, qzeros, scales, n, k, -1)


def awq_gemm(x, packed, packed_scales, packed_zeros, n):
    if torch.version.hip is not None:
        from gptqmodel.utils.allspark_hip import allspark_hip_gemm

        return allspark_hip_gemm(x, packed, packed_scales, n, b_qzeros=packed_zeros)
    from gptqmodel.utils.allspark import allspark_gemm

    return allspark_gemm(x, packed, packed_scales, n, b_qzeros=packed_zeros)


def benchmark_shape_awq(k, n, dtype, device, m_values, modes, warmup, iters, flush_buf):
    if torch.version.hip is not None:
        from gptqmodel.utils.allspark_hip import prewarm_allspark_hip_extension

        if not prewarm_allspark_hip_extension():
            raise RuntimeError("AllSpark HIP extension failed to build")
    else:
        from gptqmodel.utils.allspark import prewarm_allspark_extension

        if not prewarm_allspark_extension():
            raise RuntimeError("AllSpark CUDA extension failed to build")

    qweight, qzeros, scales, codes, zeros = _awq_tensors(k, n, dtype, device)
    packed, packed_scales, packed_zeros = pack_awq_for_runtime(
        qweight, qzeros, scales, n, k
    )

    # Exact dequantized weights from original codes/zeros/scales. This reported
    # error includes weight rounding, split reduction and output rounding.
    reference_weight = ((codes.float() - zeros.float()) * scales.float()).to(torch.float64)
    weight_bytes = k * n
    rows = []

    for m in m_values:
        x = torch.randn(m, k, dtype=dtype, device=device)

        def run():
            return awq_gemm(x, packed, packed_scales, packed_zeros, n)

        reference = (x.to(torch.float64) @ reference_weight).float()
        err = (run().float() - reference).abs().max().item() / reference.abs().max().item()

        row = {"m": m, "allspark_err": err}
        if "hot" in modes:
            row["allspark_hot_ms"] = time_per_call(run, None, warmup, iters)
        if "flushed" in modes:
            row["allspark_flushed_ms"] = time_per_call(run, flush_buf, warmup, iters)
        if "graph" in modes:
            flush_graph = capture_graph(lambda: flush_buf.sum())
            flush_only_ms = replay_ms(flush_graph)
            row["allspark_graph_ms"] = replay_ms(capture_graph(run))
            row["allspark_graph_flushed_ms"] = (
                replay_ms(capture_graph(lambda: (flush_buf.sum(), run()))) - flush_only_ms
            )

        for key, value in list(row.items()):
            if key.endswith("_ms"):
                row[key.replace("_ms", "_gbps")] = weight_bytes / (value * 1e-3) / 1e9
        rows.append(row)

    return {
        "k": k,
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "method": "awq",
        "weight_bytes": weight_bytes,
        "rows": rows,
    }


def print_report(report):
    print(f"\nK={report['k']} N={report['n']} dtype={report['dtype']} "
          f"weights={report['weight_bytes']/1e6:.1f} MB")

    def fmt(value):
        return "-" if value is None else f"{value:.4f}"

    if report.get("method") == "awq":
        header = (f"{'M':>5} | {'as hot':>7} | {'as flush':>8} | {'as GB/s':>8} | "
                  f"{'as graph':>8} | {'as gfl':>8} | {'as err':>8}")
        print(header)
        print("-" * len(header))
        for row in report["rows"]:
            print(
                f"{row['m']:>5} | "
                f"{fmt(row.get('allspark_hot_ms')):>7} | "
                f"{fmt(row.get('allspark_flushed_ms')):>8} | "
                f"{fmt(row.get('allspark_flushed_gbps')):>8} | "
                f"{fmt(row.get('allspark_graph_ms')):>8} | "
                f"{fmt(row.get('allspark_graph_flushed_ms')):>8} | "
                f"{row['allspark_err']:>8.2e}"
            )
        return

    header = (f"{'M':>5} | {'as hot':>7} {'mr hot':>7} | "
              f"{'as flush':>8} {'mr flush':>8} | {'as GB/s':>8} {'mr GB/s':>8} | "
              f"{'as graph':>8} {'mr graph':>8} | {'as err':>8} {'mr err':>8}")
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        print(
            f"{row['m']:>5} | "
            f"{fmt(row.get('allspark_hot_ms')):>7} {fmt(row.get('marlin_hot_ms')):>7} | "
            f"{fmt(row.get('allspark_flushed_ms')):>8} {fmt(row.get('marlin_flushed_ms')):>8} | "
            f"{fmt(row.get('allspark_flushed_gbps')):>8} {fmt(row.get('marlin_flushed_gbps')):>8} | "
            f"{fmt(row.get('allspark_graph_ms')):>8} {fmt(row.get('marlin_graph_ms')):>8} | "
            f"{row['allspark_err']:>8.2e} {row['marlin_err']:>8.2e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--shapes", nargs="+", default=[f"{k}x{n}" for k, n in DEFAULT_SHAPES],
                        help="one or more KxN shapes, e.g. 4096x11008")
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M),
                        help="token counts to benchmark")
    parser.add_argument("--modes", nargs="+", default=["hot", "flushed", "graph"],
                        choices=("hot", "flushed", "graph"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None, help="write results as JSON")
    parser.add_argument(
        "--awq",
        action="store_true",
        help="benchmark the AWQ 8-bit path (group_size=-1) instead of GPTQ vs Marlin",
    )
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0 or any(m < 1 for m in args.m):
        parser.error("iters and M must be positive; warmup must be nonnegative")

    if not torch.cuda.is_available():
        print("no CUDA/ROCm device visible", file=sys.stderr)
        return 1

    from gptqmodel.utils.allspark import allspark_supported

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.version.hip is not None:
        if not args.awq:
            parser.error("ROCm requires --awq here; use benchmark_allspark_hip_w8a16.py for GPTQ")
        from gptqmodel.utils.allspark_hip import allspark_hip_supported

        supported = allspark_hip_supported(device)
    else:
        supported = allspark_supported(device)

    if not supported:
        print(
            f"AllSpark is not supported on {torch.cuda.get_device_name(0)}; "
            "it needs compute capability 8.0-12.0 (CUDA) or gfx1200/gfx1201 (ROCm)",
            file=sys.stderr,
        )
        return 1

    dtype = getattr(torch, args.dtype)
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {torch.cuda.get_device_name(device)} "
          f"cc={torch.cuda.get_device_capability(device)} "
          f"SMs={properties.multi_processor_count} "
          f"L2={properties.L2_cache_size / 2**20:.0f}MiB")
    print(f"torch {torch.__version__} cuda={torch.version.cuda} hip={torch.version.hip}")

    # Raise clocks before measuring; a cold GPU reports numbers ~30% too low.
    sink = torch.randn(2048, 2048, dtype=dtype, device=device)
    warm_weight = torch.randn_like(sink) / (2048 ** 0.5)
    for _ in range(60):
        sink = sink @ warm_weight
    torch.cuda.synchronize()
    del sink, warm_weight
    torch.cuda.empty_cache()

    flush_buf = None
    if "flushed" in args.modes or "graph" in args.modes:
        flush_buf = torch.randn(FLUSH_BYTES // 4, dtype=torch.float32, device=device)

    reports = []
    for k, n in parse_shapes(args.shapes):
        bench = benchmark_shape_awq if args.awq else benchmark_shape
        report = bench(
            k, n, dtype, device, args.m, args.modes, args.warmup, args.iters, flush_buf
        )
        reports.append(report)
        print_report(report)

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
