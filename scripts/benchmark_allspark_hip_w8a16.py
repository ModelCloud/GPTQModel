#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark HIP/RDNA4 W8A16 microbenchmark (gfx1200/gfx1201).

The ROCm counterpart of ``scripts/benchmark_allspark_w8a16.py``. It exists to
make the RDNA4 numbers on PR #3490 reproducible, and to keep the fp16/bf16
asymmetry visible: both dtypes are timed with the same shapes so a regression
in either dequant path shows up as a dtype gap in the ``hot`` column.

    python scripts/benchmark_allspark_hip_w8a16.py
    python scripts/benchmark_allspark_hip_w8a16.py --triton
    python scripts/benchmark_allspark_hip_w8a16.py --awq

Protocol (identical to the CUDA script):

* K=4096, N=11008, random uint8 codes and per-output-channel scales.
* ``hot``     back-to-back calls (weights may stay in L2).
* ``flushed`` a 256 MiB *random* fp32 buffer is reduced before every timed
              call, so the weights come from device memory. Random rather than
              zero-filled: a compressible buffer would not evict the weights.
* ``graph``   one graph capture, replay timed (removes launch overhead).

Reported GB/s is ``K * N / time`` (weight bytes only), the same convention as
the PR body, so it is directly comparable with the 4090/Triton table.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from gptqmodel.utils.allspark_hip import (
    allspark_hip_repack,
    allspark_hip_supported,
    prewarm_allspark_hip_extension,
)


DEFAULT_SHAPES = ((4096, 11008),)
DEFAULT_M = (1, 2, 4, 8, 16)
FLUSH_BYTES = 256 * 2**20


def time_per_call(fn, flush_buf=None, warmup=20, iters=100):
    """Median per-call time in milliseconds, from CUDA/HIP events."""
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


def parse_shapes(values):
    shapes = []
    for value in values:
        k_text, n_text = value.lower().split("x")
        shapes.append((int(k_text), int(n_text)))
    return shapes


def build_triton(k, n, dtype):
    """The shipped Triton ROCm fused path, used as the reference column."""
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    module = TorchLinear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, pack_dtype=torch.int32,
        bias=False, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32)
        )
        module.qzeros.copy_(
            torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32)
        )
        module.scales.uniform_(0.001, 0.02)
    module.post_init()
    return module


def benchmark_shape(k, n, dtype, m_values, warmup, iters, flush_buf, use_triton,
                    awq=False):
    device = torch.device("cuda")
    torch.manual_seed(0)
    codes = torch.randint(0, 256, (k, n), dtype=torch.uint8, device=device)
    scales = (torch.rand(n, dtype=torch.float32, device=device) * 0.01 + 0.005).to(dtype)
    packed, packed_scales, _ = allspark_hip_repack(codes, scales, n, k)

    # AWQ: same K x N codes, but one zero point and one scale per output
    # channel (``group_size=-1``), which is the only grouping the kernel's
    # dequant path loads. The zero point is deliberately not 128, so a kernel
    # that silently assumes the GPTQ neutral value cannot look fast by
    # skipping the subtraction.
    awq_args = (None, -1)
    if awq:
        from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
        from gptqmodel.utils.allspark_hip import allspark_hip_repack_awq

        icodes = torch.randint(0, 256, (k, n), dtype=torch.int32, device=device)
        izeros = torch.randint(0, 256, (1, n), dtype=torch.int32, device=device)
        iscales = (torch.rand(1, n, dtype=torch.float32, device=device) * 0.01
                   + 0.005).to(dtype)
        qweight_awq, qzeros_awq = pack_awq(icodes, izeros, 8)
        # int32 AWQ words -> the uint8 (K, N) codes the kernel repacks; this is
        # the same conversion post_init performs, so the timing covers the real
        # runtime layout.
        codes = (icodes & 0xFF).to(torch.uint8).contiguous()
        packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
            qweight_awq, qzeros_awq, iscales, n, k, -1
        )
        awq_args = (packed_zeros, -1)
        del qweight_awq, qzeros_awq, icodes, izeros

    triton = build_triton(k, n, dtype) if use_triton else None
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    weight_bytes = k * n
    rows = []

    for m in m_values:
        x = (torch.randn(m, k, dtype=torch.float32, device=device) * 0.5).to(dtype)
        with torch.inference_mode():
            packed_zeros, gsz = awq_args

            # Call the raw op so Python attribute lookups do not pollute timing.
            def hip_fn():
                return torch.ops.gptqmodel_allspark_hip.allspark_w8a16_gemm(
                    x, packed, packed_scales, packed_zeros, n, gsz, sm_count,
                    props.major * 10 + props.minor, 1024,
                    packed_zeros is not None, True,
                )

            hip_fn()  # build/warm any lazy state before the graph capture
            hot = time_per_call(hip_fn, warmup=warmup, iters=iters)
            flushed = time_per_call(hip_fn, flush_buf=flush_buf, warmup=warmup, iters=iters)
            graph = replay_ms(capture_graph(hip_fn), warmup=warmup, iters=iters)

            row = {
                "dtype": str(dtype).split(".")[-1],
                "m": m,
                "awq": awq,
                "hot_ms": hot,
                "flushed_ms": flushed,
                "graph_ms": graph,
                "hot_gbs": weight_bytes / hot / 1e6,
                "flushed_gbs": weight_bytes / flushed / 1e6,
                "graph_gbs": weight_bytes / graph / 1e6,
            }

            if triton is not None:
                def tri_fn():
                    return triton(x)

                tri_fn()
                tri_hot = time_per_call(tri_fn, warmup=warmup, iters=iters)
                tri_flushed = time_per_call(tri_fn, flush_buf=flush_buf, warmup=warmup, iters=iters)
                tri_graph = replay_ms(capture_graph(tri_fn), warmup=warmup, iters=iters)
                row.update(
                    triton_hot_ms=tri_hot,
                    triton_flushed_ms=tri_flushed,
                    triton_graph_ms=tri_graph,
                    speedup_hot=tri_hot / hot,
                    speedup_flushed=tri_flushed / flushed,
                    speedup_graph=tri_graph / graph,
                )
        rows.append(row)
        line = (
            f"{row['dtype']:9s} M={m:2d} | "
            f"hot {hot * 1e3:7.2f}us {row['hot_gbs']:6.1f}GB/s  "
            f"fl {flushed * 1e3:7.2f}us {row['flushed_gbs']:6.1f}  "
            f"gr {graph * 1e3:7.2f}us {row['graph_gbs']:6.1f}"
        )
        if triton is not None:
            line += (
                f" | triton hot {tri_hot * 1e3:7.2f}us {weight_bytes / tri_hot / 1e6:6.1f}  "
                f"fl {tri_flushed * 1e3:7.2f}us {weight_bytes / tri_flushed / 1e6:6.1f}  "
                f"gr {tri_graph * 1e3:7.2f}us {weight_bytes / tri_graph / 1e6:6.1f}"
                f" | x{row['speedup_hot']:.2f}/{row['speedup_flushed']:.2f}/{row['speedup_graph']:.2f}"
            )
        print(line, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", nargs="+", default=[f"{k}x{n}" for k, n in DEFAULT_SHAPES])
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--triton", action="store_true", help="also time the Triton ROCm path")
    parser.add_argument(
        "--awq", action="store_true",
        help="also time the AWQ 8-bit path (per-output-channel zeros, group_size=-1)",
    )
    parser.add_argument("--json", type=str, default=None, help="write raw results here")
    args = parser.parse_args()

    if not allspark_hip_supported():
        raise SystemExit(
            "AllSpark HIP needs an RDNA4 device (gfx1200/gfx1201); "
            f"found {getattr(torch.cuda.get_device_properties(0), 'gcnArchName', 'no device')}"
        )
    if not prewarm_allspark_hip_extension():
        raise SystemExit("AllSpark HIP extension failed to build")

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} ({props.gcnArchName}), {props.multi_processor_count} CUs")
    print(f"torch: {torch.__version__}, hip: {torch.version.hip}")

    # Random, not zero: a compressible buffer would not evict the weights and
    # would make the flushed column meaningless.
    flush_buf = torch.rand(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    dram_gbs = (FLUSH_BYTES / 1e6) / time_per_call(flush_buf.sum, warmup=5, iters=20)
    print(f"device read baseline ({FLUSH_BYTES // 2**20} MiB random sum): {dram_gbs:.1f} GB/s")

    results = {
        "device": props.name,
        "arch": props.gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dram_read_gbs": dram_gbs,
        "rows": [],
    }
    for k, n in parse_shapes(args.shapes):
        for dtype in (torch.float16, torch.bfloat16):
            print(f"\nK={k} N={n} {str(dtype).split('.')[-1]}")
            results["rows"] += benchmark_shape(
                k, n, dtype, args.m, args.warmup, args.iters, flush_buf, args.triton
            )
            if args.awq:
                print("  AWQ group_size=-1")
                results["rows"] += benchmark_shape(
                    k, n, dtype, args.m, args.warmup, args.iters, flush_buf,
                    args.triton, awq=True,
                )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
