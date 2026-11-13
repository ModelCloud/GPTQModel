#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ w4a16 decode microbenchmark over real model shapes.

Kernel work needs a shape list that matches what inference actually asks for, at
the batch sizes decode actually runs: M=1..8. This measures Marlin on those
shapes and prints the weight-read bandwidth, so the gap to the card's memory roof
is visible per shape instead of assumed.

Shapes come from two sources and are stored per preset:

``qwen3_8b``      read from the released config: hidden 4096, intermediate
                  12288, 32 query heads and 8 KV heads of width 128.
``qwen3_4b``      hidden 2560, intermediate 9728, 8 KV heads of width 128.
``deepseek_v41``  read from DeepSeek-V4.1-Flash's text_config at revision
                  314113f8: hidden 5120, expert FFN 2304, MLA q_proj 5120->32768,
                  kv_a_proj 5120->576.

Protocol, matching ``benchmark_allspark_w8a16.py`` so numbers are comparable:

``hot``    back-to-back calls; the weights stay in L2 when the card has room.
           Latency only, not a bandwidth measurement.
``cold``   a 256 MiB random fp32 buffer is reduced before each timed call, so the
           weights come from device memory. Reported GB/s uses the fastest cold
           call (see below), since that is the number the kernel determines.

Both report the minimum, the median and a spread. This is a laptop die: a
sustained run of memory-bound kernels reaches its power limit and downclocks, so
the median tracks the thermal state and the minimum is the stable one.

Reported GB/s counts the weight read only (``K * N * bits / 8 / time``).
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_M = (1, 2, 4, 8)
FLUSH_BYTES = 256 * 2**20
BITS = 4
GROUP_SIZE = 128


@dataclass(frozen=True)
class Shape:
    name: str
    k: int
    n: int


# (k, n): k is the input width, n the output width, matching a Linear(k, n).
PRESETS: dict[str, tuple[Shape, ...]] = {
    "qwen3_8b": (
        Shape("q_proj", 4096, 4096),
        Shape("kv_proj", 4096, 1024),
        Shape("o_proj", 4096, 4096),
        Shape("gate_up", 4096, 12288),
        Shape("down", 12288, 4096),
    ),
    "qwen3_4b": (
        Shape("q_proj", 2560, 2560),
        Shape("kv_proj", 2560, 1024),
        Shape("o_proj", 2560, 2560),
        Shape("gate_up", 2560, 9728),
        Shape("down", 9728, 2560),
    ),
    # deepseek-ai/DeepSeek-V4.1-Flash, revision 314113f8: text_config declares
    # hidden 5120, 40 layers, 384 routed experts of FFN 2304 plus one shared
    # expert, and MLA with 64 heads of width 512 (kv heads 1, rope 64).
    "deepseek_v41": (
        Shape("q_proj", 5120, 32768),
        Shape("kv_a_proj", 5120, 576),
        Shape("o_proj", 32768, 5120),
        Shape("expert_gate_up", 5120, 2304),
        Shape("expert_down", 2304, 5120),
        Shape("shared_gate_up", 5120, 2304),
        Shape("shared_down", 2304, 5120),
    ),
}


@dataclass(frozen=True)
class Fusion:
    """A set of projections that inference engines merge into one GEMM.

    vLLM and sglang call these fused_qkv and merged gate/up. The point of
    measuring them here is that the parts are small enough to lose bandwidth on
    their own, so the fused shape can be much faster than the sum of its parts.
    """

    name: str
    parts: tuple[Shape, ...]
    fused: Shape


FUSIONS: dict[str, tuple[Fusion, ...]] = {
    "qwen3_8b": (
        Fusion(
            "qkv",
            parts=(Shape("q_proj", 4096, 4096), Shape("kv_proj", 4096, 1024),
                   Shape("kv_proj", 4096, 1024)),
            fused=Shape("qkv_fused", 4096, 6144),
        ),
        Fusion(
            "gate_up",
            parts=(Shape("gate_up", 4096, 12288), Shape("gate_up", 4096, 12288)),
            fused=Shape("gateup_fused", 4096, 24576),
        ),
    ),
    "qwen3_4b": (
        Fusion(
            "qkv",
            parts=(Shape("q_proj", 2560, 2560), Shape("kv_proj", 2560, 1024),
                   Shape("kv_proj", 2560, 1024)),
            fused=Shape("qkv_fused", 2560, 4608),
        ),
        Fusion(
            "gate_up",
            parts=(Shape("gate_up", 2560, 9728), Shape("gate_up", 2560, 9728)),
            fused=Shape("gateup_fused", 2560, 19456),
        ),
    ),
    "deepseek_v41": (
        Fusion(
            "expert_gate_up",
            parts=(Shape("expert_gate_up", 5120, 2304),
                   Shape("expert_gate_up", 5120, 2304)),
            fused=Shape("expert_gateup_fused", 5120, 4608),
        ),
    ),
}


def build_marlin(shape: Shape, device: str, dtype: torch.dtype):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.utils.backend import BACKEND

    module = MarlinLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=shape.k,
        out_features=shape.n,
        bias=False,
        dtype=dtype,
        backend=BACKEND.MARLIN,
    ).to(device=device, dtype=dtype)
    # Fills g_idx_sort_indices and the scratch workspace, and sets is_k_full.
    module.post_init()
    return module


def marlin_call(module, x):
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


_FLUSH = None


def flush_buffer(device: str) -> torch.Tensor:
    """One 256 MiB fp32 buffer, allocated once.

    Random rather than zero filled: a compressible buffer can fail to evict the
    weights and turn a memory measurement into an L2 measurement.
    """
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.rand(FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    return _FLUSH


def _samples(fn, iters: int, warmup: int, flush) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        if flush is not None:
            flush.sum()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))
    return sorted(out)


def _stats(samples: list[float]) -> tuple[float, float, float]:
    """(min, median, spread) where spread is (p90 - p10) / median.

    The minimum is the headline: this is a laptop die, and a sustained run of
    memory-bound kernels hits its power limit and downclocks, so the median
    drifts with the thermal state while the minimum stays put.
    """
    med = statistics.median(samples)
    lo = samples[int(0.1 * (len(samples) - 1))]
    hi = samples[int(0.9 * (len(samples) - 1))]
    spread = (hi - lo) / med if med > 0 else 0.0
    return samples[0], med, spread


def time_hot(fn, iters: int, warmup: int) -> tuple[float, float, float]:
    """Back-to-back calls. The weights can stay in L2; latency only."""
    return _stats(_samples(fn, iters, warmup, None))


def time_flushed(fn, iters: int, warmup: int, device: str) -> tuple[float, float, float]:
    return _stats(_samples(fn, iters, warmup, flush_buffer(device)))


def gbps(shape: Shape, ms: float) -> float:
    weight_bytes = shape.k * shape.n * BITS / 8
    return weight_bytes / (ms * 1e-3) / 1e9


def warm_up_gpu(device: str, seconds: float = 1.0) -> None:
    """Pull the GPU out of its idle clock state before measuring anything.

    The first measurement in a process is otherwise slower than every later one
    on the same kernel - 29.5us vs 21.5us for the same fused GEMM here - because
    the card starts at idle clocks and only boosts once it has work.
    """
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        a @ b
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


def _measure(shape: Shape, m: int, args, device: str, dtype) -> dict:
    module = build_marlin(shape, device, dtype)
    x = torch.randn(m, shape.k, dtype=dtype, device=device)
    fn = functools.partial(marlin_call, module, x)
    hot_min, hot_med, hot_spread = time_hot(fn, args.iters, args.warmup)
    cold_min, cold_med, cold_spread = time_flushed(fn, args.iters, args.warmup, device)
    del module
    torch.cuda.empty_cache()
    return {
        "preset": args.preset,
        "shape": shape.name,
        "k": shape.k,
        "n": shape.n,
        "m": m,
        "hot_min_ms": hot_min,
        "hot_median_ms": hot_med,
        "hot_spread": hot_spread,
        "cold_min_ms": cold_min,
        "cold_median_ms": cold_med,
        "cold_spread": cold_spread,
        "cold_gbps": gbps(shape, cold_min),
    }


def _best_of(best: dict, key, row: dict) -> None:
    previous = best.get(key)
    if previous is None or row["cold_min_ms"] < previous["cold_min_ms"]:
        best[key] = row


def run(args) -> list[dict]:
    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    warm_up_gpu(device)
    shapes = PRESETS[args.preset]
    if args.shapes:
        shapes = tuple(
            Shape(name, int(k), int(n))
            for name, k, n in (value.split(":") for value in args.shapes)
        )

    # Measure each configuration once per pass. Shuffle with a fixed seed per
    # pass so a shape does not always follow the same predecessor, while runs
    # remain reproducible.
    configs = [(shape, m) for shape in shapes for m in args.m]
    best: dict[tuple[Shape, int], dict] = {}
    for pass_index in range(args.passes):
        order = configs.copy()
        random.Random(pass_index).shuffle(order)
        for shape, m in order:
            _best_of(best, (shape, m), _measure(shape, m, args, device, dtype))

    rows = []
    for shape in shapes:
        for m in args.m:
            row = best[(shape, m)]
            rows.append(row)
            print(
                f"{row['shape']:16s} k={row['k']:5d} n={row['n']:5d} m={row['m']:2d} "
                f"hot={row['hot_min_ms']*1000:6.1f}us cold={row['cold_min_ms']*1000:6.1f}us "
                f"(median {row['cold_median_ms']*1000:6.1f}us)  {row['cold_gbps']:6.1f} GB/s "
                f"spread {row['cold_spread']*100:4.1f}%",
                flush=True,
            )
    return rows


def run_fusions(args) -> list[dict]:
    """Compare one fused GEMM against the separate calls it replaces.

    The parts are timed individually and their times summed; inference engines
    (vLLM, sglang) issue one GEMM over the concatenated weights instead.
    """
    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    warm_up_gpu(device)
    groups = FUSIONS[args.preset]
    rows = []
    for group in groups:
        for m in args.m:
            variants = (*group.parts, group.fused)
            best: dict[int, dict] = {}
            for pass_index in range(args.passes):
                order = list(enumerate(variants))
                random.Random(pass_index).shuffle(order)
                for index, shape in order:
                    _best_of(best, index, _measure(shape, m, args, device, dtype))
            part_rows = [best[index] for index in range(len(group.parts))]
            fused = best[len(group.parts)]

            parts_ms = sum(r["cold_min_ms"] for r in part_rows)
            parts_bytes = sum(r["k"] * r["n"] for r in part_rows) * BITS / 8
            fused_bytes = group.fused.k * group.fused.n * BITS / 8
            parts_gbps = parts_bytes / (parts_ms * 1e-3) / 1e9

            row = {
                "preset": args.preset,
                "group": group.name,
                "m": m,
                "parts": [r["shape"] for r in part_rows],
                "parts_calls": len(part_rows),
                "parts_ms": parts_ms,
                "parts_gbps": parts_gbps,
                "fused_shape": fused["shape"],
                "fused_ms": fused["cold_min_ms"],
                "fused_gbps": fused["cold_gbps"],
                "speedup": parts_ms / fused["cold_min_ms"],
                "bytes_equal": abs(parts_bytes - fused_bytes) < 1e-6 * fused_bytes,
            }
            rows.append(row)
            print(
                f"{group.name:16s} m={m:2d}  parts {len(part_rows)} calls "
                f"{parts_ms*1000:6.1f}us {parts_gbps:6.1f} GB/s   fused 1 call "
                f"{fused['cold_min_ms']*1000:6.1f}us {fused['cold_gbps']:6.1f} GB/s   "
                f"speedup {row['speedup']:4.2f}x",
                flush=True,
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen3_8b")
    parser.add_argument("--shapes", nargs="*", default=None, help="name:k:n overrides")
    parser.add_argument("--m", nargs="*", type=int, default=list(DEFAULT_M))
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--passes", type=int, default=3, help="shuffled repeats")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--fusion", action="store_true",
                        help="compare fused qkv / gate_up against the separate calls")
    parser.add_argument("--json", default=None, help="write rows to this path")
    args = parser.parse_args()
    if not args.m or any(m < 1 for m in args.m):
        parser.error("--m requires one or more positive batch sizes")
    if args.iters < 1 or args.passes < 1 or args.warmup < 0:
        parser.error("--iters and --passes must be positive; --warmup cannot be negative")

    if not torch.cuda.is_available():
        print("CUDA device required")
        return 1

    rows = run_fusions(args) if args.fusion else run(args)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
