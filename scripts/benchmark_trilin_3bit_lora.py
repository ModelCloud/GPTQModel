# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.utils.trilin import trilin_matmul, trilin_matmul_lora


K = 4096
N = 4096
GROUP_SIZE = 128
SUPPORTED_RANKS = (32, 64, 128, 256)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the supported fused TriLin 3-bit plus LoRA decode path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--rank", type=int, choices=SUPPORTED_RANKS, default=128)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="Emit matched unfused/fused NVTX ranges with this many calls after warmup; skips CUDA-event timing.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile_nearest(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round(percentile * (len(ordered) - 1))]


def _benchmark(fn, *, warmup: int, iterations: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for index in range(iterations):
            starts[index].record()
            fn()
            ends[index].record()
        torch.cuda.synchronize()
    samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iterations)]
    mean_us = statistics.mean(samples_us)
    return {
        "p50_us": statistics.median(samples_us),
        "mean_us": mean_us,
        "p95_us": _percentile_nearest(samples_us, 0.95),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "operations_per_second": 1_000_000.0 / mean_us,
    }


def _print_table(results: dict) -> None:
    environment = results["environment"]
    shape = environment["shape"]
    shape_label = f"{shape['m']}x{shape['k']}x{shape['n']}"
    dtype_label = environment["dtype"].removeprefix("torch.")
    rows = []
    for name in ("unfused", "fused"):
        stats = results[name]
        rows.append(
            (
                environment["name"],
                dtype_label,
                shape_label,
                shape["rank"],
                shape["group_size"],
                name,
                stats["p50_us"],
                stats["mean_us"],
                stats["p95_us"],
                stats["operations_per_second"],
            )
        )
    border = "+------------------+----------+--------------+------+-------+---------+----------+----------+----------+------------+"
    print(border)
    print("| GPU              | dtype    | MxKxN        | rank | group | path    | p50 us   | mean us  | p95 us   | calls/s    |")
    print(border)
    for gpu, dtype, problem, rank, group, name, p50, mean, p95, throughput in rows:
        print(
            f"| {gpu:<16} | {dtype:<8} | {problem:<12} | {rank:4d} | {group:5d} | {name:<7} | "
            f"{p50:8.3f} | {mean:8.3f} | {p95:8.3f} | {throughput:10.1f} |"
        )
    print(border)
    speedup = results["speedup"]
    print(
        f"speedup: p50={speedup['p50']:.3f}x mean={speedup['mean']:.3f}x "
        f"p95={speedup['p95']:.3f}x"
    )


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.profile_iterations < 0:
        raise ValueError("warmup/profile iterations must be non-negative and timing iterations must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("TriLin+LoRA benchmarking requires CUDA")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(f"TriLin+LoRA requires sm_80, got sm_{properties.major}{properties.minor}")

    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K // 32 * 3, N),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((K // GROUP_SIZE, N), dtype=torch.float16, device=device).mul_(0.02).add_(0.001)
    x = torch.randn((1, K), dtype=dtype, device=device)
    lora_a = torch.randn((K, args.rank), dtype=dtype, device=device).mul_(0.02)
    lora_b = torch.randn((args.rank, N), dtype=dtype, device=device).mul_(0.02)
    adapter = Lora(rank=args.rank, lora_A=lora_a, lora_B=lora_b)
    workspace = torch.empty((args.rank,), dtype=torch.float32, device=device)

    def unfused():
        return adapter.apply(x, trilin_matmul(x, qweight, scales))

    def fused():
        return trilin_matmul_lora(x, qweight, scales, lora_a, lora_b, workspace)

    with torch.inference_mode():
        reference = unfused()
        actual = fused()
    torch.testing.assert_close(actual, reference, rtol=0.002, atol=0.5 if dtype == torch.bfloat16 else 0.0625)

    environment = {
        "device": str(device),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "shape": {"m": 1, "k": K, "n": N, "rank": args.rank, "group_size": GROUP_SIZE},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
    }

    if args.profile_iterations:
        with torch.inference_mode():
            for _ in range(args.warmup):
                unfused()
                fused()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"trilin_lora_rank_{args.rank}_unfused")
            for _ in range(args.profile_iterations):
                unfused()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"trilin_lora_rank_{args.rank}_fused")
            for _ in range(args.profile_iterations):
                fused()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        payload = {"environment": environment, "profile_iterations": args.profile_iterations}
    else:
        unfused_stats = _benchmark(unfused, warmup=args.warmup, iterations=args.iterations)
        fused_stats = _benchmark(fused, warmup=args.warmup, iterations=args.iterations)
        payload = {
            "environment": environment,
            "unfused": unfused_stats,
            "fused": fused_stats,
            "speedup": {
                "p50": unfused_stats["p50_us"] / fused_stats["p50_us"],
                "mean": unfused_stats["mean_us"] / fused_stats["mean_us"],
                "p95": unfused_stats["p95_us"] / fused_stats["p95_us"],
            },
        }
        _print_table(payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
