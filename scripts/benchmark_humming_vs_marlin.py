#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Humming GPTQ GEMM against Marlin on a single physical CUDA device."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_idle_preflight import (  # noqa: E402
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)


_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

import torch  # noqa: E402
import torch.nn as nn
from tabulate import tabulate  # noqa: E402

from gptqmodel.nn_modules.qlinear.humming import HummingGptqLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchLinear  # noqa: E402


BITS = 4
GROUP_SIZE = 128


@dataclass(frozen=True)
class TimingStats:
    samples: int
    median_us: float
    mean_us: float
    std_us: float
    p95_us: float
    min_us: float
    max_us: float
    batch_event_samples_us: tuple[float, ...]
    batch_event_median_us: float
    batch_event_mean_us: float
    batch_event_p95_us: float
    batch_event_min_us: float
    batch_event_max_us: float
    wall_samples_us: tuple[float, ...]
    wall_median_us: float
    wall_mean_us: float
    wall_p95_us: float
    wall_min_us: float
    wall_max_us: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Humming GPTQ GEMM against Marlin on a single physical CUDA device."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--json-out", type=Path)
    add_gpu_idle_preflight_args(parser)
    return parser.parse_args()


def _resolve_dtypes(name: str) -> tuple[torch.dtype, ...]:
    if name == "fp16":
        return (torch.float16,)
    if name == "bf16":
        return (torch.bfloat16,)
    return (torch.float16, torch.bfloat16)


def _dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "bf16"


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _nvidia_smi_inventory() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _pack_reference(
    *,
    device: torch.device,
    dtype: torch.dtype,
    size_k: int,
    size_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a dense [N, K] weight to 4-bit GPTQ and pack it into the canonical layout."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    linear = nn.Linear(size_k, size_n, bias=False).to(torch.float16)
    weight = linear.weight.data

    w = weight.view(size_n, size_k // GROUP_SIZE, GROUP_SIZE)
    scale = w.abs().amax(dim=-1, keepdim=True) / (2 ** (BITS - 1) - 1)
    q = torch.round(w / scale).clamp(-(2 ** (BITS - 1) - 1), 2 ** (BITS - 1) - 1).to(torch.int32)
    q = q + (2 ** (BITS - 1))
    zero = torch.full((size_n, size_k // GROUP_SIZE), 2 ** (BITS - 1), dtype=torch.float16)

    scales = scale.squeeze(-1).contiguous().to(torch.float16)
    zeros = zero.contiguous().to(torch.float16)
    g_idx = torch.tensor([i // GROUP_SIZE for i in range(size_k)], dtype=torch.int32)

    torch_linear = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=size_k,
        out_features=size_n,
        bias=False,
        pack_dtype=torch.int32,
    )
    torch_linear.pack(linear, scales, zeros, g_idx)

    input_tensor = torch.randn((1, size_k), device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed))
    return input_tensor, weight.to(device, dtype=torch.float16), torch_linear.qweight, torch_linear.scales, torch_linear.qzeros, torch_linear.g_idx


def _build_humming(
    *,
    device: torch.device,
    dtype: torch.dtype,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> HummingGptqLinear:
    size_k = qweight.size(0) * 8
    size_n = qweight.size(1)
    module = HummingGptqLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=size_k,
        out_features=size_n,
        bias=False,
        pack_dtype=torch.int32,
        dtype=dtype,
    ).to(device)
    module.qweight.copy_(qweight)
    module.scales.copy_(scales)
    module.qzeros.copy_(qzeros)
    module.g_idx.copy_(g_idx)
    module.eval()
    module.post_init()
    return module


def _build_marlin(
    *,
    device: torch.device,
    dtype: torch.dtype,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> MarlinLinear:
    size_k = qweight.size(0) * 8
    size_n = qweight.size(1)
    module = MarlinLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=size_k,
        out_features=size_n,
        bias=False,
        pack_dtype=torch.int32,
        dtype=dtype,
    ).to(device)
    module.qweight.copy_(qweight)
    module.scales.copy_(scales)
    module.qzeros.copy_(qzeros)
    module.g_idx.copy_(g_idx)
    module.eval()
    module.post_init()
    return module


def _reference_output(
    input: torch.Tensor,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    """Dense matmul using the dequantized weight from the canonical GPTQ buffers."""
    torch_linear = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        pack_dtype=torch.int32,
    )
    torch_linear.qweight = qweight
    torch_linear.scales = scales
    torch_linear.qzeros = qzeros
    torch_linear.g_idx = g_idx
    torch_linear.post_init()
    weight_dequant = torch_linear.dequantize_weight().to(device=input.device, dtype=input.dtype)
    return input @ weight_dequant


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    rounds: int,
    pre_timing_check: Callable[[], None] | None = None,
) -> dict[str, TimingStats]:
    for function in functions.values():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize(device)
    if pre_timing_check is not None:
        pre_timing_check()

    event_samples: dict[str, list[float]] = {name: [] for name in functions}
    batch_event_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    names = list(functions)

    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            function = functions[name]
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            last_output = None
            for index in range(iters):
                starts[index].record()
                last_output = function()
                ends[index].record()
            torch.cuda.synchronize(device)
            if last_output is None:
                raise AssertionError("benchmark produced no output")
            event_samples[name].extend(
                starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)
            )

            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            batch_start.record()
            for _ in range(iters):
                last_output = function()
            batch_end.record()
            torch.cuda.synchronize(device)
            batch_event_samples[name].append(batch_start.elapsed_time(batch_end) * 1000.0 / iters)

            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            for _ in range(iters):
                last_output = function()
            torch.cuda.synchronize(device)
            wall_samples[name].append((time.perf_counter() - wall_start) * 1e6 / iters)

    results = {}
    for name in names:
        samples = event_samples[name]
        batch_samples = batch_event_samples[name]
        synchronized_wall_samples = wall_samples[name]
        results[name] = TimingStats(
            samples=len(samples),
            median_us=statistics.median(samples),
            mean_us=statistics.mean(samples),
            std_us=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            p95_us=_percentile(samples, 0.95),
            min_us=min(samples),
            max_us=max(samples),
            batch_event_samples_us=tuple(batch_samples),
            batch_event_median_us=statistics.median(batch_samples),
            batch_event_mean_us=statistics.mean(batch_samples),
            batch_event_p95_us=_percentile(batch_samples, 0.95),
            batch_event_min_us=min(batch_samples),
            batch_event_max_us=max(batch_samples),
            wall_samples_us=tuple(synchronized_wall_samples),
            wall_median_us=statistics.median(synchronized_wall_samples),
            wall_mean_us=statistics.mean(synchronized_wall_samples),
            wall_p95_us=_percentile(synchronized_wall_samples, 0.95),
            wall_min_us=min(synchronized_wall_samples),
            wall_max_us=max(synchronized_wall_samples),
        )
    return results


def _benchmark_dtype(
    *,
    device: torch.device,
    dtype: torch.dtype,
    size_m: int,
    size_k: int,
    size_n: int,
    seed: int,
    warmup: int,
    iters: int,
    rounds: int,
) -> dict:
    input_tensor, dense_weight, qweight, scales, qzeros, g_idx = _pack_reference(
        device=device,
        dtype=dtype,
        size_k=size_k,
        size_n=size_n,
        seed=seed,
    )

    humming_module = _build_humming(
        device=device,
        dtype=dtype,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        g_idx=g_idx,
    )
    marlin_module = _build_marlin(
        device=device,
        dtype=dtype,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        g_idx=g_idx,
    )

    reference = _reference_output(
        input_tensor,
        dense_weight,
        qweight.to(device),
        scales.to(device),
        qzeros.to(device),
        g_idx.to(device),
    )

    functions = {
        "humming": lambda: humming_module(input_tensor),
        "marlin": lambda: marlin_module(input_tensor),
    }

    with torch.inference_mode():
        humming_output = functions["humming"]()
        marlin_output = functions["marlin"]()
        torch.cuda.synchronize(device)
        humming_error = (humming_output.to(torch.float32) - reference.to(torch.float32)).abs()
        marlin_error = (marlin_output.to(torch.float32) - reference.to(torch.float32)).abs()
        cross_error = (humming_output.to(torch.float32) - marlin_output.to(torch.float32)).abs()
        if not torch.isfinite(humming_output).all() or not torch.isfinite(marlin_output).all():
            raise AssertionError("non-finite benchmark output")
        timing = _measure(
            functions,
            device=device,
            warmup=warmup,
            iters=iters,
            rounds=rounds,
            pre_timing_check=(
                (lambda: recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT))
                if _GPU_IDLE_PREFLIGHT is not None
                else None
            ),
        )

    qweight_bytes = qweight.numel() * qweight.element_size()
    scales_bytes = scales.numel() * scales.element_size()
    activation_bytes = input_tensor.numel() * input_tensor.element_size()
    output_bytes = size_n * size_m * input_tensor.element_size()
    logical_bytes = qweight_bytes + scales_bytes + activation_bytes + output_bytes
    marlin_batch_median = timing["marlin"].batch_event_median_us

    rows = []
    for name, stats in timing.items():
        effective_gbs = logical_bytes / (stats.batch_event_median_us * 1000.0)
        rows.append(
            {
                "kernel": name,
                "dtype": _dtype_name(dtype),
                "m": size_m,
                "k": size_k,
                "n": size_n,
                **asdict(stats),
                "effective_gbs": effective_gbs,
                "speedup_vs_marlin": marlin_batch_median / stats.batch_event_median_us,
            }
        )

    return {
        "dtype": _dtype_name(dtype),
        "shape": {"m": size_m, "k": size_k, "n": size_n},
        "quantization": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "sym": True,
            "desc_act": False,
            "pack_dtype": "torch.int32",
        },
        "layout_bytes": {
            "qweight": qweight_bytes,
            "scales": scales_bytes,
            "logical_bytes": logical_bytes,
        },
        "errors_vs_dense": {
            "humming_max_abs": humming_error.max().item(),
            "humming_mean_abs": humming_error.mean().item(),
            "marlin_max_abs": marlin_error.max().item(),
            "marlin_mean_abs": marlin_error.mean().item(),
            "humming_vs_marlin_max_abs": cross_error.max().item(),
            "humming_vs_marlin_mean_abs": cross_error.mean().item(),
        },
        "results": rows,
    }


def _print_result(result: dict) -> None:
    errors = result["errors_vs_dense"]
    print(
        f"\n{result['dtype']} correctness: "
        f"Humming max/mean abs={errors['humming_max_abs']:.9f}/{errors['humming_mean_abs']:.9f}, "
        f"Marlin={errors['marlin_max_abs']:.9f}/{errors['marlin_mean_abs']:.9f}, "
        f"cross={errors['humming_vs_marlin_max_abs']:.9f}/{errors['humming_vs_marlin_mean_abs']:.9f}"
    )
    table = []
    for row in result["results"]:
        table.append(
            [
                row["kernel"],
                row["dtype"],
                row["m"],
                row["k"],
                row["n"],
                row["samples"],
                f"{row['median_us']:.3f}",
                f"{row['mean_us']:.3f}",
                f"{row['std_us']:.3f}",
                f"{row['p95_us']:.3f}",
                f"{row['min_us']:.3f}",
                f"{row['max_us']:.3f}",
                f"{row['batch_event_median_us']:.3f}",
                f"{row['batch_event_mean_us']:.3f}",
                f"{row['wall_median_us']:.3f}",
                f"{row['wall_mean_us']:.3f}",
                f"{row['effective_gbs']:.2f}",
                f"{row['speedup_vs_marlin']:.3f}x",
            ]
        )
    print(
        tabulate(
            table,
            headers=(
                "kernel",
                "dtype",
                "M",
                "K",
                "N",
                "samples",
                "p50 us",
                "mean us",
                "std us",
                "p95 us",
                "min us",
                "max us",
                "batch event median us",
                "batch event mean us",
                "wall median us",
                "wall mean us",
                "effective GB/s",
                "speedup vs Marlin",
            ),
            tablefmt="grid",
        )
    )


def main() -> None:
    args = _parse_args()
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.k <= 0 or args.k % GROUP_SIZE != 0:
        raise ValueError(f"--k must be positive and divisible by group size {GROUP_SIZE}")
    if args.n <= 0 or args.n % 64 != 0:
        raise ValueError("--n must be positive and divisible by 64")
    if args.warmup < 1 or args.iters < 2 or args.rounds < 1:
        raise ValueError("--warmup must be positive, --iters at least 2, and --rounds positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Humming and Marlin require a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        raise RuntimeError(f"Humming benchmark requires compute capability >= 8.0, got {properties.major}.{properties.minor}")
    if torch.bfloat16 in _resolve_dtypes(args.dtype) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested BF16 benchmark but the selected CUDA device does not support BF16")

    hardware = {
        "device_argument": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_revision": _git_revision(),
        "nvidia_smi_inventory": _nvidia_smi_inventory(),
        "gpu_idle_preflight": (
            _GPU_IDLE_PREFLIGHT.as_dict() if _GPU_IDLE_PREFLIGHT is not None else None
        ),
    }
    print(json.dumps(hardware, indent=2))

    results = []
    for dtype_index, dtype in enumerate(_resolve_dtypes(args.dtype)):
        result = _benchmark_dtype(
            device=device,
            dtype=dtype,
            size_m=args.m,
            size_k=args.k,
            size_n=args.n,
            seed=args.seed + dtype_index,
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        results.append(result)
        _print_result(result)

    payload = {
        "hardware": hardware,
        "benchmark": {
            "warmup": args.warmup,
            "iters_per_round": args.iters,
            "rounds": args.rounds,
            "timing": (
                "per-iteration and batched CUDA events plus batched synchronized wall time; "
                "speedup uses the median batched CUDA-event round"
            ),
        },
        "dtypes": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out}")


if __name__ == "__main__":
    main()
