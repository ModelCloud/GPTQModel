#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Focused Amplin M=16 N64 tile4 benchmark against splitk24 and Marlin.

Runs the Laguna/GLM/Kimi shape list for M=16 and compares the new shared-A
tile4 (and tile8 where legal) kernel with the current best M16 N64 splitk24
kernel and the Marlin baseline. This is a fast companion to the full model
sweep for evaluating the M16 large-N gap.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmark_amplin_model_shapes import SHAPES, ShapeSpec  # noqa: E402
from benchmark_amplin_vs_marlin import (  # noqa: E402
    BITS,
    GROUP_SIZE,
    _build_marlin,
    _dequantized_reference,
    _dtype_name,
    _make_case,
    _nvidia_smi_inventory,
    _resolve_dtypes,
)
from gpu_idle_preflight import (  # noqa: E402
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)

_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight()

import torch  # noqa: E402
from gptqmodel.utils import amplin  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Amplin M=16 shared-A N64 tile4 vs splitk24 and Marlin."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="fp16")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--model", choices=("all", "laguna-s-2.1", "glm-5.2", "kimi-k2.5"), default="all")
    parser.add_argument("--json-out", type=Path)
    add_gpu_idle_preflight_args(parser)
    return parser.parse_args()


def _measure_median_us(
    fn,
    device: torch.device,
    warmup: int,
    iters: int,
    rounds: int,
    pre_timing_check,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    round_medians = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        if pre_timing_check is not None:
            pre_timing_check()
        times = []
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end) * 1000.0)
        round_medians.append(statistics.median(times))
    return statistics.median(round_medians)


def _bench_shape(
    spec: ShapeSpec,
    dtype: torch.dtype,
    size_m: int,
    device: torch.device,
    warmup: int,
    iters: int,
    rounds: int,
    seed: int,
    pre_timing_check,
) -> dict | None:
    marlin_legal = spec.size_n % 64 == 0
    tile4_legal = spec.size_n % (4 * 64) == 0 and spec.size_k % 128 == 0
    tile8_legal = spec.size_n % (8 * 64) == 0 and spec.size_k % 128 == 0
    splitk24_legal = spec.size_n % 64 == 0 and spec.size_k % 128 == 0
    if not (tile4_legal or splitk24_legal or marlin_legal):
        return None

    input_tensor, canonical_qweight, canonical_scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=size_m,
        size_k=spec.size_k,
        size_n=spec.size_n,
        seed=seed,
    )
    reference = _dequantized_reference(input_tensor, canonical_qweight, canonical_scales)
    packed_n64_qweight = amplin.pack_mma_lane_n64_qweight(canonical_qweight)
    _, packed_scales = amplin.pack_hmma_weights(canonical_qweight, canonical_scales)

    functions: dict[str, tuple[callable, str]] = {}
    if splitk24_legal:
        functions["splitk24"] = (
            lambda: amplin.mma_lane_m16_n64_splitk24_pipe2_interleaved(
                input_tensor,
                packed_n64_qweight,
                packed_scales,
                logical_n=spec.size_n,
            ),
            "splitk24",
        )
    if tile4_legal:
        functions["tile4"] = (
            lambda: amplin.mma_lane_m16_n64_tile4_shared_a(
                input_tensor,
                packed_n64_qweight,
                packed_scales,
                logical_n=spec.size_n,
            ),
            "tile4",
        )
    if tile8_legal:
        functions["tile8"] = (
            lambda: amplin.mma_lane_m16_n64_tile8_shared_a(
                input_tensor,
                packed_n64_qweight,
                packed_scales,
                logical_n=spec.size_n,
            ),
            "tile8",
        )
    if marlin_legal:
        marlin_module = _build_marlin(
            device=device,
            dtype=dtype,
            qweight=canonical_qweight,
            scales=canonical_scales,
        )
        functions["marlin"] = (lambda: marlin_module(input_tensor), "marlin")

    results: dict[str, float] = {}
    errors: dict[str, float] = {}
    for name, (fn, _) in functions.items():
        median_us = _measure_median_us(
            fn,
            device,
            warmup,
            iters,
            rounds,
            pre_timing_check,
        )
        out = fn()
        err = (out.to(torch.float32) - reference).abs().max().item()
        results[name] = median_us
        errors[name] = err

    return {
        "model": spec.model,
        "role": spec.role,
        "m": size_m,
        "k": spec.size_k,
        "n": spec.size_n,
        "dtype": _dtype_name(dtype),
        "results_us": results,
        "max_abs_error": errors,
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA required")
    torch.cuda.set_device(device)

    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(f"Requires sm_80, got {properties.major}.{properties.minor}")

    pre_timing_check = (
        (lambda: recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT))
        if _GPU_IDLE_PREFLIGHT is not None
        else None
    )

    shapes = [spec for spec in SHAPES if args.model == "all" or spec.model == args.model]
    all_results = []
    for dtype in _resolve_dtypes(args.dtype):
        for spec in shapes:
            result = _bench_shape(
                spec,
                dtype,
                args.m,
                device,
                args.warmup,
                args.iters,
                args.rounds,
                args.seed,
                pre_timing_check,
            )
            if result is not None:
                all_results.append(result)
                row = (
                    f"{result['model']:15s} {result['role']:25s} "
                    f"M={result['m']:2d} K={result['k']:6d} N={result['n']:7d} "
                    + " ".join(f"{n}={t:.1f}us" for n, t in result["results_us"].items())
                )
                print(row)

    if args.json_out is not None:
        payload = {
            "hardware": {
                "device": str(device),
                "name": properties.name,
                "sm_count": properties.multi_processor_count,
                "nvidia_smi_inventory": _nvidia_smi_inventory(),
            },
            "quantization": {
                "bits": BITS,
                "group_size": GROUP_SIZE,
                "sym": True,
                "desc_act": False,
            },
            "results": all_results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out}")


if __name__ == "__main__":
    main()
