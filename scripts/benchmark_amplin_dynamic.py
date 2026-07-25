#!/usr/bin/env python3
"""Benchmark Amplin dynamic routing against Marlin across M=1..32 model shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmark_amplin_model_shapes import SHAPES  # noqa: E402
from benchmark_amplin_vs_marlin import (  # noqa: E402
    _build_marlin,
    _dequantized_reference,
    _dtype_name,
    _make_case,
    _nvidia_smi_inventory,
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
        description="Benchmark Amplin dynamic router vs Marlin across target model shapes."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--m-values",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 6, 8, 16, 32],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--model",
        choices=("all", "laguna-s-2.1", "glm-5.2", "kimi-k2.5"),
        default="all",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--routing-table-out", type=Path)
    add_gpu_idle_preflight_args(parser)
    return parser.parse_args()


def _measure_median_us(fn, device, warmup, iters, rounds, pre_timing_check):
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


def _bench_shape(spec, dtype, size_m, device, warmup, iters, rounds, seed, pre_timing_check):
    size_n = spec.size_n
    size_k = spec.size_k
    if size_k % 128 != 0 or size_n % 64 != 0:
        return None

    input_tensor, canonical_qweight, canonical_scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        seed=seed,
    )
    reference = _dequantized_reference(input_tensor, canonical_qweight, canonical_scales)

    functions: dict[str, Callable[[], torch.Tensor]] = {}

    def dynamic_fn() -> torch.Tensor:
        return amplin.dynamic(
            input_tensor,
            canonical_qweight,
            canonical_scales,
            warmup=3,
            iters=5,
        )

    functions["dynamic"] = dynamic_fn

    try:
        marlin_module = _build_marlin(
            device=device,
            dtype=dtype,
            qweight=canonical_qweight,
            scales=canonical_scales,
        )
        functions["marlin"] = lambda mod=marlin_module: mod(input_tensor)
    except Exception:
        pass

    results_us = {}
    errors = {}
    selected_kernel = None
    for name, fn in functions.items():
        try:
            median_us = _measure_median_us(fn, device, warmup, iters, rounds, pre_timing_check)
            out = fn()
            err = (out.to(torch.float32) - reference).abs().max().item()
            results_us[name] = median_us
            errors[name] = err
            if name == "dynamic":
                key = (size_m, size_k, size_n, _dtype_name(dtype))
                selected_kernel = amplin.get_routing_table().get(key)
        except Exception as e:
            print(f"  {name} failed for {spec.model} {spec.role} M={size_m} K={size_k} N={size_n}: {e}")

    return {
        "model": spec.model,
        "role": spec.role,
        "m": size_m,
        "k": size_k,
        "n": size_n,
        "dtype": _dtype_name(dtype),
        "results_us": results_us,
        "max_abs_error": errors,
        "selected_kernel": selected_kernel,
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

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    shapes = [spec for spec in SHAPES if args.model == "all" or spec.model == args.model]

    # Reset static and dynamic routing tables so the benchmark produces a fresh
    # selection that includes newly-added candidates such as ``marlin_style``.
    amplin.set_routing_table({})
    amplin.clear_dynamic_routing_table()

    print(
        f"Device: {device} {properties.name} sm_{properties.major}{properties.minor} "
        f"SMs={properties.multi_processor_count}"
    )
    print(
        f"dtype={args.dtype} m_values={args.m_values} warmup={args.warmup} "
        f"iters={args.iters} rounds={args.rounds}"
    )
    print(f"GPU inventory:\n{chr(10).join(_nvidia_smi_inventory())}")

    all_results = []
    routing_table: dict[tuple[int, int, int, str], str] = {}
    for spec in shapes:
        for size_m in args.m_values:
            result = _bench_shape(
                spec,
                dtype,
                size_m,
                device,
                args.warmup,
                args.iters,
                args.rounds,
                args.seed,
                pre_timing_check,
            )
            if result is None:
                continue
            all_results.append(result)
            row = (
                f"{result['model']:15s} {result['role']:25s} "
                f"M={result['m']:2d} K={result['k']:6d} N={result['n']:7d} "
                + " ".join(f"{n}={t:.1f}us" for n, t in result["results_us"].items())
                + f"  selected={result['selected_kernel'] or 'unknown'}"
            )
            print(row)
            if result["selected_kernel"] is not None:
                key = (result["m"], result["k"], result["n"], result["dtype"])
                routing_table[key] = result["selected_kernel"]

    if args.json_out:
        args.json_out.write_text(json.dumps(all_results, indent=2))
        print(f"Wrote results to {args.json_out}")

    if args.routing_table_out:
        serializable = {str(k): v for k, v in routing_table.items()}
        args.routing_table_out.write_text(json.dumps(serializable, indent=2))
        print(f"Wrote routing table to {args.routing_table_out}")


if __name__ == "__main__":
    main()
