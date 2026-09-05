#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep exact FP16-input/FP32-output GEMM dispatches for Qwen3.8-27B P32 shapes."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

REQUESTED_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
QWEN38_27B_SHAPES = (
    ("full_q_gate", 5120, 12288),
    ("full_kv", 5120, 1024),
    ("attn_out", 6144, 5120),
    ("linear_qkv", 5120, 10240),
    ("linear_z", 5120, 6144),
    ("mlp_gate_up", 5120, 17408),
    ("mlp_down", 17408, 5120),
)
BLAS_LIBRARIES = ("default", "cublas", "cublaslt", "ck")
SPLITS = (2, 4, 8, 16)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--m-values", type=int, nargs="+", default=REQUESTED_M)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_dispatch_sweep_gfx950.json"),
    )
    args = parser.parse_args()
    if any(m not in REQUESTED_M for m in args.m_values):
        parser.error(f"--m-values must be drawn from {REQUESTED_M}")
    if min(args.warmup, args.iterations, args.idle_samples) <= 0:
        parser.error("warmup, iterations, and idle samples must be positive")
    return args


def _rocm_snapshot(physical_gpu: int) -> dict[str, object]:
    command = [
        "rocm-smi",
        "-d",
        str(physical_gpu),
        "--showuse",
        "--showmeminfo",
        "vram",
        "--showpids",
        "--showbus",
        "--showuniqueid",
        "--showdriverversion",
        "--showproductname",
        "--json",
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    card = payload.get(f"card{physical_gpu}")
    if not isinstance(card, dict):
        raise TypeError(f"rocm-smi did not report physical GPU {physical_gpu}")
    process_ids = []
    for key, value in payload.get("system", {}).items():
        if not isinstance(key, str) or not key.startswith("PID") or not isinstance(value, str):
            continue
        fields = [field.strip() for field in value.split(",")]
        if len(fields) != 5 or not key[3:].isdigit():
            continue
        gpu_ids = {int(gpu) for gpu in fields[1].split() if gpu.isdigit()}
        if physical_gpu in gpu_ids and fields[2].isdigit() and int(fields[2]) > 0:
            process_ids.append(int(key[3:]))
    return {
        "physical_gpu": physical_gpu,
        "name": card.get("Card Series", "unknown"),
        "pci_bus_id": card.get("PCI Bus", "unknown"),
        "unique_id": card.get("Unique ID", "unknown"),
        "driver": payload.get("system", {}).get("Driver version", "unknown"),
        "utilization_percent": int(card["GPU use (%)"]),
        "vram_total_bytes": int(card["VRAM Total Memory (B)"]),
        "vram_used_bytes": int(card["VRAM Total Used Memory (B)"]),
        "process_ids": sorted(process_ids),
    }


def _idle_preflight(args: argparse.Namespace) -> dict[str, object]:
    tolerance = args.idle_memory_tolerance_mib * 1024 * 1024
    accepted = None
    violations = []
    for sample in range(args.idle_samples):
        accepted = _rocm_snapshot(args.physical_gpu)
        if accepted["utilization_percent"] != 0:
            violations.append(f"sample {sample + 1}: utilization={accepted['utilization_percent']}%")
        if accepted["vram_used_bytes"] > tolerance:
            violations.append(
                f"sample {sample + 1}: VRAM={accepted['vram_used_bytes'] / 2**20:.1f}MiB"
                f">{args.idle_memory_tolerance_mib}MiB"
            )
        if accepted["process_ids"]:
            violations.append(f"sample {sample + 1}: foreign_pids={accepted['process_ids']}")
        if sample + 1 < args.idle_samples:
            time.sleep(args.idle_interval)
    assert accepted is not None
    if violations:
        raise RuntimeError("ROCm idle gate failed: " + "; ".join(violations))
    print(
        "ROCm idle gate: "
        f"physical={accepted['physical_gpu']} pci={accepted['pci_bus_id']} unique_id={accepted['unique_id']} "
        f"utilization=0% vram={accepted['vram_used_bytes'] / 2**20:.1f}MiB "
        f"samples={args.idle_samples} valid=True",
        flush=True,
    )
    return accepted


def _timings(torch, fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    return {
        "min_ms": values[0],
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _errors(actual, reference) -> dict[str, float]:
    difference = actual.float() - reference.float()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": difference.norm().div(reference.float().norm().clamp_min(1e-12)).item(),
    }


def _mm(torch, x, weight, library: str):
    torch.backends.cuda.preferred_blas_library(library)
    return torch.mm(x, weight, out_dtype=torch.float32)


def _n_split_bmm(torch, grouped_x, grouped_weight, m: int, n: int):
    torch.backends.cuda.preferred_blas_library("default")
    return torch.bmm(grouped_x, grouped_weight, out_dtype=torch.float32).permute(1, 0, 2).reshape(m, n)


def _m_split_bmm(torch, grouped_x, grouped_weight, m: int, n: int):
    torch.backends.cuda.preferred_blas_library("default")
    return torch.bmm(grouped_x, grouped_weight, out_dtype=torch.float32).reshape(m, n)


def _print_progress(completed: int, total: int, row: dict[str, object]) -> None:
    winner = row["winner"]
    print(
        f"[{completed:02d}/{total}] {row['shape']:12s} M={row['m']:4d} K={row['k']:5d} N={row['n']:5d} "
        f"base={row['baseline_ms']:.6f}ms best={winner['median_ms']:.6f}ms "
        f"speedup={winner['speedup']:.3f}x {winner['name']}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    hardware = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)

    import torch

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    permitted_pids = set(_rocm_snapshot(args.physical_gpu)["process_ids"])
    rows = []
    total = len(QWEN38_27B_SHAPES) * len(args.m_values)

    for shape, k, n in QWEN38_27B_SHAPES:
        # P32's persistent cache is contiguous [N, K]; the current GEMM consumes
        # its zero-copy transposed [K, N] view. Values do not affect GEMM dispatch.
        weight_nk = torch.randn((n, k), dtype=torch.float16, device=device).mul_(0.02)
        weight_kn = weight_nk.t().contiguous()
        weight_view = weight_nk.t()
        for m in args.m_values:
            x = torch.randn((m, k), dtype=torch.float16, device=device).mul_(0.02)
            torch.backends.cuda.preferred_blas_library("default")
            reference = torch.mm(x, weight_view, out_dtype=torch.float32)
            candidates: list[tuple[str, object]] = []

            for library in BLAS_LIBRARIES:
                candidates.append((f"view_{library}", functools.partial(_mm, torch, x, weight_view, library)))
                candidates.append((f"contiguous_{library}", functools.partial(_mm, torch, x, weight_kn, library)))

            for groups in SPLITS:
                if n % groups == 0:
                    width = n // groups
                    grouped_weight = weight_nk.reshape(groups, width, k).transpose(1, 2)
                    grouped_x = x.unsqueeze(0).expand(groups, -1, -1)
                    candidates.append(
                        (f"n_split_{groups}", functools.partial(_n_split_bmm, torch, grouped_x, grouped_weight, m, n))
                    )
                if m % groups == 0:
                    grouped_x = x.reshape(groups, m // groups, k)
                    grouped_weight = weight_view.unsqueeze(0).expand(groups, -1, -1)
                    candidates.append(
                        (f"m_split_{groups}", functools.partial(_m_split_bmm, torch, grouped_x, grouped_weight, m, n))
                    )

            results = []
            for name, fn in candidates:
                try:
                    actual = fn()
                except RuntimeError as exc:
                    results.append({"name": name, "supported": False, "error": str(exc)})
                    continue
                error = _errors(actual, reference)
                if error["max_abs"] > 2e-3:
                    raise AssertionError(f"{shape} M={m} {name} max_abs={error['max_abs']}")
                timing = _timings(torch, fn, args.warmup, args.iterations)
                results.append({"name": name, "supported": True, **timing, "accuracy": error})

            baseline = next(result for result in results if result["name"] == "view_default")
            for result in results:
                if result["supported"]:
                    result["speedup"] = baseline["median_ms"] / result["median_ms"]
            winner = min((result for result in results if result["supported"]), key=lambda result: result["median_ms"])
            row = {
                "shape": shape,
                "m": m,
                "k": k,
                "n": n,
                "baseline_ms": baseline["median_ms"],
                "winner": winner,
                "candidates": results,
            }
            rows.append(row)
            _print_progress(len(rows), total, row)

        del weight_kn, weight_nk, weight_view, x, reference
        torch.cuda.empty_cache()

    added_pids = set(_rocm_snapshot(args.physical_gpu)["process_ids"]) - permitted_pids
    if added_pids:
        raise RuntimeError(f"benchmark invalidated by new GPU processes: {sorted(added_pids)}")

    best_speedups = [row["winner"]["speedup"] for row in rows]
    result = {
        "schema": "qvq_p32_amd_dispatch_sweep_v1",
        "benchmark_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "accuracy_gate": 2e-3,
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": properties.name,
            "gcn_arch": properties.gcnArchName,
            "compute_units": properties.multi_processor_count,
        },
        "method": {
            "dtype": "fp16 inputs, fp32 output",
            "cache_layout": "contiguous [N,K] with transposed [K,N] GEMM view",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing": "per-iteration device events, one synchronization after sample collection",
            "seed": args.seed,
        },
        "requested_m": list(args.m_values),
        "shapes": [{"name": name, "k": k, "n": n} for name, k, n in QWEN38_27B_SHAPES],
        "summary": {
            "cases": len(rows),
            "winner_speedup_min": min(best_speedups),
            "winner_speedup_geomean": math.exp(statistics.mean(math.log(value) for value in best_speedups)),
            "winner_speedup_max": max(best_speedups),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
