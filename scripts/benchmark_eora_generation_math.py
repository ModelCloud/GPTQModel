#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel.eora.eora import _eora_compute_svd


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_SVD_ALGOS = ("exact", "auto", "lowrank")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the calibration-time SVD used to generate EoRA LoRA factors."
    )
    parser.add_argument("matrices", nargs="+", type=Path, help="2-D tensors saved with torch.save().")
    parser.add_argument("--device", default="cuda:0", help="Logical device within CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--algos", default="exact,auto,lowrank", help="Comma-separated EoRA SVD algorithms.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _parse_algos(raw: str) -> list[str]:
    algos = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [algo for algo in algos if algo not in _SVD_ALGOS]
    if invalid or not algos:
        raise ValueError(f"Invalid SVD algorithms {invalid or algos}; expected a subset of {_SVD_ALGOS}.")
    return algos


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def _load_matrix(path: Path, device: torch.device) -> torch.Tensor:
    matrix = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(matrix) or matrix.ndim != 2:
        raise ValueError(f"Expected one 2-D tensor in {path}, got {type(matrix)!r}.")
    return matrix.to(device=device, dtype=torch.float32)


def _time_svd(
    matrix: torch.Tensor,
    *,
    rank: int,
    algo: str,
    warmup: int,
    iters: int,
) -> tuple[list[float], int]:
    if warmup < 0 or iters < 1:
        raise ValueError("--warmup must be non-negative and --iters must be positive.")

    device = matrix.device
    with torch.inference_mode():
        for _ in range(warmup):
            output = _eora_compute_svd(matrix, rank, algo=algo)
            del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)

        samples_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = _eora_compute_svd(matrix, rank, algo=algo)
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
            del output

        peak_bytes = max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
    return samples_ms, peak_bytes


def _objective_metrics(
    matrix: torch.Tensor,
    *,
    rank: int,
    algo: str,
    exact_residual_sq: torch.Tensor,
) -> dict[str, float | bool]:
    with torch.inference_mode():
        u, singular_values, vh = _eora_compute_svd(matrix, rank, algo=algo)
        total_energy = matrix.square().sum(dtype=torch.float64)
        residual_sq = _rank_residual_sq(matrix, u, singular_values, vh, rank)
        denominator = exact_residual_sq.clamp_min(torch.finfo(torch.float64).tiny)
        extra_residual_fraction = float(((residual_sq - exact_residual_sq) / total_energy).item())
        if abs(extra_residual_fraction) < 1e-6:
            extra_residual_fraction = 0.0
        metrics = {
            "all_finite": bool(
                torch.isfinite(u[:, :rank]).all().item()
                and torch.isfinite(singular_values[:rank]).all().item()
                and torch.isfinite(vh[:rank]).all().item()
            ),
            "captured_energy_fraction": float((1 - residual_sq / total_energy).item()),
            "residual_ratio_to_exact": float((residual_sq / denominator).item()),
            "extra_residual_fraction": extra_residual_fraction,
        }
    return metrics


def _rank_residual_sq(
    matrix: torch.Tensor,
    u: torch.Tensor,
    singular_values: torch.Tensor,
    vh: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Measure the rank-r reconstruction residual without a full-size temporary."""

    used_rank = min(rank, singular_values.numel())
    scaled_u = u[:, :used_rank] * singular_values[:used_rank].unsqueeze(0)
    vh_r = vh[:used_rank]
    residual_sq = torch.zeros((), dtype=torch.float64, device=matrix.device)
    for row_start in range(0, matrix.shape[0], 256):
        row_end = min(matrix.shape[0], row_start + 256)
        reconstruction = scaled_u[row_start:row_end] @ vh_r
        error = matrix[row_start:row_end] - reconstruction
        residual_sq.add_(error.square().sum(dtype=torch.float64))
    return residual_sq


def _environment(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices.")
    algos = _parse_algos(args.algos)

    records: list[dict[str, object]] = []
    for path in args.matrices:
        matrix = _load_matrix(path, device)
        if args.rank < 1 or args.rank > min(matrix.shape):
            raise ValueError(f"Rank {args.rank} is invalid for {tuple(matrix.shape)}.")

        with torch.inference_mode():
            exact_u, exact_singular_values, exact_vh = torch.linalg.svd(matrix, full_matrices=False)
            exact_residual_sq = _rank_residual_sq(
                matrix,
                exact_u,
                exact_singular_values,
                exact_vh,
                args.rank,
            )
        del exact_u, exact_vh

        case_records = []
        for algo in algos:
            samples_ms, peak_bytes = _time_svd(
                matrix,
                rank=args.rank,
                algo=algo,
                warmup=args.warmup,
                iters=args.iters,
            )
            metrics = _objective_metrics(
                matrix,
                rank=args.rank,
                algo=algo,
                exact_residual_sq=exact_residual_sq,
            )
            record = {
                "case": path.stem,
                "path": str(path),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "rank": args.rank,
                "algo": algo,
                "mean_ms": statistics.fmean(samples_ms),
                "p50_ms": statistics.median(samples_ms),
                "p95_ms": _percentile(samples_ms, 0.95),
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "peak_temporary_bytes": peak_bytes,
                **metrics,
            }
            records.append(record)
            case_records.append(record)

        exact_record = next((record for record in case_records if record["algo"] == "exact"), None)
        for record in case_records:
            record["speedup_vs_exact"] = (
                exact_record["p50_ms"] / record["p50_ms"] if exact_record is not None else None
            )

        del matrix, exact_singular_values, exact_residual_sq
        torch.cuda.empty_cache()

    table = [
        [
            record["case"],
            "x".join(str(value) for value in record["shape"]),
            record["algo"],
            f"{record['p50_ms']:.3f}",
            f"{record['p95_ms']:.3f}",
            f"{record['speedup_vs_exact']:.2f}" if record["speedup_vs_exact"] is not None else "n/a",
            f"{record['captured_energy_fraction']:.8f}",
            f"{record['extra_residual_fraction']:.3e}",
            str(record["all_finite"]),
        ]
        for record in records
    ]
    print(
        tabulate(
            table,
            headers=("case", "shape", "algo", "p50 ms", "p95 ms", "speedup", "energy", "extra residual", "finite"),
            tablefmt="grid",
        ),
        flush=True,
    )

    payload = {
        "environment": _environment(device),
        "rank": args.rank,
        "warmup": args.warmup,
        "iterations": args.iters,
        "results": records,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
