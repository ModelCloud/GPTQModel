#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate

from gptqmodel.quantization.paroquant.optimization import build_random_rotation_buffers
from gptqmodel.utils.paroquant import (
    _rotation_launch_config,
    apply_paroquant_rotation,
    apply_paroquant_rotation_reference,
    clear_paroquant_rotation_extension_cache,
    prewarm_paroquant_rotation_extension,
)


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    rows: int
    hidden: int
    group_size: int = 128
    krot: int = 8
    pair_ratio: float = 0.5


DEFAULT_CASES: tuple[BenchCase, ...] = (
    BenchCase("decode_h2048_r1", rows=1, hidden=2048),
    BenchCase("decode_h2048_r4", rows=4, hidden=2048),
    BenchCase("decode_h2048_r8", rows=8, hidden=2048),
    BenchCase("prefill_h2048_r128", rows=128, hidden=2048),
    BenchCase("batch_h2048_r512", rows=512, hidden=2048),
    BenchCase("batch_h2048_r2048", rows=2048, hidden=2048),
    BenchCase("decode_h4096_r1", rows=1, hidden=4096),
    BenchCase("decode_h4096_r8", rows=8, hidden=4096),
    BenchCase("prefill_h4096_r128", rows=128, hidden=4096),
    BenchCase("batch_h4096_r512", rows=512, hidden=4096),
    BenchCase("batch_h4096_r2048", rows=2048, hidden=4096),
    BenchCase("decode_h8192_r1", rows=1, hidden=8192),
    BenchCase("decode_h8192_r4", rows=4, hidden=8192),
    BenchCase("decode_h8192_r8", rows=8, hidden=8192),
    BenchCase("prefill_h8192_r128", rows=128, hidden=8192),
    BenchCase("batch_h8192_r512", rows=512, hidden=8192),
    BenchCase("batch_h8192_r1024", rows=1024, hidden=8192),
    BenchCase("batch_h8192_r2048", rows=2048, hidden=8192),
)

QUICK_CASES: tuple[BenchCase, ...] = (
    BenchCase("decode_h2048_r1", rows=1, hidden=2048),
    BenchCase("prefill_h2048_r128", rows=128, hidden=2048),
    BenchCase("batch_h4096_r512", rows=512, hidden=4096),
    BenchCase("batch_h8192_r1024", rows=1024, hidden=8192),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the fused ParoQuant CUDA rotation kernel.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index within the current visible set.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the full payload as JSON.")
    parser.add_argument("--force-rebuild-extension", action="store_true")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _torch_device_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    return torch.cuda.get_device_name(device)


def _subset_cases(cases: tuple[BenchCase, ...], shard_index: int, num_shards: int) -> list[BenchCase]:
    if num_shards <= 0:
        raise ValueError("`num_shards` must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"`shard_index` must be in [0, {num_shards - 1}].")
    return [case for idx, case in enumerate(cases) if idx % num_shards == shard_index]


def _rotation_bandwidth_gbps(case: BenchCase, dtype: torch.dtype, elapsed_ms: float) -> float:
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = case.rows * case.hidden * element_size * 2
    return total_bytes / (elapsed_ms * 1e-3) / 1e9


def _benchmark_ms(fn, device: torch.device, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3 / iters


def _make_case_inputs(case: BenchCase, dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.randn((case.rows, case.hidden), generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    pairs, _mask = build_random_rotation_buffers(
        in_features=case.hidden,
        group_size=case.group_size,
        krot=case.krot,
        pair_ratio=case.pair_ratio,
        seed=seed,
        device=device,
    )
    theta = torch.empty((case.krot, case.hidden // 2), dtype=torch.float32, device="cpu")
    theta.uniform_(-0.25, 0.25, generator=generator)
    theta = theta.to(device=device, dtype=dtype)
    scales = torch.empty((1, case.hidden), dtype=torch.float32, device="cpu")
    scales.uniform_(0.75, 1.25, generator=generator)
    scales = scales.to(device=device, dtype=dtype)
    return {
        "x": x.contiguous(),
        "pairs": pairs.contiguous(),
        "theta": theta.contiguous(),
        "scales": scales.contiguous(),
    }


def run(device: torch.device, dtype: torch.dtype, warmup: int, iters: int, quick: bool, seed: int, shard_index: int, num_shards: int) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the ParoQuant rotation benchmark.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    selected_cases = _subset_cases(QUICK_CASES if quick else DEFAULT_CASES, shard_index=shard_index, num_shards=num_shards)
    rows: list[dict[str, Any]] = []

    for case_index, case in enumerate(selected_cases):
        case_seed = seed + (case_index * 17)
        inputs = _make_case_inputs(case, dtype=dtype, device=device, seed=case_seed)
        x = inputs["x"]
        pairs = inputs["pairs"]
        theta = inputs["theta"]
        scales = inputs["scales"]

        with torch.inference_mode():
            fused = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=case.group_size)
            reference = apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=case.group_size)
        cta_m, row_pad = _rotation_launch_config(x, pairs, theta, scales=scales, group_size=case.group_size)

        diff = (fused - reference).abs()
        fp32_metrics = {
            "fused_fp32_max_abs": None,
            "fused_fp32_mean_abs": None,
            "reference_fp32_max_abs": None,
            "reference_fp32_mean_abs": None,
        }
        if dtype != torch.float32:
            fp32_inputs = _make_case_inputs(case, dtype=torch.float32, device=device, seed=case_seed)
            with torch.inference_mode():
                reference_fp32 = apply_paroquant_rotation_reference(
                    fp32_inputs["x"],
                    fp32_inputs["pairs"],
                    fp32_inputs["theta"],
                    scales=fp32_inputs["scales"],
                    group_size=case.group_size,
                )
            fused_fp32_diff = (fused.float() - reference_fp32).abs()
            reference_fp32_diff = (reference.float() - reference_fp32).abs()
            fp32_metrics = {
                "fused_fp32_max_abs": fused_fp32_diff.max().item(),
                "fused_fp32_mean_abs": fused_fp32_diff.mean().item(),
                "reference_fp32_max_abs": reference_fp32_diff.max().item(),
                "reference_fp32_mean_abs": reference_fp32_diff.mean().item(),
            }
        elapsed_ms = _benchmark_ms(
            lambda: apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=case.group_size),
            device=device,
            warmup=warmup,
            iters=iters,
        )
        rows.append(
            {
                **asdict(case),
                "dtype": str(dtype).replace("torch.", ""),
                "cta_m": cta_m,
                "row_pad": row_pad,
                "latency_ms": elapsed_ms,
                "gbps": _rotation_bandwidth_gbps(case, dtype, elapsed_ms),
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
                **fp32_metrics,
            }
        )

    geo_mean_ms = math.exp(sum(math.log(row["latency_ms"]) for row in rows) / len(rows)) if rows else float("nan")
    geo_mean_gbps = math.exp(sum(math.log(row["gbps"]) for row in rows) / len(rows)) if rows else float("nan")

    return {
        "device": _torch_device_name(device),
        "cuda_device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
        "quick": quick,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "geo_mean_ms": geo_mean_ms,
        "geo_mean_gbps": geo_mean_gbps,
        "rows": rows,
    }


def _print_ascii(results: dict[str, Any]) -> None:
    print(f"Device: {results['device']} ({results['cuda_device']})")
    print(
        tabulate(
            [
                [
                    row["case_id"],
                    row["rows"],
                    row["hidden"],
                    row["dtype"],
                    row["cta_m"],
                    row["row_pad"],
                    f"{row['latency_ms']:.3f}",
                    f"{row['gbps']:.1f}",
                    f"{row['max_abs']:.6f}",
                    f"{row['mean_abs']:.6f}",
                    "-" if row["fused_fp32_max_abs"] is None else f"{row['fused_fp32_max_abs']:.6f}",
                    "-" if row["fused_fp32_mean_abs"] is None else f"{row['fused_fp32_mean_abs']:.6f}",
                    "-" if row["reference_fp32_max_abs"] is None else f"{row['reference_fp32_max_abs']:.6f}",
                    "-" if row["reference_fp32_mean_abs"] is None else f"{row['reference_fp32_mean_abs']:.6f}",
                ]
                for row in results["rows"]
            ],
            headers=[
                "case",
                "rows",
                "hidden",
                "dtype",
                "cta_m",
                "row_pad",
                "latency_ms",
                "gbps",
                "fused vs ref max_abs",
                "fused vs ref mean_abs",
                "fused vs fp32 max_abs",
                "fused vs fp32 mean_abs",
                "ref vs fp32 max_abs",
                "ref vs fp32 mean_abs",
            ],
            tablefmt="plain",
        )
    )
    print(f"geo_mean_ms {results['geo_mean_ms']:.3f}")
    print(f"geo_mean_gbps {results['geo_mean_gbps']:.1f}")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ParoQuant rotation benchmark.")

    device = torch.device(f"cuda:{args.device}")
    if args.force_rebuild_extension:
        build_root = Path("/tmp") / (
            f"paroquant_ext_{os.getpid()}_dev{args.device}_shard{args.shard_index}_of_{args.num_shards}"
        )
        os.environ["GPTQMODEL_PAROQUANT_BUILD_ROOT"] = str(build_root)
        os.environ["GPTQMODEL_PAROQUANT_FORCE_REBUILD"] = "1"
        clear_paroquant_rotation_extension_cache()
    else:
        os.environ.pop("GPTQMODEL_PAROQUANT_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_PAROQUANT_FORCE_REBUILD", None)

    if not prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=device,
    ):
        raise RuntimeError("Failed to build/load the fused ParoQuant CUDA rotation extension.")

    results = run(
        device=device,
        dtype=_resolve_dtype(args.dtype),
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    _print_ascii(results)
    if args.json:
        print(json.dumps(results, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
