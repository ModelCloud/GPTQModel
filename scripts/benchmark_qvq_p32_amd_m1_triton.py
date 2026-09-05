#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tune a one-pass row-major FP16 GEMV for folded Qwen3.8-27B weights."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_qvq_p32_amd import (
    _idle_preflight,
    _rocm_snapshot,
    _timing_recheck,
    _timings,
)
from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    (1, 256, 4),
    (1, 256, 8),
    (1, 512, 4),
    (1, 512, 8),
    (1, 1024, 4),
    (1, 1024, 8),
    (2, 256, 4),
    (2, 256, 8),
    (2, 512, 4),
    (2, 512, 8),
    (4, 256, 4),
    (4, 256, 8),
    (4, 512, 4),
    (4, 512, 8),
    (4, 1024, 4),
    (4, 1024, 8),
    (8, 256, 4),
    (8, 256, 8),
    (8, 512, 4),
    (8, 512, 8),
    (8, 1024, 4),
    (8, 1024, 8),
    (16, 256, 4),
    (16, 256, 8),
    (16, 512, 4),
    (16, 512, 8),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_m1_triton_tune_gfx950.json"),
    )
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.idle_samples) <= 0:
        parser.error("warmup, iterations, and idle samples must be positive")
    return args


def main() -> None:
    args = _parse_args()
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-mi355x")

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def dense_gemv_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        size_k: tl.constexpr,
        size_n: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        xcd_swizzle: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if xcd_swizzle:
            num_pid = tl.num_programs(0)
            pid = (pid % 8) * (num_pid // 8) + pid // 8
        rows = pid * block_n + tl.arange(0, block_n)
        row_mask = rows < size_n
        accumulator = tl.zeros((block_n,), dtype=tl.float32)
        for k_block in range(size_k // block_k):
            columns = k_block * block_k + tl.arange(0, block_k)
            activation = tl.load(input_ptr + columns).to(tl.float32)
            weight = tl.load(
                weight_ptr + rows[:, None] * size_k + columns[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(weight * activation[None, :], axis=1)
        tl.store(output_ptr + rows, accumulator, mask=row_mask)

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    rows = []
    permitted_pids = set(hardware["process_ids"]) | set(_rocm_snapshot(args.physical_gpu)["process_ids"])
    for shape, k, n in QWEN38_27B_SHAPES:
        if (k, n) not in {(5120, 12288), (5120, 1024), (5120, 10240), (5120, 6144)}:
            continue
        x = (torch.randn((k,), dtype=torch.float16, device="cuda", generator=generator) * 0.01).contiguous()
        weight = (
            torch.randn((n, k), dtype=torch.float16, device="cuda", generator=generator) * 0.01
        ).contiguous()
        output = torch.empty((n,), dtype=torch.float16, device="cuda")
        reference = torch.mv(weight.float(), x.float())
        torch.cuda.synchronize()

        def rocblas_mv(weight=weight, x=x):
            return torch.mv(weight, x)

        timings = {"rocblas_mv": _timings(torch, rocblas_mv, warmup=args.warmup, iterations=args.iterations)}
        configs = []
        for block_n, block_k, num_warps in CONFIGS:
            for xcd_swizzle in (False, True):
                name = f"bn{block_n}_bk{block_k}_w{num_warps}_x{int(xcd_swizzle)}"
                grid = (triton.cdiv(n, block_n),)

                def candidate(
                    grid=grid,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=num_warps,
                    xcd_swizzle=xcd_swizzle,
                    x=x,
                    weight=weight,
                    output=output,
                    k=k,
                    n=n,
                ):
                    dense_gemv_kernel[grid](
                        x,
                        weight,
                        output,
                        size_k=k,
                        size_n=n,
                        block_n=block_n,
                        block_k=block_k,
                        xcd_swizzle=xcd_swizzle,
                        num_warps=num_warps,
                        num_stages=1,
                        waves_per_eu=0,
                    )
                    return output

                actual = candidate().clone()
                torch.cuda.synchronize()
                difference = actual.float() - reference
                error = {
                    "max_abs": difference.abs().max().item(),
                    "mean_abs": difference.abs().mean().item(),
                    "relative_l2": difference.norm().div(reference.norm().clamp_min(1e-12)).item(),
                }
                timing = _timings(torch, candidate, warmup=args.warmup, iterations=args.iterations)
                timings[name] = timing
                configs.append(
                    {
                        "name": name,
                        "block_n": block_n,
                        "block_k": block_k,
                        "num_warps": num_warps,
                        "xcd_swizzle": xcd_swizzle,
                        "timing": timing,
                        "accuracy": error,
                        "accuracy_pass": error["max_abs"] <= 2e-3,
                        "speedup_vs_rocblas_mv": timings["rocblas_mv"]["median_ms"] / timing["median_ms"],
                    }
                )
        snapshot, timing_valid = _timing_recheck(args, permitted_pids)
        valid = valid and timing_valid
        best = min(configs, key=lambda config: config["timing"]["median_ms"])
        row = {
            "shape": shape,
            "m": 1,
            "k": k,
            "n": n,
            "rocblas_mv": timings["rocblas_mv"],
            "best": best,
            "configs": configs,
            "timing_snapshot": snapshot,
        }
        rows.append(row)
        print(
            f"[{len(rows)}] {shape:12s} rocBLAS={timings['rocblas_mv']['median_ms'] * 1e3:.3f}us "
            f"best={best['name']}:{best['timing']['median_ms'] * 1e3:.3f}us "
            f"speedup={best['speedup_vs_rocblas_mv']:.3f}x max_abs={best['accuracy']['max_abs']:.7g}",
            flush=True,
        )
        del x, weight, output, reference
        torch.cuda.empty_cache()

    result = {
        "schema": "qvq_p32_amd_m1_triton_tune_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": valid,
        "accuracy_gate": 2e-3,
        "model": "Qwen/Qwen3.8-27B",
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
