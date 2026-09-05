#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure the full-shape ceiling from folding QVQ transforms into the AMD cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_qvq_p32_amd_dispatch_sweep import (
    QWEN38_27B_SHAPES,
    REQUESTED_M,
    _rocm_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_AXES = {
    # Qwen3.8 architecture policy. full_kv uses the stricter K-projection
    # policy; V skips its output Hadamard and therefore has a lower ceiling.
    "full_q_gate": (True, True),
    "full_kv": (True, True),
    "attn_out": (True, True),
    "linear_qkv": (True, True),
    "linear_z": (True, True),
    "mlp_gate_up": (True, False),
    "mlp_down": (False, True),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--dispatch-sweep",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_dispatch_sweep_repeat_gfx950.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_fold_ceiling_gfx950.json"),
    )
    return parser.parse_args()


def _idle_gate(physical_gpu: int) -> dict[str, object]:
    accepted = None
    for sample in range(3):
        accepted = _rocm_snapshot(physical_gpu)
        if accepted["utilization_percent"] != 0 or accepted["process_ids"]:
            raise RuntimeError(f"ROCm idle gate failed on sample {sample + 1}: {accepted}")
        if sample < 2:
            time.sleep(1.0)
    assert accepted is not None
    print(
        f"ROCm idle gate: physical={physical_gpu} pci={accepted['pci_bus_id']} "
        f"unique_id={accepted['unique_id']} utilization=0% samples=3 valid=True",
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
    }


def main() -> None:
    args = _parse_args()
    hardware = _idle_gate(args.physical_gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)

    import torch

    from gptqmodel.nn_modules.qlinear.qvq import _qvq_hadamard_fused

    dispatch = json.loads(args.dispatch_sweep.read_text())
    dispatch_rows = {(row["shape"], row["m"]): row for row in dispatch["rows"]}
    rows = []
    speedups = []
    for shape, k, n in QWEN38_27B_SHAPES:
        input_hadamard, output_hadamard = SHAPE_AXES[shape]
        su = torch.ones(k, dtype=torch.float16, device="cuda")
        sv = torch.ones(n, dtype=torch.float16, device="cuda")
        for m in REQUESTED_M:
            x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
            inner_output = torch.zeros((m, n), dtype=torch.float32, device="cuda")

            def input_transform(x=x, su=su, input_hadamard=input_hadamard):
                if input_hadamard:
                    return _qvq_hadamard_fused(x, pre_scale=su, scale_mode=2)
                return x * su

            def output_recovery(
                inner_output=inner_output,
                sv=sv,
                n=n,
                output_hadamard=output_hadamard,
            ):
                if output_hadamard:
                    mode = 3 if n >= 2048 else 4
                    recovered = _qvq_hadamard_fused(
                        inner_output,
                        post_scale=sv,
                        scale_mode=mode,
                    )
                else:
                    recovered = inner_output * sv
                return recovered.to(torch.float16)

            input_timing = _timings(torch, input_transform, args.warmup, args.iterations)
            output_timing = _timings(torch, output_recovery, args.warmup, args.iterations)
            gemm_ms = float(dispatch_rows[(shape, m)]["baseline_ms"])
            current_estimate_ms = input_timing["median_ms"] + gemm_ms + output_timing["median_ms"]
            speedup = current_estimate_ms / gemm_ms
            speedups.append(speedup)
            rows.append(
                {
                    "shape": shape,
                    "m": m,
                    "k": k,
                    "n": n,
                    "input_hadamard": input_hadamard,
                    "output_hadamard": output_hadamard,
                    "input_transform": input_timing,
                    "inner_gemm_ms": gemm_ms,
                    "output_recovery": output_timing,
                    "current_full_estimate_ms": current_estimate_ms,
                    "folded_gemm_estimate_ms": gemm_ms,
                    "folding_ceiling_speedup": speedup,
                }
            )
            print(
                f"[{len(rows):02d}/91] {shape:12s} M={m:4d} "
                f"input={input_timing['median_ms']:.6f}ms gemm={gemm_ms:.6f}ms "
                f"output={output_timing['median_ms']:.6f}ms ceiling={speedup:.3f}x",
                flush=True,
            )
        del su, sv, x, inner_output
        torch.cuda.empty_cache()

    result = {
        "schema": "qvq_p32_amd_fold_ceiling_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip},
        "requested_m": list(REQUESTED_M),
        "method": "stage-additive ceiling; folded candidate retains only the measured current GEMM",
        "summary": {
            "minimum_speedup": min(speedups),
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(value) for value in speedups)),
            "maximum_speedup": max(speedups),
            "cases_at_least_1_5x": sum(value >= 1.5 for value in speedups),
            "case_count": len(speedups),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
