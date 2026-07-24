#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare the retained AWQ M32 Qwen prefill tile with its M16 fallback."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from tabulate import tabulate

from benchmark_paroquant_qwen3_8b import QWEN3_8B_PROJECTIONS, _build_module, _make_buffers
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear


_DISABLE_ENV = "GPTQMODEL_AWQ_DISABLE_M32_QWEN_PREFILL"


def _stats(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    return {
        "p50_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _select_m32(enabled: bool) -> None:
    if enabled:
        os.environ.pop(_DISABLE_ENV, None)
    else:
        os.environ[_DISABLE_ENV] = "1"


def _measure(module, x: torch.Tensor, *, m32: bool, warmup: int, iters: int) -> dict[str, float]:
    _select_m32(m32)
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(x.device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            module(x)
            ends[index].record()
        ends[-1].synchronize()
    return _stats([starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--abba-pairs", type=int, default=4)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")

    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cases = []
    table_rows = []
    for projection_index, projection in enumerate(QWEN3_8B_PROJECTIONS):
        buffers = _make_buffers(projection, dtype=dtype, seed=10_000 + projection_index)
        module = _build_module(ParoLinear, projection, buffers, device=device, dtype=dtype)
        torch.manual_seed(20_000 + projection_index)
        torch.cuda.manual_seed_all(20_000 + projection_index)
        x = torch.randn((1, args.rows, projection.in_features), device=device, dtype=dtype)

        _select_m32(False)
        with torch.inference_mode():
            m16_output = module(x)
        _select_m32(True)
        with torch.inference_mode():
            m32_output = module(x)
        torch.testing.assert_close(m32_output, m16_output, rtol=0, atol=0)

        block_results = {"m16": [], "m32": []}
        orders = (("m16", "m32"), ("m32", "m16")) * args.abba_pairs
        for order in orders:
            for label in order:
                block_results[label].append(
                    _measure(
                        module,
                        x,
                        m32=label == "m32",
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )

        summary = {
            label: {
                metric: statistics.median(block[metric] for block in results)
                for metric in ("p50_us", "mean_us", "p95_us")
            }
            for label, results in block_results.items()
        }
        p50_speedup = summary["m16"]["p50_us"] / summary["m32"]["p50_us"]
        mean_speedup = summary["m16"]["mean_us"] / summary["m32"]["mean_us"]
        case = {
            "projection": projection.name,
            "in_features": projection.in_features,
            "out_features": projection.out_features,
            "calls_per_layer": projection.calls_per_layer,
            "exact": bool(torch.equal(m32_output, m16_output)),
            "blocks": block_results,
            "summary": summary,
            "p50_speedup": p50_speedup,
            "mean_speedup": mean_speedup,
        }
        cases.append(case)
        table_rows.append(
            [
                projection.name,
                f"{projection.in_features}->{projection.out_features}",
                f"{summary['m16']['p50_us']:.3f}/{summary['m32']['p50_us']:.3f}",
                f"{p50_speedup:.3f}x",
                f"{summary['m16']['mean_us']:.3f}/{summary['m32']['mean_us']:.3f}",
                f"{mean_speedup:.3f}x",
            ]
        )
        del module, buffers, x, m16_output, m32_output
        torch.cuda.empty_cache()

    weighted = {}
    for metric in ("p50_us", "mean_us"):
        m16 = sum(case["summary"]["m16"][metric] * case["calls_per_layer"] for case in cases)
        m32 = sum(case["summary"]["m32"][metric] * case["calls_per_layer"] for case in cases)
        weighted[metric] = {"m16": m16, "m32": m32, "speedup": m16 / m32}

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": properties.multi_processor_count,
        "dtype": str(dtype).removeprefix("torch."),
        "rows": args.rows,
        "warmup": args.warmup,
        "iters": args.iters,
        "abba_pairs": args.abba_pairs,
        "cases": cases,
        "weighted": weighted,
    }
    print(
        tabulate(
            table_rows,
            headers=("projection", "K->N", "M16/M32 p50 us", "p50 speedup", "M16/M32 mean us", "mean speedup"),
            tablefmt="grid",
        )
    )
    print(
        f"Weighted: p50 {weighted['p50_us']['speedup']:.3f}x, "
        f"mean {weighted['mean_us']['speedup']:.3f}x"
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
