#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure the GGUF Torch fused-vs-dense crossover used by the static gates.

``GGUFTorchLinear`` picks between ``_forward_fused_k`` and
``_forward_dequant_matmul`` with the static ``KernelConfig.gguf_fused_*`` shape
gates only; it never times the two paths at run time (#4590). This script is
the offline tool that produced the defaults: it interleaves the two paths on
one module and reports both timings so the row ceiling and matrix-size floor
can be re-derived after an arithmetic change.

The shipped CPU and CUDA defaults were derived with this script on the target
hardware; re-run it after any arithmetic change to the two paths.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


@dataclass(frozen=True)
class BenchCase:
    name: str
    bits: str
    in_features: int
    out_features: int
    rows: int
    group_size: int = -1


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep, fmt(headers), sep]
    for row in rows:
        out.append(fmt(row))
    out.append(sep)
    return "\n".join(out)


def _build_module(case: BenchCase, *, dtype: torch.dtype, device: str) -> GGUFTorchLinear:
    linear = nn.Linear(case.in_features, case.out_features, bias=False, dtype=dtype).cpu().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)

    module = GGUFTorchLinear(
        bits=case.bits,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=False,
        register_buffers=False,
    )
    module.pack_original(linear, scales=torch.empty(0), zeros=torch.empty(0), g_idx=None)
    module.post_init()
    # Every case must be gate-eligible, otherwise the measurement compares a
    # static gate with itself.
    module.gguf_fused_cuda_max_rows = max(case.rows, 1)
    module.gguf_fused_cuda_min_matrix_elements = 0
    module.gguf_fused_cpu_max_rows = max(case.rows, 1)
    module.gguf_fused_cpu_min_matrix_elements = 0
    return module.to(device).eval()


def _bench_once(fn, *, sync_cuda: bool) -> float:
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if sync_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _measure(fn, *, sync_cuda: bool, trials: int) -> list[float]:
    return [_bench_once(fn, sync_cuda=sync_cuda) for _ in range(trials)]


def _run_case(
    case: BenchCase,
    *,
    dtype: torch.dtype,
    device: str,
    trials: int,
    warmup: int,
) -> None:
    sync_cuda = device == "cuda"
    x = torch.randn(case.rows, case.in_features, device=device, dtype=dtype)
    module = _build_module(case, dtype=dtype, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            module._forward_dequant_matmul(x)
            module._forward_fused_k(x)

        # Alternate the two paths in one loop: a load spike that hits both
        # samples equally cancels out instead of biasing the comparison.
        fused_trials: list[float] = []
        dense_trials: list[float] = []
        for _ in range(trials):
            fused_trials.append(_bench_once(lambda: module._forward_fused_k(x), sync_cuda=sync_cuda))
            dense_trials.append(_bench_once(lambda: module._forward_dequant_matmul(x), sync_cuda=sync_cuda))

        fused_out = module._forward_fused_k(x)
        dense_out = module._forward_dequant_matmul(x)
    diff = (fused_out.to(torch.float32) - dense_out.to(torch.float32)).abs()

    fused_median = statistics.median(fused_trials)
    dense_median = statistics.median(dense_trials)
    print()
    print(
        f"CASE {case.name} device={device} bits={case.bits} rows={case.rows} "
        f"shape={case.out_features}x{case.in_features} dtype={str(dtype).removeprefix('torch.')}"
    )
    print(
        _ascii_table(
            ["path", "median_ms", "min_ms", "max_ms", "speedup_vs_fused"],
            [
                ["fused", f"{fused_median:.3f}", f"{min(fused_trials):.3f}", f"{max(fused_trials):.3f}", "-"],
                [
                    "dense",
                    f"{dense_median:.3f}",
                    f"{min(dense_trials):.3f}",
                    f"{max(dense_trials):.3f}",
                    f"{fused_median / dense_median:.2f}x",
                ],
            ],
        )
    )
    print(f"path_drift: mae={diff.mean().item():.6f} max_abs={diff.max().item():.6f}")
    print(f"selected_path: {'fused' if module._should_use_fused_k_forward(x) else 'dense'}")


def _parse_case(spec: str) -> BenchCase:
    parts = spec.split(":")
    if len(parts) not in (5, 6):
        raise ValueError(
            f"Invalid case `{spec}`. Expected name:bits:in_features:out_features:rows[:group_size]."
        )
    name, bits, in_features, out_features, rows, *rest = parts
    return BenchCase(
        name=name,
        bits=bits,
        in_features=int(in_features),
        out_features=int(out_features),
        rows=int(rows),
        group_size=int(rest[0]) if rest else -1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the GGUF Torch fused-vs-dense crossover behind the static dispatch gates."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=[],
        help="Benchmark case as name:bits:in_features:out_features:rows[:group_size].",
    )
    parser.add_argument("--trials", type=int, default=9, help="Measured trials per path.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup forwards per path.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16", help="Benchmark dtype.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "both"),
        default="auto",
        help="Benchmark device selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if args.device == "auto":
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    elif args.device == "both":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    else:
        devices = [args.device]

    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")

    cases = (
        [_parse_case(spec) for spec in args.cases]
        if args.cases
        else [
            BenchCase("attn_q4_k_m_r1", "q4_k_m", 2048, 2048, 1),
            BenchCase("attn_q5_k_m_r1", "q5_k_m", 2048, 2048, 1),
            BenchCase("attn_q6_k_r1", "q6_k", 2048, 2048, 1),
            BenchCase("mlp_q4_k_m_r8", "q4_k_m", 2048, 8192, 8),
            BenchCase("mlp_q5_k_m_r8", "q5_k_m", 2048, 8192, 8),
            BenchCase("mlp_q6_k_r8", "q6_k", 2048, 8192, 8),
        ]
    )

    print(
        f"devices={','.join(devices)} dtype={str(dtype).removeprefix('torch.')} "
        f"trials={args.trials} warmup={args.warmup}"
    )
    for device in devices:
        for case in cases:
            _run_case(case, dtype=dtype, device=device, trials=args.trials, warmup=args.warmup)


if __name__ == "__main__":
    main()
