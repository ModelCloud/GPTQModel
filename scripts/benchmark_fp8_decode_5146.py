#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the lean FP8 W8A16 decode path (#5146).

Compares, for Llama-3.1-8B projection shapes:

* a dense FP16 ``nn.Linear`` (the format the FP8 model replaces),
* the pre-fix ``TorchFP8Linear`` forward (full FP32 decode + transpose copy),
* the shipped lean forward (cast codes, transposed-view GEMM, FP32 epilogue).

The pre-fix column is produced by disabling the lean branch on an otherwise
identical module, so both FP8 columns share the same packed payload. ``--check``
exits non-zero when the lean forward is more than 1.8x the issue's cast + GEMM +
epilogue emulation, which is the regression this issue fixes.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


DEFAULT_SHAPES = (
    ("q_proj", 4096, 4096),
    ("gate_proj", 4096, 14336),
    ("down_proj", 14336, 4096),
)
DEFAULT_ROWS = (1, 2048)


def _dense(in_features: int, out_features: int, device: torch.device) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    ).eval()


def _fp8(
    in_features: int,
    out_features: int,
    device: torch.device,
    *,
    pre_fix: bool,
) -> TorchFP8Linear:
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32).eval()
    kernel = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format="float8_e4m3fn",
        weight_scale_method="row",
        register_buffers=False,
    )
    kernel.pack_original(linear, None, None)
    kernel = kernel.to(device=device).eval()
    if pre_fix:
        kernel._lean_epilogue_eligible = lambda _x: False
    return kernel


def _timed_us(fn, *, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the lean forward is >1.8x the cast+GEMM+epilogue emulation",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this benchmark needs a CUDA device")
    print(f"device: {torch.cuda.get_device_name(device)}  torch {torch.__version__}")
    print(
        f"{'rows':>6} {'projection':>11} {'dense_us':>9} {'pre_fix_us':>11} "
        f"{'lean_us':>9} {'lean/dense':>10} {'pre_fix/lean':>12}"
    )

    failures: list[str] = []
    for rows in args.rows:
        for name, in_features, out_features in DEFAULT_SHAPES:
            x = torch.randn(rows, in_features, device=device, dtype=torch.float16)
            dense = _dense(in_features, out_features, device)
            pre_fix = _fp8(in_features, out_features, device, pre_fix=True)
            lean = _fp8(in_features, out_features, device, pre_fix=False)
            with torch.inference_mode():
                dense_us = _timed_us(lambda: dense(x))
                pre_fix_us = _timed_us(lambda: pre_fix(x))
                lean_us = _timed_us(lambda: lean(x))
                scale = lean.weight_scale_inv.view(-1).to(torch.float32)

                def emulation():
                    # Per-call cast, as the shipped lean path does it.
                    codes = lean.weight.to(torch.float16)
                    return torch.mm(x, codes.t(), out_dtype=torch.float32) / scale

                emulation_us = _timed_us(emulation)
            print(
                f"{rows:>6} {name:>11} {dense_us:>9.1f} {pre_fix_us:>11.1f} "
                f"{lean_us:>9.1f} {lean_us / dense_us:>10.2f} "
                f"{pre_fix_us / lean_us:>12.2f}"
            )
            if args.check and lean_us > 1.8 * emulation_us:
                failures.append(
                    f"rows={rows} {name}: lean {lean_us:.1f}us is "
                    f"{lean_us / emulation_us:.2f}x the emulation {emulation_us:.1f}us"
                )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if args.check:
        print("\nOK: lean forward stays within 1.8x of the cast+GEMM+epilogue emulation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
