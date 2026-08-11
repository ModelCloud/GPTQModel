#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark asynchronous MPS Hessian accumulation through the GPTQ lifecycle."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import gptqmodel.quantization.gptq as gptq_module
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _median_ms(call, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        torch.mps.synchronize()
        started = time.perf_counter()
        result = call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    return statistics.median(samples), result


def _case(tokens: int, columns: int, batches: int, rows: int, repeats: int):
    torch.manual_seed(tokens + columns + batches + rows)
    device = torch.device("mps")
    weight = torch.randn(rows, columns, device=device, dtype=torch.float16)
    calibrations = [
        torch.randn(tokens, columns, device=device, dtype=torch.float16)
        for _ in range(batches)
    ]
    qcfg = QuantizeConfig(
        bits=4,
        group_size=min(128, columns),
        sym=False,
        desc_act=False,
        act_group_aware=False,
        damp_percent=0.05,
        offload_to_disk=False,
    )

    def make_task():
        layer = nn.Linear(columns, rows, bias=False, device=device, dtype=torch.float16)
        layer.weight.data.copy_(weight)
        task = GPTQ(layer, qcfg)
        task.quantizer.configure(perchannel=True)
        return task

    def accumulate(async_enabled: bool):
        task = make_task()
        gptq_module._USE_GPTQ_MPS_ASYNC_HESSIAN = async_enabled
        for calibration in calibrations:
            task.add_batch(calibration, None)
        torch.mps.synchronize()
        return next(iter(task._device_hessian_partials.values()))

    def lifecycle(async_enabled: bool):
        task = make_task()
        gptq_module._USE_GPTQ_MPS_ASYNC_HESSIAN = async_enabled
        for calibration in calibrations:
            task.add_batch(calibration, None)
        return task.quantize(blocksize=128)

    # Complete allocator/JIT warmup for both paths before collecting samples.
    accumulate(False)
    accumulate(True)
    reference = lifecycle(False)
    candidate = lifecycle(True)
    torch.mps.synchronize()
    for expected, actual in zip(reference[:4], candidate[:4]):
        if not torch.equal(expected, actual):
            raise RuntimeError("asynchronous accumulation changed GPTQ output")

    sync_accum_ms, _ = _median_ms(lambda: accumulate(False), repeats)
    async_accum_ms, _ = _median_ms(lambda: accumulate(True), repeats)
    sync_lifecycle_ms, _ = _median_ms(lambda: lifecycle(False), repeats)
    async_lifecycle_ms, _ = _median_ms(lambda: lifecycle(True), repeats)
    return sync_accum_ms, async_accum_ms, sync_lifecycle_ms, async_lifecycle_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("This benchmark requires an available MPS device.")

    print(f"Device: mps | Torch: {torch.__version__}")
    print(
        "Method: GPTQ 4-bit asymmetric, group=columns<=128, FP16 calibration/weights, FP32 Hessian"
    )
    print(
        "+--------+---------+---------+------------+------------+---------+-------------+-------------+---------+"
    )
    print(
        "| Tokens | Columns | Batches | Sync accum | Async accum| Speedup | Sync full   | Async full  | Speedup |"
    )
    print(
        "+--------+---------+---------+------------+------------+---------+-------------+-------------+---------+"
    )
    try:
        for tokens, columns, batches, rows in (
            (4, 128, 16, 128),
            (32, 128, 16, 128),
            (128, 128, 16, 128),
            (128, 512, 8, 128),
            (128, 1024, 4, 128),
        ):
            sync_accum, async_accum, sync_full, async_full = _case(
                tokens, columns, batches, rows, args.repeats
            )
            print(
                f"| {tokens:6d} | {columns:7d} | {batches:7d} | {sync_accum:8.3f} ms | "
                f"{async_accum:8.3f} ms | {sync_accum / async_accum:6.2f}x | {sync_full:9.3f} ms | "
                f"{async_full:9.3f} ms | {sync_full / async_full:6.2f}x |"
            )
    finally:
        gptq_module._USE_GPTQ_MPS_ASYNC_HESSIAN = True
    print(
        "+--------+---------+---------+------------+------------+---------+-------------+-------------+---------+"
    )


if __name__ == "__main__":
    main()
