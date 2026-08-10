#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native Metal GPTQ correction against eager MPS dispatch."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.utils.gptq_block_mps import gptq_block_mps, gptq_block_mps_supported


def serial_block(weights, hessian_inverse, scale, zero, maxq, group_size):
    quantized = torch.empty_like(weights)
    errors = torch.empty_like(weights)
    for column in range(weights.shape[1]):
        group = column // group_size
        column_scale = scale[:, group]
        column_zero = zero[:, group]
        weight = weights[:, column]
        q = column_scale * (
            torch.clamp(torch.round(weight / column_scale) + column_zero, 0, maxq)
            - column_zero
        )
        quantized[:, column] = q
        error = (weight - q) / hessian_inverse[column, column]
        errors[:, column] = error
        weights[:, column:] = torch.addr(
            weights[:, column:],
            error,
            hessian_inverse[column, column:],
            alpha=-1,
        )
    return quantized, errors


def timed(call, repeats):
    samples = []
    for _ in range(repeats):
        torch.mps.synchronize()
        started = time.perf_counter()
        call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    return statistics.median(samples)


def benchmark(rows, columns, group_size, repeats):
    torch.manual_seed(rows + columns + group_size)
    weights = torch.randn(rows, columns, device="mps")
    hessian_inverse = torch.triu(torch.randn(columns, columns, device="mps") * 0.03)
    hessian_inverse.diagonal().copy_(torch.rand(columns, device="mps") + 0.5)
    scale = torch.rand(rows, columns // group_size, device="mps") * 0.2 + 0.01
    zero = torch.randint(0, 16, scale.shape, device="mps").float()
    output = (torch.empty_like(weights), torch.empty_like(weights))

    serial_block(weights.clone(), hessian_inverse, scale, zero, 15, group_size)
    gptq_block_mps(
        weights.clone(), hessian_inverse, scale, zero, 15, group_size, out=output
    )
    torch.mps.synchronize()

    eager_ms = timed(
        lambda: serial_block(
            weights.clone(), hessian_inverse, scale, zero, 15, group_size
        ),
        repeats,
    )
    metal_ms = timed(
        lambda: gptq_block_mps(
            weights.clone(), hessian_inverse, scale, zero, 15, group_size, out=output
        ),
        repeats,
    )
    return eager_ms, metal_ms


def benchmark_lifecycle(rows, columns, group_size, repeats):
    torch.manual_seed(rows + columns + group_size + 1)
    weights = torch.randn(rows, columns, device="mps")
    hessian_inverse = torch.triu(torch.randn(columns, columns, device="mps") * 0.03)
    hessian_inverse.diagonal().copy_(torch.rand(columns, device="mps") + 0.5)
    importance = torch.rand(columns // group_size, group_size, device="mps")
    quantizer = Quantizer(
        QuantizeConfig(bits=4, group_size=group_size), name="benchmark"
    )
    quantizer.configure(perchannel=True)
    output = (torch.empty_like(weights), torch.empty_like(weights))
    fused_scale = torch.empty(rows, columns // group_size, device="mps")
    fused_zero = torch.empty_like(fused_scale)

    def eager():
        scale, zero = quantizer.find_params_batched(
            weights.reshape(rows, columns // group_size, group_size),
            weight=True,
            hessian=importance,
        )
        serial_block(weights.clone(), hessian_inverse, scale, zero, 15, group_size)

    def fused():
        gptq_block_mps(
            weights.clone(),
            hessian_inverse,
            fused_scale,
            fused_zero,
            15,
            group_size,
            find_params=True,
            scale_search="activation",
            importance=importance,
            candidate_count=80,
            out=output,
        )

    eager()
    fused()
    torch.mps.synchronize()
    return timed(eager, repeats), timed(fused, repeats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not gptq_block_mps_supported():
        raise SystemExit(
            "This benchmark requires macOS, MPS, and torch.mps.compile_shader."
        )

    print(f"Device: mps | Torch: {torch.__version__}")
    print("GPTQ correction with precomputed scales")
    print("+------+---------+-------+----------+----------+---------+")
    print("| Rows | Columns | Group | Eager ms | Metal ms | Speedup |")
    print("+------+---------+-------+----------+----------+---------+")
    for rows, columns, group_size in [
        (128, 128, 32),
        (512, 128, 128),
        (4096, 128, 128),
    ]:
        eager_ms, metal_ms = benchmark(rows, columns, group_size, args.repeats)
        print(
            f"| {rows:4d} | {columns:7d} | {group_size:5d} | {eager_ms:8.3f} | "
            f"{metal_ms:8.3f} | {eager_ms / metal_ms:6.2f}x |"
        )
    print("+------+---------+-------+----------+----------+---------+")

    print("\nActivation scale search + GPTQ correction lifecycle")
    print("+------+---------+-------+----------+----------+---------+")
    print("| Rows | Columns | Group | Eager ms | Metal ms | Speedup |")
    print("+------+---------+-------+----------+----------+---------+")
    for rows, columns, group_size in [
        (128, 128, 32),
        (512, 128, 128),
        (4096, 128, 128),
    ]:
        eager_ms, metal_ms = benchmark_lifecycle(
            rows, columns, group_size, args.repeats
        )
        print(
            f"| {rows:4d} | {columns:7d} | {group_size:5d} | {eager_ms:8.3f} | "
            f"{metal_ms:8.3f} | {eager_ms / metal_ms:6.2f}x |"
        )
    print("+------+---------+-------+----------+----------+---------+")


if __name__ == "__main__":
    main()
