# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark production RTN quantization against main's Torch path."""

import argparse
import gc
import statistics
import time

import torch

from gptqmodel.quantization import rtn as rtn_module
from gptqmodel.quantization.config import RTNConfig
from gptqmodel.quantization.rtn import RTN
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _measure_pair(torch_call, mlx_call, samples):
    torch_call()
    mlx_call()
    timings = [[], []]
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for path in order:
            start = time.perf_counter()
            (torch_call, mlx_call)[path]()
            timings[path].append((time.perf_counter() - start) * 1000)
    return statistics.median(timings[0]), statistics.median(timings[1])


def _codes(result, columns, group_size):
    weight, scales, zeros = result[:3]
    expanded_scales = scales.repeat_interleave(group_size, dim=1)[:, :columns]
    expanded_zeros = zeros.repeat_interleave(group_size, dim=1)[:, :columns]
    return torch.round(weight.float() / expanded_scales + expanded_zeros)


def run(samples):
    rows_out = []
    print(
        "projection | dtype | shape | Torch ms | MLX ms | speedup | "
        "code mismatches | max weight drift | max scale drift | max zero drift",
        flush=True,
    )
    original_minimum = rtn_module._MLX_RTN_MIN_ELEMENTS
    try:
        for name, rows, columns in QWEN38_27B_PROJECTIONS:
            generator = torch.Generator().manual_seed(380027 + rows + columns)
            source = torch.randn((rows, columns), generator=generator) * 0.025
            for label, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
                layer = torch.nn.Linear(columns, rows, bias=False, dtype=dtype)
                layer.weight.data.copy_(source)
                task = RTN(
                    layer,
                    RTNConfig(bits=4, group_size=128, sym=True),
                )
                element_count = layer.weight.numel()

                def torch_call(
                    active_task=task,
                    active_element_count=element_count,
                ):
                    rtn_module._MLX_RTN_MIN_ELEMENTS = active_element_count + 1
                    return active_task.quantize()

                def mlx_call(active_task=task):
                    rtn_module._MLX_RTN_MIN_ELEMENTS = 0
                    return active_task.quantize()

                expected = torch_call()
                actual = mlx_call()
                mismatch_count = int(
                    torch.count_nonzero(
                        _codes(actual, columns, 128) != _codes(expected, columns, 128)
                    )
                )
                drifts = [
                    float((observed.float() - reference.float()).abs().max())
                    for observed, reference in zip(actual[:3], expected[:3])
                ]
                torch_ms, mlx_ms = _measure_pair(torch_call, mlx_call, samples)
                row = (
                    name,
                    label,
                    rows,
                    columns,
                    torch_ms,
                    mlx_ms,
                    torch_ms / mlx_ms,
                    mismatch_count,
                    *drifts,
                )
                rows_out.append(row)
                print(
                    f"{name} | {label} | {rows}x{columns} | {torch_ms:.3f} | "
                    f"{mlx_ms:.3f} | {torch_ms / mlx_ms:.3f}x | "
                    f"{mismatch_count} | {drifts[0]:.9g} | {drifts[1]:.9g} | "
                    f"{drifts[2]:.9g}",
                    flush=True,
                )
                del layer, task, expected, actual
                gc.collect()
            del source
            gc.collect()
    finally:
        rtn_module._MLX_RTN_MIN_ELEMENTS = original_minimum

    print(f"median speedup: {statistics.median(row[6] for row in rows_out):.3f}x")
    print(f"minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"maximum speedup: {max(row[6] for row in rows_out):.3f}x")
    print(f"total code mismatches: {sum(row[7] for row in rows_out)}")
    print(f"maximum weight drift: {max(row[8] for row in rows_out):.9g}")
    print(f"maximum scale drift: {max(row[9] for row in rows_out):.9g}")
    print(f"maximum zero drift: {max(row[10] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    run(parser.parse_args().samples)
