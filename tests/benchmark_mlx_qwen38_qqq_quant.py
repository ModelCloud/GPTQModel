# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark QQQ processor-stage MLX quantization against its Torch oracle."""

import argparse
import gc
import statistics
import time
from functools import partial

import numpy as np
import torch

from gptqmodel.quantization.qqq import (
    Quantizer,
    _qqq_quantize_weight_mlx_to_torch,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_qqq_quantization import (
    _boundary_margin,
    _codes,
    _torch_banded_oracle,
)


def _measure_pair(torch_call, mlx_call, samples):
    for _ in range(10):
        torch_call()
        mlx_call()
    timings = [[], []]
    for sample in range(samples):
        order = (
            ((0, torch_call), (1, mlx_call))
            if sample % 2 == 0
            else ((1, mlx_call), (0, torch_call))
        )
        for path, function in order:
            start = time.perf_counter()
            function()
            timings[path].append(1000 * (time.perf_counter() - start))
    return statistics.median(timings[0]), statistics.median(timings[1])


def _mlx_processor_stage(weight, original_weight, factor, group_size):
    result = list(
        _qqq_quantize_weight_mlx_to_torch(
            weight,
            factor,
            group_size=group_size,
        )
    )
    if group_size != -1:
        quantizer = Quantizer()
        quantizer.configure(
            bits=8,
            perchannel=True,
            groupsize=-1,
            sym=True,
            mse=False,
        )
        quantizer.find_params(original_weight.clone(), weight=True)
        result[3] = quantizer.scale
    return tuple(result)


def main(samples, projection, dtype_filter, group_filter):
    rows_out = []
    print(
        "projection | dtype | mode | shape | changed codes | direct ties | "
        "tie descendants | max direct margin | max weight drift | max drift away "
        "from ties | max scale drift | max extra-scale drift | loss drift | "
        "Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        if projection and name != projection:
            continue
        rng = np.random.default_rng(744 + rows + columns)
        source = torch.from_numpy(
            rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
        )
        factor = torch.eye(columns, dtype=torch.float32)
        factor.diagonal(offset=1).fill_(0.05)
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            if dtype_filter and dtype_name.lower() != dtype_filter.lower():
                continue
            weight = source.to(dtype).float().contiguous()
            original_weight = source.to(dtype).contiguous()
            resident = weight.numpy()
            for group_size in (-1, 128):
                if group_filter is not None and group_size != group_filter:
                    continue
                torch_call = partial(
                    _torch_banded_oracle,
                    resident,
                    group_size,
                    return_loss=True,
                )
                mlx_call = partial(
                    _mlx_processor_stage,
                    weight,
                    original_weight,
                    factor,
                    group_size,
                )
                expected = torch_call()
                actual = tuple(
                    None if value is None else value.cpu().numpy()
                    for value in mlx_call()
                )
                changed = _codes(actual[:4], group_size) != _codes(
                    expected[:4], group_size
                )
                coordinates = np.argwhere(changed)
                margins = [
                    float(
                        _boundary_margin(
                            resident[row],
                            expected[0][row],
                            expected[1][
                                row, 0 if group_size == -1 else column // 128
                            ],
                            column,
                        )
                    )
                    for row, column in coordinates
                ]
                direct_margins = [margin for margin in margins if margin < 0.0002]
                drift = np.abs(actual[0] - expected[0])
                extra_drift = (
                    0.0
                    if expected[3] is None
                    else float(np.max(np.abs(actual[3] - expected[3])))
                )
                torch_ms, mlx_ms = _measure_pair(torch_call, mlx_call, samples)
                result = (
                    name,
                    dtype_name,
                    group_size,
                    rows,
                    columns,
                    len(coordinates),
                    len(direct_margins),
                    len(margins) - len(direct_margins),
                    max(direct_margins, default=0),
                    float(np.max(drift)),
                    float(np.max(drift[~changed])),
                    float(np.max(np.abs(actual[1] - expected[1]))),
                    extra_drift,
                    abs(float(actual[4]) - float(expected[4])),
                    torch_ms,
                    mlx_ms,
                    torch_ms / mlx_ms,
                )
                rows_out.append(result)
                print(
                    " | ".join(
                        (
                            name,
                            dtype_name,
                            str(group_size),
                            f"{rows}x{columns}",
                            str(result[5]),
                            str(result[6]),
                            str(result[7]),
                            f"{result[8]:.9g}",
                            f"{result[9]:.9g}",
                            f"{result[10]:.9g}",
                            f"{result[11]:.9g}",
                            f"{result[12]:.9g}",
                            f"{result[13]:.9g}",
                            f"{result[14]:.3f}",
                            f"{result[15]:.3f}",
                            f"{result[16]:.3f}x",
                        )
                    ),
                    flush=True,
                )
                del expected, actual, torch_call, mlx_call
                gc.collect()
            del weight, original_weight, resident
            gc.collect()
        del source, factor
        gc.collect()

    print(
        "\n| Projection | Dtype | Group | Shape | Changed codes | Direct ties | "
        "Tie descendants | Max direct margin | Max weight drift | Max drift "
        "away | Max scale drift | Max extra-scale drift | Loss drift | Torch "
        "ms | MLX ms | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]}x{row[4]} | "
            f"{row[5]} | {row[6]} | {row[7]} | {row[8]:.9g} | "
            f"{row[9]:.9g} | {row[10]:.9g} | {row[11]:.9g} | "
            f"{row[12]:.9g} | {row[13]:.9g} | {row[14]:.3f} | "
            f"{row[15]:.3f} | {row[16]:.3f}x |"
        )
    print(f"\nMedian speedup: {statistics.median(row[16] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[16] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[16] for row in rows_out):.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    parser.add_argument("--projection")
    parser.add_argument("--dtype", choices=("fp16", "bf16"))
    parser.add_argument("--group-size", type=int, choices=(-1, 128))
    args = parser.parse_args()
    main(args.samples, args.projection, args.dtype, args.group_size)
