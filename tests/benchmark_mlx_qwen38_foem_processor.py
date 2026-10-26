# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark production FOEM quantization against its Torch processor path."""

import argparse
import gc
import statistics
import time

import torch

from gptqmodel.quantization import foem as foem_module
from gptqmodel.quantization.config import FOEMConfig, GPTQConfig
from gptqmodel.quantization.foem import FOEM
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_foem_quantization import _adjudicate_banded_foem_row


def _measure_pair(torch_call, mlx_call, samples):
    torch_call()
    mlx_call()
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            start = time.perf_counter()
            (torch_call, mlx_call)[path]()
            timings[path].append((time.perf_counter() - start) * 1000)
    return statistics.median(timings[0]), statistics.median(timings[1])


def _codes(result, columns, group_size):
    weight, scales, zeros = result[:3]
    expanded_scales = scales.repeat_interleave(group_size, dim=1)[:, :columns]
    expanded_zeros = zeros.repeat_interleave(group_size, dim=1)[:, :columns]
    return torch.round(weight.float() / expanded_scales + expanded_zeros)


def _make_task(source, factor, config):
    layer = torch.nn.Linear(
        source.shape[1], source.shape[0], bias=False, dtype=source.dtype
    )
    layer.weight.data.copy_(source)
    task = FOEM(layer, config)
    task.quantizer.configure(perchannel=True, grid=100, maxshrink=0.8)
    task.nsamples = 4
    task.hessian_inverse = lambda _hessian: (factor, config.damp_percent)
    return task


def run(samples, projection):
    config = GPTQConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        mse=0,
        foem=FOEMConfig(alpha=0, beta=0.2),
    )
    rows_out = []
    original_minimum = foem_module._MLX_FOEM_MIN_ELEMENTS
    print(
        "projection | dtype | shape | Torch ms | MLX ms | speedup | "
        "code mismatches | direct ties | tie-propagated | unresolved | max Q drift | "
        "max Q drift away ties | max scale drift | max zero drift | loss drift",
        flush=True,
    )
    try:
        for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
            if projection and name != projection:
                continue
            generator = torch.Generator().manual_seed(380027 + index)
            source_fp32 = (
                torch.randn((rows, columns), generator=generator, dtype=torch.float32)
                * 0.025
            )
            factor = torch.eye(columns, dtype=torch.float32)
            factor.diagonal(offset=1).fill_(0.05)
            for label, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
                source = source_fp32.to(dtype)
                task = _make_task(source, factor, config)

                def torch_call(active_task=task, active_factor=factor):
                    foem_module._MLX_FOEM_MIN_ELEMENTS = (
                        active_task.rows * active_task.columns + 1
                    )
                    active_task.H = active_factor
                    return active_task.quantize(blocksize=128)

                def mlx_call(active_task=task, active_factor=factor):
                    foem_module._MLX_FOEM_MIN_ELEMENTS = 0
                    active_task.H = active_factor
                    return active_task.quantize(blocksize=128)

                expected = torch_call()
                actual = mlx_call()
                expected_codes = _codes(expected, columns, 128)
                actual_codes = _codes(actual, columns, 128)
                mismatches = torch.nonzero(actual_codes != expected_codes)
                direct_positions = []
                propagated_positions = []
                for row in torch.unique(mismatches[:, 0]).tolist():
                    direct, propagated = _adjudicate_banded_foem_row(
                        source[row].float().numpy(),
                        expected_codes[row].numpy(),
                        actual_codes[row].numpy(),
                    )
                    direct_positions.extend((row, column) for column in direct)
                    propagated_positions.extend((row, column) for column in propagated)
                accepted_positions = direct_positions + propagated_positions
                unresolved = len(mismatches) - len(accepted_positions)
                q_delta = (actual[0].float() - expected[0].float()).abs()
                max_q_drift = float(q_delta.max())
                if accepted_positions:
                    accepted_mask = torch.zeros_like(q_delta, dtype=torch.bool)
                    accepted_rows, accepted_columns = zip(*accepted_positions)
                    accepted_mask[list(accepted_rows), list(accepted_columns)] = True
                    max_q_away_ties = float(q_delta[~accepted_mask].max())
                else:
                    max_q_away_ties = max_q_drift
                scale_drift = float((actual[1] - expected[1]).abs().max())
                zero_drift = float((actual[2] - expected[2]).abs().max())
                loss_drift = abs(actual[5] - expected[5])
                torch_ms, mlx_ms = _measure_pair(torch_call, mlx_call, samples)
                row_out = (
                    name,
                    label,
                    rows,
                    columns,
                    torch_ms,
                    mlx_ms,
                    torch_ms / mlx_ms,
                    len(mismatches),
                    len(direct_positions),
                    len(propagated_positions),
                    unresolved,
                    max_q_drift,
                    max_q_away_ties,
                    scale_drift,
                    zero_drift,
                    loss_drift,
                )
                rows_out.append(row_out)
                print(
                    f"{name} | {label} | {rows}x{columns} | {torch_ms:.3f} | "
                    f"{mlx_ms:.3f} | {torch_ms / mlx_ms:.3f}x | "
                    f"{len(mismatches)} | {len(direct_positions)} | "
                    f"{len(propagated_positions)} | {unresolved} | "
                    f"{max_q_drift:.9g} | {max_q_away_ties:.9g} | "
                    f"{scale_drift:.9g} | {zero_drift:.9g} | {loss_drift:.9g}",
                    flush=True,
                )
                del source, task, expected, actual, expected_codes, actual_codes
                gc.collect()
            del source_fp32, factor
            gc.collect()
    finally:
        foem_module._MLX_FOEM_MIN_ELEMENTS = original_minimum

    print(f"median speedup: {statistics.median(row[6] for row in rows_out):.3f}x")
    print(f"minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"maximum speedup: {max(row[6] for row in rows_out):.3f}x")
    print(f"total code mismatches: {sum(row[7] for row in rows_out)}")
    print(f"direct natural ties: {sum(row[8] for row in rows_out)}")
    print(f"tie-propagated differences: {sum(row[9] for row in rows_out)}")
    print(f"unresolved mismatches: {sum(row[10] for row in rows_out)}")
    print(f"maximum Q drift: {max(row[11] for row in rows_out):.9g}")
    print(f"maximum Q drift away ties: {max(row[12] for row in rows_out):.9g}")
    print(f"maximum scale drift: {max(row[13] for row in rows_out):.9g}")
    print(f"maximum zero drift: {max(row[14] for row in rows_out):.9g}")
    print(f"maximum loss drift: {max(row[15] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--projection")
    arguments = parser.parse_args()
    run(arguments.samples, arguments.projection)
