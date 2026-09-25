# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native MLX FOEM against the existing Torch group-update math."""

import argparse
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_foem import foem_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _torch_foem_group_update(weight, factor, group_size=128, beta=0.2, *, record_ties=False):
    """Run main's alpha-zero, blocksize=group_size FOEM update on CPU."""
    rows, columns = weight.shape
    raw = weight.float()
    working = raw.clone()
    output = torch.empty_like(working)
    scales = torch.empty((rows, columns // group_size), dtype=torch.float32)
    zeros = torch.empty_like(scales)
    near_tie = torch.empty_like(working, dtype=torch.bool) if record_ties else None
    for start in range(0, columns, group_size):
        end = start + group_size
        group = working[:, start:end].clone()
        minimum = torch.minimum(group.amin(1), torch.zeros(rows))
        maximum = torch.maximum(group.amax(1), torch.zeros(rows))
        maximum = torch.maximum(minimum.abs(), maximum)
        minimum = torch.where(minimum < 0, -maximum, minimum)
        empty = (minimum == 0) & (maximum == 0)
        minimum = torch.where(empty, -1, minimum)
        maximum = torch.where(empty, 1, maximum)
        scale = (maximum - minimum) / 15
        zero = torch.full_like(scale, 8)
        scales[:, start // group_size] = scale
        zeros[:, start // group_size] = zero
        errors = torch.empty_like(group)
        for offset in range(group_size):
            value = group[:, offset].clone()
            scaled = value / scale
            if record_ties:
                near_tie[:, start + offset] = (
                    (scaled - torch.floor(scaled) - 0.5).abs() <= 2e-6
                )
            code = (torch.round(scaled) + zero).clamp(0, 15)
            q = scale * (code - zero)
            error = ((value - q) - (value - raw[:, start + offset]) * beta) / factor[
                start + offset, start + offset
            ]
            output[:, start + offset] = q
            errors[:, offset] = error
            group[:, offset:] -= error[:, None] * factor[
                start + offset, start + offset:end
            ]
            if offset + 1 < group_size:
                group[:, offset + 1] -= beta * (
                    group[:, offset + 1] - raw[:, start + offset + 1]
                )
        if end < columns:
            working[:, end:] -= errors @ factor[start:end, end:]
    return output, scales, zeros, near_tie


def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), result


def _accuracy_metrics(mlx_result, torch_result, group_size=128, show_mismatches=False):
    torch_output, torch_scales_tensor, torch_zeros_tensor, near_tie_tensor = torch_result
    torch_scales = torch_scales_tensor.numpy()
    torch_zeros = torch_zeros_tensor.numpy()
    mlx_output, mlx_scales, mlx_zeros = (np.asarray(value) for value in mlx_result)
    rows, _ = mlx_output.shape
    expected = torch_output.numpy()
    near_tie = near_tie_tensor.numpy()
    difference = mlx_output - expected
    max_q_abs = float(np.max(np.abs(difference)))
    rel_q_l2 = float(np.linalg.norm(difference) / np.linalg.norm(expected))
    max_scale_abs = float(np.max(np.abs(mlx_scales - torch_scales)))
    zero_mismatches = int(np.count_nonzero(mlx_zeros != torch_zeros))
    code_mismatches = 0
    tie_mismatches = 0
    non_tie_max_q_abs = 0.0
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        mlx_codes = np.rint(
            mlx_output[start:stop].reshape(stop - start, -1, group_size)
            / mlx_scales[start:stop, :, None] + mlx_zeros[start:stop, :, None]
        ).astype(np.uint8)
        torch_codes = np.rint(
            expected[start:stop].reshape(stop - start, -1, group_size)
            / torch_scales[start:stop, :, None] + torch_zeros[start:stop, :, None]
        ).astype(np.uint8)
        mismatch = mlx_codes != torch_codes
        tie_slice = near_tie[start:stop].reshape(stop - start, -1, group_size)
        code_mismatches += int(np.count_nonzero(mismatch))
        tie_mismatches += int(np.count_nonzero(mismatch & tie_slice))
        non_tie_max_q_abs = max(
            non_tie_max_q_abs,
            float(np.max(np.abs(difference[start:stop][~near_tie[start:stop]]), initial=0)),
        )
        if show_mismatches:
            for row, group, offset in np.argwhere(mismatch):
                column = group * group_size + offset
                print(
                    f"mismatch row={start + row} column={column} "
                    f"mlx_code={mlx_codes[row, group, offset]} "
                    f"torch_code={torch_codes[row, group, offset]} "
                    f"near_tie={tie_slice[row, group, offset]}", file=sys.stderr,
                )
        if np.any(np.abs(mlx_codes.astype(np.int16) - torch_codes.astype(np.int16))[mismatch & tie_slice] > 1):
            raise AssertionError("FOEM tie codes differ by more than one level")
    if max_scale_abs > 1e-6 or zero_mismatches or non_tie_max_q_abs > 1e-6 or tie_mismatches != code_mismatches:
        raise AssertionError(
            f"FOEM accuracy limit exceeded: {max_q_abs=}, {non_tie_max_q_abs=}, "
            f"{max_scale_abs=}, {code_mismatches=}, {tie_mismatches=}, {zero_mismatches=}"
        )
    return max_q_abs, rel_q_l2, max_scale_abs, code_mismatches, tie_mismatches, non_tie_max_q_abs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--projection", help="Only benchmark one projection role")
    parser.add_argument("--show-mismatches", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(
        "projection,out_features,in_features,mlx_ms,torch_ms,speedup,"
        "max_q_abs,rel_q_l2,max_scale_abs,code_mismatches,tie_mismatches,non_tie_max_q_abs", flush=True,
    )
    for index, (name, out_features, in_features) in enumerate(QWEN38_27B_PROJECTIONS):
        if args.projection and args.projection != name:
            continue
        mx.random.seed(9000 + index)
        weight = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
        factor = mx.eye(in_features) + 0.05 * mx.eye(in_features, k=1)
        mx.eval(weight, factor)
        torch_weight = torch.from_numpy(np.asarray(weight.astype(mx.float32)))
        torch_factor = torch.eye(in_features)
        torch_factor.diagonal(offset=1).fill_(0.05)

        def run_mlx():
            result = foem_quantize_weight_mlx(weight, factor, group_size=128, beta=0.2)
            mx.eval(*result)
            return result

        def run_torch():
            return _torch_foem_group_update(torch_weight, torch_factor)

        run_mlx()
        run_torch()
        mlx_ms, mlx_result = _median_ms(run_mlx, args.repeats)
        torch_ms, _ = _median_ms(run_torch, args.repeats)
        torch_result = _torch_foem_group_update(torch_weight, torch_factor, record_ties=True)
        max_q_abs, rel_q_l2, max_scale_abs, code_mismatches, tie_mismatches, non_tie_max_q_abs = _accuracy_metrics(
            mlx_result, torch_result, show_mismatches=args.show_mismatches,
        )
        print(
            f"{name},{out_features},{in_features},"
            f"{mlx_ms:.3f},{torch_ms:.3f},{torch_ms / mlx_ms:.1f},"
            f"{max_q_abs:.3e},{rel_q_l2:.3e},{max_scale_abs:.3e},"
            f"{code_mismatches},{tie_mismatches},{non_tie_max_q_abs:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
