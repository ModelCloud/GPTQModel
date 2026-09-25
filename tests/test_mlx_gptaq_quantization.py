# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx

"""Independent Torch-oracle checks for the MLX GPTAQ correction path."""

import sys
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_gptaq import (
    gptaq_correction_mlx,
    gptaq_quantize_weight_mlx,
)


def _torch_params(group, bits, sym):
    minimum = torch.minimum(group.amin(dim=1), torch.zeros(group.shape[0]))
    maximum = torch.maximum(group.amax(dim=1), torch.zeros(group.shape[0]))
    if sym:
        maximum = torch.maximum(minimum.abs(), maximum)
        minimum = torch.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = torch.where(empty, -1, minimum)
    maximum = torch.where(empty, 1, maximum)
    scale = (maximum - minimum) / (2**bits - 1)
    zero = (
        torch.full_like(scale, 2 ** (bits - 1))
        if sym
        else torch.round(-minimum / scale)
    )
    return scale, zero


def _torch_oracle(weight, inverse_hessian, correction, bits, group_size, sym):
    raw = torch.from_numpy(weight.copy())
    factor = torch.from_numpy(inverse_hessian.copy())
    projection = torch.from_numpy(correction.copy())
    columns = raw.shape[1]
    remaining = raw.clone()
    quantized, scales, zeros = [], [], []
    for start in range(0, columns, group_size):
        end = start + group_size
        group = remaining[:, :group_size].clone()
        scale, zero = _torch_params(group, bits, sym)
        output = torch.empty_like(group)
        errors = torch.empty_like(group)
        for offset in range(group_size):
            w = group[:, offset].clone()
            d = factor[start + offset, start + offset]
            code = (torch.round(w / scale) + zero).clamp(0, 2**bits - 1)
            q = scale * (code - zero)
            output[:, offset] = q
            error = (w - q) / d
            errors[:, offset] = error
            group[:, offset:] -= (
                error[:, None] * factor[start + offset, start + offset : end]
                - w[:, None] * projection[start + offset, start + offset : end]
            )
        quantized.append(output)
        scales.append(scale[:, None])
        zeros.append(zero[:, None])
        if end < columns:
            remaining = (
                remaining[:, group_size:]
                - errors @ factor[start:end, end:]
                + group @ projection[start:end, end:]
            )
    return tuple(
        torch.cat(values, dim=1).numpy() for values in (quantized, scales, zeros)
    )


@pytest.mark.parametrize("width", [32, 128])
@pytest.mark.parametrize("alpha", [-0.5, 0.0, 0.25, 1.5])
def test_gptaq_correction_matches_torch(width, alpha):
    rng = np.random.default_rng(113 + width)
    delta = rng.normal(0, 0.003, (width, width)).astype(np.float32)
    factor = np.eye(width, dtype=np.float32) + np.triu(
        rng.normal(0, 0.002, (width, width)).astype(np.float32),
        1,
    )
    expected = (
        alpha
        * torch.triu(torch.from_numpy(delta) @ torch.from_numpy(factor).T, diagonal=1)
        @ torch.from_numpy(factor)
    )
    actual = gptaq_correction_mlx(mx.array(delta), mx.array(factor), alpha=alpha)
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("sym", [False, True])
def test_gptaq_group_updates_match_torch(bits, sym):
    rng = np.random.default_rng(518 + bits)
    source = rng.normal(0, 0.2, (7, 64)).astype(np.float32)
    source[0] = 0
    factor = np.eye(64, dtype=np.float32) + np.triu(
        rng.normal(0, 0.002, (64, 64)).astype(np.float32),
        1,
    )
    projection = np.triu(rng.normal(0, 0.001, (64, 64)).astype(np.float32), 1)
    expected = _torch_oracle(source, factor, projection, bits, 32, sym)
    actual = gptaq_quantize_weight_mlx(
        mx.array(source),
        mx.array(factor),
        mx.array(projection),
        bits=bits,
        group_size=32,
        sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("sym", [False, True])
def test_gptaq_code_boundaries_and_input_casts(sym):
    source = torch.zeros(3, 64, dtype=torch.float32)
    source[:, 0] = -1
    source[:, 1] = 1
    scale = torch.tensor(2 / 15, dtype=torch.float32)
    midpoint = torch.arange(-7, 8, dtype=torch.float32) * scale + scale / 2
    source[:, 2:17] = torch.nextafter(
        midpoint, torch.full_like(midpoint, -float("inf"))
    )
    source[:, 17:32] = midpoint
    source[:, 32:47] = torch.nextafter(
        midpoint, torch.full_like(midpoint, float("inf"))
    )
    factor = np.eye(64, dtype=np.float32)
    projection = np.zeros((64, 64), dtype=np.float32)
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        weight = mx.array(source.numpy()).astype(dtype)
        resident = np.asarray(weight.astype(mx.float32))
        expected = _torch_oracle(resident, factor, projection, 4, 64, sym)
        actual = gptaq_quantize_weight_mlx(
            weight, mx.array(factor), mx.array(projection), group_size=64, sym=sym
        )
        for index in range(3):
            np.testing.assert_allclose(
                np.asarray(actual[index]), expected[index], atol=1e-6, rtol=1e-6
            )


def _torch_banded_oracle(source, group_size=128):
    """Independent Torch update for unit diagonal and one off-diagonal."""
    weight = torch.from_numpy(source.copy())
    rows, columns = weight.shape
    output = torch.empty_like(weight)
    scales = torch.empty((rows, columns // group_size), dtype=torch.float32)
    zeros = torch.empty_like(scales)
    for start in range(0, columns, group_size):
        group_index = start // group_size
        scale, zero = _torch_params(weight[:, start : start + group_size], 4, True)
        scales[:, group_index] = scale
        zeros[:, group_index] = zero
        for column in range(start, start + group_size):
            value = weight[:, column].clone()
            code = (torch.round(value / scale) + zero).clamp(0, 15)
            q = scale * (code - zero)
            output[:, column] = q
            error = value - q
            if column + 1 < columns:
                # GPTAQ's cross-group correction reads W1 after updating
                # the last column; within a group it reads the earlier w.
                projected = q if (column + 1) % group_size == 0 else value
                weight[:, column + 1] -= error * 0.05 - projected * 0.002
            weight[:, column] = q
    return output.numpy(), scales.numpy(), zeros.numpy()


def _codes(result, group_size):
    quantized, scales, zeros = result
    expanded_scale = np.repeat(scales, group_size, axis=1)
    expanded_zero = np.repeat(zeros, group_size, axis=1)
    return (
        np.rint(quantized / expanded_scale + expanded_zero).clip(0, 15).astype(np.uint8)
    )


def _exact_banded_tie_margin(source_row, target, group_size=128):
    """Recompute one disputed GPTAQ code from stored inputs in 80-digit math."""
    with localcontext() as context:
        context.prec = 80
        working = [Decimal.from_float(float(value)) for value in source_row]
        h = Decimal.from_float(float(np.float32(0.05)))
        p = Decimal.from_float(float(np.float32(0.002)))
        for start in range(0, target + 1, group_size):
            group = working[start : start + group_size]
            minimum = min(Decimal(0), min(group))
            maximum = max(Decimal(0), max(group))
            maximum = max(abs(minimum), maximum)
            if minimum < 0:
                minimum = -maximum
            if minimum == maximum == 0:
                minimum, maximum = Decimal(-1), Decimal(1)
            scale = (maximum - minimum) / 15
            for column in range(start, min(start + group_size, target + 1)):
                value = working[column]
                quotient = value / scale
                if column == target:
                    half = quotient.to_integral_value(rounding=ROUND_FLOOR) + Decimal(
                        "0.5"
                    )
                    return abs(quotient - half)
                code = max(
                    0,
                    min(
                        15,
                        int(quotient.to_integral_value(rounding=ROUND_HALF_EVEN)) + 8,
                    ),
                )
                quantized = scale * (code - 8)
                if column + 1 < len(working):
                    projected = quantized if (column + 1) % group_size == 0 else value
                    working[column + 1] -= (value - quantized) * h - projected * p
    raise AssertionError("target column was not reached")


@pytest.mark.parametrize("name,rows,columns", QWEN38_27B_PROJECTIONS)
def test_gptaq_qwen38_projection_oracle(name, rows, columns):
    rng = np.random.default_rng(3270 + rows + columns)
    source = rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    source = np.asarray(weight.astype(mx.float32))
    factor = np.eye(columns, dtype=np.float32)
    projection = np.zeros((columns, columns), dtype=np.float32)
    np.fill_diagonal(factor[:, 1:], 0.05)
    np.fill_diagonal(projection[:, 1:], 0.002)
    expected = _torch_banded_oracle(source)
    actual = gptaq_quantize_weight_mlx(weight, mx.array(factor), mx.array(projection))
    observed = tuple(np.asarray(tensor) for tensor in actual)
    np.testing.assert_allclose(
        observed[1], expected[1], atol=1e-6, rtol=1e-6, err_msg=f"{name} scales"
    )
    np.testing.assert_array_equal(
        observed[2], expected[2], err_msg=f"{name} zero points"
    )
    actual_codes = _codes(observed, 128)
    expected_codes = _codes(expected, 128)
    mismatches = actual_codes != expected_codes
    np.testing.assert_allclose(
        observed[0][~mismatches],
        expected[0][~mismatches],
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} weights away from ties",
    )
    last_mismatch = {}
    for row, column in np.argwhere(mismatches):
        assert (
            abs(int(actual_codes[row, column]) - int(expected_codes[row, column])) == 1
        )
        margin = _exact_banded_tie_margin(source[row], column)
        if margin >= Decimal("1e-30"):
            # One superdiagonal is nonzero. A code selected at an exact tie
            # can therefore affect only its immediately following column;
            # any further difference must continue that adjacent chain.
            assert last_mismatch.get(int(row)) == column - 1
        last_mismatch[int(row)] = int(column)
