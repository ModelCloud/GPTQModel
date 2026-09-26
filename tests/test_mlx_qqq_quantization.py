# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ method: Meituan, Ying Zhang et al., https://arxiv.org/abs/2406.09904
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Independent Torch-oracle checks for native MLX QQQ quantization."""

import sys
from decimal import ROUND_FLOOR, Decimal, localcontext

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_qqq import qqq_quantize_weight_mlx  # noqa: E402


def _params(group, group_size):
    minimum = torch.minimum(group.amin(dim=1), torch.zeros(group.shape[0]))
    maximum = torch.maximum(group.amax(dim=1), torch.zeros(group.shape[0]))
    maximum = torch.maximum(minimum.abs(), maximum)
    minimum = torch.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = torch.where(empty, -1, minimum)
    maximum = torch.where(empty, 1, maximum)
    if group_size == -1:
        return maximum / 7, torch.zeros_like(maximum)
    return (maximum - minimum) / 15, torch.full_like(maximum, 8)


def _torch_oracle(source, factor, group_size):
    weight = torch.from_numpy(source.copy())
    hinv = torch.from_numpy(factor.copy())
    rows, columns = weight.shape
    result = torch.empty_like(weight)
    scale_extra = (
        torch.where(
            weight.abs().amax(dim=1) == 0,
            1,
            weight.abs().amax(dim=1),
        )[:, None]
        / 127
    )
    fixed = _params(weight, -1) if group_size == -1 else None
    scales, zeros = [], []
    block = 128 if group_size == -1 else group_size
    for start in range(0, columns, block):
        end = min(start + block, columns)
        scale, zero = fixed if fixed is not None else _params(weight[:, start:end], 128)
        if fixed is None:
            scales.append(scale[:, None])
            zeros.append(zero[:, None])
        errors = torch.empty((rows, end - start), dtype=torch.float32)
        for column in range(start, end):
            value = weight[:, column].clone()
            lower, upper = (-7, 7) if group_size == -1 else (0, 15)
            code = (torch.round(value / scale) + zero).clamp(lower, upper)
            q = scale * (code - zero)
            result[:, column] = q
            error = (value - q) / hinv[column, column]
            errors[:, column - start] = error
            weight[:, column:end] -= error[:, None] * hinv[column, column:end]
        if end < columns:
            weight[:, end:] -= errors @ hinv[start:end, end:]
    if fixed is not None:
        scale, zero = fixed
        return result.numpy(), scale[:, None].numpy(), zero[:, None].numpy(), None
    return (
        result.numpy(),
        torch.cat(scales, axis=1).numpy(),
        torch.cat(zeros, axis=1).numpy(),
        scale_extra.numpy(),
    )


def _observe(weight, factor, group_size):
    return tuple(
        None if value is None else np.asarray(value)
        for value in qqq_quantize_weight_mlx(
            weight, mx.array(factor), group_size=group_size
        )
    )


def _codes(result, group_size):
    quantized, scales, zeros, _ = result
    repeats = quantized.shape[1] if group_size == -1 else 128
    return np.rint(
        quantized / np.repeat(scales, repeats, axis=1)
        + np.repeat(zeros, repeats, axis=1)
    ).astype(np.int8)


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_qqq_rounding_boundaries_and_casts(group_size, dtype):
    source = torch.zeros((4, 256), dtype=torch.float32)
    source[:, 0] = -1
    source[:, 1] = 1
    step = torch.tensor(1 / 7 if group_size == -1 else 2 / 15)
    half = torch.arange(-8, 9, dtype=torch.float32) * step + step / 2
    source[1:, 2:19] = torch.nextafter(half, torch.full_like(half, -float("inf")))
    source[1:, 19:36] = half
    source[1:, 36:53] = torch.nextafter(half, torch.full_like(half, float("inf")))
    source[2, 54] = 100
    source[3, 55] = -100
    weight = mx.array(source.numpy()).astype(dtype)
    resident = np.asarray(weight.astype(mx.float32))
    factor = np.eye(256, dtype=np.float32)
    expected = _torch_oracle(resident, factor, group_size)
    actual = _observe(weight, factor, group_size)
    np.testing.assert_array_equal(
        _codes(actual, group_size), _codes(expected, group_size)
    )
    for index in range(4):
        if expected[index] is not None:
            np.testing.assert_allclose(
                actual[index], expected[index], atol=1e-6, rtol=1e-6
            )


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_group_updates_match_torch(group_size):
    rng = np.random.default_rng(744)
    source = rng.normal(0, 0.3, (7, 256)).astype(np.float32)
    source[0] = 0
    factor = np.eye(256, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.001, (256, 256)).astype(np.float32), 1)
    expected = _torch_oracle(source, factor, group_size)
    actual = _observe(mx.array(source), factor, group_size)
    np.testing.assert_array_equal(
        _codes(actual, group_size), _codes(expected, group_size)
    )
    for index in range(4):
        if expected[index] is not None:
            np.testing.assert_allclose(
                actual[index], expected[index], atol=1e-6, rtol=1e-6
            )


def test_qqq_dynamic_fused_extrema_rows():
    source = np.zeros((4, 128), dtype=np.float32)
    source[1] = np.linspace(0.25, 2.0, 128, dtype=np.float32)
    source[2] = np.linspace(-3.0, -0.5, 128, dtype=np.float32)
    source[3] = np.linspace(-2.0, 1.5, 128, dtype=np.float32)
    factor = np.eye(128, dtype=np.float32)
    expected = _torch_oracle(source, factor, 128)
    actual = _observe(mx.array(source), factor, 128)
    for index in range(4):
        np.testing.assert_array_equal(actual[index], expected[index])


def _torch_banded_oracle(source, group_size):
    """Torch oracle specialized to unit diagonal and one 0.05 superdiagonal."""
    weight = torch.from_numpy(source.copy())
    columns = weight.shape[1]
    output = torch.empty_like(weight)
    fixed = _params(weight, -1) if group_size == -1 else None
    scale_extra = (
        torch.where(
            weight.abs().amax(dim=1) == 0,
            1,
            weight.abs().amax(dim=1),
        )[:, None]
        / 127
    )
    scales, zeros = [], []
    for start in range(0, columns, 128):
        end = min(start + 128, columns)
        scale, zero = fixed if fixed is not None else _params(weight[:, start:end], 128)
        if fixed is None:
            scales.append(scale[:, None])
            zeros.append(zero[:, None])
        for column in range(start, end):
            value = weight[:, column].clone()
            code = (torch.round(value / scale) + zero).clamp(
                -7 if fixed is not None else 0, 7 if fixed is not None else 15
            )
            q = scale * (code - zero)
            output[:, column] = q
            if column + 1 < columns:
                weight[:, column + 1] -= (value - q) * 0.05
            weight[:, column] = q
    if fixed is not None:
        scale, zero = fixed
        return output.numpy(), scale[:, None].numpy(), zero[:, None].numpy(), None
    return (
        output.numpy(),
        torch.cat(scales, dim=1).numpy(),
        torch.cat(zeros, dim=1).numpy(),
        scale_extra.numpy(),
    )


def _boundary_margin(source_row, quantized_row, scale, column):
    """Measure a disputed quotient using its full row recurrence."""
    with localcontext() as context:
        context.prec = 80
        def decimal(value):
            return Decimal.from_float(float(value))

        h = decimal(np.float32(0.05))
        error = Decimal(0)
        for index in range(column + 1):
            corrected = decimal(source_row[index]) - error * h
            error = corrected - decimal(quantized_row[index])
        quotient = corrected / decimal(scale)
        half = quotient.to_integral_value(rounding=ROUND_FLOOR) + Decimal("0.5")
        return abs(quotient - half)


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize(
    "dtype",
    [mx.float16, mx.bfloat16],
    ids=["fp16", "bf16"],
)
@pytest.mark.parametrize("name,rows,columns", QWEN38_27B_PROJECTIONS)
def test_qqq_qwen38_projection_oracle(name, rows, columns, dtype, group_size):
    rng = np.random.default_rng(744 + rows + columns)
    source = rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
    weight = mx.array(source).astype(dtype)
    mx.eval(weight)
    resident = np.asarray(weight.astype(mx.float32))
    factor = np.eye(columns, dtype=np.float32)
    np.fill_diagonal(factor[:, 1:], 0.05)
    expected = _torch_banded_oracle(resident, group_size)
    actual = _observe(weight, factor, group_size)
    for index in (1, 2, 3):
        if expected[index] is not None:
            np.testing.assert_allclose(
                actual[index],
                expected[index],
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"{name} metadata {index}",
            )
    actual_codes = _codes(actual, group_size)
    expected_codes = _codes(expected, group_size)
    changed = actual_codes != expected_codes
    last_changed = {}
    for row, column in np.argwhere(changed):
        assert (
            abs(int(actual_codes[row, column]) - int(expected_codes[row, column])) == 1
        )
        scale = expected[1][row, 0 if group_size == -1 else column // 128]
        margin = _boundary_margin(resident[row], expected[0][row], scale, column)
        if margin >= Decimal("0.0002"):
            # One superdiagonal can carry a prior tie decision only to the
            # immediately following column.
            assert last_changed.get(int(row)) == column - 1, (
                f"{name}: unexplained code drift {margin} at {(row, column)}"
            )
        last_changed[int(row)] = int(column)
    np.testing.assert_allclose(
        actual[0][~changed],
        expected[0][~changed],
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} quantized weights away from rounding boundaries",
    )


def test_qqq_rejects_invalid_inputs():
    source = mx.zeros((2, 128), dtype=mx.float32)
    factor = mx.eye(128, dtype=mx.float32)
    with pytest.raises(ValueError, match="group_size"):
        qqq_quantize_weight_mlx(source, factor, group_size=64)
    with pytest.raises(ValueError, match="inverse_hessian"):
        qqq_quantize_weight_mlx(source, mx.eye(64), group_size=128)
