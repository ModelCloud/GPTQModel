# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation math: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle tests for ParoQuant's MLX rotation kernel."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_paroquant import paroquant_pack_weight_mlx
from gptqmodel.quantization.mlx_paroquant_quant import (
    paroquant_quantize_weight_mlx,
)
from gptqmodel.quantization.mlx_paroquant_rotation import paroquant_rotate_mlx


def _pair_schedule(columns, group_size, krot, seed):
    rng = np.random.default_rng(seed)
    groups = columns // group_size
    pairs = np.empty((krot, columns), dtype=np.int16)
    for stage in range(krot):
        for group in range(groups):
            start = group * group_size
            pairs[stage, start : start + group_size] = rng.permutation(group_size)
    return pairs


def _torch_oracle(weight, pairs, theta, scales, *, group_size, inverse=False):
    """Independent FP32 Torch implementation of the grouped Givens transform."""
    rows, columns = weight.shape
    groups = columns // group_size
    half_group = group_size // 2
    output = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32)).reshape(
        rows, columns
    )
    theta = torch.from_numpy(np.ascontiguousarray(theta, dtype=np.float32))
    pair_values = np.asarray(pairs, dtype=np.int64)
    scale = (
        None
        if scales is None
        else torch.from_numpy(np.asarray(scales, dtype=np.float32))
    )

    if not inverse and scale is not None:
        output = output * scale.reshape(1, columns)

    stages = range(theta.shape[0] - 1, -1, -1) if inverse else range(theta.shape[0])
    for stage in stages:
        stage_pairs = torch.from_numpy(
            pair_values[stage].reshape(groups, half_group, 2).copy()
        )
        left_indices = stage_pairs[:, :, 0]
        right_indices = stage_pairs[:, :, 1]
        angles = theta[stage].reshape(groups, half_group)
        if inverse:
            angles = -angles
        cosine = torch.cos(angles).unsqueeze(0)
        sine = torch.sin(angles).unsqueeze(0)

        grouped = output.reshape(rows, groups, group_size)
        index_shape = (rows, groups, half_group)
        left = grouped.gather(2, left_indices.unsqueeze(0).expand(index_shape))
        right = grouped.gather(2, right_indices.unsqueeze(0).expand(index_shape))
        rotated_left = left * cosine + right * sine
        rotated_right = -left * sine + right * cosine
        next_grouped = torch.empty_like(grouped)
        next_grouped.scatter_(
            2, left_indices.unsqueeze(0).expand(index_shape), rotated_left
        )
        next_grouped.scatter_(
            2, right_indices.unsqueeze(0).expand(index_shape), rotated_right
        )
        output = next_grouped.reshape(rows, columns)

    if inverse and scale is not None:
        output = output / scale.reshape(1, columns)
    return output.reshape(weight.shape).numpy()


def _run(weight, pairs, theta, scales, *, group_size, inverse=False):
    return np.asarray(
        paroquant_rotate_mlx(
            weight,
            pairs,
            theta,
            group_size=group_size,
            channel_scales=scales,
            inverse=inverse,
        )
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_paroquant_rotation_small_oracle(dtype, inverse):
    rng = np.random.default_rng(20260925)
    group_size, columns, rows, krot = 16, 64, 7, 4
    source = mx.array(rng.normal(0, 0.4, (rows, columns)).astype(np.float32)).astype(
        dtype
    )
    theta_array = mx.array(rng.uniform(-0.45, 0.45, (krot, columns // 2))).astype(dtype)
    scales_array = mx.array(rng.uniform(0.5, 1.5, columns).astype(np.float32)).astype(
        dtype
    )
    mx.eval(source, theta_array, scales_array)
    source_f32 = np.asarray(source.astype(mx.float32))
    theta_f32 = np.asarray(theta_array.astype(mx.float32))
    scales_f32 = np.asarray(scales_array.astype(mx.float32))
    pairs = _pair_schedule(columns, group_size, krot, seed=883)

    expected = _torch_oracle(
        source_f32,
        pairs,
        theta_f32,
        scales_f32,
        group_size=group_size,
        inverse=inverse,
    )
    actual = _run(
        source,
        pairs,
        theta_array,
        scales_array,
        group_size=group_size,
        inverse=inverse,
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("inverse", [False, True])
def test_paroquant_rotation_zero_and_quarter_turn_boundaries(inverse):
    group_size, columns = 16, 32
    pairs = np.tile(np.arange(group_size, dtype=np.int16), (2, columns // group_size))
    theta = np.zeros((2, columns // 2), dtype=np.float32)
    theta[0, 0] = 0.0
    theta[0, 1] = np.nextafter(np.float32(0), np.float32(1))
    theta[0, 2] = np.nextafter(np.float32(0), np.float32(-1))
    theta[1, 0] = np.float32(np.pi / 2)
    theta[1, 1] = np.nextafter(theta[1, 0], np.float32(-np.inf))
    theta[1, 2] = np.nextafter(theta[1, 0], np.float32(np.inf))
    values = np.arange(1, 97, dtype=np.float32).reshape(3, columns) / 13
    scales = np.linspace(0.7, 1.3, columns, dtype=np.float32)

    expected = _torch_oracle(
        values, pairs, theta, scales, group_size=group_size, inverse=inverse
    )
    actual = _run(
        mx.array(values),
        pairs,
        mx.array(theta),
        mx.array(scales),
        group_size=group_size,
        inverse=inverse,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_paroquant_rotation_rank_three_without_channel_scales():
    group_size, columns, krot = 16, 32, 2
    rng = np.random.default_rng(717)
    source = rng.normal(0, 0.2, (2, 3, columns)).astype(np.float32)
    pairs = _pair_schedule(columns, group_size, krot, seed=718)
    theta = rng.uniform(-0.2, 0.2, (krot, columns // 2)).astype(np.float32)
    expected = _torch_oracle(
        source.reshape(-1, columns),
        pairs,
        theta,
        None,
        group_size=group_size,
    ).reshape(source.shape)
    actual = _run(
        mx.array(source),
        pairs,
        mx.array(theta),
        None,
        group_size=group_size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_paroquant_rotation_output_can_be_quantized_packed_and_saved(tmp_path):
    group_size, columns, rows, krot = 128, 256, 32, 3
    rng = np.random.default_rng(913)
    source = mx.array(rng.normal(0, 0.2, (rows, columns)).astype(np.float32)).astype(
        mx.bfloat16
    )
    pairs = _pair_schedule(columns, group_size, krot, seed=914)
    theta = mx.array(rng.uniform(-0.2, 0.2, (krot, columns // 2)).astype(np.float32))
    channel_scales = mx.array(rng.uniform(0.8, 1.2, columns).astype(np.float32))
    learned_scales = mx.array(rng.uniform(0.015, 0.07, (rows, 2)).astype(np.float32))

    transformed = paroquant_rotate_mlx(
        source,
        pairs,
        theta,
        group_size=group_size,
        channel_scales=channel_scales,
    )
    pseudo_weight = paroquant_quantize_weight_mlx(
        transformed,
        learned_scales,
        bits=4,
        group_size=group_size,
    )
    qweight, qzeros, stored_scales = paroquant_pack_weight_mlx(
        pseudo_weight,
        learned_scales,
        group_size=group_size,
    )
    path = tmp_path / "paroquant_rotation_export.npz"
    mx.savez(str(path), qweight=qweight, qzeros=qzeros, scales=stored_scales)
    restored = mx.load(str(path))
    np.testing.assert_array_equal(np.asarray(restored["qweight"]), np.asarray(qweight))
    np.testing.assert_array_equal(np.asarray(restored["qzeros"]), np.asarray(qzeros))
    np.testing.assert_array_equal(
        np.asarray(restored["scales"]), np.asarray(stored_scales)
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("name,rows,columns", QWEN38_27B_PROJECTIONS)
def test_paroquant_rotation_qwen38_projection_oracle(name, rows, columns, inverse):
    group_size, krot = 128, 8
    seed = 8821 + rows + columns
    rng = np.random.default_rng(seed)
    source = mx.array(rng.normal(0, 0.2, (rows, columns)).astype(np.float32)).astype(
        mx.bfloat16
    )
    theta_array = mx.array(
        rng.uniform(-0.35, 0.35, (krot, columns // 2)).astype(np.float32)
    )
    scales_array = mx.array(rng.uniform(0.7, 1.3, columns).astype(np.float32))
    mx.eval(source, theta_array, scales_array)
    source_f32 = np.asarray(source.astype(mx.float32))
    theta_f32 = np.asarray(theta_array)
    scales_f32 = np.asarray(scales_array)
    pairs = _pair_schedule(columns, group_size, krot, seed=seed + 17)

    expected = _torch_oracle(
        source_f32,
        pairs,
        theta_f32,
        scales_f32,
        group_size=group_size,
        inverse=inverse,
    )
    actual = _run(
        source,
        pairs,
        theta_array,
        scales_array,
        group_size=group_size,
        inverse=inverse,
    )
    error = np.abs(actual - expected)
    allowed = 1e-6 + 1e-6 * np.abs(expected)
    outside_tolerance = int(np.count_nonzero(error > allowed))
    assert outside_tolerance == 0, (
        f"{name} inverse={inverse}: {outside_tolerance} values outside tolerance"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} inverse={inverse}",
    )
    del source, theta_array, scales_array, expected, actual, error
    gc.collect()


def test_paroquant_rotation_rejects_invalid_pairs_and_scales():
    weight = mx.zeros((2, 32), dtype=mx.float32)
    theta = mx.zeros((2, 16), dtype=mx.float32)
    scales = mx.ones((32,), dtype=mx.float32)
    pairs = _pair_schedule(32, 16, 2, seed=77)

    invalid_pairs = pairs.copy()
    invalid_pairs[0, 0] = invalid_pairs[0, 1]
    with pytest.raises(ValueError, match="every local channel"):
        paroquant_rotate_mlx(weight, invalid_pairs, theta, group_size=16)
    with pytest.raises(ValueError, match="positive"):
        paroquant_rotate_mlx(
            weight,
            pairs,
            theta,
            group_size=16,
            channel_scales=mx.zeros_like(scales),
        )
    with pytest.raises(ValueError, match="finite"):
        bad_theta = mx.full((2, 16), float("nan"), dtype=mx.float32)
        paroquant_rotate_mlx(weight, pairs, bad_theta, group_size=16)
    with pytest.raises(ValueError, match="weight must be finite"):
        bad_weight = mx.full((2, 32), float("inf"), dtype=mx.float32)
        paroquant_rotate_mlx(bad_weight, pairs, theta, group_size=16)
    bad_width_weight = mx.zeros((2, 48), dtype=mx.float32)
    bad_width_pairs = _pair_schedule(48, 16, 2, seed=78)
    bad_width_theta = mx.zeros((2, 24), dtype=mx.float32)
    with pytest.raises(ValueError, match="divisible"):
        paroquant_rotate_mlx(
            bad_width_weight,
            bad_width_pairs,
            bad_width_theta,
            group_size=32,
        )
