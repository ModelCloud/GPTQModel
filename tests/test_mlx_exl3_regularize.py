# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Torch-oracle checks for EXL3's fused native MLX regularization kernel."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_hadamard import _torch_hadamard_oracle
from tests.test_mlx_exl3_rms import _normalized_rms_drift, _torch_block_rms_oracle

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_regularize import (
    exl3_input_regularize_mlx,
    exl3_output_regularize_mlx,
)

_CODEBOOK_SCALE = 1.24371088


def _torch_output_regularize_oracle(weight, signs, rms, mean, apply_scales):
    signs_t = torch.from_numpy(np.ascontiguousarray(signs, dtype=np.float32))
    scales = torch.from_numpy(np.ascontiguousarray(rms, dtype=np.float32)).clone()
    if mean > 1e-30:
        scales /= mean
    zero_channels = scales.abs() < 1e-30
    if apply_scales:
        scales[zero_channels] = 0.1
        stored = (signs_t * scales + 1e-10).float()
    else:
        stored = signs_t.clone()
    scaled = np.ascontiguousarray(weight / stored.numpy(), dtype=np.float32)
    stored[zero_channels] = 0.0
    transformed = _torch_hadamard_oracle(scaled, axis=1)
    return transformed, stored.numpy(), mean <= 1e-30


def _torch_input_regularize_oracle(weight, signs, rms):
    signs_t = torch.from_numpy(np.ascontiguousarray(signs, dtype=np.float32))
    scales = torch.from_numpy(np.ascontiguousarray(rms, dtype=np.float32)).clone()
    scales[scales.abs() < 1e-30] = 0.1
    stored = (signs_t * scales / -_CODEBOOK_SCALE + 1e-10).float()
    scaled = np.ascontiguousarray(weight / stored.numpy(), dtype=np.float32)
    transformed = _torch_hadamard_oracle(scaled, axis=0)
    return transformed, stored.numpy()


def _assert_matches(actual, expected, *, err_msg=None):
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
        err_msg=err_msg,
    )
    assert np.isfinite(actual).all(), err_msg
    assert _normalized_rms_drift(actual, expected) <= 1e-6, err_msg


@pytest.mark.parametrize("apply_scales", [False, True])
def test_exl3_output_regularize_small_torch_oracle(apply_scales):
    rng = np.random.default_rng(4401 + int(apply_scales))
    weight = rng.normal(0.0, 0.2, (256, 384)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(1, 384)).astype(np.float32)
    rms = _torch_block_rms_oracle(weight, 0)
    mean = float(torch.from_numpy(rms).mean().item())
    expected_weight, expected_scales, expected_zero = _torch_output_regularize_oracle(
        weight, signs, rms, mean, apply_scales
    )
    actual_weight, actual_scales, actual_zero = exl3_output_regularize_mlx(
        mx.array(weight),
        mx.array(signs),
        mx.array(rms),
        mean=mean,
        apply_scales=apply_scales,
    )
    _assert_matches(np.asarray(actual_weight), expected_weight)
    _assert_matches(np.asarray(actual_scales), expected_scales)
    assert actual_zero is expected_zero


def test_exl3_input_regularize_small_torch_oracle():
    rng = np.random.default_rng(5513)
    weight = rng.normal(0.0, 0.2, (256, 384)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(256, 1)).astype(np.float32)
    rms = _torch_block_rms_oracle(weight, 1)
    expected_weight, expected_scales = _torch_input_regularize_oracle(
        weight, signs, rms
    )
    actual_weight, actual_scales = exl3_input_regularize_mlx(
        mx.array(weight), mx.array(signs), mx.array(rms)
    )
    _assert_matches(np.asarray(actual_weight), expected_weight)
    _assert_matches(np.asarray(actual_scales), expected_scales)


@pytest.mark.parametrize("mode", ["output", "input"])
def test_exl3_regularize_float32_boundaries(mode):
    below = np.nextafter(np.float32(1e-30), np.float32(0))
    threshold = np.float32(1e-30)
    above = np.nextafter(np.float32(1e-30), np.float32(np.inf))
    rms_values = np.array(
        [
            0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            below,
            threshold,
            above,
            0.1,
            np.nextafter(np.float32(1), np.float32(0)),
            1.0,
            np.nextafter(np.float32(1), np.float32(np.inf)),
        ],
        dtype=np.float32,
    )
    if mode == "output":
        rms = np.resize(rms_values, (1, 128)).astype(np.float32)
        row_signs = np.resize(
            np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float32), (128, 1)
        )
        weight = (row_signs * rms).astype(np.float32)
        signs = np.resize(np.array([-1.0, 1.0], dtype=np.float32), (1, 128))
        expected_weight, expected_scales, expected_zero = (
            _torch_output_regularize_oracle(weight, signs, rms, 1.0, True)
        )
        actual_weight, actual_scales, actual_zero = exl3_output_regularize_mlx(
            mx.array(weight),
            mx.array(signs),
            mx.array(rms),
            mean=1.0,
            apply_scales=True,
        )
        assert actual_zero is expected_zero
    else:
        rms = np.resize(rms_values, (128, 1)).astype(np.float32)
        column_signs = np.resize(
            np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float32), (1, 128)
        )
        weight = (rms * column_signs).astype(np.float32)
        signs = np.resize(np.array([-1.0, 1.0], dtype=np.float32), (128, 1))
        expected_weight, expected_scales = _torch_input_regularize_oracle(
            weight, signs, rms
        )
        actual_weight, actual_scales = exl3_input_regularize_mlx(
            mx.array(weight), mx.array(signs), mx.array(rms)
        )
    _assert_matches(np.asarray(actual_weight), expected_weight)
    _assert_matches(np.asarray(actual_scales), expected_scales)


def test_exl3_output_regularize_all_zero_mean():
    weight = np.zeros((128, 128), dtype=np.float32)
    signs = np.ones((1, 128), dtype=np.float32)
    rms = np.zeros((1, 128), dtype=np.float32)
    for apply_scales in (False, True):
        expected_weight, expected_scales, expected_zero = (
            _torch_output_regularize_oracle(weight, signs, rms, 0.0, apply_scales)
        )
        actual_weight, actual_scales, actual_zero = exl3_output_regularize_mlx(
            mx.array(weight),
            mx.array(signs),
            mx.array(rms),
            mean=0.0,
            apply_scales=apply_scales,
        )
        _assert_matches(np.asarray(actual_weight), expected_weight)
        np.testing.assert_array_equal(np.asarray(actual_scales), expected_scales)
        assert actual_zero is expected_zero


@pytest.mark.parametrize("mode", ["output", "input"])
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_regularize_qwen38_projection_oracle(
    name, out_features, in_features, mode
):
    rng = np.random.default_rng(15331 + out_features + in_features)
    weight = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
    if mode == "output":
        signs = rng.choice([-1.0, 1.0], size=(1, out_features)).astype(np.float32)
        rms = _torch_block_rms_oracle(weight, 0)
        mean = float(torch.from_numpy(rms).mean().item())
        expected_weight, expected_scales, expected_zero = (
            _torch_output_regularize_oracle(weight, signs, rms, mean, True)
        )
        actual_weight, actual_scales, actual_zero = exl3_output_regularize_mlx(
            mx.array(weight),
            mx.array(signs),
            mx.array(rms),
            mean=mean,
            apply_scales=True,
        )
        assert actual_zero is expected_zero
    else:
        signs = rng.choice([-1.0, 1.0], size=(in_features, 1)).astype(np.float32)
        rms = _torch_block_rms_oracle(weight, 1)
        expected_weight, expected_scales = _torch_input_regularize_oracle(
            weight, signs, rms
        )
        actual_weight, actual_scales = exl3_input_regularize_mlx(
            mx.array(weight), mx.array(signs), mx.array(rms)
        )

    _assert_matches(
        np.asarray(actual_weight), expected_weight, err_msg=f"{name} {mode} weight"
    )
    _assert_matches(
        np.asarray(actual_scales), expected_scales, err_msg=f"{name} {mode} scales"
    )
    del (
        weight,
        signs,
        rms,
        expected_weight,
        expected_scales,
        actual_weight,
        actual_scales,
    )
    gc.collect()
    mx.clear_cache()


def test_exl3_regularize_rejects_invalid_inputs():
    weight = mx.zeros((128, 128), dtype=mx.float32)
    out_signs = mx.ones((1, 128), dtype=mx.float32)
    out_rms = mx.ones((1, 128), dtype=mx.float32)
    in_signs = mx.ones((128, 1), dtype=mx.float32)
    in_rms = mx.ones((128, 1), dtype=mx.float32)

    with pytest.raises(ValueError, match="rank-two"):
        exl3_input_regularize_mlx(mx.zeros((128,), dtype=mx.float32), in_signs, in_rms)
    with pytest.raises(ValueError, match="float32"):
        exl3_output_regularize_mlx(
            weight.astype(mx.float16),
            out_signs,
            out_rms,
            mean=1.0,
            apply_scales=True,
        )
    with pytest.raises(ValueError, match="divisible by 128"):
        exl3_output_regularize_mlx(
            mx.zeros((128, 129), dtype=mx.float32),
            out_signs,
            out_rms,
            mean=1.0,
            apply_scales=True,
        )
    with pytest.raises(ValueError, match="shape"):
        exl3_output_regularize_mlx(
            weight, in_signs, out_rms, mean=1.0, apply_scales=True
        )
    for mean in (-1.0, float("inf")):
        with pytest.raises(ValueError, match="mean"):
            exl3_output_regularize_mlx(
                weight, out_signs, out_rms, mean=mean, apply_scales=True
            )
    for mean in (None, True):
        with pytest.raises(TypeError, match="mean"):
            exl3_output_regularize_mlx(
                weight, out_signs, out_rms, mean=mean, apply_scales=True
            )
    with pytest.raises(TypeError, match="boolean"):
        exl3_output_regularize_mlx(weight, out_signs, out_rms, mean=1.0, apply_scales=1)
