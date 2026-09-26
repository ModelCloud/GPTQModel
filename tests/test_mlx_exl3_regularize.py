# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Torch-oracle checks for EXL3's fused native MLX regularization kernel."""

import gc
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_hadamard import _torch_hadamard_oracle
from tests.test_mlx_exl3_rms import _normalized_rms_drift, _torch_block_rms_oracle

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_fallback import exl3_fallback_quantize_mlx  # noqa: E402
from gptqmodel.quantization.mlx_exl3_regularize import (  # noqa: E402
    exl3_input_regularize_mlx,
    exl3_output_regularize_mlx,
    exl3_regularize_transforms_mlx,
)
from tests.test_mlx_exl3_tiles import (  # noqa: E402
    _torch_from_tiles_oracle,
    _torch_to_tiles_oracle,
)
from tests.test_mlx_exl3_viterbi import _torch_viterbi_oracle_tensors  # noqa: E402

_CODEBOOK_SCALE = 1.24371088


@lru_cache(maxsize=1)
def _float64_hadamard():
    hadamard = np.ones((1, 1), dtype=np.float64)
    while hadamard.shape[0] < 128:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    return hadamard / np.sqrt(128.0)


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


def _torch_regularize_transforms_oracle(
    weight,
    input_signs,
    output_signs,
    *,
    force_output_scales=None,
    hessian_diagonal=None,
    fallback=False,
):
    if not fallback and hessian_diagonal is not None:
        diagonal = torch.from_numpy(
            np.ascontiguousarray(hessian_diagonal, dtype=np.float32)
        ).sqrt()
        diagonal, _ = torch.sort(diagonal, descending=True)
        cutoff = diagonal.shape[0] // 50
        skew = diagonal[:cutoff].sum() / diagonal.sum()
        apply_output_scales = (
            skew.item() < 0.15
            if force_output_scales is None
            else force_output_scales
        )
    else:
        apply_output_scales = (
            True if force_output_scales is None else force_output_scales
        )
    if fallback:
        apply_output_scales = force_output_scales

    output_rms = _torch_block_rms_oracle(weight, 0)
    output_mean = float(torch.from_numpy(output_rms).mean().item())
    if output_mean <= 1e-30 and force_output_scales is not None:
        apply_output_scales = True
    transformed, output_scales, _ = _torch_output_regularize_oracle(
        weight,
        output_signs,
        output_rms,
        output_mean,
        bool(apply_output_scales),
    )
    input_rms = _torch_block_rms_oracle(transformed, 1)
    transformed, input_scales = _torch_input_regularize_oracle(
        transformed,
        input_signs,
        input_rms,
    )
    return apply_output_scales, transformed, input_scales, output_scales


def _float64_regularize_transforms_oracle(weight, input_signs, output_signs):
    """Evaluate the transform in float64 to adjudicate natural float32 drift."""
    hadamard = _float64_hadamard()

    transformed = weight.astype(np.float64)
    output_rms = np.sqrt(np.mean(np.square(transformed), axis=0, keepdims=True))
    output_scales = (
        output_signs.astype(np.float64) * output_rms / np.mean(output_rms) + 1e-10
    )
    transformed = transformed / output_scales
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, transformed.shape[1], 128):
            transformed[:, start : start + 128] = (
                transformed[:, start : start + 128] @ hadamard
            )

    input_rms = np.sqrt(np.mean(np.square(transformed), axis=1, keepdims=True))
    input_scales = (
        input_signs.astype(np.float64) * input_rms / -_CODEBOOK_SCALE + 1e-10
    )
    transformed = transformed / input_scales
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, transformed.shape[0], 128):
            transformed[start : start + 128] = (
                hadamard @ transformed[start : start + 128]
            )
    assert np.isfinite(transformed).all()
    return transformed


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


def _assert_hadamard_matches(actual, expected, scaled, axis, *, err_msg=None):
    """Adjudicate butterfly/dense-order drift with selected FP64 outputs."""
    assert np.isfinite(actual).all(), err_msg
    assert _normalized_rms_drift(actual, expected) <= 1e-6, err_msg
    error = np.abs(actual - expected)
    differing = np.argwhere(error > 1e-6 + 1e-6 * np.abs(expected))
    if not differing.size:
        return

    hadamard = _float64_hadamard()
    ideal = np.empty(differing.shape[0], dtype=np.float64)
    if axis == 0:
        for index, (row, column) in enumerate(differing):
            start = (row // 128) * 128
            ideal[index] = (
                hadamard[row % 128]
                @ scaled[start : start + 128, column].astype(np.float64)
            )
    else:
        for index, (row, column) in enumerate(differing):
            start = (column // 128) * 128
            ideal[index] = (
                scaled[row, start : start + 128].astype(np.float64)
                @ hadamard[:, column % 128]
            )
    actual_values = actual[tuple(differing.T)].astype(np.float64)
    expected_values = expected[tuple(differing.T)].astype(np.float64)
    np.testing.assert_allclose(
        actual_values, ideal, rtol=1e-6, atol=1e-6, err_msg=err_msg
    )
    assert np.all(
        np.abs(actual_values - ideal) <= np.abs(expected_values - ideal)
    ), err_msg


def _assert_composed_weight_matches(actual, expected, *, err_msg=None):
    """Bound natural error after both RMS and Hadamard stages have composed."""
    assert np.isfinite(actual).all(), err_msg
    assert _normalized_rms_drift(actual, expected) <= 1e-6, err_msg
    np.testing.assert_allclose(
        actual,
        expected,
        atol=4e-6,
        rtol=1e-6,
        err_msg=err_msg,
    )


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


@pytest.mark.parametrize(
    "force_output_scales,fallback,hessian_kind",
    [
        (None, False, "uniform"),
        (None, False, "skewed"),
        (False, False, "skewed"),
        (True, False, "uniform"),
        (False, True, None),
        (True, True, None),
    ],
)
def test_exl3_regularize_transforms_small_torch_oracle(
    force_output_scales, fallback, hessian_kind
):
    rng = np.random.default_rng(18401 + fallback)
    weight = rng.normal(0.0, 0.2, (256, 384)).astype(np.float32)
    input_signs = rng.choice([-1.0, 1.0], size=(256, 1)).astype(np.float32)
    output_signs = rng.choice([-1.0, 1.0], size=(1, 384)).astype(np.float32)
    if hessian_kind == "uniform":
        hessian_diagonal = np.ones(256, dtype=np.float32)
    elif hessian_kind == "skewed":
        hessian_diagonal = np.full(256, np.float32(1e-6), dtype=np.float32)
        hessian_diagonal[:8] = np.float32(1e6)
    else:
        hessian_diagonal = None

    expected = _torch_regularize_transforms_oracle(
        weight,
        input_signs,
        output_signs,
        force_output_scales=force_output_scales,
        hessian_diagonal=hessian_diagonal,
        fallback=fallback,
    )
    actual = exl3_regularize_transforms_mlx(
        mx.array(weight),
        mx.array(input_signs),
        mx.array(output_signs),
        force_output_scales=force_output_scales,
        hessian_diagonal=(
            None if hessian_diagonal is None else mx.array(hessian_diagonal)
        ),
        fallback=fallback,
    )
    assert actual[0] is expected[0]
    _assert_composed_weight_matches(np.asarray(actual[1]), expected[1])
    for actual_array, expected_array in zip(actual[2:], expected[2:]):
        _assert_matches(np.asarray(actual_array), expected_array)


def test_exl3_regularize_transforms_skew_threshold_boundaries():
    rows = 128
    weight = np.zeros((rows, 128), dtype=np.float32)
    weight[:, 0] = 1.0
    input_signs = np.ones((rows, 1), dtype=np.float32)
    output_signs = np.ones((1, 128), dtype=np.float32)
    boundary = np.float32(18.9 / 1.7)

    for top_value in (
        np.nextafter(boundary, np.float32(0)),
        boundary,
        np.nextafter(boundary, np.float32(np.inf)),
    ):
        roots = np.ones(rows, dtype=np.float32)
        roots[:2] = top_value
        hessian_diagonal = np.square(roots, dtype=np.float32)
        expected = _torch_regularize_transforms_oracle(
            weight,
            input_signs,
            output_signs,
            hessian_diagonal=hessian_diagonal,
        )
        actual = exl3_regularize_transforms_mlx(
            mx.array(weight),
            mx.array(input_signs),
            mx.array(output_signs),
            hessian_diagonal=mx.array(hessian_diagonal),
        )
        assert actual[0] is expected[0]
        actual_weight = np.asarray(actual[1])
        ideal = _float64_regularize_transforms_oracle(
            weight, input_signs, output_signs
        )
        assert _normalized_rms_drift(actual_weight, ideal) <= 1e-6
        assert np.linalg.norm(actual_weight.astype(np.float64) - ideal) <= (
            np.linalg.norm(expected[1].astype(np.float64) - ideal)
        )
        for actual_array, expected_array in zip(actual[2:], expected[2:]):
            _assert_matches(np.asarray(actual_array), expected_array)


def test_exl3_regularize_butterfly_cancellation_matches_fp64():
    weight = np.zeros((128, 128), dtype=np.float32)
    weight[:, 0] = 1.0
    input_signs = np.ones((128, 1), dtype=np.float32)
    output_signs = np.ones((1, 128), dtype=np.float32)
    expected = _torch_regularize_transforms_oracle(
        weight, input_signs, output_signs
    )[1]
    actual = np.asarray(
        exl3_regularize_transforms_mlx(
            mx.array(weight), mx.array(input_signs), mx.array(output_signs)
        )[1]
    )
    ideal = _float64_regularize_transforms_oracle(
        weight, input_signs, output_signs
    )

    assert _normalized_rms_drift(actual, ideal) <= 1e-6
    actual_error = np.linalg.norm(actual.astype(np.float64) - ideal)
    torch_error = np.linalg.norm(expected.astype(np.float64) - ideal)
    assert actual_error < torch_error


def test_exl3_regularize_transforms_all_zero_forces_configured_scales():
    weight = np.zeros((128, 128), dtype=np.float32)
    input_signs = np.ones((128, 1), dtype=np.float32)
    output_signs = np.ones((1, 128), dtype=np.float32)
    actual = exl3_regularize_transforms_mlx(
        mx.array(weight),
        mx.array(input_signs),
        mx.array(output_signs),
        force_output_scales=False,
    )
    expected = _torch_regularize_transforms_oracle(
        weight,
        input_signs,
        output_signs,
        force_output_scales=False,
    )
    assert actual[0] is expected[0] is True
    _assert_composed_weight_matches(np.asarray(actual[1]), expected[1])
    for actual_array, expected_array in zip(actual[2:], expected[2:]):
        _assert_matches(np.asarray(actual_array), expected_array)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_exl3_regularize_transforms_preserve_downstream_quantization(bits):
    rng = np.random.default_rng(29401 + bits)
    weight = rng.normal(0.0, 0.2, (128, 128)).astype(np.float32)
    input_signs = rng.choice([-1.0, 1.0], size=(128, 1)).astype(np.float32)
    output_signs = rng.choice([-1.0, 1.0], size=(1, 128)).astype(np.float32)
    hessian_diagonal = rng.uniform(0.01, 2.0, size=128).astype(np.float32)
    expected_regularized = _torch_regularize_transforms_oracle(
        weight,
        input_signs,
        output_signs,
        hessian_diagonal=hessian_diagonal,
    )[1]
    actual_regularized = exl3_regularize_transforms_mlx(
        mx.array(weight),
        mx.array(input_signs),
        mx.array(output_signs),
        hessian_diagonal=mx.array(hessian_diagonal),
    )[1]

    expected_tiles = _torch_to_tiles_oracle(expected_regularized)
    expected_quantized_tiles, expected_encoded = _torch_viterbi_oracle_tensors(
        expected_tiles.reshape(-1, 256), bits, "mcg"
    )
    actual_quantized, actual_encoded = exl3_fallback_quantize_mlx(
        actual_regularized,
        bits=bits,
        codebook="mcg",
    )
    expected_quantized = _torch_from_tiles_oracle(
        expected_quantized_tiles.numpy().reshape(expected_tiles.shape)
    )
    actual_encoded = np.asarray(actual_encoded)
    expected_encoded = expected_encoded.numpy().reshape(actual_encoded.shape)
    actual_quantized = np.asarray(actual_quantized)
    if np.array_equal(actual_encoded, expected_encoded):
        np.testing.assert_array_equal(actual_quantized, expected_quantized)
        return
    if np.array_equal(actual_quantized, expected_quantized):
        return

    ideal = _float64_regularize_transforms_oracle(
        weight, input_signs, output_signs
    )
    changed = np.argwhere(actual_quantized != expected_quantized)
    changed_tiles = {(int(row // 16), int(column // 16)) for row, column in changed}
    assert changed_tiles
    for tile_row, tile_column in changed_tiles:
        tile = np.s_[
            tile_row * 16 : (tile_row + 1) * 16,
            tile_column * 16 : (tile_column + 1) * 16,
        ]
        expected_error = np.sum(
            np.square(expected_quantized[tile].astype(np.float64) - ideal[tile])
        )
        actual_error = np.sum(
            np.square(actual_quantized[tile].astype(np.float64) - ideal[tile])
        )
        assert actual_error <= expected_error, (
            f"MLX changed tile {(tile_row, tile_column)} without improving the "
            f"float64 objective: MLX={actual_error}, Torch={expected_error}"
        )


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_regularize_transforms_qwen38_projection_oracle(
    name, out_features, in_features
):
    rng = np.random.default_rng(25401 + out_features + in_features)
    weight = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
    input_signs = rng.choice([-1.0, 1.0], size=(in_features, 1)).astype(np.float32)
    output_signs = rng.choice([-1.0, 1.0], size=(1, out_features)).astype(
        np.float32
    )
    hessian_diagonal = rng.uniform(0.01, 2.0, size=in_features).astype(np.float32)
    expected = _torch_regularize_transforms_oracle(
        weight,
        input_signs,
        output_signs,
        hessian_diagonal=hessian_diagonal,
    )
    actual = exl3_regularize_transforms_mlx(
        mx.array(weight),
        mx.array(input_signs),
        mx.array(output_signs),
        hessian_diagonal=mx.array(hessian_diagonal),
    )
    assert actual[0] is expected[0], name
    _assert_composed_weight_matches(
        np.asarray(actual[1]), expected[1], err_msg=f"{name} weight"
    )
    for label, actual_array, expected_array in zip(
        ("input scales", "output scales"), actual[2:], expected[2:]
    ):
        _assert_matches(
            np.asarray(actual_array), expected_array, err_msg=f"{name} {label}"
        )

    del weight, input_signs, output_signs, hessian_diagonal, expected, actual
    gc.collect()
    mx.clear_cache()


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

    scaled = np.ascontiguousarray(
        weight / expected_scales, dtype=np.float32
    )
    _assert_hadamard_matches(
        np.asarray(actual_weight),
        expected_weight,
        scaled,
        1 if mode == "output" else 0,
        err_msg=f"{name} {mode} weight",
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
    with pytest.raises(TypeError, match="boolean or None"):
        exl3_regularize_transforms_mlx(
            weight, in_signs, out_signs, force_output_scales="auto"
        )
    with pytest.raises(TypeError, match="fallback"):
        exl3_regularize_transforms_mlx(
            weight, in_signs, out_signs, fallback=1
        )
    with pytest.raises(ValueError, match="one value"):
        exl3_regularize_transforms_mlx(
            weight,
            in_signs,
            out_signs,
            hessian_diagonal=mx.ones((127,), dtype=mx.float32),
        )
    with pytest.raises(ValueError, match="float32"):
        exl3_regularize_transforms_mlx(
            weight,
            in_signs,
            out_signs,
            hessian_diagonal=mx.ones((128,), dtype=mx.float16),
        )
    invalid_diagonal = mx.ones((128,), dtype=mx.float32).at[0].add(float("inf"))
    with pytest.raises(ValueError, match="finite nonnegative"):
        exl3_regularize_transforms_mlx(
            weight,
            in_signs,
            out_signs,
            hessian_diagonal=invalid_diagonal,
        )
