# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for Metal GGUF block packing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

_source = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_gguf.py"
_spec = importlib.util.spec_from_file_location("gptqmodel_mlx_gguf_test", _source)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def _torch_gguf_q4_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 32)
    selected = blocks.gather(1, blocks.abs().argmax(dim=1, keepdim=True))
    scale = selected / -8.0
    reciprocal = torch.where(scale == 0, 0, 1.0 / scale)
    # GGUF's Q4_0 reference multiplies in float64 after float32 scale
    # selection and reciprocal calculation.
    codes = torch.trunc(blocks.double() * reciprocal.double() + 8.5)
    codes = codes.clamp(0, 15).to(torch.uint8)
    payload = codes[:, :16] | (codes[:, 16:] << 4)
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((scale_bytes, payload.numpy()), axis=1)
    return packed.reshape(rows, columns // 32 * packed.shape[1])


@pytest.mark.parametrize("rows,width", [(13, 32), (13, 256), (257, 512)])
def test_gguf_q4_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(86).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :32] = 1
    weight[2, :32] = np.arange(-16, 16, dtype=np.float32) / 8
    weight[3, :32] = np.tile(
        (np.arange(-8, 8, dtype=np.float32) + 0.5) * 0.25, 2
    )
    weight[3, 0] = -2.0
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0")
    expected = _torch_gguf_q4_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q4_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(19).standard_normal((8, 64))).astype(
        dtype
    )
    expected = _torch_gguf_q4_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q4_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_0_packing_at_float32_rounding_boundaries():
    rng = np.random.default_rng(819)
    rows = 64
    maxima = rng.uniform(0.03, 20.0, rows).astype(np.float32)
    signed_maxima = np.where(rng.integers(0, 2, rows), maxima, -maxima)
    weight = np.zeros((rows, 32), dtype=np.float32)
    weight[:, 0] = signed_maxima
    reciprocal = np.float32(1.0) / (signed_maxima / np.float32(-8.0))
    for column in range(1, 32):
        code = rng.integers(2, 14, rows)
        boundary = ((code - 8.5) / reciprocal.astype(np.float64)).astype(
            np.float32
        )
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        weight[:, column] = np.where(
            rng.integers(0, 3, rows) == 0,
            np.nextafter(boundary, direction),
            boundary,
        )

    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0")
    expected = _torch_gguf_q4_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_0_packing_validates_format_and_shape():
    with pytest.raises(ValueError, match="supports Q4_0"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 32)), "Q5_K")
    with pytest.raises(ValueError, match="divisible by 32"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 33)), "Q4_0")
    with pytest.raises(ValueError, match="nonzero input width"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 0)), "Q4_0")


def test_gguf_q4_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _quantize_gguf_tensor_numpy,
    )

    weight = np.random.default_rng(17).standard_normal((4, 128)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0"))
    expected = _quantize_gguf_tensor_numpy(weight, "Q4_0")
    np.testing.assert_array_equal(packed, expected)
    dequantized = _dequantize_gguf_tensor_numpy(packed, "Q4_0")
    assert dequantized.shape == weight.shape
    assert np.isfinite(dequantized).all()
