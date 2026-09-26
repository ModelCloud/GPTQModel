# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle checks for native EXL3 fallback quantization."""

import gc
import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_codebook import _torch_codebook_oracle
from tests.test_mlx_exl3_gss import _patterned_weight
from tests.test_mlx_exl3_tiles import (
    _torch_from_tiles_oracle,
    _torch_to_tiles_oracle,
)
from tests.test_mlx_exl3_viterbi import (
    _half_rounding_boundary,
    _torch_viterbi_oracle,
)

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_fallback import (  # noqa: E402
    exl3_fallback_quantize_mlx,
)


def _torch_fallback_quantize_oracle(weight, *, bits, codebook):
    tiles = _torch_to_tiles_oracle(weight)
    flat_tiles = tiles.reshape(-1, 256)
    unique, inverse = np.unique(flat_tiles, axis=0, return_inverse=True)
    quantized_bank, encoded_bank = _torch_viterbi_oracle(unique, bits, codebook)
    quantized_tiles = quantized_bank[inverse].reshape(tiles.shape)
    encoded = encoded_bank[inverse].reshape(tiles.shape)
    quantized_weight = _torch_from_tiles_oracle(quantized_tiles)
    return quantized_weight, encoded


def _torch_patterned_fallback_oracle(weight, *, bits, codebook):
    rows, columns = weight.shape
    tile_rows = rows // 16
    tile_columns = columns // 16
    pattern_tiles = _torch_to_tiles_oracle(weight[:16, :64]).reshape(4, 256)
    quantized_bank, encoded_bank = _torch_viterbi_oracle(pattern_tiles, bits, codebook)
    repeats = (tile_columns + 3) // 4
    quantized_tiles = np.tile(
        quantized_bank.reshape(1, 4, 256), (tile_rows, repeats, 1)
    )[:, :tile_columns]
    encoded = np.tile(encoded_bank.reshape(1, 4, 256), (tile_rows, repeats, 1))[
        :, :tile_columns
    ]
    return _torch_from_tiles_oracle(quantized_tiles), encoded


def _assert_bits_equal(actual, expected, *, err_msg=None):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
        err_msg=err_msg,
    )


@pytest.mark.parametrize("bits", (2, 4, 8))
def test_exl3_fallback_small_torch_oracle(bits):
    source = _patterned_weight(32, 128, seed=9317 + bits)
    expected_weight, expected_encoded = _torch_fallback_quantize_oracle(
        source, bits=bits, codebook="mcg"
    )
    actual_weight, actual_encoded = exl3_fallback_quantize_mlx(
        mx.array(source), bits=bits
    )
    assert actual_weight.dtype == mx.float32
    assert actual_encoded.dtype == mx.int16
    _assert_bits_equal(actual_weight, expected_weight)
    np.testing.assert_array_equal(np.asarray(actual_encoded), expected_encoded)


@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
def test_exl3_fallback_rounding_boundaries(codebook):
    tie = _half_rounding_boundary(codebook)
    lut = _torch_codebook_oracle(codebook).numpy()
    values = np.array(
        [
            np.nextafter(tie, np.float16(-np.inf), dtype=np.float16),
            tie,
            np.nextafter(tie, np.float16(np.inf), dtype=np.float16),
            -0.0,
            0.0,
            lut.min(),
            lut.max(),
        ],
        dtype=np.float32,
    )
    source = np.resize(values, (16, 128)).astype(np.float32, copy=False)
    expected_weight, expected_encoded = _torch_fallback_quantize_oracle(
        source, bits=4, codebook=codebook
    )
    actual_weight, actual_encoded = exl3_fallback_quantize_mlx(
        mx.array(source), bits=4, codebook=codebook
    )
    _assert_bits_equal(actual_weight, expected_weight)
    np.testing.assert_array_equal(np.asarray(actual_encoded), expected_encoded)


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_fallback_qwen38_projection_oracle(name, out_features, in_features):
    source = _patterned_weight(
        in_features,
        out_features,
        seed=43167 + out_features + in_features,
    )
    expected_weight, expected_encoded = _torch_patterned_fallback_oracle(
        source, bits=4, codebook="mcg"
    )
    actual_weight, actual_encoded = exl3_fallback_quantize_mlx(mx.array(source), bits=4)
    _assert_bits_equal(actual_weight, expected_weight, err_msg=f"{name} weight")
    np.testing.assert_array_equal(
        np.asarray(actual_encoded), expected_encoded, err_msg=f"{name} states"
    )

    del source, expected_weight, expected_encoded, actual_weight, actual_encoded
    gc.collect()
    mx.clear_cache()


def test_exl3_fallback_rejects_invalid_inputs():
    valid = mx.zeros((16, 128), dtype=mx.float32)
    with pytest.raises(ValueError, match="rank-two"):
        exl3_fallback_quantize_mlx(mx.zeros((128,), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="nonempty"):
        exl3_fallback_quantize_mlx(mx.zeros((0, 128), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="input features"):
        exl3_fallback_quantize_mlx(mx.zeros((17, 128), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="output features"):
        exl3_fallback_quantize_mlx(mx.zeros((16, 144), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="float32"):
        exl3_fallback_quantize_mlx(mx.zeros((16, 128), dtype=mx.float16), bits=4)
    for bits in (0, 9, 4.0, True):
        with pytest.raises(ValueError, match="bits"):
            exl3_fallback_quantize_mlx(valid, bits=bits)
    for codebook in ("unknown", 1, None):
        with pytest.raises(ValueError, match="codebook"):
            exl3_fallback_quantize_mlx(valid, bits=4, codebook=codebook)
    with pytest.raises(ValueError, match="positive integer"):
        exl3_fallback_quantize_mlx(valid, bits=4, workspace_bytes=0)
