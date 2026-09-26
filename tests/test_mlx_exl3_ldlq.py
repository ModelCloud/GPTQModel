# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for native MLX EXL3 LDLQ."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_codebook import _torch_codebook_oracle
from tests.test_mlx_exl3_fallback import _torch_patterned_fallback_oracle
from tests.test_mlx_exl3_gss import _patterned_weight
from tests.test_mlx_exl3_tiles import _torch_tensor_core_permutation
from tests.test_mlx_exl3_viterbi import (
    _half_rounding_boundary,
    _torch_viterbi_oracle_tensors,
)

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_ldlq import (  # noqa: E402
    _exl3_ldlq_product_mlx,
    exl3_ldlq_quantize_mlx,
)


def _torch_quantize_rows_oracle(rows, *, bits, codebook):
    """Quantize matrix rows without calling any MLX implementation."""
    source = rows.to(dtype=torch.float32).contiguous()
    row_count, columns = source.shape
    permutation = _torch_tensor_core_permutation()
    tiles = (
        source.reshape(row_count // 16, 16, columns // 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(row_count // 16, columns // 16, 256)
    )[:, :, permutation]
    quantized_tiles, encoded = _torch_viterbi_oracle_tensors(
        tiles, bits, codebook
    )
    inverse = torch.argsort(permutation)
    quantized = (
        quantized_tiles[:, :, inverse]
        .reshape(row_count // 16, columns // 16, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(row_count, columns)
        .contiguous()
    )
    return quantized, encoded.contiguous()


def _torch_ldlq_oracle(
    weight,
    ldl_factor,
    *,
    bits,
    codebook="mcg",
    buffer_rows=128,
):
    """Reproduce EXL3 LDLQ's buffered update order with Torch float32 math."""
    source = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32))
    factor = torch.from_numpy(np.ascontiguousarray(ldl_factor, dtype=np.float32))
    rows, _ = source.shape
    pending = torch.zeros_like(source)
    quantized_chunks = []
    encoded_chunks = []

    for stop in range(rows, 0, -buffer_rows):
        start = stop - buffer_rows
        source_chunk = source[start:stop]
        chunk_compensation = pending[start:stop]
        factor_chunk = factor[start:stop]
        quantized_suffix = None
        encoded_blocks = []

        for block_stop in range(buffer_rows, 0, -16):
            block_start = block_stop - 16
            compensation = chunk_compensation[block_start:block_stop]
            if quantized_suffix is not None:
                suffix_error = source_chunk[block_stop:] - quantized_suffix
                coupling = factor_chunk[
                    block_stop:, start + block_start : start + block_stop
                ]
                compensation = compensation + coupling.T @ suffix_error

            quantized_block, encoded = _torch_quantize_rows_oracle(
                source_chunk[block_start:block_stop] + compensation,
                bits=bits,
                codebook=codebook,
            )
            quantized_suffix = (
                quantized_block
                if quantized_suffix is None
                else torch.cat((quantized_block, quantized_suffix), dim=0)
            )
            encoded_blocks.append(encoded)

        encoded_chunk = torch.cat(tuple(reversed(encoded_blocks)), dim=0)
        quantized_chunks.append(quantized_suffix)
        encoded_chunks.append(encoded_chunk)
        if start:
            chunk_error = source_chunk - quantized_suffix
            pending = pending[:start] + factor_chunk[:, :start].T @ chunk_error

    return (
        torch.cat(tuple(reversed(quantized_chunks)), dim=0).numpy(),
        torch.cat(tuple(reversed(encoded_chunks)), dim=0).numpy(),
    )


def _normalized_drift(actual, expected):
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    return float(
        np.linalg.norm(actual64 - expected64)
        / max(np.linalg.norm(expected64), np.finfo(np.float64).tiny)
    )


def _block_ldl_factor(rows, *, seed):
    generator = torch.Generator().manual_seed(seed)
    factor = torch.eye(rows, dtype=torch.float32)
    blocks = rows // 16
    for block_row in range(1, blocks):
        for block_column in range(block_row):
            factor[
                block_row * 16 : (block_row + 1) * 16,
                block_column * 16 : (block_column + 1) * 16,
            ] = torch.randn((16, 16), generator=generator) * 0.015625
    return factor.numpy()


def _assert_quantization_matches(actual_weight, actual_encoded, expected_weight, expected_encoded, *, name):
    actual_weight = np.asarray(actual_weight)
    actual_encoded = np.asarray(actual_encoded)
    assert np.isfinite(actual_weight).all(), name
    np.testing.assert_array_equal(actual_encoded, expected_encoded, err_msg=f"{name} states")
    np.testing.assert_allclose(
        actual_weight,
        expected_weight,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"{name} weight",
    )
    assert _normalized_drift(actual_weight, expected_weight) <= 1e-6, name


@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("buffer_rows", (16, 32))
def test_exl3_ldlq_dense_feedback_matches_torch_oracle(bits, buffer_rows):
    rng = np.random.default_rng(3171 + bits + buffer_rows)
    source = rng.normal(0.0, 0.4, (64, 128)).astype(np.float32)
    factor = _block_ldl_factor(64, seed=93171 + bits)
    expected_weight, expected_encoded = _torch_ldlq_oracle(
        source,
        factor,
        bits=bits,
        buffer_rows=buffer_rows,
    )
    actual_weight, actual_encoded = exl3_ldlq_quantize_mlx(
        mx.array(source),
        mx.array(factor),
        bits=bits,
        buffer_rows=buffer_rows,
    )
    _assert_quantization_matches(
        actual_weight,
        actual_encoded,
        expected_weight,
        expected_encoded,
        name=f"bits={bits}, buffer_rows={buffer_rows}",
    )


@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
def test_exl3_ldlq_rounding_boundaries_with_nonzero_feedback(codebook):
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
    source = np.resize(values, (32, 128)).astype(np.float32, copy=False)
    factor = np.eye(32, dtype=np.float32)
    factor[16:, :16] = np.eye(16, dtype=np.float32) * np.float32(0.25)
    expected_weight, expected_encoded = _torch_ldlq_oracle(
        source,
        factor,
        bits=4,
        codebook=codebook,
        buffer_rows=16,
    )
    actual_weight, actual_encoded = exl3_ldlq_quantize_mlx(
        mx.array(source),
        mx.array(factor),
        bits=4,
        codebook=codebook,
        buffer_rows=16,
    )
    _assert_quantization_matches(
        actual_weight,
        actual_encoded,
        expected_weight,
        expected_encoded,
        name=codebook,
    )


def test_exl3_ldlq_product_meets_float64_oracle_limit():
    rng = np.random.default_rng(73171)
    factor = rng.normal(0.0, 0.04, (128, 96)).astype(np.float32)
    error = rng.normal(0.0, 0.3, (128, 257)).astype(np.float32)
    expected = torch.from_numpy(factor).to(torch.float64).T @ torch.from_numpy(
        error
    ).to(torch.float64)
    actual = np.asarray(
        _exl3_ldlq_product_mlx(mx.array(factor), mx.array(error))
    )
    assert np.isfinite(actual).all()
    assert _normalized_drift(actual, expected.numpy()) <= 1e-6


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_ldlq_qwen38_projection_oracle(name, out_features, in_features):
    source = _patterned_weight(
        in_features,
        out_features,
        seed=53171 + out_features + in_features,
    )
    expected_weight, expected_encoded = _torch_patterned_fallback_oracle(
        source, bits=4, codebook="mcg"
    )
    factor = mx.eye(in_features, dtype=mx.float32) + np.float32(2**-16) * mx.eye(
        in_features, k=-16, dtype=mx.float32
    )
    actual_weight, actual_encoded = exl3_ldlq_quantize_mlx(
        mx.array(source), factor, bits=4
    )
    _assert_quantization_matches(
        actual_weight,
        actual_encoded,
        expected_weight,
        expected_encoded,
        name=name,
    )

    del source, expected_weight, expected_encoded, factor
    del actual_weight, actual_encoded
    gc.collect()
    mx.clear_cache()


def test_exl3_ldlq_rejects_invalid_inputs():
    weight = mx.zeros((32, 128), dtype=mx.float32)
    factor = mx.eye(32, dtype=mx.float32)
    with pytest.raises(ValueError, match="rank-two"):
        exl3_ldlq_quantize_mlx(mx.zeros((128,), dtype=mx.float32), factor, bits=4)
    with pytest.raises(ValueError, match="nonempty"):
        exl3_ldlq_quantize_mlx(mx.zeros((0, 128), dtype=mx.float32), factor, bits=4)
    with pytest.raises(ValueError, match="float32"):
        exl3_ldlq_quantize_mlx(weight.astype(mx.float16), factor, bits=4)
    with pytest.raises(ValueError, match="float32"):
        exl3_ldlq_quantize_mlx(weight, factor.astype(mx.float16), bits=4)
    with pytest.raises(ValueError, match="square"):
        exl3_ldlq_quantize_mlx(weight, mx.eye(16, dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="input features"):
        exl3_ldlq_quantize_mlx(
            mx.zeros((17, 128), dtype=mx.float32),
            mx.eye(17, dtype=mx.float32),
            bits=4,
        )
    with pytest.raises(ValueError, match="output features"):
        exl3_ldlq_quantize_mlx(
            mx.zeros((32, 144), dtype=mx.float32), factor, bits=4
        )
    for buffer_rows in (0, 15, 17, 16.0, True):
        with pytest.raises(ValueError, match="positive multiple"):
            exl3_ldlq_quantize_mlx(
                weight, factor, bits=4, buffer_rows=buffer_rows
            )
    with pytest.raises(ValueError, match="divisible by buffer_rows"):
        exl3_ldlq_quantize_mlx(weight, factor, bits=4, buffer_rows=48)
    for bits in (0, 9, 4.0, True):
        with pytest.raises(ValueError, match="bits"):
            exl3_ldlq_quantize_mlx(weight, factor, bits=bits, buffer_rows=16)
    for codebook in ("unknown", 1, None):
        with pytest.raises(ValueError, match="codebook"):
            exl3_ldlq_quantize_mlx(
                weight, factor, bits=4, codebook=codebook, buffer_rows=16
            )
    with pytest.raises(ValueError, match="positive integer"):
        exl3_ldlq_quantize_mlx(
            weight, factor, bits=4, buffer_rows=16, workspace_bytes=0
        )
