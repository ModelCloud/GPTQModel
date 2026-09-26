# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Independent bitsandbytes CPU-oracle checks for MLX weight quantization."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
bnb = pytest.importorskip("bitsandbytes")

from gptqmodel.quantization.mlx_bitsandbytes import (  # noqa: E402
    quantize_4bit_weight_mlx,
    quantize_int8_weight_mlx,
)


BLOCK_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
SOURCE_DTYPES = (torch.float16, torch.bfloat16)


def _check_4bit(source, quant_type, block_size, compressed):
    weight = mx.array(source.float().numpy()).astype({
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.float32: mx.float32,
    }[source.dtype])
    expected, state = bnb.functional.quantize_4bit(
        source, quant_type=quant_type, blocksize=block_size,
        compress_statistics=compressed, quant_storage=torch.uint8,
    )
    actual, scales = quantize_4bit_weight_mlx(
        weight, quant_type=quant_type, block_size=block_size,
        compress_statistics=compressed,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
    if compressed:
        _check_nested_codes(source, block_size, np.asarray(scales["absmax"]), state.absmax.numpy())
        np.testing.assert_array_equal(np.asarray(scales["nested_code"]), state.state2.code.numpy())
        np.testing.assert_allclose(
            np.asarray(scales["nested_absmax"]), state.state2.absmax.numpy(), rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(np.asarray(scales["offset"]), state.offset.numpy(), rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_allclose(np.asarray(scales), state.absmax.numpy(), rtol=1e-6, atol=1e-6)


def _check_nested_codes(source, block_size, actual, expected):
    differing = np.flatnonzero(actual != expected)
    if not differing.size:
        return
    weight = source.float().abs().flatten()
    padded = (-weight.numel()) % block_size
    if padded:
        weight = torch.nn.functional.pad(weight, (0, padded))
    raw = weight.reshape(-1, block_size).amax(dim=1).numpy().astype(np.float64)
    if padded:
        raw[-1] = max(raw[-1], 1e-38)
    exact_offset = raw.mean()
    centered = raw - exact_offset
    padded_scales = (-centered.size) % 256
    nested_scale = np.max(np.abs(np.pad(centered, (0, padded_scales)).reshape(-1, 256)), axis=1)
    position = (centered[differing] / nested_scale[differing // 256] + 1) * 32767.5
    distance_to_boundary = np.abs(position - (np.floor(position) + 0.5))
    # The offset and nested block maximum each use a parallel reduction.
    # Different valid reduction orders can move a lookup by up to two FP32
    # steps when the normalized value is directly beside a code boundary.
    np.testing.assert_array_less(
        distance_to_boundary,
        2 * np.spacing(position.astype(np.float32)),
    )
    np.testing.assert_array_equal(
        np.abs(actual[differing].astype(np.int16) - expected[differing].astype(np.int16)),
        np.ones(differing.size, dtype=np.int16),
    )


def _check_int8(source):
    expected, stats, outliers = bnb.functional.int8_vectorwise_quant(source, threshold=0.0)
    assert outliers is None or outliers.numel() == 0
    weight = mx.array(source.float().numpy()).astype({
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.float32: mx.float32,
    }[source.dtype])
    actual, actual_stats = quantize_int8_weight_mlx(weight)
    actual_codes = np.asarray(actual)
    expected_codes = expected.numpy()
    np.testing.assert_allclose(np.asarray(actual_stats), stats.numpy(), rtol=1e-6, atol=1e-6)
    rows, cols = np.nonzero(actual_codes != expected_codes)
    if rows.size:
        # Both backends multiply a rounded float32 reciprocal. At exact
        # half steps, one ULP can choose a neighboring int8 code.
        value = source.float().numpy()[rows, cols].astype(np.float64)
        maximum = stats.numpy()[rows].astype(np.float64)
        exact = value * 127.0 / maximum
        np.testing.assert_array_equal(exact - np.floor(exact), np.full(rows.size, 0.5))
        np.testing.assert_array_equal(
            np.abs(actual_codes[rows, cols].astype(np.int16) - expected_codes[rows, cols].astype(np.int16)),
            np.ones(rows.size, dtype=np.int16),
        )


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_4bit_small_dtypes_and_zero_blocks(quant_type, compressed, dtype):
    torch.manual_seed(3800)
    source = torch.randn((4, 65), dtype=torch.float32).to(dtype)
    source[0] = 0
    source[0, 0] = -0.0
    _check_4bit(source, quant_type, 32, compressed)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
def test_4bit_all_block_sizes_low_precision(
    block_size, quant_type, compressed, dtype
):
    torch.manual_seed(4100 + block_size)
    source = torch.randn((2, block_size + 1), dtype=torch.float32).to(dtype)
    source[0, :block_size] = 0
    _check_4bit(source, quant_type, block_size, compressed)


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_4bit_all_boundaries_and_neighbors(quant_type):
    code = bnb.functional.get_4bit_type(quant_type, device="cpu")
    sorted_code = code.sort().values
    midpoints = (sorted_code[:-1] + sorted_code[1:]) / 2
    values = torch.cat([
        torch.tensor([1.0, -1.0, 0.0, -0.0]),
        torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
        midpoints,
        torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
    ])
    source = torch.cat([values, torch.zeros(128 - values.numel())]).reshape(2, 64)
    _check_4bit(source, quant_type, 64, False)


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_4bit_subnormal_and_zero_blocks(quant_type):
    source = torch.zeros((4, 32), dtype=torch.float32)
    source[0, :6] = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1.0, -1.0])
    _check_4bit(source, quant_type, 32, False)
    _check_4bit(source, quant_type, 32, True)
    _check_4bit(torch.zeros((1, 33), dtype=torch.float32), quant_type, 32, True)
    tiny = torch.zeros((1, 33), dtype=torch.float32)
    tiny[0, :2] = torch.tensor([1e-40, -1e-40])
    _check_4bit(tiny, quant_type, 32, True)


def test_int8_zero_rows_and_rounding_boundaries():
    source = torch.zeros((4, 128), dtype=torch.float16)
    source[1, :6] = torch.tensor([1.48046875, 2.9609375, -1.48046875, -2.9609375, 0, -0.0])
    source[2, :4] = torch.tensor([1.376953125, 2.75390625, -1.376953125, -2.75390625])
    _check_int8(source)


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
@pytest.mark.parametrize("name,rows,cols", QWEN38_27B_PROJECTIONS)
def test_4bit_qwen38_27b_full_projection(
    quant_type, compressed, dtype, name, rows, cols
):
    del name
    torch.manual_seed(380027 + rows + cols)
    source = torch.randn((rows, cols), dtype=torch.float32).to(dtype)
    _check_4bit(source, quant_type, 64, compressed)
    mx.clear_cache()
    gc.collect()


@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
@pytest.mark.parametrize("name,rows,cols", QWEN38_27B_PROJECTIONS)
def test_int8_qwen38_27b_full_projection(dtype, name, rows, cols):
    del name
    torch.manual_seed(380027 + rows + cols)
    source = torch.randn((rows, cols), dtype=torch.float32).to(dtype)
    _check_int8(source)
    mx.clear_cache()
    gc.collect()


def test_nested_lookup_neighbors_on_mlx_generated_mlp_gate():
    rows, cols = 17408, 5120
    mx.random.seed(380027 + rows + cols)
    weight = mx.random.normal((rows, cols)).astype(mx.bfloat16)
    source = torch.from_numpy(np.asarray(weight.astype(mx.float32))).to(torch.bfloat16)
    expected, state = bnb.functional.quantize_4bit(
        source, quant_type="nf4", blocksize=64, compress_statistics=True,
    )
    actual, scales = quantize_4bit_weight_mlx(weight, quant_type="nf4", compress_statistics=True)
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
    nested_codes = np.asarray(scales["absmax"])
    expected_codes = state.absmax.numpy()
    _check_nested_codes(source, 64, nested_codes, expected_codes)
    assert abs(float(np.asarray(scales["offset"])) - float(state.offset)) <= 1e-6
    mx.clear_cache()


def test_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        quantize_4bit_weight_mlx(mx.zeros((0, 64)))
    with pytest.raises(ValueError, match="block_size"):
        quantize_4bit_weight_mlx(mx.zeros((2, 64)), block_size=12)
    with pytest.raises(ValueError, match="finite"):
        quantize_int8_weight_mlx(mx.array([[float("nan"), 1.0]]))
