# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Check packed BitsAndBytes MLX kernels on Qwen3.8-27B projection shapes."""

import gc
import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import (  # noqa: E402
    MlxBitsAndBytesLinear,
    _four_bit_kernel,
    repack_int8_affine,
)


_CODEBOOKS = {
    "nf4": np.array([
        -1.0, -0.6961928, -0.52507305, -0.3949175,
        -0.28444138, -0.18477343, -0.09105003625154495, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.33791524,
        0.44070983, 0.562617, 0.72295684, 1.0,
    ], dtype=np.float32),
    "fp4": np.array([
        0.0, 0.0052083335, 0.6666667, 1.0,
        0.33333334, 0.5, 0.16666667, 0.25,
        0.0, -0.0052083335, -0.6666667, -1.0,
        -0.33333334, -0.5, -0.16666667, -0.25,
    ], dtype=np.float32),
}


def _weights(format_name, out_features, in_features, block_size=64):
    scale = np.float32(0.01875)
    if format_name in ("nf4", "fp4"):
        codebook = _CODEBOOKS[format_name]
        codes = np.arange(in_features, dtype=np.uint8) & np.uint8(15)
        packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
        packed = np.tile(packed_row, out_features)
        scales = np.full(
            out_features * ((in_features + block_size - 1) // block_size),
            scale,
            dtype=np.float32,
        )
        row_weight = (codebook[codes] * scale).astype(np.float16)
        bits = 4
    else:
        codebook = None
        codes = ((np.arange(in_features, dtype=np.int32) % 255) - 127).astype(np.int8)
        packed = np.tile(codes, out_features)
        scales = np.full(out_features, scale, dtype=np.float32)
        row_weight = (codes.astype(np.float32) * (scale / 127)).astype(np.float16)
        bits = 8
    dense = np.tile(row_weight, (out_features, 1))
    payload = {}
    if bits == 8:
        packed, scales, payload["affine_biases"], payload["block_size"] = repack_int8_affine(
            packed, scales, in_features, out_features,
        )
    return packed, scales, codebook, dense, row_weight, bits, payload


def _four_bit_unrounded(layer, x):
    rows = x.size // layer.in_features
    row_tile = 1 if rows == 1 else min(8, rows)
    small_decode = (
        rows == 1 and layer.out_features <= 2048 and layer.in_features >= 4096
    )
    if small_decode and x.dtype == mx.float16:
        threads = 128
    elif small_decode and x.dtype == mx.bfloat16:
        threads = 256
    else:
        threads = 32
    return _four_bit_kernel()(
        inputs=[x, layer.weight, layer.scales, layer.codebook, layer.bias],
        template=[
            ("K", layer.in_features), ("N", layer.out_features),
            ("BLOCK", layer.block_size), ("ROWS", rows),
            ("RTILE", row_tile), ("THREADS", threads),
            ("EVEN", layer.in_features % 2 == 0), ("GROUPS", threads // 32),
        ],
        grid=(threads, layer.out_features, (rows + row_tile - 1) // row_tile),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, layer.out_features)],
        output_dtypes=[mx.float32],
    )[0]


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("format_name", ("nf4", "fp4"))
def test_bnb_four_code_decode_codebook_and_block_boundaries(format_name, dtype):
    out_features, in_features, block_size = 5, 128, 32
    codebook = _CODEBOOKS[format_name]
    codes = np.resize(
        np.array([0, 15, 1, 14, 7, 8, 6, 9], dtype=np.uint8),
        in_features,
    )
    packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
    packed = np.tile(packed_row, out_features)
    scales = np.linspace(
        0.006, 0.024, out_features * (in_features // block_size),
        dtype=np.float32,
    ).reshape(out_features, -1)
    bias = np.linspace(-0.003, 0.003, out_features, dtype=np.float32).astype(
        np.float16
    )
    layer = MlxBitsAndBytesLinear(
        packed,
        scales,
        in_features=in_features,
        out_features=out_features,
        bits=4,
        block_size=block_size,
        codebook=codebook,
        bias=bias,
    )
    boundary = np.array(
        [-1.0, np.nextafter(-1.0, 0.0), -0.0, 0.0,
         np.nextafter(1.0, 0.0), 1.0, -0.5, 0.5],
        dtype=np.float32,
    )
    x = mx.array(np.resize(boundary, (2, in_features))).astype(dtype)
    internal = _four_bit_unrounded(layer, x)
    actual = layer(x)
    main_rounded = internal.astype(dtype)
    mx.eval(internal, actual, main_rounded)

    dense = codebook[codes][None, :] * np.repeat(scales, block_size, axis=1)
    dense = dense.astype(np.float16).astype(np.float64)
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    raw = (torch_input @ torch.from_numpy(dense).double().T).numpy()
    raw += bias.astype(np.float64)[None, :]
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    rounded = torch.from_numpy(raw).to(target).float().numpy()

    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(main_rounded.astype(mx.float32)),
    )
    np.testing.assert_allclose(np.asarray(internal), raw, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), rounded, rtol=2e-3, atol=2e-3,
    )


@pytest.mark.parametrize("block_size", (32, 64, 128, 256, 512, 1024, 2048, 4096))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("format_name", ("nf4", "fp4"))
def test_bnb_small_decode_all_4bit_block_sizes(format_name, dtype, block_size):
    out_features, in_features = 96, 4096
    packed, scales, codebook, _, row_weight, bits, _ = _weights(
        format_name, out_features, in_features, block_size,
    )
    layer = MlxBitsAndBytesLinear(
        packed, scales, in_features=in_features, out_features=out_features,
        bits=bits, block_size=block_size, codebook=codebook,
    )
    positions = np.arange(in_features, dtype=np.float32)
    source = (np.sin(positions * 0.013) * 0.08)[None]
    x = mx.array(source).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    raw = np.sum(
        np.asarray(x.astype(mx.float32)).astype(np.float64)
        * row_weight.astype(np.float64)[None, :],
        axis=1,
    )
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    expected = torch.from_numpy(np.repeat(raw[:, None], out_features, axis=1)).to(
        torch_dtype,
    ).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    assert actual.dtype == dtype
    np.testing.assert_allclose(visible, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("format_name", ("nf4", "fp4", "int8"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_bnb_qwen38_native_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, format_name, dtype, record_property,
):
    packed, scales, codebook, dense, row_weight, bits, payload = _weights(
        format_name, out_features, in_features,
    )
    bias = np.linspace(-0.002, 0.002, out_features, dtype=np.float32).astype(np.float16)
    native = MlxBitsAndBytesLinear(
        packed, scales, in_features=in_features, out_features=out_features,
        bits=bits, block_size=payload.get("block_size", 64), codebook=codebook,
        affine_biases=payload.get("affine_biases"), bias=bias,
    )
    main = nn.Linear(in_features, out_features, bias=True)
    main.weight = mx.array(dense)
    main.bias = mx.array(bias)
    del dense

    rng = np.random.default_rng(sum(name.encode()) + out_features + in_features + bits)
    x = mx.array(rng.normal(0, 0.15, (1, in_features)).astype(np.float32)).astype(dtype)
    # Merged main decodes to FP16 dense weights, then preserves activation dtype.
    main_output = main(x).astype(dtype)
    actual = native(x)
    internal = _four_bit_unrounded(native, x) if bits == 4 else None
    main_rounded = internal.astype(dtype) if internal is not None else None
    mx.eval(
        main_output, actual,
        *(() if internal is None else (internal, main_rounded)),
    )
    assert actual.dtype == dtype

    input_values = np.asarray(x.astype(mx.float32)).astype(np.float64)
    raw_row = np.sum(input_values * row_weight.astype(np.float64)[None, :], axis=1)
    raw_oracle = raw_row[:, None] + bias.astype(np.float64)[None, :]
    rounded_oracle = torch.from_numpy(raw_oracle).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    main_values = np.asarray(main_output.astype(mx.float32))
    visible = np.asarray(actual.astype(mx.float32))
    record_property("main_output_dtype", str(main_output.dtype))
    record_property("max_abs_main_vs_rounded_torch", float(np.max(np.abs(main_values - rounded_oracle))))
    record_property("max_abs_main_vs_fp64", float(np.max(np.abs(main_values - raw_oracle))))
    record_property("max_abs_native_vs_main", float(np.max(np.abs(visible - main_values))))
    record_property("max_abs_native_vs_rounded_torch", float(np.max(np.abs(visible - rounded_oracle))))
    record_property("max_abs_native_vs_fp64", float(np.max(np.abs(visible - raw_oracle))))
    if internal is not None:
        internal_values = np.asarray(internal)
        record_property(
            "max_abs_internal_fp32",
            float(np.max(np.abs(internal_values - raw_oracle))),
        )
        np.testing.assert_allclose(
            internal_values, raw_oracle, rtol=2e-3, atol=2e-3,
        )
        np.testing.assert_array_equal(
            visible, np.asarray(main_rounded.astype(mx.float32)),
        )
    np.testing.assert_allclose(main_values, rounded_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded_oracle, rtol=2e-3, atol=2e-3)
    del main, native, main_output, actual, packed
    mx.clear_cache()
    gc.collect()
