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
bnb = pytest.importorskip("bitsandbytes")

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import (  # noqa: E402
    MlxBitsAndBytesLinear,
    repack_int8_affine,
)


def _weights(format_name, out_features, in_features):
    scale = np.float32(0.01875)
    if format_name in ("nf4", "fp4"):
        codebook = bnb.functional.get_4bit_type(format_name, device="cpu").numpy().astype(np.float32)
        codes = np.arange(in_features, dtype=np.uint8) & np.uint8(15)
        packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
        packed = np.tile(packed_row, out_features)
        scales = np.full(out_features * in_features // 64, scale, dtype=np.float32)
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
    x = mx.array(rng.normal(0, 0.15, (3, in_features)).astype(np.float32)).astype(dtype)
    # Merged main decodes to FP16 dense weights, then preserves activation dtype.
    main_output = main(x).astype(dtype)
    actual = native(x)
    mx.eval(main_output, actual)
    assert actual.dtype == dtype

    input_values = np.asarray(x.astype(mx.float32)).astype(np.float64)
    raw_row = input_values @ row_weight.astype(np.float64)
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
    np.testing.assert_allclose(main_values, rounded_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded_oracle, rtol=2e-3, atol=2e-3)
    del main, native, main_output, actual, packed
    mx.clear_cache()
    gc.collect()
