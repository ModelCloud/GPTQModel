# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 encoding oracle: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Packed MLX FP8 inference accuracy across formats, scales, and Qwen shapes."""

import gc

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")

FORMATS = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
)
SCALE_LAYOUTS = (("tensor", None), ("row", None), ("block", (16, 32)))
QWEN_CASES = (
    ("float8_e4m3fn", "block"),
    ("float8_e5m2", "tensor"),
    ("float8_e4m3fnuz", "row"),
    ("float8_e5m2fnuz", "block"),
    ("float8_e8m0fnu", "row"),
)


def _source(fmt, method, block_size, out_features=64, in_features=128):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    source = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=True,
        dtype=torch.float16, format=fmt, weight_scale_method=method,
        weight_block_size=block_size,
    )
    if fmt == "float8_e8m0fnu":
        codes = (torch.arange(source.weight.numel(), dtype=torch.int64) % 31 + 111)
        source.weight.copy_(codes.to(torch.uint8).reshape_as(source.weight).view(source.fp8_dtype))
        source.weight_scale_inv.fill_(64)
        source.bias.copy_(torch.linspace(-0.02, 0.02, out_features))
    else:
        torch.manual_seed(800 + len(fmt) + len(method))
        source.pack_original(torch.nn.Linear(in_features, out_features, bias=True).half(), None, None)
    return source


def _rounded_oracle(source, inputs, dtype):
    weight = source.dequantize_weight(device="cpu", dtype=torch.float32).T
    output = torch.from_numpy(np.asarray(inputs.astype(mx.float32))) @ weight.T
    output += source.bias.float()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return output.to(target).float().numpy()


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("method,block_size", SCALE_LAYOUTS)
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
def test_fp8_packed_all_formats_scale_layouts_and_output_dtypes(
    fmt, method, block_size, dtype,
):
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear

    source = _source(fmt, method, block_size)
    payload = FP8MlxQuantLinear.packed_payload(source)
    layer = MlxFP8PackedLinear(**payload)
    inputs = mx.array(
        np.random.default_rng(812).normal(0, 0.1, (2, 3, 128)).astype(np.float32),
    ).astype(dtype)
    actual = layer(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    assert layer.weight.nbytes == source.weight.numel()
    expected = _rounded_oracle(source, inputs, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_fp8_packed_decodes_every_byte(fmt):
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear

    torch_dtype = getattr(torch, fmt)
    codes = np.arange(256, dtype=np.uint8).reshape(256, 1)
    codebook = torch.arange(256, dtype=torch.uint8).view(torch_dtype).float().numpy()
    layer = MlxFP8PackedLinear(
        codes, np.ones(256, dtype=np.float32), codebook,
        in_features=1, out_features=256, scale_method="row",
    )
    actual = layer(mx.ones((1, 1), dtype=mx.float32))
    mx.eval(actual)
    if fmt == "float8_e8m0fnu":
        # Apple Metal flushes the sole float32-subnormal E8M0 value. The old
        # FP16 dense transfer also rounds this value to zero.
        codebook[0] = 0
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), codebook, rtol=0, atol=0, equal_nan=True,
    )


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("rows", (1, 16))
@pytest.mark.parametrize(
    "projection,fmt,method",
    [
        (projection, *QWEN_CASES[index % len(QWEN_CASES)])
        for index, projection in enumerate(QWEN38_27B_PROJECTIONS)
    ],
    ids=[projection[0] for projection in QWEN38_27B_PROJECTIONS],
)
def test_fp8_packed_qwen38_27b_shapes(projection, fmt, method, rows, dtype):
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear

    _, out_features, in_features = projection
    torch_dtype = getattr(torch, fmt)
    max_code = 120 if fmt != "float8_e8m0fnu" else 31
    offset = 0 if fmt != "float8_e8m0fnu" else 111
    base = (np.arange(in_features, dtype=np.uint32) % max_code + offset).astype(np.uint8)
    weight = np.broadcast_to(base, (out_features, in_features))
    codebook = torch.arange(256, dtype=torch.uint8).view(torch_dtype).float().numpy()
    block_size = (128, 128) if method == "block" else None
    if method == "tensor":
        scales = np.array(97, dtype=np.float32)
    elif method == "row":
        scales = 90 + np.arange(out_features, dtype=np.float32) % 31
    else:
        scales = np.full(
            (out_features // 128, in_features // 128), 97, dtype=np.float32,
        )
    layer = MlxFP8PackedLinear(
        weight, scales, codebook,
        in_features=in_features, out_features=out_features,
        scale_method=method, block_size=block_size,
    )
    rng = np.random.default_rng(380027 + in_features + out_features)
    inputs = mx.array(rng.normal(0, 0.01, (rows, in_features)).astype(np.float32)).astype(dtype)
    actual = layer(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    decoded = codebook[base].astype(np.float64)
    host_input = np.asarray(inputs.astype(mx.float32)).astype(np.float64)
    with np.errstate(all="ignore"):
        if method == "block":
            partials = np.stack([
                host_input[:, first:first + 128] @ decoded[first:first + 128]
                for first in range(0, in_features, 128)
            ], axis=1)
            expected = partials @ (1.0 / scales[0].astype(np.float64))
            expected = np.repeat(expected[:, None], out_features, axis=1)
        else:
            dot = host_input @ decoded
            expected = np.broadcast_to(dot[:, None], (rows, out_features)) / scales
    expected = torch.from_numpy(np.array(expected)).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )
    del layer, actual
    mx.clear_cache()
    gc.collect()


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize(
    "projection",
    [
        projection for projection in QWEN38_27B_PROJECTIONS
        if projection[2] <= 8192 and projection[1] >= 8192
    ],
    ids=[
        projection[0] for projection in QWEN38_27B_PROJECTIONS
        if projection[2] <= 8192 and projection[1] >= 8192
    ],
)
def test_fp8_wide_prefill_tile_is_exact_to_main(projection, dtype):
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear

    _, out_features, in_features = projection
    base = (np.arange(in_features, dtype=np.uint32) % 120).astype(np.uint8)
    weight = np.broadcast_to(base, (out_features, in_features))
    codebook = torch.arange(256, dtype=torch.uint8).view(torch.float8_e5m2).float().numpy()
    scales = 600000 + np.arange(out_features, dtype=np.float32) % 10000
    layer = MlxFP8PackedLinear(
        weight, scales, codebook,
        in_features=in_features, out_features=out_features, scale_method="row",
    )
    inputs = mx.array(
        np.random.default_rng(3816 + in_features + out_features).normal(
            0, 0.01, (16, in_features),
        ).astype(np.float32),
    ).astype(dtype)
    main = (layer._prefill(inputs, 16, tile_columns=16) + layer.bias).astype(dtype)
    actual = layer(inputs)
    mx.eval(main, actual)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(main.astype(mx.float32)),
    )
