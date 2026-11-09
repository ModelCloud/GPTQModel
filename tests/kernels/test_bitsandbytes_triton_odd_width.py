# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Packed 4-bit rows with odd K start on alternating byte nibbles."""

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("triton")

from gptqmodel.nn_modules.qlinear import bitsandbytes_triton as bt


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [(128, 64), (129, 64), (130, 64), (131, 64), (129, 65)],
)
@pytest.mark.parametrize("rows_shape", [(16,), (2, 8)])
def test_dequant_and_fused_matmul_follow_global_nibble_parity(
    dtype, in_features, out_features, rows_shape
):
    element_count = out_features * in_features
    indices = torch.arange(element_count, device="cuda", dtype=torch.int64)
    codes = ((indices * 7 + 3) % 16).to(torch.uint8)
    packed = torch.zeros((element_count + 1) // 2, device="cuda", dtype=torch.uint8)
    packed[:] = codes[0::2] << 4
    packed[: codes[1::2].numel()] |= codes[1::2]
    codebook = torch.arange(16, device="cuda", dtype=torch.float32) - 8
    quant_state = SimpleNamespace(
        shape=(out_features, in_features),
        dtype=dtype,
        absmax=torch.ones((element_count + 63) // 64, device="cuda"),
        code=codebook,
        blocksize=64,
    )
    bias = (torch.arange(out_features, device="cuda") % 3 - 1).to(dtype)
    inputs = torch.ones(rows_shape + (in_features,), device="cuda", dtype=dtype)

    # Decode the flat packed stream by global element index. This oracle does
    # not call either Triton function or reuse their reconstructed values.
    bytes_for_elements = packed[indices // 2]
    decoded_codes = torch.where(
        indices % 2 == 0, (bytes_for_elements >> 4) & 0xF, bytes_for_elements & 0xF
    ).long()
    dense = codebook[decoded_codes].reshape(out_features, in_features)
    reference = (
        inputs.reshape(-1, in_features).to(torch.float64) @ dense.T.to(torch.float64)
        + bias.to(torch.float64)
    ).reshape(rows_shape + (out_features,)).to(dtype)

    dequantized = bt.dequantize_4bit(packed, quant_state, out_dtype=torch.float32)
    actual = bt.matmul_4bit(inputs, packed, quant_state, bias=bias, out_dtype=dtype)
    repeated = bt.matmul_4bit(inputs, packed, quant_state, bias=bias, out_dtype=dtype)

    assert dequantized.shape == dense.shape
    torch.testing.assert_close(dequantized, dense, rtol=0, atol=0)
    assert actual.shape == reference.shape
    assert actual.dtype == dtype
    assert torch.equal(actual, repeated)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["padded_rows", "column_stride"])
def test_fused_matmul_accepts_strided_activation_views(dtype, layout):
    rows, in_features, out_features = 16, 128, 64
    generator = torch.Generator(device="cuda").manual_seed(472)
    codes = torch.randint(16, (out_features * in_features,), device="cuda", generator=generator, dtype=torch.uint8)
    packed = (codes[0::2] << 4) | codes[1::2]
    codebook = torch.arange(16, device="cuda", dtype=torch.float32) - 8
    quant_state = SimpleNamespace(
        shape=(out_features, in_features),
        dtype=dtype,
        absmax=torch.ones(out_features * in_features // 64, device="cuda"),
        code=codebook,
        blocksize=64,
    )
    storage_width = in_features + 8 if layout == "padded_rows" else in_features * 2
    storage = torch.randint(0, 3, (rows, storage_width), device="cuda", generator=generator).to(dtype) - 1
    x = storage[:, :in_features] if layout == "padded_rows" else storage[:, ::2]
    assert not x.is_contiguous()
    bias = (torch.arange(out_features, device="cuda") % 3 - 1).to(dtype)

    dense = codebook[codes.long()].reshape(out_features, in_features)
    reference = (x.double() @ dense.T.double() + bias.double()).to(dtype)
    actual = bt.matmul_4bit(x, packed, quant_state, bias=bias, out_dtype=dtype)
    torch.cuda.synchronize()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operand", ["qweight", "codebook", "absmax", "bias"])
def test_fused_matmul_and_dequant_accept_strided_packed_operands(dtype, operand):
    rows, in_features, out_features, blocksize = 16, 128, 64, 64
    count = out_features * in_features
    indices = torch.arange(count, device="cuda")
    codes = ((indices * 7 + 3) % 16).to(torch.uint8)
    packed = (codes[0::2] << 4) | codes[1::2]
    codebook = torch.arange(16, device="cuda", dtype=torch.float32) - 8
    absmax = torch.tensor([0.5, 1.0, 2.0], device="cuda")[
        torch.arange(count // blocksize, device="cuda") % 3
    ]
    bias = (torch.arange(out_features, device="cuda") % 5 - 2).to(dtype)
    x = ((torch.arange(rows * in_features, device="cuda").reshape(rows, in_features) * 3) % 5 - 2).to(dtype)
    arguments = {"qweight": packed, "codebook": codebook, "absmax": absmax, "bias": bias}
    source = arguments[operand]
    storage = torch.empty((source.numel() * 2,), device="cuda", dtype=source.dtype)
    view = storage[::2]
    view.copy_(source)
    assert not view.is_contiguous()
    arguments[operand] = view
    quant_state = SimpleNamespace(
        shape=(out_features, in_features), dtype=dtype,
        absmax=arguments["absmax"], code=arguments["codebook"], blocksize=blocksize,
    )

    dense = (codebook[codes.long()] * absmax[indices // blocksize]).reshape(out_features, in_features)
    reference = (x.double() @ dense.T.double() + bias.double()).to(dtype)
    dequantized = bt.dequantize_4bit(arguments["qweight"], quant_state, out_dtype=torch.float32)
    actual = bt.matmul_4bit(x, arguments["qweight"], quant_state, bias=arguments["bias"], out_dtype=dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(dequantized, dense, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
