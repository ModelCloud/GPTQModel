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
