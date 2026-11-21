# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CUDA forwards over out-axis Tenstorrent block-float checkpoint bytes."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding, TTBlockfloatLinear, TTBlockfloatTiedHead
from gptqmodel.utils import tt_blockfloat_cuda
from gptqmodel.utils.backend import BACKEND


def _layer(bits, signed, exponents, dtype, device, bias=None):
    k, n = signed.shape
    offset = (1 << (bits - 1)) - 1
    codes = (signed + offset).to(torch.uint8)
    if bits == 4:
        codes = codes[:, ::2] | (codes[:, 1::2] << 4)
    layer = TTBlockfloatLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        bias=bias is not None,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=dtype,
        bias_dtype=torch.float32,
        block_axis="out",
    )
    layer.set_packed_state(codes.flatten(), exponents)
    if bias is not None:
        layer.bias.copy_(bias)
    # Compute W^T independently of GPT-QModel's pack, unpack and recompose paths.
    powers = exponents.to(torch.int32).repeat_interleave(16, dim=1) - (bits - 2)
    grid = torch.ldexp(signed.double(), powers)
    return layer.to(device), grid.to(device)


@pytest.fixture
def device():
    pytest.importorskip("triton")
    if not tt_blockfloat_cuda.supported():
        pytest.skip("requires NVIDIA CUDA sm_80 or newer")
    return torch.device("cuda", torch.cuda.current_device())


def _reject_materialization(*_args, **_kwargs):
    raise AssertionError("eligible out-axis forward materialized a dense weight")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,k", [(1, 272), (3, 272), (7, 784), (36, 272)])
def test_out_axis_linear_uses_packed_storage(device, monkeypatch, bits, dtype, m, k):
    n = 48
    generator = torch.Generator().manual_seed(43)
    offset = (1 << (bits - 1)) - 1
    signed = torch.randint(-offset, offset + 1, (k, n), generator=generator)
    exponents = torch.randint(-8, 0, (k, n // 16), generator=generator, dtype=torch.int8)
    bias = torch.linspace(-0.7, 0.4, n) if m % 2 else None
    layer, grid = _layer(bits, signed, exponents, dtype, device, bias)
    operands = torch.randn(m, k * 2, generator=generator).to(device=device, dtype=dtype) / 8
    x = operands[:, ::2]
    assert x.stride(-1) == 2
    state_before = {name: (value.data_ptr(), value.clone()) for name, value in layer.named_buffers()}
    monkeypatch.setattr(layer, "dequantize", _reject_materialization)

    with torch.inference_mode():
        for actual_x in (x, (-x).unsqueeze(0)):
            expected = actual_x.double() @ grid
            if bias is not None:
                expected += bias.double().to(device)
            actual = layer(actual_x)
            assert actual.dtype == dtype
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-3, atol=2e-3)

    assert set(state_before) == set(dict(layer.named_buffers()))
    for name, value in layer.named_buffers():
        pointer, saved = state_before[name]
        assert pointer == value.data_ptr()
        assert torch.equal(value, saved)
    assert not list(layer.parameters())


@pytest.mark.parametrize("m", [1, 9])
def test_out_axis_float16_range_and_wide_bias(device, monkeypatch, m):
    signed = torch.zeros(32, 16, dtype=torch.int32)
    signed[0, :] = 64
    exponents = torch.zeros(32, 1, dtype=torch.int8)
    exponents[0, :] = 16
    bias = -65536 + torch.arange(16, dtype=torch.float32)
    layer, _ = _layer(8, signed, exponents, torch.float16, device, bias)
    x = torch.zeros(m, 32, dtype=torch.float16, device=device)
    x[:, 0] = 1
    monkeypatch.setattr(layer, "dequantize", _reject_materialization)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = layer(x)
    assert torch.equal(actual, torch.arange(16, dtype=x.dtype, device=device).expand(m, -1))


@pytest.mark.parametrize("m", [1, 9])
def test_out_axis_preserves_subnormal_grid_values(device, monkeypatch, m):
    signed = torch.ones(32, 16, dtype=torch.int32)
    exponents = torch.full((32, 1), -127, dtype=torch.int8)
    layer, _ = _layer(8, signed, exponents, torch.float32, device)
    x = torch.zeros(m, 32, dtype=torch.float32, device=device)
    x[:, 0] = 2.0**126
    monkeypatch.setattr(layer, "dequantize", _reject_materialization)
    with torch.inference_mode():
        actual = layer(x)
    # The FP32 subnormal 2^-133 contributes 1/128 to each output.
    assert torch.equal(actual, torch.full_like(actual, 1.0 / 128.0))


def test_out_axis_activation_gradient_uses_portable_path(device):
    signed = torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16) % 15 - 7
    exponents = torch.full((32, 1), -2, dtype=torch.int8)
    layer, grid = _layer(4, signed, exponents, torch.float32, device)
    x = torch.randn(4, 32, device=device, requires_grad=True)
    residual = torch.linspace(-1, 1, 16, device=device)
    layer(x).backward(residual.expand(4, -1))
    expected = residual.double() @ grid.T
    torch.testing.assert_close(x.grad, expected.float().expand(4, -1))


def test_out_axis_tied_head_reads_the_same_packed_embedding(device, monkeypatch):
    embedding = TTBlockfloatEmbedding(16, 32, bits=4, dtype=torch.float16, block_axis="out").to(device)
    # Stored out-axis rows 0 and 1 are +65536 and -65536 respectively.
    codes = torch.full_like(embedding.codes.reshape(32, 8), 0x77)
    codes[0, :] = 0xBB
    codes[1, :] = 0x33
    exponent = torch.zeros_like(embedding.exponents)
    exponent[:2] = 16
    embedding.set_packed_state(codes.flatten(), exponent)
    head = TTBlockfloatTiedHead(embedding)
    monkeypatch.setattr(embedding, "dequantize", _reject_materialization)

    hidden = torch.ones(9, 32, dtype=torch.float16, device=device)
    with torch.inference_mode():
        assert torch.equal(head(hidden), torch.zeros(9, 16, dtype=hidden.dtype, device=device))
    assert not head.state_dict()
