# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Block-float forwards must not narrow weights or the stored FP32 bias early.

Regression for #3863.  The linear fallback dequantised the packed grid and the
stored float32 bias to the activation dtype before the reduction, and the
native bindings rounded the bias to the activation dtype before their float32
epilogue.  A finite result then came back as ``0.0`` for the bf16 cancellation
case or ``-inf`` for the fp16 range case even though a float32 accumulation
over the same grid is exact.  The tied output head had the same early weight
narrowing without a bias.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tt_blockfloat import (
    TTBlockfloatEmbedding,
    TTBlockfloatLinear,
    TTBlockfloatTiedHead,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_blockfloat_quant import decompose_blockfloat


def _packed(weight: torch.Tensor, bias: torch.Tensor | None, *, bits: int = 8) -> TTBlockfloatLinear:
    source = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        source.weight.copy_(weight)
        if bias is not None:
            source.bias.copy_(bias)
    return TTBlockfloatLinear(
        bits=bits,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=bias is not None,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.float32,
    ).pack(source)


def _oracle(layer: TTBlockfloatLinear, x: torch.Tensor) -> torch.Tensor:
    """Independent float64 GEMM over the exact grid plus the stored FP32 bias."""

    grid = layer.dequantize(dtype=torch.float32).to(torch.float64)
    out = x.to(torch.float64) @ grid.T
    if layer.bias is not None:
        out = out + layer.bias.to(torch.float64)
    return out.to(x.dtype)


def _bf16_cancellation():
    weight = torch.zeros(1, 16)
    weight[0, 0] = 1024.0
    bias = torch.tensor([-1023.5])
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[0, 0] = 1
    return weight, bias, x


def _fp16_range():
    weight = torch.full((1, 16), 4096.0)
    bias = torch.tensor([-65536.0])
    x = torch.ones(1, 16, dtype=torch.float16)
    return weight, bias, x


def _cuda_kernel_available() -> bool:
    from tests.accelerator_gates import cuda_only_skip_reason

    if cuda_only_skip_reason("block-float") is not None:
        return False
    if not torch.cuda.is_available():
        return False
    from gptqmodel.utils import tt_blockfloat_cuda as bf_cuda

    return bf_cuda.supported()


def _hip_kernel_available() -> bool:
    from gptqmodel.utils.tt_blockfloat_hip import tt_blockfloat_hip_supported

    return tt_blockfloat_hip_supported()


@pytest.mark.parametrize("case", [_bf16_cancellation, _fp16_range], ids=["bf16-cancellation", "fp16-range"])
def test_dense_fallback_accumulates_grid_and_bias_in_float32(case):
    weight, bias, x = case()
    layer = _packed(weight, bias)

    # CPU has no block-float kernel, so this is the dense fallback.
    assert layer._kernel_linear(x, layer.bias) is None

    got = layer(x)
    expected = _oracle(layer, x)

    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)
    # The fallback releases its temporary dense weight.
    assert "weight" not in vars(layer)


def test_tied_head_fallback_does_not_narrow_the_shared_grid():
    grid = torch.zeros(1, 16)
    grid[0, 0] = 65536.0
    grid[0, 1] = -65536.0

    embedding = TTBlockfloatEmbedding(1, 16, bits=8, dtype=torch.float16)
    embedding.set_quantized_state(*decompose_blockfloat(grid, 8))
    head = TTBlockfloatTiedHead(embedding)

    hidden = torch.zeros(1, 16, dtype=torch.float16)
    hidden[0, 0] = 1
    hidden[0, 1] = 1

    got = head(hidden)
    expected = (hidden.float() @ grid.T).to(torch.float16)
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)


@pytest.mark.skipif(not _cuda_kernel_available(), reason="block-float CUDA kernels need sm_80+")
@pytest.mark.parametrize("case", [_bf16_cancellation, _fp16_range], ids=["bf16-cancellation", "fp16-range"])
def test_cuda_kernel_adds_the_float32_bias_before_the_final_cast(case):
    weight, bias, x = case()
    layer = _packed(weight, bias).cuda()
    x = x.cuda()

    assert layer._kernel_linear(x, layer.bias) is not None

    got = layer(x)
    expected = _oracle(layer, x)

    assert got.dtype is x.dtype
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)


@pytest.mark.skipif(not _hip_kernel_available(), reason="block-float HIP kernels need a ROCm GPU")
@pytest.mark.parametrize("case", [_bf16_cancellation, _fp16_range], ids=["bf16-cancellation", "fp16-range"])
def test_hip_kernel_adds_the_float32_bias_before_the_final_cast(case):
    weight, bias, x = case()
    layer = _packed(weight, bias).cuda()
    x = x.cuda()

    assert layer._kernel_linear(x, layer.bias) is not None

    got = layer(x)
    expected = _oracle(layer, x)

    assert got.dtype is x.dtype
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)
