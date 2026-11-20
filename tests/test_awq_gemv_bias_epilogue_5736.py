# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for issue #5736.

The native AWQ GEMV/GEMV_FAST paths rounded their product to FP16 before the
bias add, and ``gemmv2`` stored every split-K partial in FP16.  Both roundings
can turn a finite contract result into ``inf`` (or ``nan`` after a split-K
cancellation).  These tests drive real packed layers and the raw ``gemmv2``
entry point on CUDA.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, pack_intweight
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq_columns
from gptqmodel.utils.awq import awq_gemmv2_forward, awq_runtime_available
from gptqmodel.utils.gemv import calculate_zeros_width


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AWQ GEMV kernels require CUDA"
)

GROUP_SIZE = 128
IN_FEATURES = 128
OUT_FEATURES = 128
# 128 channels * weight 15 * activation 52 = 99,840, above the FP16 max 65,504.
PRE_BIAS_SUM = 99_840.0
BIAS = -50_000.0
# -50,000 rounds to -49,984 in FP16, so the contract sum is 49,856.
EXPECTED = 49_856.0


def _require_runtime():
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA JIT extension entrypoint unavailable")


def _gemv_buffers(in_features, out_features, codes, scales, zeros):
    qweight = pack_awq_columns(codes, 4)
    zeros_width = calculate_zeros_width(in_features, GROUP_SIZE, 8)
    qscales = torch.zeros((out_features, zeros_width * 8), dtype=torch.float16)
    qscales[:, : scales.shape[1]] = scales
    padded_zeros = torch.zeros((out_features, zeros_width * 8), dtype=torch.int32)
    padded_zeros[:, : zeros.shape[1]] = zeros
    qzeros = pack_awq_columns(padded_zeros, 4)
    return qweight, qscales, qzeros


def _gemv_module():
    module = AwqGEMVLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
    ).cuda()
    codes = torch.full((OUT_FEATURES, IN_FEATURES), 15, dtype=torch.int32)
    zeros = torch.zeros((OUT_FEATURES, 1), dtype=torch.int32)
    scales = torch.ones((OUT_FEATURES, 1), dtype=torch.float16)
    qweight, qscales, qzeros = _gemv_buffers(
        IN_FEATURES, OUT_FEATURES, codes, scales, zeros
    )
    module.qweight = qweight.cuda()
    module.scales = qscales.cuda()
    module.qzeros = qzeros.cuda()
    module.bias = torch.full((OUT_FEATURES,), BIAS, dtype=torch.float16, device="cuda")
    return module


def _gemv_fast_module():
    module = AwqGEMVFastLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
    ).cuda()
    codes = torch.full((OUT_FEATURES, IN_FEATURES), 15, dtype=torch.int32)
    zeros = torch.zeros((1, OUT_FEATURES), dtype=torch.int32)
    scales = torch.ones((1, OUT_FEATURES), dtype=torch.float16)
    offsets = (-scales.float() * zeros.float()).half()
    zeros_width = calculate_zeros_width(IN_FEATURES, GROUP_SIZE, 8)
    qscales = torch.zeros((OUT_FEATURES, zeros_width * 8), dtype=torch.float16)
    qscales[:, :1] = scales.T
    qzeros = torch.zeros((OUT_FEATURES, zeros_width * 8), dtype=torch.float16)
    qzeros[:, :1] = offsets.T
    module.qweight = pack_intweight(codes, interleave=4, kstride=64).cuda()
    module.scales = qscales.T.contiguous().cuda()
    module.qzeros = qzeros.T.contiguous().cuda()
    module.bias = torch.full((OUT_FEATURES,), BIAS, dtype=torch.float16, device="cuda")
    return module


def _contract_oracle(rows):
    device = "cuda"
    pre_bias = torch.full(
        (rows, OUT_FEATURES), PRE_BIAS_SUM, dtype=torch.float64, device=device
    )
    bias = (
        torch.full((OUT_FEATURES,), BIAS, dtype=torch.float64, device=device)
        .half()
        .double()
    )
    return (pre_bias + bias.unsqueeze(0)).to(torch.float16)


@CUDA_REQUIRED
def test_gemv_bias_joins_the_fp32_reduction():
    _require_runtime()
    module = _gemv_module()
    bias_before = module.bias.clone()

    out = module(torch.full((1, IN_FEATURES), 52.0, dtype=torch.float16, device="cuda"))

    assert torch.isfinite(out).all()
    assert torch.equal(out, _contract_oracle(1))
    assert torch.equal(module.bias, bias_before)


@CUDA_REQUIRED
def test_gemv_fast_decode_bias_joins_the_fp32_reduction():
    _require_runtime()
    module = _gemv_fast_module()
    bias_before = module.bias.clone()

    out = module(
        torch.full((1, 1, IN_FEATURES), 52.0, dtype=torch.float16, device="cuda")
    )

    assert out.shape == (1, 1, OUT_FEATURES)
    assert torch.isfinite(out).all()
    assert torch.equal(out.reshape(1, OUT_FEATURES), _contract_oracle(1))
    assert torch.equal(module.bias, bias_before)


@CUDA_REQUIRED
def test_gemv_fast_prefill_bias_joins_the_fp32_reduction():
    _require_runtime()
    module = _gemv_fast_module()

    out = module(
        torch.full((1, 2, IN_FEATURES), 52.0, dtype=torch.float16, device="cuda")
    )

    assert out.shape == (1, 2, OUT_FEATURES)
    assert torch.isfinite(out).all()
    assert torch.equal(out.reshape(2, OUT_FEATURES), _contract_oracle(2))


@CUDA_REQUIRED
def test_gemv_split_k_module_bias_joins_the_fp32_reduction():
    _require_runtime()
    module = _gemv_module()

    out = module(
        torch.full((16, IN_FEATURES), 52.0, dtype=torch.float16, device="cuda")
    )

    assert out.shape == (16, OUT_FEATURES)
    assert torch.isfinite(out).all()
    assert torch.equal(out, _contract_oracle(16))


@CUDA_REQUIRED
def test_gemv_fast_large_prefill_bias_joins_the_fp32_reduction():
    _require_runtime()
    module = _gemv_fast_module()

    out = module(
        torch.full((1, 256, IN_FEATURES), 52.0, dtype=torch.float16, device="cuda")
    )

    assert out.shape == (1, 256, OUT_FEATURES)
    assert torch.isfinite(out).all()
    assert torch.equal(out.reshape(256, OUT_FEATURES), _contract_oracle(256))


@CUDA_REQUIRED
def test_gemmv2_keeps_split_k_partials_in_fp32():
    _require_runtime()
    in_features = 256
    out_features = 128
    codes = torch.full((out_features, in_features), 15, dtype=torch.int32)
    codes[:, in_features // 2 :] = 0
    zeros = torch.zeros((out_features, 2), dtype=torch.int32)
    zeros[:, 1] = 15
    scales = torch.ones((out_features, 2), dtype=torch.float16)
    qweight, qscales, qzeros = _gemv_buffers(
        in_features, out_features, codes, scales, zeros
    )

    # Each 32-column split-K partial is +-144,000, above the FP16 max, while
    # the eight partials cancel to exactly zero.
    x = torch.full((16, in_features), 300.0, dtype=torch.float16, device="cuda")
    half_out = awq_gemmv2_forward(
        x, qweight.cuda(), qscales.cuda(), qzeros.cuda(), GROUP_SIZE, 8
    )

    assert half_out.dtype == torch.float16
    assert torch.isfinite(half_out).all()
    assert torch.count_nonzero(half_out) == 0

    wide_out = awq_gemmv2_forward(
        x,
        qweight.cuda(),
        qscales.cuda(),
        qzeros.cuda(),
        GROUP_SIZE,
        8,
        fp32_output=True,
    )

    assert wide_out.dtype == torch.float32
    assert torch.count_nonzero(wide_out) == 0
