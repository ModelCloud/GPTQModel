# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Marlin OCP microscaling (E8M0 scales, group 32): MXFP8 and MXFP4.

Two defects kept both formats from working, even though their kernels are
instantiated and dispatched:

* ``gptq_marlin.cu`` gated every ``float8_e8m0fnu`` path on
  ``__has_include(<ATen/native/quantized/Float8_e8m0fnu.h>)``, a header that
  does not exist; the type lives in ``c10/util``. E8M0 scales were therefore
  always rejected (bf16: ``expected scalar type BFloat16 but found
  Float8_e8m0fnu``), so neither MXFP8 nor MXFP4 could run.
* ``marlin_template.h`` reads 8-bit scales in a paired layout: one 16-bit slot
  holds two adjacent groups' scales, read as ``int2``. It applied that layout
  only when the *weight* was FP4 (``w_type == kFE2M1f``). MXFP8 (FP8 weights,
  E8M0 scales) used the 16-bit strides and ``int4`` reads over a 1-byte scale
  tensor, so every group read the wrong scales and outputs were off by about
  100%. vLLM keys these sites on ``is_8bit_scale`` (vllm-project/vllm#34664).

The tests check both formats on the real kernel against an FP64 oracle, check
that fp16 compute (which has no E8M0 kernels) is rejected with a clear error,
and check that the host-side scale interleave is a pure permutation.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils.marlin import (
    MXFP8_GROUP_SIZE,
    gptq_marlin_gemm,
    marlin_import_exception,
    marlin_interleave_8bit_scales,
    marlin_make_workspace_new,
    marlin_runtime_available,
    marlin_runtime_error,
    mxfp4_marlin_prepare,
    mxfp8_marlin_prepare,
)
from gptqmodel.utils.marlin_scalar_type import scalar_types


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
_SHAPES = [(2048, 2048), (4096, 1024), (1024, 4096), (256, 512)]
_M = [1, 7, 16, 64, 128, 512, 2048]


def _skip_reason(dtype: torch.dtype) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the Marlin microscaling kernels"
    if marlin_import_exception is not None:
        return f"Marlin kernel unavailable: {marlin_import_exception}"
    if not marlin_runtime_available(dtype):
        return marlin_runtime_error(dtype)
    return None


def _e8m0_scales(size_n: int, size_k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # A different exponent per group (2**-8 .. 2**7), so a scale read from the wrong group shows up.
    exps = torch.randint(119, 135, (size_n, size_k // MXFP8_GROUP_SIZE), generator=g, dtype=torch.int32)
    scale = torch.pow(2.0, exps.double() - 127).repeat_interleave(MXFP8_GROUP_SIZE, dim=1)
    return exps.to(torch.uint8).view(torch.float8_e8m0fnu), scale


def _mxfp8_operands(size_n: int, size_k: int, device: torch.device, seed: int):
    g = torch.Generator().manual_seed(seed)
    # Every finite E4M3 code (sign, exponent, mantissa); 0x7F/0xFF are NaN.
    codes = torch.randint(0, 256, (size_n, size_k), generator=g, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes).to(torch.uint8)
    weight = codes.view(torch.float8_e4m3fn)
    scales, scale = _e8m0_scales(size_n, size_k, g)
    qweight, marlin_scales = mxfp8_marlin_prepare(weight.to(device), scales.to(device))
    return qweight, marlin_scales, scalar_types.float8_e4m3fn, weight.double() * scale


def _mxfp4_operands(size_n: int, size_k: int, device: torch.device, seed: int):
    g = torch.Generator().manual_seed(seed)
    nibbles = torch.randint(0, 16, (size_n, size_k), generator=g)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)  # low nibble = even k
    scales, scale = _e8m0_scales(size_n, size_k, g)
    qweight, marlin_scales = mxfp4_marlin_prepare(packed.to(device), scales.to(device))
    return qweight, marlin_scales, scalar_types.float4_e2m1f, _E2M1[nibbles] * scale


_OPERANDS = {"mxfp8": _mxfp8_operands, "mxfp4": _mxfp4_operands}


def _gemm(x, qweight, scales, b_q_type, size_n, size_k):
    return gptq_marlin_gemm(
        a=x, c=None, b_q_weight=qweight, b_bias=None, b_scales=scales, global_scale=None,
        b_zeros=None, g_idx=None, perm=None, workspace=marlin_make_workspace_new(x.device),
        b_q_type=b_q_type, size_m=x.shape[0], size_n=size_n, size_k=size_k, use_fp32_reduce=True,
    )


@pytest.mark.parametrize("size_k,size_n", _SHAPES)
def test_marlin_interleave_8bit_scales_is_a_permutation(size_k, size_n):
    groups = size_k // MXFP8_GROUP_SIZE
    s = torch.arange(groups * size_n, dtype=torch.int64).remainder(251).to(torch.uint8).view(groups, size_n)
    out = marlin_interleave_8bit_scales(s)
    assert out.shape == s.shape and out.dtype == s.dtype
    assert torch.equal(out.flatten().sort().values, s.flatten().sort().values)


@pytest.mark.cuda
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("size_m", _M)
@pytest.mark.parametrize("size_k,size_n", _SHAPES)
def test_marlin_microscaling_matches_fp64_oracle(fmt, size_m, size_k, size_n):
    reason = _skip_reason(torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda")
    qweight, scales, b_q_type, dequant = _OPERANDS[fmt](size_n, size_k, device, seed=size_k * 7 + size_n)
    torch.manual_seed(size_m)
    x = (torch.randn(size_m, size_k) * 0.25).to(torch.bfloat16)
    oracle = x.double() @ dequant.T
    out = _gemm(x.to(device), qweight, scales, b_q_type, size_n, size_k).double().cpu()

    assert torch.isfinite(out).all()
    # bf16 output rounding (2**-8 relative) on top of fp32 accumulation, relative to each row's scale.
    err = (out - oracle).abs()
    tol = 2.0 ** -7 * oracle.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    bad = err > tol
    assert not bad.any(), (
        f"{fmt} Marlin M={size_m} K={size_k} N={size_n}: {int(bad.sum())} outputs off; "
        f"max err {err.max().item():.4g} vs |oracle| max {oracle.abs().max().item():.4g}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
def test_marlin_microscaling_rejects_fp16_compute(fmt):
    reason = _skip_reason(torch.float16) or _skip_reason(torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda")
    size_k, size_n = 256, 512
    qweight, scales, b_q_type, _ = _OPERANDS[fmt](size_n, size_k, device, seed=1)
    x = torch.randn(16, size_k, device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match=r"requires bfloat16 compute \(MX"):
        _gemm(x, qweight, scales, b_q_type, size_n, size_k)
