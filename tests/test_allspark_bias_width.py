# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The AllSpark kernel bias must be as narrow as it is lossless.

A bias buffer in the activation dtype is what keeps a biased layer on the fast
prefill path; an FP32 buffer forces the HIP fallback onto an FP32 matmul. The
width is therefore a correctness decision, not a cosmetic one: narrowing a
value the dtype cannot hold (an FP32 -65536 is -inf in FP16) would silently
change the result.
"""

from __future__ import annotations

import torch

from gptqmodel.utils.allspark_bias import narrow_bias_for_kernel


def test_none_passes_through():
    assert narrow_bias_for_kernel(None, torch.float16) is None


def test_matching_dtype_is_returned_unchanged():
    bias = torch.tensor([1.0, -2.0], dtype=torch.float16)
    assert narrow_bias_for_kernel(bias, torch.float16) is bias


def test_representable_fp32_bias_is_narrowed():
    bias = torch.tensor([-32768.0, 0.5, 1024.0], dtype=torch.float32)
    for dtype in (torch.float16, torch.bfloat16):
        narrowed = narrow_bias_for_kernel(bias, dtype)
        assert narrowed.dtype == dtype
        assert torch.equal(narrowed.to(torch.float32), bias)


def test_out_of_range_fp32_bias_stays_fp32():
    # -65536 is -inf in FP16; the epilogue needs the finite FP32 value.
    bias = torch.tensor([-65536.0, 0.0], dtype=torch.float32)
    assert narrow_bias_for_kernel(bias, torch.float16) is bias


def test_inexact_fp32_bias_stays_fp32():
    # 100000 needs 17 mantissa bits and is outside the FP16 range.
    bias = torch.tensor([-100000.0], dtype=torch.float32)
    for dtype in (torch.float16, torch.bfloat16):
        assert narrow_bias_for_kernel(bias, dtype) is bias


def test_nan_bias_stays_fp32():
    bias = torch.tensor([float("nan")], dtype=torch.float32)
    assert narrow_bias_for_kernel(bias, torch.float16) is bias


def test_non_float_bias_is_not_touched():
    bias = torch.zeros(4, dtype=torch.int32)
    assert narrow_bias_for_kernel(bias, torch.float16) is bias


def test_non_half_target_dtype_is_not_touched():
    bias = torch.zeros(4, dtype=torch.float32)
    assert narrow_bias_for_kernel(bias, torch.float64) is bias


def test_other_half_width_is_widened_only_when_lossless():
    # bf16 -> fp16 is exact for these values.
    bias = torch.tensor([0.5, -2.0], dtype=torch.bfloat16)
    narrowed = narrow_bias_for_kernel(bias, torch.float16)
    assert narrowed.dtype == torch.float16
    assert torch.equal(narrowed.to(torch.float32), bias.to(torch.float32))
