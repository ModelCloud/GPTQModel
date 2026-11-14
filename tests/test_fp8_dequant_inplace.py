# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FP8 native dequantization must not materialise a full N x K scale.

The module dequantizes on every forward. ``_dequantize_native`` used to build a
full-size fp32 scale with two ``repeat_interleave`` calls before dividing. The
in-place ``view(...).div_()`` formulation produces bitwise-identical output for
divisible block shapes; non-divisible shapes keep the padded/cropped expansion
as a fallback.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module


def _kernel(*, in_features=64, out_features=64, method="row", block=None, seed=0):
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=True).eval()
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method=method,
        weight_block_size=block,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    return kernel.eval()


def _replicate_pad(expanded: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if expanded.shape[0] < rows:
        expanded = torch.cat([expanded, expanded[-1:].expand(rows - expanded.shape[0], -1)], dim=0)
    if expanded.shape[1] < cols:
        expanded = torch.cat([expanded, expanded[:, -1:].expand(-1, cols - expanded.shape[1])], dim=1)
    return expanded[:rows, :cols]


def _expanded_fp64(kernel) -> torch.Tensor:
    scale = kernel.weight_scale_inv.to(torch.float64)
    if kernel.weight_scale_method == "tensor":
        return scale
    if kernel.weight_scale_method == "row":
        return scale.view(-1, 1)
    block_rows, block_cols = kernel.weight_block_size
    expanded = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)
    return _replicate_pad(expanded, *kernel.weight.shape)


def _old_formulation(kernel, dtype: torch.dtype) -> torch.Tensor:
    """The pre-change fp32 arithmetic: ``weight.float() / expanded_scale``."""

    weight = kernel.weight.to(torch.float32)
    scale = kernel.weight_scale_inv.to(torch.float32)
    if kernel.weight_scale_method == "tensor":
        expanded = scale
    elif kernel.weight_scale_method == "row":
        expanded = scale.view(-1, 1)
    else:
        block_rows, block_cols = kernel.weight_block_size
        expanded = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)
        expanded = _replicate_pad(expanded, *kernel.weight.shape)
    return (weight / expanded).to(dtype).transpose(0, 1).contiguous()


@pytest.mark.parametrize("method,block", [("tensor", None), ("row", None), ("block", (16, 16))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_native_dequant_is_bitwise_identical_to_old_formulation(method, block, dtype):
    kernel = _kernel(method=method, block=block)
    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=dtype)
    expected = _old_formulation(kernel, dtype)
    assert torch.equal(actual, expected), "in-place div changed the fp32 division result"


@pytest.mark.parametrize("method,block", [("tensor", None), ("row", None), ("block", (16, 16))])
def test_fp8_native_dequant_matches_fp64_oracle(method, block):
    kernel = _kernel(method=method, block=block)
    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=torch.float32)
    oracle = (kernel.weight.to(torch.float64) / _expanded_fp64(kernel)).transpose(0, 1).contiguous()
    torch.testing.assert_close(actual, oracle.to(torch.float32), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("method,block", [("tensor", None), ("row", None), ("block", (16, 16))])
def test_fp8_native_dequant_does_not_expand_the_scale(monkeypatch, method, block):
    """Divisible shapes must divide the fp32 weight buffer in place.

    Before the change ``_dequantize_native`` always called
    ``_expanded_scale_inv``; this test fails if that allocation returns.
    """

    kernel = _kernel(method=method, block=block)

    def forbidden(*args, **kwargs):
        raise AssertionError("full-size fp32 scale expansion was materialised")

    monkeypatch.setattr(kernel, "_expanded_scale_inv", forbidden)
    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=torch.float16)
    expected = _old_formulation(kernel, torch.float16)
    assert torch.equal(actual, expected)


def test_fp8_block_dequant_crops_non_divisible_rows():
    kernel = _kernel(in_features=64, out_features=64, method="block", block=(16, 16))
    # A ceil-shaped checkpoint scale for a 100-row weight: 7 blocks -> 112 rows.
    weight = torch.randn((100, 64), dtype=torch.float32)
    kernel.weight = weight.to(torch.float8_e4m3fn)
    kernel.weight_scale_inv = torch.rand((7, 4), dtype=torch.float32) + 0.5
    kernel.out_features = 100

    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=torch.float32)
    expanded = kernel.weight_scale_inv.repeat_interleave(16, 0).repeat_interleave(16, 1)[:100, :64]
    expected = (kernel.weight.to(torch.float32) / expanded).transpose(0, 1).contiguous()
    assert torch.equal(actual, expected)


def test_fp8_block_dequant_pads_short_non_divisible_rows():
    kernel = _kernel(in_features=64, out_features=64, method="block", block=(16, 16))
    # A floor-shaped scale leaves 4 rows uncovered; the fallback pads them.
    weight = torch.randn((100, 64), dtype=torch.float32)
    kernel.weight = weight.to(torch.float8_e4m3fn)
    kernel.weight_scale_inv = torch.rand((6, 4), dtype=torch.float32) + 0.5
    kernel.out_features = 100

    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=torch.float32)
    expanded = kernel.weight_scale_inv.repeat_interleave(16, 0).repeat_interleave(16, 1)
    expanded = _replicate_pad(expanded, 100, 64)
    expected = (kernel.weight.to(torch.float32) / expanded).transpose(0, 1).contiguous()
    assert torch.equal(actual, expected)


def test_fp8_block_dequant_crops_non_divisible_columns():
    kernel = _kernel(in_features=64, out_features=64, method="block", block=(16, 16))
    weight = torch.randn((64, 100), dtype=torch.float32)
    kernel.weight = weight.to(torch.float8_e4m3fn)
    kernel.weight_scale_inv = torch.rand((4, 7), dtype=torch.float32) + 0.5
    kernel.in_features = 100

    actual = kernel._dequantize_native(target_device=torch.device("cpu"), target_dtype=torch.float32)
    expanded = kernel.weight_scale_inv.repeat_interleave(16, 0).repeat_interleave(16, 1)[:, :100]
    expected = (kernel.weight.to(torch.float32) / expanded).transpose(0, 1).contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("method,block", [("tensor", None), ("row", None), ("block", (16, 16))])
def test_fp8_native_dequant_cuda_is_bitwise_identical(monkeypatch, method, block):
    kernel = _kernel(method=method, block=block).to("cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("full-size fp32 scale expansion was materialised on CUDA")

    monkeypatch.setattr(kernel, "_expanded_scale_inv", forbidden)
    actual = kernel._dequantize_native(target_device=torch.device("cuda"), target_dtype=torch.float16)
    expected = _old_formulation(kernel, torch.float16)
    assert torch.equal(actual, expected)
