# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4743.

``GGUFCudaKernel`` (AUTO for GGUF Q4_0/Q4_K on CUDA) had no empty-input guard,
so a zero-row activation - an MoE expert that received no tokens, or an empty
batch slice - reached the ggml plan builder and raised

    RuntimeError: GGUFCudaKernel GGML plan has no scratch tensors.

Every other kernel returns an empty ``(..., N)`` output for M=0, and so does the
CPU sibling ``GGUFCppKernel``. These tests pin the guard, the output contract
and - most importantly - that the guard runs before any ggml or plan work.

The ggml CUDA backend is not required: the empty path must not touch it at all,
which is what ``test_zero_rows_never_touch_the_ggml_bridge`` asserts.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel


IN_FEATURES = 64
OUT_FEATURES = 16


def _layer() -> GGUFCudaKernel:
    """A kernel instance with only the state the empty path may read."""

    layer = object.__new__(GGUFCudaKernel)
    object.__setattr__(layer, "in_features", IN_FEATURES)
    object.__setattr__(layer, "out_features", OUT_FEATURES)
    object.__setattr__(layer, "adapter", None)
    object.__setattr__(layer, "_ggml_cuda_plans", OrderedDict())
    return layer


@pytest.mark.parametrize(
    "shape",
    [(0, IN_FEATURES), (2, 0, IN_FEATURES), (0, 3, IN_FEATURES)],
    ids=["m0", "batch-m0", "m0-seq"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_rows_return_an_empty_output(shape, dtype):
    layer = _layer()
    x = torch.empty(shape, dtype=dtype)

    out = layer.forward(x)

    assert tuple(out.shape) == tuple(shape[:-1]) + (OUT_FEATURES,)
    assert out.dtype is dtype
    assert out.device == x.device
    assert out.numel() == 0
    # No plan was built or cached for the empty batch.
    assert not layer._ggml_cuda_plans


@pytest.mark.parametrize("shape", [(0, IN_FEATURES), (2, 0, IN_FEATURES)])
def test_zero_rows_never_touch_the_ggml_bridge(shape, monkeypatch):
    def _fail():
        raise AssertionError("a zero-row input must not reach the ggml bridge")

    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", _fail)

    out = _layer().forward(torch.empty(shape, dtype=torch.float16))

    assert out.shape[-1] == OUT_FEATURES
    assert out.numel() == 0


def test_non_empty_input_still_requires_cuda(monkeypatch):
    """The guard is narrow: a real matmul keeps the CUDA-only diagnostic."""

    def _fail():
        raise AssertionError("a non-CUDA input must be rejected before the bridge")

    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", _fail)

    with pytest.raises(RuntimeError, match="only supports CUDA inference"):
        _layer().forward(torch.empty((1, IN_FEATURES), dtype=torch.float16))
