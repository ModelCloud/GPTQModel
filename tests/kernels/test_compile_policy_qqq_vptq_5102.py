# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""QQQ/VPTQ portable kernels opt into compiled static-cache generation (#5102).

`QQQTorchLinear` and `VPTQLinear` are trace-clean under `torch.compile` and a
model-level static-cache `generate()` adds one graph with zero breaks and
decodes several times faster than the eager path. They used to keep
`SUPPORTS_COMPILE = False`, either citing a number measured on a different
kernel (`QQQLinear`) or recording that nobody had timed the portable path yet,
so the #4669 policy forced every ROCm AUTO QQQ/VPTQ checkpoint, and CUDA bf16
QQQ, onto that slower eager path.

This file is red on unmodified `main` and green with the opt-in. The fused
`VPTQCudaLinear` sibling is the counter-case: it inherits from `VPTQLinear`, so
it has to re-declare the policy off or the parent's opt-in would leak to the
kernel whose static-cache decode is slower than eager.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear._compile_support import (
    COMPILE_POLICY_FLOOR,
    compile_level,
)
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear


pytestmark = [pytest.mark.cpu]


class _RecordingModel(nn.Module):
    """The inner HF model: records the generation kwargs it receives."""

    def __init__(self, projection: nn.Module):
        super().__init__()
        self.projection = projection
        self.device = torch.device("cpu")
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, 1, dtype=torch.long)


class _Harness:
    """``BaseQModel``'s policy helpers on a fake model holding one kernel."""

    _kernels_support_compile = BaseQModel._kernels_support_compile
    _generation_compile_kwargs = BaseQModel._generation_compile_kwargs

    def __init__(self, projection: nn.Module):
        self.model = _RecordingModel(projection)
        self.tokenizer = None


def _build_qqq_torch() -> QQQTorchLinear:
    return QQQTorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
    )


def _build_vptq_torch() -> VPTQLinear:
    return VPTQLinear(
        bits=2,
        in_features=64,
        out_features=64,
        bias=False,
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=64,
        register_buffers=True,
        dtype=None,
    )


CASES = [
    pytest.param(QQQTorchLinear, _build_qqq_torch, id="QQQTorchLinear"),
    pytest.param(VPTQLinear, _build_vptq_torch, id="VPTQLinear"),
]


@pytest.mark.parametrize("kernel_cls,build", CASES)
def test_portable_kernel_declares_compile_support(kernel_cls, build):
    """The policy flag must follow the measured static-cache win."""

    assert kernel_cls.SUPPORTS_COMPILE is True
    assert not (kernel_cls.SUPPORTS_COMPILE_EAGER_REASON or "").strip(), (
        "an opted-in kernel must not keep the old measured-loss reason"
    )
    assert compile_level(kernel_cls) >= COMPILE_POLICY_FLOOR


@pytest.mark.parametrize("kernel_cls,build", CASES)
def test_policy_no_longer_forces_eager(kernel_cls, build):
    """A static-cache generate() must keep the caller's compile policy."""

    harness = _Harness(build())

    assert harness._kernels_support_compile() is True
    assert harness._generation_compile_kwargs(
        {"cache_implementation": "static"}
    ) == {"cache_implementation": "static"}


def test_the_fp16_cuda_qqq_kernel_stays_opted_out():
    """The opt-in belongs to the torch kernel, not its FP16 CUDA parent."""

    assert QQQLinear is not QQQTorchLinear
    assert QQQLinear.SUPPORTS_COMPILE is False


def test_vptq_cuda_sibling_does_not_inherit_the_torch_opt_in():
    """The fused CUDA kernel recompiles (23 graphs, 33 breaks) and loses."""

    assert issubclass(VPTQCudaLinear, VPTQLinear)
    assert VPTQCudaLinear.SUPPORTS_COMPILE is False
    assert (VPTQCudaLinear.SUPPORTS_COMPILE_EAGER_REASON or "").strip()
