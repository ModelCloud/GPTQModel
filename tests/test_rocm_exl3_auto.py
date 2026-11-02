# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU coverage for the EXL3 AUTO backend dispatch on ROCm."""

import inspect

import pytest
import torch

from gptqmodel.models import loader
from gptqmodel.models._const import DEVICE
from gptqmodel.models.loader import _resolve_exl3_auto_backend
from gptqmodel.utils.backend import BACKEND


@pytest.mark.parametrize(
    ("cuda_available", "family", "expected"),
    [
        (True, DEVICE.CUDA, BACKEND.EXL3_EXLLAMA_V3),
        (True, DEVICE.ROCM, BACKEND.EXL3_TORCH),
        (False, DEVICE.ROCM, BACKEND.EXL3_TORCH),
        (True, DEVICE.CPU, BACKEND.EXL3_TORCH),
    ],
)
def test_exl3_auto_backend_never_selects_exllama_v3_on_rocm(
    monkeypatch, cuda_available, family, expected
):
    monkeypatch.setattr(loader, "selector_device_family", lambda device: family)

    assert _resolve_exl3_auto_backend(
        torch.device("cuda:0"), cuda_available=cuda_available
    ) == expected

def test_exl3_auto_branch_delegates_to_the_helper():
    """Guard the AUTO branch itself: it must not inline the old CUDA/HIP test."""

    source = inspect.getsource(loader.ModelLoader)
    start = source.index("if format_code == FORMAT.EXL3:")
    branch = source[start : start + 800]

    auto_branch = branch[: branch.index("if backend == BACKEND.EXL3_EXLLAMA_V3:")]
    assert "_resolve_exl3_auto_backend(device)" in auto_branch
    assert "DEVICE.ROCM" not in auto_branch

