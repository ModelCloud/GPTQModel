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
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
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


@pytest.mark.parametrize("device", [DEVICE.ROCM, DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_native_exl3_rejects_hip_requests(device, monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    assert ExllamaV3Linear.SUPPORTS_DEVICES == [DEVICE.CUDA]
    assert DEVICE.ALL in ExllamaV3TorchLinear.SUPPORTS_DEVICES
    with pytest.raises(NotImplementedError, match="EXL3_TORCH"):
        ExllamaV3Linear.validate_device(device)


@pytest.mark.parametrize("device", [DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_native_exl3_accepts_nvidia_requests(device, monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", False)
    ExllamaV3Linear.validate_device(device)


def test_native_exl3_rejects_hip_before_importing_extension(monkeypatch):
    import builtins
    from types import SimpleNamespace

    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    module = ExllamaV3Linear(in_features=128, out_features=128, name="test", tensors={})
    module.trellis = SimpleNamespace(device=torch.device("cuda:0"))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "exllamav3.modules.quant.exl3", "must reject before importing native extension"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(NotImplementedError, match="EXL3_TORCH"):
        module.post_init()

