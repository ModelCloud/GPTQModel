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
    ("cuda_available", "family", "native", "expected"),
    [
        (True, DEVICE.CUDA, True, BACKEND.EXL3_EXLLAMA_V3),
        (True, DEVICE.CUDA, False, BACKEND.EXL3_EXLLAMA_V3),
        (True, DEVICE.ROCM, True, BACKEND.EXL3_EXLLAMA_V3),
        (True, DEVICE.ROCM, False, BACKEND.EXL3_TORCH),
        (False, DEVICE.ROCM, True, BACKEND.EXL3_TORCH),
        (True, DEVICE.CPU, True, BACKEND.EXL3_TORCH),
    ],
)
def test_exl3_auto_backend_uses_native_only_on_a_usable_rocm_target(
    monkeypatch, cuda_available, family, native, expected
):
    monkeypatch.setattr(loader, "selector_device_family", lambda device: family)
    monkeypatch.setattr(loader, "_selector_devices", lambda device: (torch.device("cuda:0"),))
    monkeypatch.setattr(ExllamaV3Linear, "supports_device", classmethod(lambda cls, device: native))

    assert _resolve_exl3_auto_backend(
        torch.device("cuda:0"), cuda_available=cuda_available
    ) == expected


def test_exl3_auto_rocm_requires_every_selected_device(monkeypatch):
    monkeypatch.setattr(loader, "selector_device_family", lambda device: DEVICE.ROCM)
    monkeypatch.setattr(
        loader,
        "_selector_devices",
        lambda device: (torch.device("cuda:0"), torch.device("cuda:1")),
    )
    monkeypatch.setattr(
        ExllamaV3Linear,
        "supports_device",
        classmethod(lambda cls, device: device.index == 0),
    )

    assert _resolve_exl3_auto_backend(
        (torch.device("cuda:0"), torch.device("cuda:1")), cuda_available=True
    ) == BACKEND.EXL3_TORCH


def test_exl3_auto_branch_delegates_to_the_helper():
    """Guard the AUTO branch itself: it must not inline the old CUDA/HIP test."""

    source = inspect.getsource(loader.ModelLoader)
    start = source.index("if format_code == FORMAT.EXL3:")
    branch = source[start : start + 800]

    auto_branch = branch[: branch.index("if backend == BACKEND.EXL3_EXLLAMA_V3:")]
    assert "_resolve_exl3_auto_backend(device)" in auto_branch
    assert "DEVICE.ROCM" not in auto_branch


@pytest.mark.parametrize("device", [DEVICE.ROCM, DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_native_exl3_accepts_rdna4_hip_requests(device, monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_available",
        lambda _device=None: True,
    )
    assert ExllamaV3Linear.SUPPORTS_DEVICES == [DEVICE.CUDA, DEVICE.ROCM]
    assert DEVICE.ALL in ExllamaV3TorchLinear.SUPPORTS_DEVICES
    ExllamaV3Linear.validate_device(device)


@pytest.mark.parametrize("device", [DEVICE.ROCM, DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_native_exl3_rejects_hip_without_native_gemm(device, monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_available",
        lambda _device=None: False,
    )
    with pytest.raises(NotImplementedError, match="EXL3_TORCH"):
        ExllamaV3Linear.validate_device(device)


def test_native_exl3_validates_each_selected_ordinal(monkeypatch):
    """A device map may mix ordinals; each one is probed, not GPU 0."""

    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)

    def available(device=None):
        return torch.device(device).index == 1

    monkeypatch.setattr("gptqmodel.exllamav3.ext.exl3_native_hip_gemm_available", available)
    with pytest.raises(NotImplementedError, match="EXL3_TORCH"):
        ExllamaV3Linear.validate_device((torch.device("cuda:1"), torch.device("cuda:2")))
    ExllamaV3Linear.validate_device((torch.device("cuda:1"),))


def test_native_exl3_supports_device_requires_a_usable_extension(monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_usable",
        lambda _device=None: True,
    )
    assert ExllamaV3Linear.supports_device(torch.device("cuda:1"))

    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_usable",
        lambda _device=None: False,
    )
    assert not ExllamaV3Linear.supports_device(torch.device("cuda:1"))


@pytest.mark.parametrize("device", [DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_native_exl3_accepts_nvidia_requests(device, monkeypatch):
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", False)
    ExllamaV3Linear.validate_device(device)


def test_native_exl3_rejects_hip_before_importing_extension(monkeypatch):
    import builtins
    from types import SimpleNamespace

    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_available",
        lambda _device=None: False,
    )
    module = ExllamaV3Linear(in_features=128, out_features=128, name="test", tensors={})
    module.trellis = SimpleNamespace(device=torch.device("cuda:0"))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "exllamav3.modules.quant.exl3", "must reject before importing native extension"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(NotImplementedError, match="EXL3_TORCH"):
        module.post_init()
