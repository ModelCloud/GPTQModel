# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The shared kernel gates must not read HIP's CUDA namespace as NVIDIA CUDA."""

import torch

from tests import accelerator_gates


def _forbidden_toolchain_lookup(name):
    raise AssertionError(f"{name} must not be probed for an unusable device")


def _forbidden_capability(*args, **kwargs):
    raise AssertionError("get_device_capability must not classify a HIP device")


def test_hip_build_is_rejected_before_any_capability_probe(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _forbidden_capability)

    assert "ROCm" in accelerator_gates.swordfish_skip_reason()
    assert "ROCm" in accelerator_gates.nf12_skip_reason()


def test_hip_build_is_never_reported_as_a_missing_device(monkeypatch):
    """A HIP host must not be skipped as if no accelerator were present."""
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert "CUDA-only" in accelerator_gates.cuda_only_skip_reason("Swordfish")


def test_nvidia_build_passes_the_family_gate(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)

    assert accelerator_gates.cuda_only_skip_reason("Swordfish") is None


def _fake_devices(monkeypatch, capabilities, supported):
    """Install ``capabilities`` per ordinal and a matching support probe."""
    import gptqmodel.utils.swordfish as swordfish

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: capabilities[
            device if isinstance(device, int) else getattr(device, "index", None) or 0
        ],
    )
    monkeypatch.setattr(
        swordfish,
        "_validate_swordfish_device_support",
        lambda device=None: (device.index if device is not None else 0) in supported,
    )


def test_swordfish_reports_the_real_capability_on_nvidia(monkeypatch):
    _fake_devices(monkeypatch, [(8, 9)], supported=())

    assert accelerator_gates.swordfish_skip_reason() == (
        "Swordfish requires Blackwell (sm100+); no visible device is supported "
        "(cuda:0=sm89)"
    )


def test_swordfish_rejects_an_explicit_unsupported_ordinal(monkeypatch):
    _fake_devices(monkeypatch, [(10, 0), (8, 9)], supported=(0,))

    assert accelerator_gates.swordfish_skip_reason(torch.device("cuda:1")) == (
        "Swordfish requires Blackwell (sm100+); found cuda:1=sm89"
    )
    assert accelerator_gates.swordfish_skip_reason(torch.device("cuda:0")) is None
    assert accelerator_gates.swordfish_skip_reason() is None


def test_swordfish_targets_the_first_supported_ordinal_on_a_mixed_host(monkeypatch):
    """A Blackwell GPU behind an older one must still be exercised."""
    _fake_devices(monkeypatch, [(8, 9), (10, 0), (10, 3)], supported=(1, 2))

    assert accelerator_gates.swordfish_supported_devices() == [1, 2]
    assert accelerator_gates.swordfish_test_device() == torch.device("cuda:1")


def test_swordfish_mixed_host_with_no_supported_device_lists_every_ordinal(monkeypatch):
    _fake_devices(monkeypatch, [(7, 5), (8, 9)], supported=())

    reason = accelerator_gates.swordfish_skip_reason()

    assert "cuda:0=sm75" in reason
    assert "cuda:1=sm89" in reason


def test_swordfish_reports_a_raising_device_probe(monkeypatch):
    import gptqmodel.utils.swordfish as swordfish

    def raising(device=None):
        raise RuntimeError("no HIP kernel")

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(swordfish, "_validate_swordfish_device_support", raising)

    # An explicit target reports the probe's own error.
    assert accelerator_gates.swordfish_skip_reason(torch.device("cuda:1")) == (
        "Swordfish device validation failed: no HIP kernel"
    )
    # A sweep treats a raising probe as "this ordinal is not supported".
    assert accelerator_gates.swordfish_skip_reason() == (
        "Swordfish requires Blackwell (sm100+); no visible device is supported "
        "(cuda:0=sm89, cuda:1=sm89)"
    )


def test_swordfish_without_a_cuda_device_is_skipped(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert accelerator_gates.swordfish_skip_reason() == "CUDA is unavailable"


def test_nf12_rejects_a_pre_sm80_device_before_any_build(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(accelerator_gates.shutil, "which", _forbidden_toolchain_lookup)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: {(0,): (8, 0), (1,): (7, 5), (2,): (6, 1)}[
            (device if isinstance(device, int) else getattr(device, "index", None),)
        ],
    )

    assert accelerator_gates.nf12_skip_reason() == (
        "NF12 needs SM80 or newer on every visible device; unsupported: "
        "cuda:1=sm75, cuda:2=sm61"
    )


def test_nf12_reports_the_missing_source_build_tools(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(accelerator_gates.shutil, "which", lambda name: None)

    reason = accelerator_gates.nf12_skip_reason()

    assert reason is not None
    for tool in ("cmake", "ninja", "nvcc"):
        assert tool in reason
    assert "build is unavailable" in reason


def test_nf12_reports_a_failed_probe_once(monkeypatch):
    import gptqmodel_nf12 as nf12

    attempts = []

    def failing_load():
        attempts.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(accelerator_gates.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(accelerator_gates, "_nvcc", lambda: "/usr/bin/nvcc")
    monkeypatch.setattr(accelerator_gates, "_nf12_build_error", None)
    monkeypatch.setattr(accelerator_gates, "_nf12_probed", False)
    monkeypatch.setattr(nf12, "load", failing_load)

    assert "boom" in accelerator_gates.nf12_skip_reason()
    assert "boom" in accelerator_gates.nf12_skip_reason()
    assert len(attempts) == 1


def test_nf12_passes_when_the_native_load_succeeds(monkeypatch):
    import gptqmodel_nf12 as nf12

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(accelerator_gates.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(accelerator_gates, "_nvcc", lambda: "/usr/bin/nvcc")
    monkeypatch.setattr(accelerator_gates, "_nf12_build_error", None)
    monkeypatch.setattr(accelerator_gates, "_nf12_probed", False)
    monkeypatch.setattr(nf12, "load", lambda: None)

    assert accelerator_gates.nf12_skip_reason() is None
