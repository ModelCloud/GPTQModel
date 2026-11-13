# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The shared kernel gates must not read HIP's CUDA namespace as NVIDIA CUDA."""

import shutil
import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import _const
from tests import accelerator_gates


def _set_family(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(_const, "IS_ROCM", hip is not None)


def _forbidden_toolchain_lookup(name):
    raise AssertionError(f"{name} must not be probed for an unusable device")


def _forbidden_capability(*args, **kwargs):
    raise AssertionError("get_device_capability must not classify a HIP device")


@pytest.fixture
def nf12_loader(monkeypatch, tmp_path):
    import gptqmodel_nf12 as nf12
    from gptqmodel_nf12 import _jit, _library

    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(torch.ops, "nf12", SimpleNamespace())
    monkeypatch.setattr(_library, "_loaded", threading.Event())
    monkeypatch.setattr(_library, "_register_fakes", lambda: None)
    monkeypatch.setattr(nf12, "__path__", [str(tmp_path / "package")])
    monkeypatch.setattr(_jit, "_directory", None)
    monkeypatch.setattr(_jit, "_source_root", lambda: tmp_path)
    monkeypatch.setattr(_jit, "_cache_key", lambda *_: "fixture")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return SimpleNamespace(
        library=_library,
        jit=_jit,
        package=tmp_path / "package",
        cache=tmp_path / "cache" / "gptqmodel" / "nf12" / "fixture",
    )


def _ready_nf12_runtime(loader, mode):
    if mode == "loaded":
        loader.library._loaded.set()
    elif mode == "registered":
        torch.ops.nf12 = SimpleNamespace(**{
            name: lambda *args: None for name in (
                "encode", "decode", "linear", "embedding", "encode_moe", "moe_linear", "_fused_moe"
            )
        })
    else:
        directory = loader.package if mode == "packaged" else loader.cache / "build"
        directory.mkdir(parents=True)
        for name in ("libnf12.so", "libnf12_torch.so"):
            (directory / name).touch()
        if mode == "cached":
            (loader.cache / "complete").touch()


@pytest.mark.parametrize("mode", ["loaded", "registered", "packaged", "cached"])
def test_nf12_ready_runtime_does_not_require_build_tools(monkeypatch, nf12_loader, mode):
    _ready_nf12_runtime(nf12_loader, mode)
    loaded = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    monkeypatch.setattr(shutil, "which", _forbidden_toolchain_lookup)

    assert accelerator_gates.nf12_skip_reason() is None
    assert nf12_loader.library._loaded.is_set()
    assert len(loaded) == (2 if mode in ("packaged", "cached") else 0)


@pytest.mark.parametrize("mode", ["packaged", "cached"])
def test_nf12_binary_load_failure_is_not_hidden_by_missing_tools(monkeypatch, nf12_loader, mode):
    _ready_nf12_runtime(nf12_loader, mode)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def broken_binary(path):
        raise RuntimeError("native binary ABI mismatch")

    monkeypatch.setattr(torch.ops, "load_library", broken_binary)

    with pytest.raises(RuntimeError, match="native binary ABI mismatch"):
        accelerator_gates.nf12_skip_reason()


def test_nf12_source_compile_failure_is_not_a_missing_tool_skip(monkeypatch, nf12_loader):
    jit = nf12_loader.jit
    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")
    attempts = []

    def broken_compile(command, **kwargs):
        attempts.append(command)
        raise jit.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(jit.subprocess, "run", broken_compile)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="JIT compilation failed"):
            accelerator_gates.nf12_skip_reason()
    assert len(attempts) == 2


def test_hip_build_is_rejected_before_any_capability_probe(monkeypatch):
    _set_family(monkeypatch, "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _forbidden_capability)

    assert "ROCm" in accelerator_gates.swordfish_skip_reason()
    assert "ROCm" in accelerator_gates.nf12_skip_reason()


def test_hip_build_is_never_reported_as_a_missing_device(monkeypatch):
    """A HIP host must not be skipped as if no accelerator were present."""
    _set_family(monkeypatch, "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert "CUDA-only" in accelerator_gates.cuda_only_skip_reason("Swordfish")


def test_nvidia_build_passes_the_family_gate(monkeypatch):
    _set_family(monkeypatch, None)

    assert accelerator_gates.cuda_only_skip_reason("Swordfish") is None


def _fake_devices(monkeypatch, capabilities, supported):
    """Install ``capabilities`` per ordinal and a matching support probe."""
    import gptqmodel.utils.swordfish as swordfish

    _set_family(monkeypatch, None)
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

    _set_family(monkeypatch, None)
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
    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert accelerator_gates.swordfish_skip_reason() == "CUDA is unavailable"


def test_nf12_rejects_a_pre_sm80_device_before_any_build(monkeypatch):
    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(shutil, "which", _forbidden_toolchain_lookup)
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


def test_nf12_reports_the_missing_source_build_tools(monkeypatch, nf12_loader):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    reason = accelerator_gates.nf12_skip_reason()

    assert reason is not None
    for tool in ("cmake", "ninja", "nvcc"):
        assert tool in reason.lower()
    assert "build is unavailable" in reason


def test_nf12_build_failure_is_not_cached_as_a_skip(monkeypatch):
    import gptqmodel_nf12 as nf12

    attempts = []

    def failing_load():
        attempts.append(1)
        raise RuntimeError("boom")

    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(nf12, "load", failing_load)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom"):
            accelerator_gates.nf12_skip_reason()
    assert len(attempts) == 2


def test_nf12_passes_when_the_native_load_succeeds(monkeypatch):
    import gptqmodel_nf12 as nf12

    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(nf12, "load", lambda: None)

    assert accelerator_gates.nf12_skip_reason() is None


@pytest.mark.parametrize("override", ["/custom/bin/nvcc", "custom-nvcc"])
def test_nf12_nvcc_honors_cudacxx(monkeypatch, tmp_path, override):
    from torch.utils import cpp_extension

    from gptqmodel_nf12 import _jit

    monkeypatch.setenv("CUDACXX", override)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/unused/cuda")
    monkeypatch.setattr(shutil, "which", lambda name: name if name in ("cmake", "ninja", override) else None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        build = tmp_path / "build"
        build.mkdir(exist_ok=True)
        for name in _jit._LIBRARIES:
            (build / name).touch()

    monkeypatch.setattr(_jit.subprocess, "run", run)
    _jit._build(tmp_path, tmp_path, [89])

    assert f"-DCMAKE_CUDA_COMPILER={override}" in commands[0]
    assert len(commands) == 2
