# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The HIP JIT cache key must name the gfx target hipcc actually compiles for (#4815).

``TorchOpsJitExtension._cache_fingerprint`` used only CUDA inputs: the
hipcc-ignored ``TORCH_CUDA_ARCH_LIST``, the coarse ``get_device_capability()``
pair (gfx1200 and gfx1201 both report ``(12, 0)``; gfx908 and gfx90a both report
``(9, 0)``), and the empty ``resolved_cuda_arch_flags()``. A HIP binary built for
one target was therefore reused by a later process on another target, and every
launch failed with ``invalid device function``.

These tests are CPU-only: they monkeypatch ``torch.version.hip``,
``PYTORCH_ROCM_ARCH`` and the device properties, and never need ROCm hardware.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gptqmodel.utils import cpp as cpp_module


HIP_VERSION = "10.1.0"


def _make_loader(tmp_path, **overrides) -> cpp_module.TorchOpsJitExtension:
    """Construct a shared torch.ops loader with a disposable build root."""

    source = tmp_path / "unit_test.cpp"
    source.write_text("PYBIND11_MODULE(unit_test, m) {}\n", encoding="utf-8")
    params = {
        "name": "unit_test_ops",
        "namespace": "unit_test_ns",
        "required_ops": ("kernel",),
        "sources": [str(source)],
        "build_root_env": "UNIT_TEST_BUILD_ROOT",
        "default_build_root": lambda: tmp_path / "jit_build",
        "display_name": "Unit Test Kernel",
        "requires_cuda": True,
    }
    params.update(overrides)
    return cpp_module.TorchOpsJitExtension(**params)


@pytest.fixture
def hip(monkeypatch):
    """Pretend this process runs on a ROCm build with no explicit target."""

    monkeypatch.setattr(cpp_module.torch.version, "hip", HIP_VERSION, raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    return HIP_VERSION


def _fake_devices(monkeypatch, arch_names):
    """Make ``torch.cuda`` enumerate devices with the given ``gcnArchName`` values."""

    props = [SimpleNamespace(gcnArchName=name) for name in arch_names]
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: len(props))
    monkeypatch.setattr(
        cpp_module.torch.cuda,
        "get_device_properties",
        lambda index: props[index],
    )
    return props


# ---------------------------------------------------------------------------
# PYTORCH_ROCM_ARCH is the authoritative target set.
# ---------------------------------------------------------------------------


def test_rocm_arch_env_changes_the_fingerprint(monkeypatch, tmp_path, hip):
    """Two machines that set different targets must never share a cache key."""

    loader = _make_loader(tmp_path)

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1200")
    gfx1200_payload = loader._cuda_cache_fingerprint_payload()
    gfx1200_fingerprint = loader._cache_fingerprint()

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")
    gfx1201_payload = loader._cuda_cache_fingerprint_payload()
    gfx1201_fingerprint = loader._cache_fingerprint()

    assert gfx1200_payload == ["cuda_ext=1", f"hip={hip}", "rocm_arch_list=gfx1200"]
    assert gfx1201_payload == ["cuda_ext=1", f"hip={hip}", "rocm_arch_list=gfx1201"]
    assert gfx1200_fingerprint != gfx1201_fingerprint


def test_same_rocm_arch_env_is_stable_and_canonical(monkeypatch, tmp_path, hip):
    """Ordering and spacing of the same target set must not fork the cache."""

    loader = _make_loader(tmp_path)

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201;gfx1200")
    first = loader._cache_fingerprint()
    assert loader._cache_fingerprint() == first

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1200 gfx1201")
    assert loader._cache_fingerprint() == first

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", " GFX1201 ; gfx1200 ")
    assert loader._cache_fingerprint() == first


def test_hip_runtime_version_is_part_of_the_key(monkeypatch, tmp_path, hip):
    loader = _make_loader(tmp_path)
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")
    first = loader._cache_fingerprint()

    monkeypatch.setattr(cpp_module.torch.version, "hip", "10.2.0", raising=False)

    assert loader._cache_fingerprint() != first


def test_torch_cuda_arch_list_does_not_mask_the_rocm_target(monkeypatch, tmp_path, hip):
    """hipcc ignores ``TORCH_CUDA_ARCH_LIST``; it must not hide the gfx target."""

    loader = _make_loader(tmp_path)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")

    payload = loader._cuda_cache_fingerprint_payload()

    assert payload == ["cuda_ext=1", f"hip={hip}", "rocm_arch_list=gfx1201"]
    assert not any(entry.startswith("arch_list=") for entry in payload)
    assert not any(entry.startswith("visible_caps=") for entry in payload)


# ---------------------------------------------------------------------------
# Without the env override the visible gcnArchName set is the key.
# ---------------------------------------------------------------------------


def test_rocm_arch_falls_back_to_sorted_visible_gcn_arch(monkeypatch, tmp_path, hip):
    _fake_devices(
        monkeypatch,
        [
            "gfx1201:sramecc+:xnack-",
            "gfx1200",
            "",  # a device that cannot report its architecture contributes nothing
        ],
    )
    loader = _make_loader(tmp_path)

    payload = loader._cuda_cache_fingerprint_payload()

    assert payload == ["cuda_ext=1", f"hip={hip}", "rocm_arch_list=gfx1200;gfx1201"]


def test_visible_gcn_arch_order_does_not_change_the_payload(monkeypatch, tmp_path, hip):
    loader = _make_loader(tmp_path)

    _fake_devices(monkeypatch, ["gfx1201", "gfx1200"])
    forward = loader._cuda_cache_fingerprint_payload()

    _fake_devices(monkeypatch, ["gfx1200", "gfx1201"])
    reversed_order = loader._cuda_cache_fingerprint_payload()

    assert forward == reversed_order


def test_two_machine_gfx1200_and_gfx1201_do_not_share_a_key(monkeypatch, tmp_path, hip):
    """The issue's NFS-shared cache scenario: RX 9060 vs R9700."""

    loader = _make_loader(tmp_path)

    _fake_devices(monkeypatch, ["gfx1200:sramecc+:xnack-"])
    rx9060 = loader._cache_fingerprint()

    _fake_devices(monkeypatch, ["gfx1201:sramecc+:xnack-"])
    r9700 = loader._cache_fingerprint()

    assert rx9060 != r9700


def test_missing_gpu_or_properties_still_yields_a_stable_payload(monkeypatch, tmp_path, hip):
    loader = _make_loader(tmp_path)

    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: False)
    assert loader._cuda_cache_fingerprint_payload() == [
        "cuda_ext=1",
        f"hip={hip}",
        "rocm_arch_list=none",
    ]

    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)

    def _unavailable(_index):
        raise RuntimeError("device properties unavailable")

    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_properties", _unavailable)
    assert loader._cuda_cache_fingerprint_payload() == [
        "cuda_ext=1",
        f"hip={hip}",
        "rocm_arch_list=none",
    ]


# ---------------------------------------------------------------------------
# The CUDA payload stays byte-identical so existing CUDA caches survive.
# ---------------------------------------------------------------------------


def test_non_hip_payload_is_unchanged(monkeypatch, tmp_path):
    """Golden CUDA payloads; changing them would invalidate every cached binary."""

    monkeypatch.setattr(cpp_module.torch.version, "hip", None, raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda _index: (12, 0))
    monkeypatch.setattr(
        cpp_module,
        "resolved_cuda_arch_flags",
        lambda: [
            "-gencode=arch=compute_120,code=compute_120",
            "-gencode=arch=compute_120,code=sm_120",
        ],
    )
    loader = _make_loader(tmp_path)

    assert loader._cuda_cache_fingerprint_payload() == [
        "cuda_ext=1",
        "visible_caps=12.0",
        (
            "resolved_arch_flags="
            "-gencode=arch=compute_120,code=compute_120,"
            "-gencode=arch=compute_120,code=sm_120"
        ),
    ]

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
    monkeypatch.setattr(
        cpp_module,
        "_effective_cuda_arch_flags",
        lambda **_kwargs: ["-gencode=arch=compute_120,code=sm_120"],
    )
    assert loader._cuda_cache_fingerprint_payload() == [
        "cuda_ext=1",
        "arch_list=12.0",
        "resolved_arch_flags=-gencode=arch=compute_120,code=sm_120",
    ]

    assert _make_loader(tmp_path, requires_cuda=False)._cuda_cache_fingerprint_payload() == [
        "cuda_ext=0"
    ]
