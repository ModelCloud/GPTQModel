# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4815.

`TorchOpsJitExtension._cache_fingerprint` decides which cached JIT binary a
process may load. Its device payload only covered CUDA inputs, but `hipcc`
ignores `TORCH_CUDA_ARCH_LIST` and `get_device_capability()` is meaningless on
ROCm (gfx1201 and gfx1200 both report `(12, 0)`). A HIP binary built for one gfx
target was therefore reused on another, and every later kernel launch failed
with `invalid device function`.
"""

from types import SimpleNamespace

import torch

from gptqmodel.utils.cpp import TorchOpsJitExtension


HIP = "7.16.26385"


def _payload(*, hip=HIP, arch_env=None, devices=("gfx1201",)):
    payload = TorchOpsJitExtension._cuda_cache_fingerprint_payload(
        SimpleNamespace(requires_cuda=True)
    )
    assert isinstance(payload, list)
    return payload


def _rocm_payload(monkeypatch, *, hip=HIP, arch_env=None, devices=("gfx1201",)):
    monkeypatch.setattr(torch.version, "hip", hip)
    if arch_env is None:
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", arch_env)

    props = [SimpleNamespace(gcnArchName=name) for name in devices]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=None: props[int(index or 0)],
    )
    return _payload()


def test_pytorch_rocm_arch_selects_the_target(monkeypatch):
    first = _rocm_payload(monkeypatch, arch_env="gfx1100")
    second = _rocm_payload(monkeypatch, arch_env="gfx1201")

    assert any("gfx1100" in entry for entry in first)
    assert any("gfx1201" in entry for entry in second)
    assert first != second


def test_rocm_arch_list_is_normalized_and_order_independent(monkeypatch):
    ordered = _rocm_payload(monkeypatch, arch_env="gfx1201;gfx1100")
    reversed_list = _rocm_payload(monkeypatch, arch_env="gfx1100;gfx1201")
    suffixed = _rocm_payload(monkeypatch, arch_env="gfx1201:sramecc+:xnack-")

    assert ordered == reversed_list
    assert any("gfx1100,gfx1201" in entry for entry in ordered)
    assert any("gfx1201" in entry for entry in suffixed)


def test_visible_gcnarch_name_is_used_when_the_variable_is_unset(monkeypatch):
    r9700 = _rocm_payload(monkeypatch, devices=("gfx1201:sramecc+:xnack-",))
    rx9060 = _rocm_payload(monkeypatch, devices=("gfx1200",))

    assert any("gfx1201" in entry for entry in r9700)
    # The feature suffixes must not leak into the key.
    assert not any("sramecc" in entry for entry in r9700)
    assert r9700 != rx9060


def test_every_visible_device_arch_is_covered(monkeypatch):
    payload = _rocm_payload(monkeypatch, devices=("gfx90a", "gfx908"))

    assert any("gfx908,gfx90a" in entry for entry in payload)


def test_hip_version_is_part_of_the_key(monkeypatch):
    older = _rocm_payload(monkeypatch, hip="7.15.0")
    newer = _rocm_payload(monkeypatch, hip=HIP)

    assert older != newer


def test_rocm_never_reads_the_capability_pair(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("get_device_capability must not classify AMD hardware")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _fail)

    payload = _rocm_payload(monkeypatch)

    assert any(entry.startswith("rocm=") for entry in payload)
    assert not any(entry.startswith("visible_caps=") for entry in payload)


def test_cuda_payload_is_unchanged_without_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index=None: SimpleNamespace(gcnArchName="")
    )

    payload = _payload()

    assert payload[0] == "cuda_ext=1"
    assert not any(entry.startswith("rocm=") for entry in payload)
    assert any(entry == "visible_caps=12.0" for entry in payload)
