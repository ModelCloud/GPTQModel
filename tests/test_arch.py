# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Mocked coverage for the shared ROCm architecture abstraction."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils import arch


def _props(name, warp_size=64):
    return SimpleNamespace(name=name, gcnArchName=name, warp_size=warp_size)


@pytest.fixture(autouse=True)
def _reset_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _fake_device(monkeypatch, name, *, hip="7.16.26385", warp_size=64):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=0: _props(name, warp_size=warp_size),
    )
    return torch.device("cuda:0")


def test_normalize_arch_strips_features_and_lowercases():
    assert arch.normalize_arch("gfx1201:sramecc+:xnack-") == "gfx1201"
    assert arch.normalize_arch(" GFX950 ") == "gfx950"
    assert arch.normalize_arch(None) == ""
    assert arch.normalize_arch("") == ""


def test_rocm_arch_returns_normalized_name(monkeypatch):
    device = _fake_device(monkeypatch, "gfx1201:sramecc+:xnack-")
    assert arch.rocm_arch(device) == "gfx1201"
    # Cached: a later property failure must not change the answer.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=0: _props("gfx942"))
    assert arch.rocm_arch(device) == "gfx1201"


@pytest.mark.parametrize(
    ("name", "fused", "cls", "wave"),
    [
        ("gfx950", True, "cdna4", 64),
        ("gfx942", False, "other", 64),
        ("gfx1100", False, "rdna3", 32),
        ("gfx1200", True, "rdna4", 32),
        ("gfx1201:sramecc+:xnack-", True, "rdna4", 32),
        ("gfx1250", True, "rdna4", 32),
    ],
)
def test_classification_table(monkeypatch, name, fused, cls, wave):
    device = _fake_device(monkeypatch, name)
    assert arch.is_rocm_fused_arch(device) is fused
    assert arch.arch_class(device) == cls
    assert arch.wave_size(device) == wave
    assert arch.is_gfx950(device) is name.startswith("gfx950")
    assert arch.is_rdna4(device) is (cls == "rdna4")


def test_gate_limits_keep_cdna_values_and_disable_unknown(monkeypatch):
    assert arch.gate_limits("gfx950") == (16, 5120)
    # RDNA4 re-measured on gfx1201: the CDNA 5120 K-loop bound does not apply.
    assert arch.gate_limits("gfx1201:sramecc+") == (16, 17408)
    assert arch.gate_limits("gfx942") == (0, 0)
    assert arch.gate_limits("gfx1100") == (0, 0)
    assert arch.gate_limits("gfx9999") == (0, 0)


def test_unknown_and_missing_hip_are_not_fused(monkeypatch):
    device = _fake_device(monkeypatch, "gfx9999")
    assert arch.rocm_arch(device) == "gfx9999"
    assert arch.arch_class(device) == "other"
    assert arch.is_rocm_fused_arch(device) is False

    device = _fake_device(monkeypatch, "NVIDIA GeForce RTX 4090", hip=None)
    assert arch.rocm_arch(device) is None
    assert arch.arch_class(device) == "other"
    assert arch.is_rocm_fused_arch(device) is False
    assert arch.is_rdna4(device) is False
    assert arch.wave_size(device) == 32


def test_sm89_device_properties_without_gcnarch(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=0: SimpleNamespace(name="NVIDIA GeForce RTX 4090", warp_size=32),
    )
    device = torch.device("cuda:0")
    assert arch.rocm_arch(device) is None
    assert arch.is_rocm_fused_arch(device) is False
    assert arch.gate_limits(device) == (0, 0)


def test_wave_size_string_form_matches_device_form(monkeypatch):
    # gfx9x non-cdna4 targets are wave64; the string form must agree with the
    # device form instead of falling through to a device-0 query.
    _fake_device(monkeypatch, "gfx942", warp_size=64)
    assert arch.wave_size("gfx942") == 64
    assert arch.wave_size(torch.device("cuda:0")) == 64
    assert arch.wave_size("gfx90a") == 64
    assert arch.wave_size("gfx1100") == 32


def test_non_cuda_device_reports_no_rocm_arch(monkeypatch):
    _fake_device(monkeypatch, "gfx1201")

    def _forbidden_current_device():
        raise AssertionError("a non-cuda device must not initialize the GPU context")

    monkeypatch.setattr(torch.cuda, "current_device", _forbidden_current_device)
    for device in (torch.device("cpu"), torch.device("meta")):
        assert arch.rocm_arch(device) is None
        assert arch.arch_class(device) == "other"
        assert arch.gate_limits(device) == (0, 0)
        assert arch.is_rocm_fused_arch(device) is False
        assert arch.wave_size(device) == 32


def test_property_query_failure_returns_none(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")

    def boom(index=0):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert arch.rocm_arch(torch.device("cuda:0")) is None
    assert arch.is_rocm_fused_arch(torch.device("cuda:0")) is False
