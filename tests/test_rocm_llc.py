# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AMD last-level-cache *report*.

GPT-QModel only reads the cache geometry; it never patches torch or Inductor.
"""

from __future__ import annotations

import torch

from gptqmodel import rocm_llc


class _Props:
    L2_cache_size = 8 * 1024 * 1024

    class uuid:  # noqa: N801 - mirrors the pybind attribute name
        bytes = list(b"0123456789abcdef")


def test_import_leaves_torch_device_properties_alone():
    props = getattr(torch._C, "_CudaDeviceProperties", None)
    if props is not None:
        assert "last_level_cache_size" not in vars(props)
    assert not hasattr(rocm_llc, "apply")


def test_last_level_cache_size_falls_back_to_l2(monkeypatch):
    monkeypatch.setattr(rocm_llc.torch.cuda, "get_device_properties", lambda device=None: _Props())
    monkeypatch.setattr(rocm_llc, "cache_levels", lambda device=None: None)
    assert rocm_llc.last_level_cache_size(0) == _Props.L2_cache_size


def test_last_level_cache_size_prefers_outermost_level(monkeypatch):
    monkeypatch.setattr(rocm_llc.torch.cuda, "get_device_properties", lambda device=None: _Props())
    # L4 is smaller than L3 but is still the outermost level.
    monkeypatch.setattr(rocm_llc, "cache_levels", lambda device=None: (32 * 1024, 8 << 20, 64 << 20, 4 << 20))
    assert rocm_llc.last_level_cache_size(0) == 4 << 20
    monkeypatch.setattr(rocm_llc, "cache_levels", lambda device=None: (32 * 1024, 8 << 20, 64 << 20, 0))
    assert rocm_llc.last_level_cache_size(0) == 64 << 20
    monkeypatch.setattr(rocm_llc, "cache_levels", lambda device=None: (0, 8 << 20, 0, 0))
    assert rocm_llc.last_level_cache_size(0) == _Props.L2_cache_size


def test_device_uuid_rejects_non_hex(monkeypatch):
    class _Bad:
        L2_cache_size = 8 << 20

        class uuid:  # noqa: N801
            bytes = list(b"GPU-0123456789ab")

    monkeypatch.setattr(rocm_llc.torch.cuda, "get_device_properties", lambda device=None: _Bad())
    monkeypatch.setattr(rocm_llc, "_api", None)
    monkeypatch.setattr(rocm_llc, "_api_error", None)
    monkeypatch.setattr(rocm_llc, "_levels", {})
    monkeypatch.setattr(rocm_llc.torch.version, "hip", "7.0")
    assert rocm_llc.cache_levels(0) is None
