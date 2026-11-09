# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AMD last-level-cache back-port.

The interesting behaviour is all defensive: the patch must never raise, must be
idempotent, must leave ``L2_cache_size`` alone, and must decline cleanly when
the torch internals it targets have moved.
"""

from __future__ import annotations

import textwrap

import pytest
import torch

from gptqmodel import rocm_llc


class _Props:
    L2_cache_size = 8 * 1024 * 1024

    class uuid:  # noqa: N801 - mirrors the pybind attribute name
        bytes = list(b"0123456789abcdef")


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Each test starts from "not applied" regardless of package import."""

    rocm_llc.unapply()
    monkeypatch.setattr(rocm_llc, "_STATE", {"applied": False, "restore": [], "status": {}})
    monkeypatch.delenv("GPTQMODEL_NO_ROCM_LLC", raising=False)
    yield
    rocm_llc.unapply()


def test_apply_is_idempotent_and_never_raises():
    first = rocm_llc.apply()
    second = rocm_llc.apply()
    assert first == second
    expected = {"props", "benchmarker", "cudagraph"} if torch.version.hip else {"props"}
    assert set(first) == expected
    for value in first.values():
        assert isinstance(value, str) and value


def test_env_opt_out(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_NO_ROCM_LLC", "1")
    status = rocm_llc.apply()
    assert status["props"].startswith("skipped")
    assert "GPTQMODEL_NO_ROCM_LLC" in status["props"]
    assert set(status) == {"props"}


def test_non_rocm_reports_not_rocm(monkeypatch):
    monkeypatch.setattr(rocm_llc.torch.version, "hip", None)
    status = rocm_llc.apply()
    assert status == {"props": "skipped: not ROCm"}


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


def test_cudagraph_rewrite_matches_upstream_shape():
    snippet = textwrap.dedent(
        """\
        def do_bench_with_cudagraphs(self):
            if indices:
                props = device_interface.get_device_properties(device_idx)
                cache_size = next(
                    (
                        getattr(props, attr)
                        for attr in ("L2_cache_size", "last_level_cache_size")
                        if getattr(props, attr, None)
                    ),
                    256 * 1024 * 1024,
                )
                pool_size = 4
        """
    )
    patched, count = rocm_llc._CUDAGRAPH_RE.subn(rocm_llc._cudagraph_replacement, snippet)
    assert count == 1
    assert "max(" in patched
    assert "256 * 1024 * 1024" in patched
    # Still valid Python, and still keeps the surrounding block.
    compile(patched, "<test>", "exec")
    assert "pool_size = 4" in patched


def test_cudagraph_rewrite_declines_unknown_source():
    snippet = "def do_bench_with_cudagraphs(self):\n    cache_size = 42\n"
    patched, count = rocm_llc._CUDAGRAPH_RE.subn(rocm_llc._cudagraph_replacement, snippet)
    assert count == 0
    assert patched == snippet


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_props_expose_last_level_cache_after_apply():
    rocm_llc.apply()
    props = torch.cuda.get_device_properties(0)
    assert hasattr(props, "last_level_cache_size")
    assert props.last_level_cache_size >= props.L2_cache_size
    assert props.L2_cache_size == props.L2_cache_size  # untouched by the patch


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_benchmarker_uses_max_of_l2_and_llc():
    from torch._inductor.runtime.benchmarking import InductorBenchmarker

    rocm_llc.apply()
    props = torch.cuda.get_device_properties(0)
    expected = max(props.L2_cache_size, props.last_level_cache_size)
    assert InductorBenchmarker().get_device_cache_size() == expected
