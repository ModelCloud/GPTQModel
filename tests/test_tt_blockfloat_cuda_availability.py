# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""SM support alone is not block-float native availability.

Regression for #3928.  ``tests/test_tt_blockfloat_precision.py`` gated its
native-only assertions on ``tt_blockfloat_cuda.supported()``, which only reads
the GPU architecture.  On an SM80+ CUDA runtime host without a CUDA toolkit the
JIT extension cannot load or build, so the native assertions ran and failed at
``assert layer._kernel_linear(x, layer.bias) is not None`` even though the dense
fallback was working.  ``tests/test_tt_blockfloat_cuda.py`` had the same
architecture-only gate.  ``tt_blockfloat_cuda.available()`` now reports whether
the extension is already loaded/registered, a cached or configured prebuilt
binary exists, or the source build tools are present.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from gptqmodel.models import _const
from gptqmodel.utils import cpp
from gptqmodel.utils import tt_blockfloat_cuda as bf_cuda


def _set_family(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(_const, "IS_ROCM", hip is not None)


def _patch_gate(
    monkeypatch,
    *,
    supported: bool,
    loadable: bool = False,
    prebuilt: Path | None = None,
    tools: bool = False,
):
    """Install one synthetic availability state on the extension probe."""

    _set_family(monkeypatch, None)
    monkeypatch.setattr(bf_cuda, "supported", lambda device=None: supported)
    monkeypatch.setattr(bf_cuda._EXTENSION, "loadable_without_build", lambda: loadable)
    monkeypatch.setattr(bf_cuda._EXTENSION, "configured_prebuilt_library", lambda: prebuilt)
    monkeypatch.setattr(bf_cuda._EXTENSION, "cuda_source_build_available", lambda: tools)


def test_supported_sm_without_a_binary_or_toolkit_is_unavailable(monkeypatch):
    _patch_gate(monkeypatch, supported=True, loadable=False, tools=False)

    assert bf_cuda.available() is False


def test_cached_or_registered_extension_is_available_without_build_tools(monkeypatch):
    _patch_gate(monkeypatch, supported=True, loadable=True, tools=False)

    assert bf_cuda.available() is True


def test_source_build_tools_make_the_extension_available(monkeypatch):
    _patch_gate(monkeypatch, supported=True, loadable=False, prebuilt=None, tools=True)

    assert bf_cuda.available() is True


def test_unsupported_architecture_stays_unavailable_with_build_tools(monkeypatch):
    _patch_gate(monkeypatch, supported=False, loadable=False, tools=True)

    assert bf_cuda.available() is False


def test_configured_missing_prebuilt_disables_the_source_build(monkeypatch):
    """``load()`` never falls back to a build when a prebuilt library is selected."""

    _patch_gate(monkeypatch, supported=True, loadable=False, prebuilt=Path("/missing/lib.so"), tools=True)

    assert bf_cuda.available() is False


def test_registered_operators_count_as_loadable_without_build(monkeypatch):
    _set_family(monkeypatch, None)
    monkeypatch.setattr(bf_cuda._EXTENSION, "_ops_available", lambda: True)
    monkeypatch.setattr(bf_cuda._EXTENSION, "_configured_prebuilt_library", lambda: None)

    assert bf_cuda._EXTENSION.loadable_without_build() is True


def test_cached_binary_counts_as_loadable_without_build(monkeypatch, tmp_path):
    _set_family(monkeypatch, None)
    build_root = tmp_path / "cache"
    build_root.mkdir()
    (build_root / "gptqmodel_blockfloat_cuda.so").touch()
    monkeypatch.setattr(bf_cuda._EXTENSION, "_ops_available", lambda: False)
    monkeypatch.setattr(bf_cuda._EXTENSION, "_configured_prebuilt_library", lambda: None)
    monkeypatch.setattr(bf_cuda._EXTENSION, "build_root", lambda: build_root)

    assert bf_cuda._EXTENSION.loadable_without_build() is True


def test_cuda_source_build_requires_cuda_home_nvcc_and_ninja(monkeypatch, tmp_path):
    _set_family(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setattr(cpp, "CUDA_HOME", str(toolkit))
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "ninja" else f"/tools/{name}")

    # A toolkit without ninja cannot run torch's JIT build.
    assert bf_cuda._EXTENSION.cuda_source_build_available() is False

    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")

    assert bf_cuda._EXTENSION.cuda_source_build_available() is True

    # The reported host: a CUDA runtime wheel with no toolkit at all.
    monkeypatch.setattr(cpp, "CUDA_HOME", None)

    assert bf_cuda._EXTENSION.cuda_source_build_available() is False

    # CUDA_HOME present but no nvcc under it.
    monkeypatch.setattr(cpp, "CUDA_HOME", str(tmp_path / "empty"))

    assert bf_cuda._EXTENSION.cuda_source_build_available() is False


def test_native_only_precision_assertions_skip_on_a_toolkitless_host(monkeypatch):
    """The reported repro: SM80+ GPU, no toolkit, no cached/prebuilt extension."""

    _patch_gate(monkeypatch, supported=True, loadable=False, prebuilt=None, tools=False)

    from tests import test_tt_blockfloat_cuda, test_tt_blockfloat_precision

    assert test_tt_blockfloat_precision._cuda_kernel_available() is False
    reason = test_tt_blockfloat_cuda._skip_reason()
    assert reason is not None
    assert "loadable or buildable" in reason


def test_dense_fallback_precision_tests_still_run_on_a_toolkitless_host(monkeypatch):
    """Only the native assertions are gated; the fallback oracle keeps running."""

    _patch_gate(monkeypatch, supported=True, loadable=False, prebuilt=None, tools=False)

    from tests import test_tt_blockfloat_precision

    for case in (test_tt_blockfloat_precision._bf16_cancellation, test_tt_blockfloat_precision._fp16_range):
        test_tt_blockfloat_precision.test_dense_fallback_accumulates_grid_and_bias_in_float32(case)
