# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ROCm Triton distributions must still get the autotuner thread-safety patch."""

import importlib.metadata

import pytest

from gptqmodel.utils import nogil_patcher


triton = pytest.importorskip("triton")
triton_autotuner = pytest.importorskip("triton.runtime.autotuner")


def _install_fake_autotuner(monkeypatch, *, drop_module_version):
    class _FakeAutotuner:
        pass

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    if drop_module_version:
        # Simulate a build whose distribution metadata is the only source.
        monkeypatch.delattr(triton, "__version__", raising=False)
    return _FakeAutotuner


@pytest.mark.parametrize("dist_name", ["triton-rocm", "pytorch-triton-rocm"])
def test_patch_applies_for_rocm_distribution_only(monkeypatch, dist_name):
    fake_autotuner = _install_fake_autotuner(monkeypatch, drop_module_version=True)

    def fake_version(package_name):
        if package_name == dist_name:
            return "3.8.0"
        raise importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(nogil_patcher, "version", fake_version)

    assert nogil_patcher.patch_triton_autotuner() is True
    assert getattr(fake_autotuner, "_gptqmodel_threadsafe", False) is True


def test_patch_prefers_module_version_over_distribution_metadata(monkeypatch):
    fake_autotuner = _install_fake_autotuner(monkeypatch, drop_module_version=False)

    def stale_version(_package_name):
        # Distribution metadata that would have made the patch bail out.
        return "3.4.0"

    monkeypatch.setattr(nogil_patcher, "version", stale_version)

    assert nogil_patcher.patch_triton_autotuner() is True
    assert getattr(fake_autotuner, "_gptqmodel_threadsafe", False) is True


def test_patch_warns_when_triton_version_cannot_be_detected(monkeypatch):
    _install_fake_autotuner(monkeypatch, drop_module_version=True)
    warnings = []

    class _FakeLog:
        def warning(self, message, *args, **_kwargs):
            warnings.append(message % args if args else message)

    def missing_version(package_name):
        raise importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(nogil_patcher, "log", _FakeLog())
    monkeypatch.setattr(nogil_patcher, "version", missing_version)

    assert nogil_patcher.patch_triton_autotuner() is False
    assert len(warnings) == 1
    assert "NOT active" in warnings[0]
