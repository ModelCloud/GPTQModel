# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU coverage for the ROCm >= 10.1 support floor."""

import pytest
import torch

from gptqmodel.utils import rocm


class _Recorder:
    def __init__(self):
        self.messages = []

    def warn(self, message):
        self.messages.append(message)


def _patch_rocm(monkeypatch, *, release, torch_version, is_rocm=True):
    monkeypatch.setattr(rocm, "IS_ROCM", is_rocm)
    monkeypatch.setattr(torch.version, "rocm", release, raising=False)
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(rocm, "_UNSUPPORTED_ROCM_WARNED", False)


def test_rocm_version_prefers_torch_version_rocm(monkeypatch):
    _patch_rocm(monkeypatch, release="10.1.0", torch_version="2.16.0+rocm6.0")
    # ROCm 10.1 ships HIP 7.16; the HIP runtime version must never win.
    monkeypatch.setattr(torch.version, "hip", "7.16.26385", raising=False)

    assert rocm.rocm_version() == (10, 1, 0)
    assert rocm.rocm_version_string() == "10.1.0"


def test_rocm_version_falls_back_to_torch_local_tag(monkeypatch):
    _patch_rocm(monkeypatch, release=None, torch_version="2.4.0+rocm6.2.41133")

    assert rocm.rocm_version() == (6, 2, 41133)


def test_rocm_version_is_none_off_rocm(monkeypatch):
    _patch_rocm(monkeypatch, release="10.1.0", torch_version="2.16.0+rocm10.1", is_rocm=False)

    assert rocm.rocm_version() is None
    assert rocm.is_supported_rocm_version() is None
    assert rocm.warn_unsupported_rocm(_Recorder()) is False


def test_rocm_version_never_parses_the_hip_version(monkeypatch):
    _patch_rocm(monkeypatch, release=None, torch_version="2.16.0")
    monkeypatch.setattr(torch.version, "hip", "7.16.26385", raising=False)

    assert rocm.rocm_version() is None


@pytest.mark.parametrize(
    ("release", "supported"),
    [
        ("10.1.0", True),
        ("10.1", True),
        ("10.2.0", True),
        ("11.0.0", True),
        ("10.0.0", False),
        ("6.4.3", False),
        ("5.7.1", False),
        (None, None),
    ],
)
def test_rocm_support_boundary(monkeypatch, release, supported):
    _patch_rocm(
        monkeypatch,
        release=release,
        torch_version="2.16.0" if release is None else f"2.16.0+rocm{release}",
    )

    assert rocm.is_supported_rocm_version() is supported


def test_old_rocm_warns_once_with_actionable_message(monkeypatch):
    _patch_rocm(monkeypatch, release="6.4.3", torch_version="2.16.0+rocm6.4.3")
    recorder = _Recorder()

    assert rocm.warn_unsupported_rocm(recorder) is True
    assert rocm.warn_unsupported_rocm(recorder) is False

    assert len(recorder.messages) == 1
    message = recorder.messages[0]
    assert "6.4.3" in message
    assert "10.1" in message
    assert "upgrade" in message.lower()


@pytest.mark.parametrize("release", ["10.1.0", "10.2.0", None])
def test_supported_or_undetectable_rocm_does_not_warn(monkeypatch, release):
    _patch_rocm(
        monkeypatch,
        release=release,
        torch_version="2.16.0" if release is None else f"2.16.0+rocm{release}",
    )
    recorder = _Recorder()

    assert rocm.warn_unsupported_rocm(recorder) is False
    assert recorder.messages == []
