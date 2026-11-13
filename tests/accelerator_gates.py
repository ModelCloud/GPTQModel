# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Accelerator-family gates shared by kernel tests.

``torch.cuda.is_available()`` is true on ROCm because HIP reuses the CUDA
device namespace, and ``torch.cuda.get_device_capability()`` reports an
unrelated numeric pair there (gfx1201 returns ``(12, 0)``). A test that only
compares that pair therefore classifies an AMD device as an NVIDIA SM120 one.

These helpers ask the build/runtime instead: a HIP build is rejected before a
capability is read, and the NVIDIA-specific checks run only on CUDA builds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


_nf12_build_error: str | None = None
_nf12_probed = False


def cuda_only_skip_reason(kernel: str) -> str | None:
    """Return a skip reason for a CUDA-only kernel on a HIP build."""
    if torch.version.hip is not None:
        return f"{kernel} is CUDA-only; ROCm (HIP) is not supported"
    return None


def _capability(index: int) -> tuple[int, int] | None:
    try:
        return tuple(torch.cuda.get_device_capability(index))
    except Exception:
        return None


def _device_list(indices) -> str:
    described = []
    for index in indices:
        capability = _capability(index)
        described.append(
            f"cuda:{index}=sm{capability[0]}{capability[1]}"
            if capability is not None
            else f"cuda:{index}=unknown"
        )
    return ", ".join(described)


def swordfish_supported_devices() -> list[int]:
    """Return every visible ordinal that Swordfish accepts, in index order.

    A mixed host can expose a supported Blackwell part next to an older one, so
    the gate has to resolve support per GPU instead of from one capability.
    """
    if cuda_only_skip_reason("Swordfish") is not None:
        return []
    if not torch.cuda.is_available():
        return []

    from gptqmodel.utils.swordfish import _validate_swordfish_device_support

    supported = []
    for index in range(torch.cuda.device_count()):
        target = torch.device("cuda", index)
        try:
            ok = _validate_swordfish_device_support(target)
        except (RuntimeError, AssertionError):
            ok = False
        if ok:
            supported.append(index)
    return supported


def swordfish_test_device() -> torch.device:
    """Pick the first supported ordinal, so tests never target an older GPU."""
    supported = swordfish_supported_devices()
    if supported:
        return torch.device("cuda", supported[0])
    return torch.device("cuda", torch.cuda.current_device())


def swordfish_skip_reason(device: torch.device | None = None) -> str | None:
    """Return why Swordfish cannot run on ``device`` (or on any visible GPU).

    Sweeping ``device=None`` asks whether *some* visible GPU is supported; pass
    an explicit device to validate that one. Swordfish owns the Blackwell
    decision (its extension loader pins the exact cubin/PTX variants), so the
    test does not duplicate that SM table.
    """
    reason = cuda_only_skip_reason("Swordfish")
    if reason is not None:
        return reason
    if not torch.cuda.is_available():
        return "CUDA is unavailable"

    if device is None:
        if swordfish_supported_devices():
            return None
        return (
            "Swordfish requires Blackwell (sm100+); no visible device is "
            f"supported ({_device_list(range(torch.cuda.device_count()))})"
        )

    from gptqmodel.utils.swordfish import _validate_swordfish_device_support

    try:
        supported = _validate_swordfish_device_support(device)
    except (RuntimeError, AssertionError) as exc:
        return f"Swordfish device validation failed: {exc}"
    if supported:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return (
        f"Swordfish requires Blackwell (sm100+); found "
        f"{_device_list([index])}"
    )


def _nvcc() -> str | None:
    """Locate nvcc the same way ``gptqmodel_nf12._jit`` does."""
    from torch.utils.cpp_extension import CUDA_HOME

    candidate = Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else "nvcc"
    return shutil.which(str(candidate))


def nf12_skip_reason() -> str | None:
    """Return why the vendored NF12 native operators cannot be used here."""
    global _nf12_build_error, _nf12_probed

    reason = cuda_only_skip_reason("gptqmodel_nf12")
    if reason is not None:
        return reason
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    # The build derives its -gencode list from every visible GPU and rejects
    # the set if any of them is below SM80, so fail before a doomed build.
    unsupported = [
        index
        for index in range(torch.cuda.device_count())
        if (_capability(index) or (0, 0))[0] < 8
    ]
    if unsupported:
        return (
            "NF12 needs SM80 or newer on every visible device; unsupported: "
            f"{_device_list(unsupported)}"
        )

    missing = [name for name in ("cmake", "ninja") if shutil.which(name) is None]
    if _nvcc() is None:
        missing.append("nvcc")
    if missing:
        return (
            "gptqmodel_nf12 native operators require "
            f"{', '.join(missing)} (CUDA toolkit 13.0+, CMake 3.26+); "
            "the source build is unavailable"
        )

    # The toolchain can be present while the build still fails (for example an
    # older toolkit). Probe the real load once and report its error verbatim.
    if not _nf12_probed:
        try:
            import gptqmodel_nf12 as nf12

            nf12.load()
        except Exception as exc:
            _nf12_build_error = f"gptqmodel_nf12 native operators are unavailable: {exc}"
        finally:
            _nf12_probed = True
    return _nf12_build_error
