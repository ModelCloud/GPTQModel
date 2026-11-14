# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pre-import safety gate for the PyTorch runtime.

This module is deliberately standalone: it must run before ``gptqmodel.utils``
(which initializes the CUDA allocator) and before any kernel or checkpoint
code, so it cannot import the package's own submodules.

GPT-QModel loads checkpoints and executes kernels through whichever PyTorch it
is imported against. A PyTorch release carrying a known memory-corruption
advisory is therefore not a supported runtime, and starting anyway turns a
dependency problem into silent data corruption. The gate below fails closed and
names the fix.

``GPTQ_BYPASS_SAFETY_CHECKS`` is the only override. It exists for a pinned CI
matrix that deliberately covers older CUDA builds, and it is a pre-import
safety-valve rather than a behavior switch: no config object or API argument can
exist at this point in the import sequence.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from packaging.version import InvalidVersion, Version


BYPASS_ENV = "GPTQ_BYPASS_SAFETY_CHECKS"

# https://github.com/pytorch/pytorch/security/advisories/GHSA-rrmf-rvhw-rf47
# Memory corruption through ``torch.jit.script``. GitHub rates it Low (CVSS 5.3);
# what this gate encodes is the affected range and its patch boundary.
TORCH_ADVISORY_ID = "GHSA-rrmf-rvhw-rf47"
TORCH_ADVISORY_CVE = "CVE-2025-3000"
LAST_VULNERABLE_TORCH_VERSION = "2.12.1"
FIRST_SAFE_TORCH_VERSION = "2.13.0"

_TRUTHFUL = {"1", "true", "yes", "on", "y"}


class UnsafeTorchVersionError(RuntimeError):
    """The imported PyTorch carries a known memory-corruption advisory."""


def safety_checks_bypassed() -> bool:
    """Return whether ``GPTQ_BYPASS_SAFETY_CHECKS`` is set to a truthy value."""

    return os.getenv(BYPASS_ENV, "").strip().lower() in _TRUTHFUL


def _public_version(raw: str) -> Version:
    """Parse a runtime version without its local label.

    A ``+cu130``-style local label sorts after its base version under PEP 440,
    so ``Version("2.12.1+cu121") <= Version("2.12.1")`` is False and a naive
    comparison would miss exactly the CUDA builds that carry the advisory.
    """

    return Version(Version(raw).public)


def torch_version_is_vulnerable(version: Optional[str] = None) -> bool:
    """Return whether ``version`` (default: the imported Torch) is affected.

    An unparseable version raises rather than passing: an unidentifiable runtime
    cannot be cleared for checkpoint loading.
    """

    raw = torch.__version__ if version is None else version
    try:
        current = _public_version(raw)
    except InvalidVersion as exc:
        raise UnsafeTorchVersionError(
            f"GPT-QModel cannot verify the PyTorch version: {raw!r} is not a valid release "
            f"string. Install PyTorch {FIRST_SAFE_TORCH_VERSION} or newer, or set "
            f"{BYPASS_ENV}=1 to run an unverified build at your own risk."
        ) from exc
    return current <= Version(LAST_VULNERABLE_TORCH_VERSION)


def ensure_safe_torch(
    version: Optional[str] = None,
    *,
    bypass: Optional[bool] = None,
) -> None:
    """Raise ``UnsafeTorchVersionError`` on an affected PyTorch runtime.

    ``version`` and ``bypass`` exist for tests and embedding callers; production
    import uses the live ``torch.__version__`` and the environment flag.
    """

    if bypass is None:
        bypass = safety_checks_bypassed()
    if bypass:
        return
    if not torch_version_is_vulnerable(version):
        return

    raw = torch.__version__ if version is None else version
    raise UnsafeTorchVersionError(
        f"GPT-QModel requires PyTorch >= {FIRST_SAFE_TORCH_VERSION}: the imported runtime "
        f"({raw}) is affected by {TORCH_ADVISORY_ID} ({TORCH_ADVISORY_CVE}), a memory-corruption "
        "defect in torch.jit.script. Checkpoint loading and kernel execution are not supported "
        "on an affected runtime.\n"
        f"Upgrade with: pip install --upgrade 'torch>={FIRST_SAFE_TORCH_VERSION}'\n"
        f"To run this exact build anyway, set {BYPASS_ENV}=1. That override is unsupported."
    )


__all__ = [
    "BYPASS_ENV",
    "FIRST_SAFE_TORCH_VERSION",
    "LAST_VULNERABLE_TORCH_VERSION",
    "TORCH_ADVISORY_CVE",
    "TORCH_ADVISORY_ID",
    "UnsafeTorchVersionError",
    "ensure_safe_torch",
    "safety_checks_bypassed",
    "torch_version_is_vulnerable",
]
