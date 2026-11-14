# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pre-import safety gate for the PyTorch runtime.

This module is deliberately standalone: it must run before ``gptqmodel.utils``
(which initializes the CUDA allocator) and before any kernel or checkpoint
code, so it cannot import the package's own submodules.

GPT-QModel loads checkpoints and executes kernels through whichever PyTorch it
is imported against. A PyTorch release carrying a known advisory that affects
that surface is therefore not a supported runtime, and starting anyway turns a
dependency problem into silent corruption or data disclosure. The gate below
fails closed and names the fix.

Each advisory is recorded as a release range rather than as one global
"minimum version", because the affected ranges do not form a single prefix:
``torch.jit.script`` is fixed in 2.13.0, while the ``weights_only`` file read is
introduced in 2.14.0 and fixed in 2.15.0. A single ``>= X`` bound would clear
2.14.x, which is why the table exists.

``GPTQ_BYPASS_SAFETY_CHECKS`` is the only override. It exists for a pinned CI
matrix and for the pinned Tenstorrent runtime, both of which deliberately run a
Torch below a patch boundary. It is a pre-import safety-valve rather than a
behavior switch: no config object or API argument can exist at this point in the
import sequence.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Tuple

import torch
from packaging.version import InvalidVersion, Version


BYPASS_ENV = "GPTQ_BYPASS_SAFETY_CHECKS"

_Release = Tuple[int, int, int]
_FLOOR: _Release = (0, 0, 0)

_TRUTHFUL = {"1", "true", "yes", "on", "y"}


def _format_release(release: _Release) -> str:
    return ".".join(str(part) for part in release)


@dataclasses.dataclass(frozen=True)
class TorchAdvisory:
    """One PyTorch advisory expressed as a release range.

    ``introduced`` is inclusive and ``patched`` is the first release that
    contains the fix, so the affected releases are
    ``[introduced, patched)``. ``introduced=None`` means every release before
    the patch is affected.
    """

    advisory_id: str
    cve_id: Optional[str]
    summary: str
    introduced: Optional[_Release]
    patched: _Release
    url: str

    def matches(self, version: Version) -> bool:
        """Return whether ``version`` is inside this advisory's range.

        Matching uses the release tuple so that ``2.14.0.dev1`` (which predates
        the 2.14.0 release but already carries the defect) is caught. A
        pre-release of the *patched* release is treated as affected as well,
        because ``2.15.0.dev1`` predates the 2.15.0 fix.
        """

        release = _release_tuple(version)
        if release > self.patched:
            return False
        if release == self.patched and not version.is_prerelease:
            return False
        if self.introduced is not None and release < self.introduced:
            return False
        return True

    def patched_version(self) -> str:
        return _format_release(self.patched)


# https://github.com/advisories/GHSA-rrmf-rvhw-rf47
# Memory corruption through ``torch.jit.script``. GitHub rates it Low
# (CVSS 3.1 5.3); what this gate encodes is the affected range and its patch.
TORCH_ADVISORIES: Tuple[TorchAdvisory, ...] = (
    TorchAdvisory(
        advisory_id="GHSA-rrmf-rvhw-rf47",
        cve_id="CVE-2025-3000",
        summary="memory corruption in `torch.jit.script`",
        introduced=None,
        patched=(2, 13, 0),
        url="https://github.com/advisories/GHSA-rrmf-rvhw-rf47",
    ),
    # https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83
    # ``torch.load(weights_only=True)`` reads arbitrary files from a malicious
    # checkpoint. No CVE is assigned; it is a repository advisory only.
    TorchAdvisory(
        advisory_id="GHSA-v32g-mr63-3j83",
        cve_id=None,
        summary="`torch.load(weights_only=True)` arbitrary file read",
        introduced=(2, 14, 0),
        patched=(2, 15, 0),
        url="https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83",
    ),
)

# Compatibility aliases for the first advisory, which introduced this module.
# ``TORCH_ADVISORIES`` is the authoritative table; these describe only the
# ``torch.jit.script`` range and are kept because callers and tests use them.
TORCH_ADVISORY_ID = TORCH_ADVISORIES[0].advisory_id
TORCH_ADVISORY_CVE = TORCH_ADVISORIES[0].cve_id
LAST_VULNERABLE_TORCH_VERSION = "2.12.1"
FIRST_SAFE_TORCH_VERSION = TORCH_ADVISORIES[0].patched_version()


class UnsafeTorchVersionError(RuntimeError):
    """The imported PyTorch carries a known advisory."""


def safety_checks_bypassed() -> bool:
    """Return whether ``GPTQ_BYPASS_SAFETY_CHECKS`` is set to a truthy value."""

    return os.getenv(BYPASS_ENV, "").strip().lower() in _TRUTHFUL


def _release_tuple(version: Version) -> _Release:
    """Return the ``major.minor.patch`` release tuple, zero-padded."""

    parts = list(version.release[:3])
    parts.extend([0] * (3 - len(parts)))
    return parts[0], parts[1], parts[2]


def _parse(version: Optional[str]) -> Version:
    """Parse a runtime version, raising when it cannot be identified.

    A ``+cu130``-style local label sorts after its base version under PEP 440,
    so ``Version("2.12.1+cu121") <= Version("2.12.1")`` is False. The public
    version is used for range membership so a local label can never hide an
    affected base release.
    """

    raw = torch.__version__ if version is None else version
    try:
        return Version(Version(raw).public)
    except InvalidVersion as exc:
        raise UnsafeTorchVersionError(
            f"GPT-QModel cannot verify the PyTorch version: {raw!r} is not a valid release "
            f"string. Supported releases are {safe_torch_releases_hint()}, or set "
            f"{BYPASS_ENV}=1 to run an unverified build at your own risk."
        ) from exc


def _safe_release_ranges() -> Tuple[Tuple[_Release, Optional[_Release]], ...]:
    """Complement of the affected ranges, as ``[low, high)`` release pairs."""

    affected = sorted(
        ((advisory.introduced or _FLOOR, advisory.patched) for advisory in TORCH_ADVISORIES),
        key=lambda bounds: bounds[0],
    )
    ranges = []
    cursor = _FLOOR
    for low, high in affected:
        if low > cursor:
            ranges.append((cursor, low))
        if high > cursor:
            cursor = high
    ranges.append((cursor, None))
    return tuple(ranges)


def safe_torch_releases_hint() -> str:
    """Render the releases that are outside every advisory, e.g. ``2.13.x or >= 2.15.0``."""

    parts = []
    for low, high in _safe_release_ranges():
        if high is None:
            parts.append(f">= {_format_release(low)}")
        elif high[2] == 0 and high[0] == low[0] and high[1] == low[1] + 1:
            # A complete minor line, e.g. 2.13.0 <= v < 2.14.0.
            parts.append(f"{low[0]}.{low[1]}.x")
        else:
            parts.append(f">= {_format_release(low)}, < {_format_release(high)}")
    return " or ".join(parts)


def matching_torch_advisories(version: Optional[str] = None) -> Tuple[TorchAdvisory, ...]:
    """Return every advisory whose range contains ``version``."""

    current = _parse(version)
    return tuple(advisory for advisory in TORCH_ADVISORIES if advisory.matches(current))


def torch_version_is_vulnerable(version: Optional[str] = None) -> bool:
    """Return whether ``version`` (default: the imported Torch) is affected."""

    return bool(matching_torch_advisories(version))


def _unsafe_runtime_message(raw: str, matches: Tuple[TorchAdvisory, ...]) -> str:
    lines = [
        "GPT-QModel cannot start on this PyTorch runtime: the imported release "
        f"({raw}) is affected by:"
    ]
    for advisory in matches:
        cve = f" ({advisory.cve_id})" if advisory.cve_id else ""
        lines.append(
            f"  - {advisory.advisory_id}{cve}: {advisory.summary}; "
            f"patched in {advisory.patched_version()}"
        )
        lines.append(f"    {advisory.url}")
    target = max(advisory.patched for advisory in matches)
    lines.append(f"Upgrade with: pip install --upgrade 'torch>={_format_release(target)}'")
    lines.append(f"Supported releases are {safe_torch_releases_hint()}.")
    lines.append(
        f"To run this exact build anyway, set {BYPASS_ENV}=1. That override is unsupported."
    )
    return "\n".join(lines)


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
    matches = matching_torch_advisories(version)
    if not matches:
        return
    raw = torch.__version__ if version is None else version
    raise UnsafeTorchVersionError(_unsafe_runtime_message(raw, matches))


__all__ = [
    "BYPASS_ENV",
    "FIRST_SAFE_TORCH_VERSION",
    "LAST_VULNERABLE_TORCH_VERSION",
    "TORCH_ADVISORIES",
    "TORCH_ADVISORY_CVE",
    "TORCH_ADVISORY_ID",
    "TorchAdvisory",
    "UnsafeTorchVersionError",
    "ensure_safe_torch",
    "matching_torch_advisories",
    "safe_torch_releases_hint",
    "safety_checks_bypassed",
    "torch_version_is_vulnerable",
]
