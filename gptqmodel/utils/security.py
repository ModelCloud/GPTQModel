# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Public surface for the runtime safety gates.

The implementation lives in ``gptqmodel._torch_safety`` because it must run
before ``gptqmodel.utils`` is imported. Import it from here in ordinary code;
import the private module only from the package bootstrap.
"""

from __future__ import annotations

from .._torch_safety import (
    BYPASS_ENV,
    FIRST_SAFE_TORCH_VERSION,
    LAST_VULNERABLE_TORCH_VERSION,
    TORCH_ADVISORIES,
    TORCH_ADVISORY_CVE,
    TORCH_ADVISORY_ID,
    TorchAdvisory,
    UnsafeTorchVersionError,
    ensure_safe_torch,
    matching_torch_advisories,
    safe_torch_releases_hint,
    safety_checks_bypassed,
    torch_version_is_vulnerable,
)


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
