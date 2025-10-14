# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Environment variable helpers used across GPTQModel."""

from __future__ import annotations

import os


_TRUTHY = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: str | None = "0") -> bool:
    """Return ``True`` when an env var is set to a truthy value."""

    value = os.getenv(name, default)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


__all__ = ["env_flag"]
