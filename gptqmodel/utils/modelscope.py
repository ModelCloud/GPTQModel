# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Final

from .env import env_flag


_ENV_VAR: Final[str] = "GPTQMODEL_USE_MODELSCOPE"


def modelscope_requested() -> bool:
    """
    Return ``True`` when the user explicitly enabled ModelScope integration
    via the GPTQMODEL_USE_MODELSCOPE environment variable.
    """
    return env_flag(_ENV_VAR, default="0")


def ensure_modelscope_available() -> bool:
    """
    Ensure the ModelScope package is available if requested.

    Returns:
        bool: ``True`` when ModelScope was requested and successfully imported,
        otherwise ``False``.

    Raises:
        ModuleNotFoundError: When the environment variable enables ModelScope
        but the package is not installed.
    """
    if not modelscope_requested():
        return False

    try:
        import modelscope  # noqa: F401
    except Exception as exc:
        raise ModuleNotFoundError(
            "env `GPTQMODEL_USE_MODELSCOPE` used but modelscope pkg is not found: "
            "please install with `pip install modelscope`."
        ) from exc

    return True
