# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from .env import env_flag


log = logging.getLogger(__name__)

_PACK_BLOCK_EXTENSION: Optional[bool] = None
_PACK_BLOCK_EXTENSION_INITIALISED = False


def load_pack_block_extension(*, verbose: bool = False) -> Optional[object]:
    """Ensure the pack_block CPU extension is built and loaded.

    Returns ``True`` when the extension is available, ``None`` otherwise.
    The function is idempotent and caches its result to avoid repeated builds.
    """

    global _PACK_BLOCK_EXTENSION, _PACK_BLOCK_EXTENSION_INITIALISED

    if hasattr(torch.ops.gptqmodel, "pack_block_cpu"):
        _PACK_BLOCK_EXTENSION_INITIALISED = True
        _PACK_BLOCK_EXTENSION = True
        return _PACK_BLOCK_EXTENSION

    if _PACK_BLOCK_EXTENSION_INITIALISED and _PACK_BLOCK_EXTENSION is not None:
        return _PACK_BLOCK_EXTENSION

    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "pack_block_cpu.cpp"
    if not source_path.exists():
        source_path = project_root / "gptqmodel_ext" / "pack_block_cpu.cpp"
    if not source_path.exists():
        log.debug("pack_block_cpu extension source not found at %s", source_path)
        _PACK_BLOCK_EXTENSION = None
        _PACK_BLOCK_EXTENSION_INITIALISED = True
        return None

    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags: list[str] = []

    build_dir = os.getenv("GPTQMODEL_EXT_BUILD")

    if not verbose:
        verbose = env_flag("GPTQMODEL_EXT_VERBOSE", True)

    try:
        load(
            name="gptqmodel_pack_block_cpu",
            sources=[str(source_path)],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=build_dir,
            verbose=verbose,
            is_python_module=False,
        )
        log.debug("pack_block_cpu extension loaded from %s", source_path)
        _PACK_BLOCK_EXTENSION = True
    except Exception as exc:  # pragma: no cover - environment-specific
        log.debug("pack_block_cpu extension build failed: %s", exc)
        _PACK_BLOCK_EXTENSION = None
    _PACK_BLOCK_EXTENSION_INITIALISED = True
    return _PACK_BLOCK_EXTENSION
