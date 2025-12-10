# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load, _get_build_directory

from .env import env_flag

log = logging.getLogger(__name__)

_PACK_BLOCK_EXTENSION: Optional[bool] = None
_PACK_BLOCK_EXTENSION_INITIALISED = False

_cpp_ext_lock = threading.Lock()

# Used to track whether cleanup has been done already
_cpp_ext_initialized = False


def safe_load_cpp_ext(
        name,
        build_directory=None,
        verbose=False,
        **kwargs
):
    """
    A safe wrapper for torch.utils.cpp_extension.load()
    """
    global _cpp_ext_initialized

    with _cpp_ext_lock:
        # First-time initialization cleanup
        if not _cpp_ext_initialized:
            build_directory = build_directory or _get_build_directory(name, verbose=verbose)
            if os.path.exists(build_directory):
                try:
                    shutil.rmtree(build_directory)
                    if verbose:
                        log.debug(f"[safe_cpp_extension_load] Removed old build directory: {build_directory}")
                except Exception as e:
                    log.error(f"[safe_cpp_extension_load] Failed to remove build directory: {e}")

            if not os.path.exists(build_directory):
                if verbose:
                    log.debug('Creating extension directory %s...', build_directory)
                # This is like mkdir -p, i.e. will also create parent directories.
                os.makedirs(build_directory, exist_ok=True)

            # Load the extension (JIT)
            load(
                name=name,
                build_directory=build_directory,
                **kwargs
            )

            _cpp_ext_initialized = True

        return


def load_pack_block_extension(*, verbose: bool = False) -> Optional[object]:
    """Ensure the pack_block CPU extension is built and loaded."""

    global _PACK_BLOCK_EXTENSION, _PACK_BLOCK_EXTENSION_INITIALISED

    if hasattr(torch.ops.gptqmodel, "pack_block_cpu"):
        _PACK_BLOCK_EXTENSION_INITIALISED = True
        _PACK_BLOCK_EXTENSION = True
        return _PACK_BLOCK_EXTENSION

    if _PACK_BLOCK_EXTENSION_INITIALISED and _PACK_BLOCK_EXTENSION:
        return _PACK_BLOCK_EXTENSION

    try:
        spec = importlib.util.find_spec("gptqmodel_pack_block_cpu")
    except (ModuleNotFoundError, AttributeError):
        spec = None

    if spec and spec.origin:
        try:
            torch.ops.load_library(spec.origin)
            if hasattr(torch.ops.gptqmodel, "pack_block_cpu"):
                log.debug("pack_block_cpu extension loaded from %s", spec.origin)
                _PACK_BLOCK_EXTENSION = True
                _PACK_BLOCK_EXTENSION_INITIALISED = True
                return _PACK_BLOCK_EXTENSION
        except Exception as exc:  # pragma: no cover - environment-specific
            log.debug("pack_block_cpu prebuilt load failed: %s", exc)

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
        safe_load_cpp_ext(
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
