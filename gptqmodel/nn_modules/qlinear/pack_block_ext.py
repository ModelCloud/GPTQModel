# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

log = logging.getLogger(__name__)

_EXTENSION = None
_EXTENSION_INITIALISED = False


def _load_extension() -> Optional[object]:
    global _EXTENSION, _EXTENSION_INITIALISED
    if hasattr(torch.ops.gptqmodel, "pack_block_cpu"):
        _EXTENSION_INITIALISED = True
        _EXTENSION = True
        return _EXTENSION

    if _EXTENSION_INITIALISED and _EXTENSION is not None:
        return _EXTENSION

    source_path = Path(__file__).resolve().parents[3] / "pack_block_cpu.cpp"
    if not source_path.exists():
        # Fallback to repository root/gptqmodel_ext
        source_path = Path(__file__).resolve().parents[3] / "gptqmodel_ext" / "pack_block_cpu.cpp"
    if not source_path.exists():
        log.debug("pack_block_cpu extension source not found at %s", source_path)
        _EXTENSION = None
        _EXTENSION_INITIALISED = True
        return None

    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags = []

    build_dir = os.environ.get("GPTQMODEL_EXT_BUILD", None)

    try:
        load(
            name="gptqmodel_pack_block_cpu",
            sources=[str(source_path)],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=build_dir,
            verbose=False,
            is_python_module=False,
        )
        log.debug("pack_block_cpu extension loaded from %s", source_path)
        _EXTENSION = True
    except Exception as exc:  # pragma: no cover - environment-specific
        log.debug("pack_block_cpu extension build failed: %s", exc)
        _EXTENSION = None
    _EXTENSION_INITIALISED = True
    return _EXTENSION


def pack_block_cpu(
    weight: Tensor,
    scales: Tensor,
    zeros: Tensor,
    g_idx: Tensor,
    bits: int,
    word_bits: int,
    block_in: int,
    threads: int,
) -> Tuple[Tensor, Tensor]:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("pack_block_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_block_cpu(
        weight,
        scales,
        zeros,
        g_idx,
        int(bits),
        int(word_bits),
        int(block_in),
        int(threads),
    )
