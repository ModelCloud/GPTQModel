# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim over GPTQModel's lightweight vendored NF12 runtime.

New inference consumers should import :mod:`gptqmodel_nf12` directly. Keeping
this module preserves the established quantizer and qlinear call sites while
making the GPTQModel wheel the only source of native NF12 support.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import gptqmodel_nf12._jit


log = logging.getLogger(__name__)
_ERROR: str | None = None


def load() -> bool:
    """Load GPTQModel's vendored NF12 operator set if CUDA is available."""
    global _ERROR
    if not torch.cuda.is_available():
        _ERROR = "NF12-AD native kernels require an available CUDA device."
        return False
    try:
        gptqmodel_nf12.load()
    except Exception as exc:  # noqa: BLE001 - optional accelerator
        _ERROR = f"{type(exc).__name__}: {exc}"
        log.debug("NF12-AD native extension unavailable: %s", _ERROR)
        return False
    _ERROR = None
    return True


def available() -> bool:
    return load()


def last_error() -> str | None:
    return _ERROR


def build_directory() -> Path | None:
    """Return the vendored JIT build directory, if one was used."""
    return gptqmodel_nf12._jit._directory


def encode(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    result = gptqmodel_nf12.encode(weight)
    return result.packed, result.escapes, result.exp_base


def decode(packed: torch.Tensor, escapes: torch.Tensor, exp_base: int) -> torch.Tensor:
    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    return gptqmodel_nf12.decode(gptqmodel_nf12.NF12Weight(packed, escapes, exp_base))


def linear(x: torch.Tensor, packed: torch.Tensor, escapes: torch.Tensor,
           exp_base: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    return gptqmodel_nf12.linear(x, gptqmodel_nf12.NF12Weight(packed, escapes, exp_base), bias)


__all__ = ["available", "build_directory", "decode", "encode", "last_error", "linear", "load"]
