# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
import threading

import torch


_GLOBAL_WARMUP_LOCK = threading.Lock()


def _make_spd(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a small symmetric positive definite matrix."""
    base = torch.randn((size, size), device=device, dtype=dtype)
    identity = torch.eye(size, device=device, dtype=dtype)
    return base @ base.transpose(-1, -2) + identity * 1e-3


def _run_cholesky_and_eigh(device: torch.device, dtype: torch.dtype) -> None:
    spd = _make_spd(4, device, dtype)
    torch.linalg.cholesky(spd)
    torch.linalg.eigh(spd)


def _run_svd(device: torch.device, dtype: torch.dtype) -> None:
    mat = torch.randn((4, 3), device=device, dtype=dtype)
    torch.linalg.svd(mat, full_matrices=False)


def _run_qr(device: torch.device, dtype: torch.dtype) -> None:
    square = torch.randn((4, 4), device=device, dtype=dtype)
    torch.linalg.qr(square)


def run_torch_linalg_warmup(device: torch.device) -> None:
    """
    Execute the torch.linalg operators used across the project once on the worker thread.

    Serialized under a global lock to avoid races inside PyTorch's lazy wrappers. The warmup
    still runs once per physical device so backend-specific handles are initialized where needed.
    """
    with _GLOBAL_WARMUP_LOCK:
        dtypes = (torch.float32, torch.float64)
        for dtype in dtypes:
            _run_cholesky_and_eigh(device, dtype)
            _run_svd(device, dtype)
            _run_qr(device, dtype)

        if device.type == "cuda" and hasattr(torch.backends, "cuda"):
            preferred = getattr(torch.backends.cuda, "preferred_linalg_library", None)
            if callable(preferred):
                current = preferred()
                # Core warmup already ran using the currently preferred backend above.
                # Some installations fall back to MAGMA when the primary solver is unavailable,
                # so we pre-initialize MAGMA as well when it differs from the preferred backend.
                if current and current != "magma":
                    with contextlib.suppress(Exception):
                        torch.backends.cuda.preferred_linalg_library(backend="magma")
                        _run_cholesky_and_eigh(device, torch.float32)
                if current:
                    with contextlib.suppress(Exception):
                        torch.backends.cuda.preferred_linalg_library(backend=current)


__all__ = ["run_torch_linalg_warmup"]
