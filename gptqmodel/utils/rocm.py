# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from contextlib import contextmanager

import torch


IS_ROCM = torch.version.hip is not None

# hipBLASLt loads one Tensile library per (dtype, layout) pair on first use. On
# gfx950 (MI350X) each of those loads spends ~0.8-1.2 s finalizing a code object
# that carries thousands of kernel entries, and a single calibration pass
# touches four combinations, so most of a cold GPTQ/QQQ/ParoQuant quantize is
# library loading rather than math. rocBLAS covers the same work with roughly
# 0.15 s of loading and the same steady-state throughput.
QUANT_BLAS_BACKEND_ENV = "GPTQMODEL_ROCM_QUANT_BLAS"
QUANT_BLAS_BACKEND_DEFAULT = "rocblas"
QUANT_BLAS_BACKEND_DISABLED = frozenset({"", "0", "off", "default", "hipblaslt", "cublaslt"})
# torch names rocBLAS "hipblas" on ROCm and rejects "rocblas" outright.
_QUANT_BLAS_BACKEND_ALIASES = {"rocblas": "hipblas"}


def _preferred_blas_library_getter():
    return getattr(torch.backends.cuda, "preferred_blas_library", None)


def preferred_blas_library():
    """Return torch's current preferred BLAS backend, or None when unsupported."""

    getter = _preferred_blas_library_getter()
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def _is_hipblaslt(backend) -> bool:
    """Return True when ``backend`` is torch's hipBLASLt (cuBLASLt) selection."""

    name = getattr(backend, "name", None) or str(backend)
    return name.rsplit(".", 1)[-1].strip() == "Cublaslt"


@contextmanager
def quant_blas_backend():
    """Route BLAS work to rocBLAS for the duration of a quantization run.

    The override is scoped to the quantization phase: the previous backend is
    restored afterwards, so inference and every later call keep the user's
    selection. It is skipped on non-ROCm hosts, when the platform lacks
    ``torch.backends.cuda.preferred_blas_library``, and when the user already
    selected a non-hipBLASLt backend. Set ``GPTQMODEL_ROCM_QUANT_BLAS`` to
    ``hipblaslt`` to keep torch's default backend instead.
    """

    if not IS_ROCM:
        yield
        return

    requested = os.getenv(QUANT_BLAS_BACKEND_ENV, QUANT_BLAS_BACKEND_DEFAULT).strip().lower()
    setter = _preferred_blas_library_getter()
    previous = preferred_blas_library()
    if (
        setter is None
        or previous is None
        or not _is_hipblaslt(previous)
        or requested in QUANT_BLAS_BACKEND_DISABLED
    ):
        yield
        return

    try:
        setter(_QUANT_BLAS_BACKEND_ALIASES.get(requested, requested))
        applied = preferred_blas_library()
    except Exception:
        yield
        return
    if applied is None or _is_hipblaslt(applied):
        # The platform refused the switch; keep the original backend.
        yield
        return

    try:
        yield
    finally:
        try:
            setter(previous)
        except Exception:
            # Restoring is best-effort: the quantization result is already
            # computed, and a platform that refuses the restore must not turn a
            # successful quantize into a failure.
            pass
