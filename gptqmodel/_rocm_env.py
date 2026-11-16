# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Default ROCm process environment installed before the first BLAS call.

On ROCm 10.1 / gfx1201 rocBLAS forwards its GEMMs to hipBLASLt, which selects
a different GEMM kernel per process for the same shape (for example
``MT64x64x16`` vs ``MT256x128x8``, or a split-K variant). The choice is stable
within one process, so single-process determinism tests pass while two
quantization runs in separate processes diverge from the first
``torch.cholesky_inverse`` at ``quantization/gptq.py`` and produce different
checkpoints. ``ROCBLAS_USE_HIPBLASLT=0`` keeps the GEMMs in rocBLAS, which is
deterministic across processes for these shapes.

The variable is read when rocBLAS initialises, so it has to be set before the
first BLAS call. This module is deliberately standalone (``os`` plus a lazy
``torch`` import) so ``gptqmodel/__init__.py`` can apply it before importing
``gptqmodel.utils``, which initialises the runtime. A caller that ran a BLAS
operation before importing GPT-QModel has already fixed the choice; nothing
here can change that process.
"""

from __future__ import annotations

import os


ROCBLAS_USE_HIPBLASLT_ENV = "ROCBLAS_USE_HIPBLASLT"


def running_on_rocm() -> bool:
    """Whether the imported Torch is a HIP build, without touching a device."""

    try:
        import torch

        return torch.version.hip is not None
    except Exception:  # pragma: no cover - Torch is a hard dependency
        return False


def apply_default_rocm_blas_env() -> str | None:
    """Install the deterministic rocBLAS selection unless the caller chose one.

    Returns the value that was installed, or ``None`` when the host is not ROCm
    or the caller already set :data:`ROCBLAS_USE_HIPBLASLT_ENV`. An explicit
    caller value is honoured verbatim, including an empty string, so a user can
    still opt back into hipBLASLt with ``ROCBLAS_USE_HIPBLASLT=1``.
    """

    if not running_on_rocm():
        return None
    if ROCBLAS_USE_HIPBLASLT_ENV in os.environ:
        return None
    os.environ[ROCBLAS_USE_HIPBLASLT_ENV] = "0"
    return "0"


__all__ = [
    "ROCBLAS_USE_HIPBLASLT_ENV",
    "apply_default_rocm_blas_env",
    "running_on_rocm",
]
