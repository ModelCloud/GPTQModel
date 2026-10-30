# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from contextlib import contextmanager

import torch

from .arch import arch_class


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
_QUANT_BLAS_BACKEND_CHOICES = frozenset({"rocblas", "hipblaslt", "cublaslt"})
# torch names rocBLAS "hipblas" on ROCm and rejects "rocblas" outright.
_QUANT_BLAS_BACKEND_ALIASES = {"rocblas": "hipblas"}

# Quantize-phase kernel prewarm groups.  ``None`` means the historical full
# payload.  RDNA4 drops the "shape" group (cat/stack/roll/pad/repeat_interleave):
# those are plain copy kernels whose runtime init is negligible, while the cold
# payload costs measurable time.  This split has NOT been measured on gfx120*;
# the benchmark script is where the per-arch payload gets validated.
_RDNA4_PREWARM_GROUPS = ("elementwise", "fp32", "reduce", "gather", "dtype", "bitwise", "softmax")


def _requested_quant_blas_backend() -> str:
    """Return the nominal ``GPTQMODEL_ROCM_QUANT_BLAS`` value, lower-cased."""

    return os.getenv(QUANT_BLAS_BACKEND_ENV, QUANT_BLAS_BACKEND_DEFAULT).strip().lower()


def quantize_tuning(device=None) -> dict:
    """Return the quantization-phase tuning for ``device``.

    The mapping has three keys:

    ``blas_backend``
        Nominal BLAS selection for the quantize phase: ``"rocblas"`` (default),
        ``"hipblaslt"``/``"cublaslt"`` (experimental), or ``"default"`` on
        non-ROCm hosts, where no override is applied.

        hipBLASLt does ship Tensile libraries for ``gfx1200``/``gfx1201``/
        ``gfx1250`` (the ``hipblaslt/library/`` directories exist), so this is
        not a "library missing" fallback.  Those kernels are simply not tuned
        for RDNA4 yet, and the per-(dtype, layout) cold load measured on gfx950
        is the dominant cost of a quantization pass.  rocBLAS is therefore the
        default for both ``cdna4`` and ``rdna4``; setting
        ``GPTQMODEL_ROCM_QUANT_BLAS=hipblaslt`` keeps torch's hipBLASLt default
        as an opt-in experiment.

    ``amp``
        Whether the ParoQuant stage optimizer may use AMP.  ROCm keeps AMP off
        (measured slower and no more accurate on gfx950; still pending a RDNA4
        measurement, so RDNA4 also stays off).  Non-ROCm keeps the historical
        CUDA default of enabling it.

    ``prewarm_ops``
        Quantize prewarm op groups, or ``None`` for the full historical payload.
    """

    if not IS_ROCM:
        return {"blas_backend": "default", "amp": True, "prewarm_ops": None}
    return _rocm_quantize_tuning(device)


def _rocm_quantize_tuning(device=None) -> dict:
    """ROCm quantize tuning without re-checking ``IS_ROCM`` (callers assert it)."""

    requested = _requested_quant_blas_backend()
    blas_backend = requested if requested in _QUANT_BLAS_BACKEND_CHOICES else QUANT_BLAS_BACKEND_DEFAULT
    cls = arch_class(device)
    prewarm_ops = _RDNA4_PREWARM_GROUPS if cls == "rdna4" else None
    return {"blas_backend": blas_backend, "amp": False, "prewarm_ops": prewarm_ops}


def rocm_amp_disabled(device=None, *, is_rocm: bool | None = None) -> bool:
    """True when AMP must stay off for the ParoQuant stage optimizer.

    Every ROCm target currently disables AMP: the autocast overhead was
    measured slower (and no more accurate) than the plain path on MI355X
    (gfx950).  RDNA4 keeps the same conservative choice until it is measured on
    real gfx120* hardware.

    ``is_rocm`` overrides the module-level host detection; it exists for unit
    tests that patch the caller's ``IS_ROCM`` flag on a CPU-only machine.
    """

    if is_rocm is None:
        is_rocm = IS_ROCM
    if not is_rocm:
        return False
    return not _rocm_quantize_tuning(device)["amp"]


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
def quant_blas_backend(device=None):
    """Route BLAS work to rocBLAS for the duration of a quantization run.

    The override is scoped to the quantization phase: the previous backend is
    restored afterwards, so inference and every later call keep the user's
    selection. The requested backend comes from ``quantize_tuning(device)``, so
    CDNA and RDNA4 both default to rocBLAS. It is skipped on non-ROCm hosts,
    when the platform lacks ``torch.backends.cuda.preferred_blas_library``, when
    the user already selected a non-hipBLASLt backend, and when
    ``GPTQMODEL_ROCM_QUANT_BLAS`` asks to keep torch's default (``hipblaslt``).
    """

    if not IS_ROCM:
        yield
        return

    requested = quantize_tuning(device)["blas_backend"]
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
