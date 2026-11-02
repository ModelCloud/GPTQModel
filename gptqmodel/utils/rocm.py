# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import re
from contextlib import contextmanager

import torch

from .arch import arch_class
from .logger import setup_logger


IS_ROCM = torch.version.hip is not None

# GPT-QModel only supports and validates ROCm 10.1 and newer. The loader and
# package init warn once on older runtimes; see ``warn_unsupported_rocm``.
MIN_ROCM_VERSION = (10, 1)
MIN_ROCM_VERSION_STR = "10.1"

_TORCH_ROCM_TAG_RE = re.compile(r"\+rocm(\d+(?:\.\d+)*)", re.IGNORECASE)
_ROCM_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")

log = setup_logger()

_UNSUPPORTED_ROCM_WARNED = False


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def parse_rocm_version(value) -> tuple[int, ...] | None:
    """Parse a ROCm version string into comparable integer components."""

    if not value:
        return None
    match = _ROCM_VERSION_RE.search(str(value))
    if match is None:
        return None
    return _version_tuple(match.group(1))


def rocm_version() -> tuple[int, ...] | None:
    """Return the ROCm release used by torch, or ``None`` when undetectable.

    ``torch.version.rocm`` is authoritative and present on current wheels.
    Older wheels only encode the release in the local version tag
    (``2.4.0+rocm6.2...``), which is the fallback. ``torch.version.hip`` is the
    HIP runtime version rather than the ROCm release and is never parsed.
    """

    if not IS_ROCM:
        return None

    version = parse_rocm_version(getattr(torch.version, "rocm", None))
    if version is not None:
        return version

    match = _TORCH_ROCM_TAG_RE.search(str(getattr(torch, "__version__", "")))
    if match is None:
        return None
    return _version_tuple(match.group(1))


def rocm_version_string() -> str | None:
    """Return the detected ROCm release as ``"major.minor[.patch]"``."""

    version = rocm_version()
    if version is None:
        return None
    return ".".join(str(part) for part in version)


def is_supported_rocm_version() -> bool | None:
    """Return whether ROCm is new enough, or ``None`` off ROCm/undetectable."""

    if not IS_ROCM:
        return None
    version = rocm_version()
    if version is None:
        return None
    return version >= MIN_ROCM_VERSION


def warn_unsupported_rocm(logger=None) -> bool:
    """Warn once when ROCm is older than :data:`MIN_ROCM_VERSION`.

    Returns ``True`` when the warning was emitted. The check is inert off ROCm
    and when torch does not expose a ROCm release, so a supported or
    undetectable host never produces a false warning.
    """

    global _UNSUPPORTED_ROCM_WARNED

    if _UNSUPPORTED_ROCM_WARNED:
        return False
    if is_supported_rocm_version() is not False:
        return False

    _UNSUPPORTED_ROCM_WARNED = True
    target = logger or log
    target.warn(
        f"ROCm {rocm_version_string()} is not supported: GPT-QModel requires "
        f"ROCm >= {MIN_ROCM_VERSION_STR}. Please upgrade to the latest ROCm "
        "release (>= 10.1) before quantizing or loading models."
    )
    return True

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
# payload.  RDNA4 drops the "shape" group (cat/stack/roll/pad/repeat_interleave).
# Measured on an AMD Radeon AI PRO R9700 (``gfx1201``, HIP 7.16.26385), cold
# per-group incremental cost in one process: elementwise 1534 ms, fp32 2.2 ms,
# reduce 120 ms, gather 214 ms, shape 11.8 ms, dtype 2.0 ms, bitwise 40 ms,
# softmax 84 ms (sum 2006 ms); the whole payload in a fresh process is ~2.0 s.
# Dropping "shape" therefore saves ~0.6% of the cold payload: it is a neutral,
# harmless split rather than a win, kept because it never costs anything.  The
# warmup runs in the background while the CPU-bound model load is in flight.
_RDNA4_PREWARM_GROUPS = ("elementwise", "fp32", "reduce", "gather", "dtype", "bitwise", "softmax")


def _requested_quant_blas_backend() -> str:
    """Return the nominal ``GPTQMODEL_ROCM_QUANT_BLAS`` value, lower-cased."""

    return os.getenv(QUANT_BLAS_BACKEND_ENV, QUANT_BLAS_BACKEND_DEFAULT).strip().lower()


def quantize_tuning(device=None) -> dict:
    """Return the quantization-phase tuning for ``device``.

    The mapping has three keys:

    ``blas_backend``
        Nominal BLAS selection for the quantize phase: ``"rocblas"`` (default),
        ``"hipblaslt"``/``"cublaslt"`` (experimental), a disabled opt-out token
        (``""``/``"0"``/``"off"``/``"default"``/``"hipblaslt"``/``"cublaslt"``)
        preserved verbatim so ``quant_blas_backend`` leaves torch's backend
        unchanged, or ``"default"`` on non-ROCm hosts.

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
        (measured slower and no more accurate on gfx950; RDNA4 was not measured
        separately, so it inherits the same conservative choice).  Non-ROCm
        keeps the historical CUDA default of enabling it.

    ``prewarm_ops``
        Quantize prewarm op groups, or ``None`` for the full historical payload.
    """

    if not IS_ROCM:
        return {"blas_backend": "default", "amp": True, "prewarm_ops": None}
    return _rocm_quantize_tuning(device)


def _rocm_quantize_tuning(device=None) -> dict:
    """ROCm quantize tuning without re-checking ``IS_ROCM`` (callers assert it)."""

    requested = _requested_quant_blas_backend()
    # Disabled tokens (``""``/``0``/``off``/``default``) must survive resolution so
    # ``quant_blas_backend`` can treat them as "keep torch's backend". Only an
    # unrecognised *name* falls back to the rocBLAS default.
    if requested in QUANT_BLAS_BACKEND_DISABLED:
        blas_backend = requested
    elif requested in _QUANT_BLAS_BACKEND_CHOICES:
        blas_backend = requested
    else:
        blas_backend = QUANT_BLAS_BACKEND_DEFAULT
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
