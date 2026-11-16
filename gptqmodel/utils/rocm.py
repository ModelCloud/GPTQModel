# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import re
import threading
import time
from contextlib import contextmanager

import torch

from .arch import arch_class
from .kernel import get_kernel_config
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
QUANT_BLAS_BACKEND_DEFAULT = "rocblas"
QUANT_BLAS_BACKEND_DISABLED = frozenset({"", "0", "off", "default", "hipblaslt", "cublaslt"})
_QUANT_BLAS_BACKEND_CHOICES = frozenset({"rocblas", "hipblaslt", "cublaslt"})
# torch names rocBLAS "hipblas" on ROCm and rejects "rocblas" outright.
_QUANT_BLAS_BACKEND_ALIASES = {"rocblas": "hipblas"}

# rocBLAS forwards its GEMMs to hipBLASLt when this is truthy. That is the
# documented opt-in, and it is not reproducible across processes.
_HIPBLASLT_ENV = "ROCBLAS_USE_HIPBLASLT"
_HIPBLASLT_ENV_TRUTHY = frozenset({"1", "true", "yes", "on"})
_REPRODUCIBILITY_WARNED = False

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
    """Return the configured quantization BLAS backend, lower-cased.

    ``None`` keeps the reproducible ``rocblas`` default. The empty string is a
    documented opt-out spelling, so it must be preserved instead of being
    coalesced into the default.
    """

    configured = get_kernel_config().rocm_quant_blas
    if configured is None:
        return QUANT_BLAS_BACKEND_DEFAULT
    return configured.strip().lower()


def _warn_if_blas_is_not_reproducible(blas_backend: str) -> None:
    """Warn once when the configuration gives up cross-process reproducibility.

    Every spelling that skips the rocBLAS pin is covered, because all of them
    leave torch on hipBLASLt, which selects a different GEMM kernel per process
    for the same shape:

    * ``hipblaslt`` / ``cublaslt`` - the documented opt-in.
    * ``""`` / ``0`` / ``off`` / ``default`` - the documented opt-out, which
      leaves torch's backend alone (hipBLASLt by default on this stack).
    * ``ROCBLAS_USE_HIPBLASLT`` truthy - rocBLAS forwards to hipBLASLt.

    The opt-ins exist for speed, so none of them is silently made
    deterministic; they only have to be visible.
    """

    global _REPRODUCIBILITY_WARNED
    if _REPRODUCIBILITY_WARNED:
        return
    forwarded = os.getenv(_HIPBLASLT_ENV, "").strip().lower()
    if blas_backend not in QUANT_BLAS_BACKEND_DISABLED and forwarded not in _HIPBLASLT_ENV_TRUTHY:
        return
    _REPRODUCIBILITY_WARNED = True
    log.warning(
        "ROCm quantization will not pin rocBLAS (rocm_quant_blas=%s), so hipBLASLt may choose a "
        "different GEMM kernel in another process for the same shape and identical "
        "runs can save different checkpoints. The reproducible default is "
        "`rocm_quant_blas=rocblas` with `%s` unset; see docs/rocm-default-environment.md.",
        f"{blas_backend}, {_HIPBLASLT_ENV}={forwarded or 'unset'}",
        _HIPBLASLT_ENV,
    )


def quantize_tuning(device=None) -> dict:
    """Return the quantization-phase tuning for ``device``.

    The mapping has three keys:

    ``blas_backend``
        Nominal BLAS selection for the Hessian/linalg phase: ``"rocblas"``
        (default), ``"hipblaslt"``/``"cublaslt"`` (experimental), a disabled
        opt-out token
        (``""``/``"0"``/``"off"``/``"default"``/``"hipblaslt"``/``"cublaslt"``)
        preserved verbatim so ``quant_blas_backend`` leaves torch's backend
        unchanged, or ``"default"`` on non-ROCm hosts. The override is not
        applied to the fp16/bf16 forward and replay GEMMs.

        hipBLASLt does ship Tensile libraries for ``gfx1200``/``gfx1201``/
        ``gfx1250`` (the ``hipblaslt/library/`` directories exist), so this is
        not a "library missing" fallback.  Those kernels are simply not tuned
        for RDNA4 yet, and the per-(dtype, layout) cold load measured on gfx950
        is the dominant cost of a quantization pass.  rocBLAS is therefore the
        default for both ``cdna4`` and ``rdna4``; setting
        ``rocm_quant_blas="hipblaslt"`` keeps torch's hipBLASLt default
        as an opt-in experiment. That opt-in is faster for the forward/replay
        path but not reproducible across processes, so it logs a warning once.

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
    tuning = _rocm_quantize_tuning(device)
    _warn_if_blas_is_not_reproducible(tuning["blas_backend"])
    return tuning


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


# The two backends this guard can pin. ``torch`` spells rocBLAS ``hipblas``.
_QUANT_BLAS_TARGET_ROCBLAS = _QUANT_BLAS_BACKEND_ALIASES[QUANT_BLAS_BACKEND_DEFAULT]
_QUANT_BLAS_TARGET_HIPBLASLT = "cublaslt"

_QUANT_BLAS_GUARD_LOCK = threading.RLock()
_QUANT_BLAS_GUARD_COND = threading.Condition(_QUANT_BLAS_GUARD_LOCK)
_QUANT_BLAS_GUARD_DEPTH = 0
_QUANT_BLAS_GUARD_PREVIOUS = None
# Backend the active scopes require, or ``None`` when the guard is not
# controlling the process backend (the caller's selection is authoritative).
_QUANT_BLAS_GUARD_TARGET = None
_QUANT_BLAS_GUARD_RESTORE = False
# A mixed-backend overlap waits for the active run instead of switching the
# backend under it. The bound only exists so a wedged run cannot hang a CI job
# forever; the degraded path is loud rather than silent.
_QUANT_BLAS_GUARD_WAIT_SECONDS = 1800.0
_QUANT_BLAS_GUARD_WAIT_WARNED = False
# Per-thread nesting depth. A nested scope always adopts the pin its own run
# already holds; waiting for itself would deadlock.
_QUANT_BLAS_GUARD_THREAD_LOCAL = threading.local()


@contextmanager
def quant_blas_backend(device=None, *, deterministic: bool | None = None):
    """Pin the quantization BLAS backend for the duration of one run.

    ``deterministic=True`` pins rocBLAS: the method accumulates an input
    Hessian from calibration activations, and its fp32 GEMMs must be
    reproducible across processes. ``deterministic=False`` pins hipBLASLt: the
    method has no Hessian, so it keeps the fast fp16/bf16 kernels. ``None``
    resolves to ``True`` when ``quantize_tuning`` reports a rocBLAS-class
    backend, which keeps the historical behaviour for direct callers such as
    the per-call Hessian contexts in ``quantization/gptq.py``.

    The pin is skipped on non-ROCm hosts, when the platform lacks
    ``torch.backends.cuda.preferred_blas_library``, when the caller already
    selected a backend the guard must not override, and when
    ``KernelConfig.rocm_quant_blas`` is one of ``QUANT_BLAS_BACKEND_DISABLED``
    (the documented reproducibility opt-out).

    ``torch.backends.cuda.preferred_blas_library`` is process-global, so the
    guard is reference-counted under a reentrant lock: runs that need the same
    backend share it, and the backend is restored only when the last one exits
    (issue #4094). Runs that need *different* backends cannot share one
    process-wide setting, so the later run waits for the active one to finish
    instead of flipping the backend under it. Without that, a concurrent AWQ
    run quantized part of its layers on rocBLAS and the rest on hipBLASLt, and
    its checkpoint depended on when the other thread exited (issue #4123).
    """

    global _QUANT_BLAS_GUARD_DEPTH, _QUANT_BLAS_GUARD_PREVIOUS
    global _QUANT_BLAS_GUARD_TARGET, _QUANT_BLAS_GUARD_RESTORE
    global _QUANT_BLAS_GUARD_WAIT_WARNED

    if not IS_ROCM:
        yield
        return

    requested = quantize_tuning(device)["blas_backend"]
    setter = _preferred_blas_library_getter()
    if setter is None or requested in QUANT_BLAS_BACKEND_DISABLED:
        yield
        return

    if deterministic is None:
        deterministic = True
    want = _QUANT_BLAS_TARGET_ROCBLAS if deterministic else _QUANT_BLAS_TARGET_HIPBLASLT

    with _QUANT_BLAS_GUARD_COND:
        deadline = time.monotonic() + _QUANT_BLAS_GUARD_WAIT_SECONDS
        announced = False
        while (
            _QUANT_BLAS_GUARD_DEPTH > 0
            and _QUANT_BLAS_GUARD_TARGET not in (None, want)
            and getattr(_QUANT_BLAS_GUARD_THREAD_LOCAL, "depth", 0) == 0
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not announced:
                announced = True
                log.info(
                    "ROCm BLAS guard: another quantization run holds the %s backend; "
                    "waiting for it to finish so neither run changes backend mid-way.",
                    _QUANT_BLAS_GUARD_TARGET,
                )
            _QUANT_BLAS_GUARD_COND.wait(timeout=min(remaining, 60.0))

        if _QUANT_BLAS_GUARD_DEPTH == 0:
            previous = preferred_blas_library()
            _QUANT_BLAS_GUARD_PREVIOUS = previous
            _QUANT_BLAS_GUARD_TARGET = None
            _QUANT_BLAS_GUARD_RESTORE = False
            if previous is not None:
                if want == _QUANT_BLAS_TARGET_ROCBLAS and _is_hipblaslt(previous):
                    try:
                        setter(want)
                        applied = preferred_blas_library()
                    except Exception:
                        applied = None
                    if applied is not None and not _is_hipblaslt(applied):
                        _QUANT_BLAS_GUARD_TARGET = want
                        _QUANT_BLAS_GUARD_RESTORE = True
                elif want == _QUANT_BLAS_TARGET_HIPBLASLT and _is_hipblaslt(previous):
                    # Already the fast backend. Record ownership anyway so a
                    # Hessian run that starts now waits instead of flipping it.
                    _QUANT_BLAS_GUARD_TARGET = want
        elif _QUANT_BLAS_GUARD_TARGET not in (None, want):
            # The wait above expired: adopt the active backend instead of
            # changing it under the running quantization.
            if not _QUANT_BLAS_GUARD_WAIT_WARNED:
                _QUANT_BLAS_GUARD_WAIT_WARNED = True
                log.warning(
                    "ROCm BLAS guard: waited %.0fs for another quantization run pinned "
                    "to %s; continuing on that backend. The resulting checkpoint may "
                    "differ from a serial run.",
                    _QUANT_BLAS_GUARD_WAIT_SECONDS,
                    _QUANT_BLAS_GUARD_TARGET,
                )
        _QUANT_BLAS_GUARD_THREAD_LOCAL.depth = getattr(_QUANT_BLAS_GUARD_THREAD_LOCAL, "depth", 0) + 1
        _QUANT_BLAS_GUARD_DEPTH += 1

    try:
        yield
    finally:
        with _QUANT_BLAS_GUARD_COND:
            _QUANT_BLAS_GUARD_THREAD_LOCAL.depth = (
                getattr(_QUANT_BLAS_GUARD_THREAD_LOCAL, "depth", 1) - 1
            )
            _QUANT_BLAS_GUARD_DEPTH -= 1
            if _QUANT_BLAS_GUARD_DEPTH == 0:
                previous = _QUANT_BLAS_GUARD_PREVIOUS
                restore = _QUANT_BLAS_GUARD_RESTORE
                _QUANT_BLAS_GUARD_PREVIOUS = None
                _QUANT_BLAS_GUARD_TARGET = None
                _QUANT_BLAS_GUARD_RESTORE = False
                if restore and previous is not None:
                    try:
                        setter(previous)
                    except Exception:
                        # Restoring is best-effort: the quantization result is
                        # already computed, and a platform that refuses the
                        # restore must not turn a successful quantize into a
                        # failure.
                        pass
                _QUANT_BLAS_GUARD_COND.notify_all()
