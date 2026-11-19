# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

from .kernel import get_kernel_config
from .torch import dynamo_is_compiling


_REQUIRED_AVX512_BF16_FLAGS = frozenset(
    (
        "xsave",
        "avx512f",
        "avx512bw",
        "avx512dq",
        "avx512vl",
        "avx512vbmi",
        "avx512_bf16",
    )
)


# The host capability probes below are process constants, so they are resolved
# once and then published as a plain module-level ``bool``. The probe bodies
# read ``/proc/cpuinfo`` (``open()``) and mutate the process-global oneDNN
# precision under an ``RLock``; both are hard graph breaks for dynamo, and the
# quantized CPU forwards call ``has_cpu_bf16x3_matmul()`` on every invocation.
# Reading a module global is traceable (dynamo guards on its value), which is
# what keeps the CPU kernels compilable.
_AVX512_BF16: bool | None = None
_CPU_BF16X3_PRECISION: bool | None = None


def _probe_avx512_bf16() -> bool:
    if platform.system() != "Linux":
        return False
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return False

    for line in cpuinfo.splitlines():
        if not line.startswith("flags"):
            continue
        _, _, values = line.partition(":")
        flags = set(values.split())
        if _REQUIRED_AVX512_BF16_FLAGS.issubset(flags):
            return True
    return False


def has_avx512_bf16() -> bool:
    """Return whether this host exposes the AVX512-BF16 instruction set.

    GPT-QModel's ported CPU kernel is compiled with AVX512-BF16 flags. The
    Linux cpuinfo check is deliberate: ``torch.backends.cpu.get_cpu_capability``
    reports the AVX512 family but does not distinguish BF16 support.

    The answer is resolved once and then returned as a module global so a
    traced forward never reaches the ``open()``/``read_text`` probe.
    """

    global _AVX512_BF16
    if _AVX512_BF16 is None:
        _AVX512_BF16 = _probe_avx512_bf16()
    return _AVX512_BF16


def avx512_bf16_error() -> str:
    return (
        "The GPTQ CPU AVX512-BF16 kernel requires Linux on x86-64 with "
        "XSAVE, AVX512F, AVX512BW, AVX512DQ, AVX512VL, AVX512-VBMI, and "
        "AVX512-BF16."
    )


def _cpu_bf16x3_precision() -> str:
    return torch.backends.mkldnn.matmul.fp32_precision


def _set_cpu_bf16x3_precision(value: str) -> None:
    torch.backends.mkldnn.matmul.fp32_precision = value


_CPU_BF16X3_LOCK = threading.RLock()
_CPU_BF16X3_DEPTH = 0
_CPU_BF16X3_RESTORE = None


def _probe_cpu_bf16x3_precision() -> bool:
    if not torch.backends.mkldnn.is_available():
        return False

    with _CPU_BF16X3_LOCK:
        if _CPU_BF16X3_DEPTH > 0:
            return True
        try:
            previous = _cpu_bf16x3_precision()
            _set_cpu_bf16x3_precision("bf16")
            _set_cpu_bf16x3_precision(previous)
        except (AttributeError, RuntimeError, ValueError):
            return False
        return True


def has_cpu_bf16x3_precision() -> bool:
    """Return whether oneDNN accepts the BF16x3 FP32 matmul precision.

    Capability only: the environment-variable gates stay in each caller's
    predicate (``has_cpu_bf16x3_matmul`` here and ``_cpu_bf16x3_support`` in the
    EXL3 torch kernel), so both entry points probe oneDNN the same way.

    The probe mutates the process-global precision, so it takes the same owner
    lock as :func:`cpu_bf16x3_scope`. Without that lock a concurrent probe can
    snapshot IEEE, race a scope entry, and then restore its stale snapshot over
    the active scope (issue #4486). When a scope is already active the owner
    has proven BF16x3 is accepted, so there is nothing to probe.

    The result is a process constant: it is resolved once and then published as
    a module global, because the probe mutates process-global state under an
    ``RLock`` and a traced CPU forward must not enter it.
    """

    global _CPU_BF16X3_PRECISION
    if _CPU_BF16X3_PRECISION is None:
        _CPU_BF16X3_PRECISION = _probe_cpu_bf16x3_precision()
    return _CPU_BF16X3_PRECISION


def has_cpu_bf16x3_matmul() -> bool:
    """Return whether this process may run FP32 matmul as BF16x3.

    The policy is deliberately *not* cached: callers may install a different
    ``KernelConfig`` between runs and expect the next matmul to observe it.
    Only the host capability probes (``has_avx512_bf16`` and
    ``has_cpu_bf16x3_precision``) are process constants and stay cached.
    """

    if not get_kernel_config().cpu_bf16x3_matmul:
        return False
    if not has_avx512_bf16():
        return False
    return has_cpu_bf16x3_precision()


@contextmanager
def cpu_bf16x3_scope():
    """Run FP32 matmuls as oneDNN BF16x3, reference-counted process-wide.

    ``torch.backends.mkldnn.matmul.fp32_precision`` is process-wide, so
    overlapping callers share one override. The first entry records the
    caller's precision and selects BF16x3; every later entry only increments
    ``_CPU_BF16X3_DEPTH``; the last exit restores the recorded precision. The
    old per-entry save/restore let the first caller to finish restore IEEE
    while another caller was still inside, so that caller's remaining matmuls
    ran in plain FP32 (issue #4116).

    The lock is an ``RLock`` on purpose: this scope is re-entrant, and a nested
    entry on the same thread would deadlock behind a plain ``Lock``.

    Callers decide whether BF16x3 is eligible for their input; this scope only
    owns the process-global setting.
    """

    global _CPU_BF16X3_DEPTH, _CPU_BF16X3_RESTORE

    entered = False
    with _CPU_BF16X3_LOCK:
        if _CPU_BF16X3_DEPTH > 0:
            _CPU_BF16X3_DEPTH += 1
            entered = True
        else:
            previous = _cpu_bf16x3_precision()
            if previous != "bf16":
                try:
                    _set_cpu_bf16x3_precision("bf16")
                except (AttributeError, RuntimeError, ValueError):
                    pass
                else:
                    _CPU_BF16X3_RESTORE = previous
                    _CPU_BF16X3_DEPTH = 1
                    entered = True

    if not entered:
        # Already BF16x3 (the caller's own setting) or the platform refused the
        # switch: nothing to restore, so leave the count alone.
        yield
        return

    try:
        yield
    finally:
        with _CPU_BF16X3_LOCK:
            _CPU_BF16X3_DEPTH -= 1
            if _CPU_BF16X3_DEPTH == 0:
                previous = _CPU_BF16X3_RESTORE
                _CPU_BF16X3_RESTORE = None
                if previous is not None:
                    try:
                        _set_cpu_bf16x3_precision(previous)
                    except (AttributeError, RuntimeError, ValueError):
                        # Best effort: the matmul already produced its result,
                        # and a platform that refuses the restore must not turn
                        # a successful forward into a failure.
                        pass


@contextmanager
def cpu_bf16x3_matmul():
    """Use oneDNN BF16x3 for FP32 matmul while restoring caller precision.

    ``torch.compile`` cannot enter this scope. ``torch.backends.mkldnn.matmul
    .fp32_precision`` is a process-global flag, so the scope must (a) mutate it
    around the matmul and (b) hold an ``RLock`` so concurrent callers share one
    override. A traced graph may do neither: dynamo cannot enter an ``RLock``
    context manager (a hard graph break), and a global side effect has no
    representation inside a graph. A compiled CPU forward therefore runs plain
    IEEE FP32. That is the *more* accurate direction (BF16x3 is an
    approximation), and the difference is far below the 4-bit weight
    quantization error, so the compiled and eager outputs agree within the
    kernel protocol's tolerance.
    """

    if not has_cpu_bf16x3_matmul():
        yield
        return

    if dynamo_is_compiling():
        yield
        return

    with cpu_bf16x3_scope():
        yield
