# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

from .env import env_flag


_CPU_BF16X3_ENV = "GPTQMODEL_CPU_BF16X3"


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


@lru_cache(maxsize=1)
def has_avx512_bf16() -> bool:
    """Return whether this host exposes the AVX512-BF16 instruction set.

    GPT-QModel's ported CPU kernel is compiled with AVX512-BF16 flags. The
    Linux cpuinfo check is deliberate: ``torch.backends.cpu.get_cpu_capability``
    reports the AVX512 family but does not distinguish BF16 support.
    """

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


@lru_cache(maxsize=1)
def has_cpu_bf16x3_precision() -> bool:
    """Return whether oneDNN accepts the BF16x3 FP32 matmul precision.

    Capability only: the environment-variable gates stay in each caller's
    predicate (``has_cpu_bf16x3_matmul`` here and ``_cpu_bf16x3_support`` in the
    EXL3 torch kernel), so both entry points probe oneDNN the same way.
    """

    if not torch.backends.mkldnn.is_available():
        return False

    try:
        previous = _cpu_bf16x3_precision()
        _set_cpu_bf16x3_precision("bf16")
        _set_cpu_bf16x3_precision(previous)
    except (AttributeError, RuntimeError, ValueError):
        return False
    return True


@lru_cache(maxsize=1)
def has_cpu_bf16x3_matmul() -> bool:
    """Return whether this process may run FP32 matmul as BF16x3."""

    if not env_flag(_CPU_BF16X3_ENV, default=True):
        return False
    if not has_avx512_bf16():
        return False
    return has_cpu_bf16x3_precision()


_CPU_BF16X3_LOCK = threading.RLock()
_CPU_BF16X3_DEPTH = 0
_CPU_BF16X3_RESTORE = None


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
    """Use oneDNN BF16x3 for FP32 matmul while restoring caller precision."""

    if not has_cpu_bf16x3_matmul():
        yield
        return

    with cpu_bf16x3_scope():
        yield
