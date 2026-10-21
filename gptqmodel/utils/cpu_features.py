# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path


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
