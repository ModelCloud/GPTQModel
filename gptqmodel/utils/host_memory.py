# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Host-heap helpers shared by the quantization loop.

AWQ's per-layer host temporaries (token-row feature packing and the CPU pack
path) leave freed memory in the glibc heap. ``malloc_trim(0)`` returns it to
the OS; without a trim a large MoE AWQ run keeps several GiB after the packed
model is gone (issue #4990). EXL3 already carried a private copy of this
helper.

The helper is deliberately silent: a platform without glibc ``malloc_trim``
(musl, macOS, Windows) must never break quantization.
"""

from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


# ``(library handle, symbol)``. The handle is kept so the resolved function
# pointer stays valid; ``None`` means this platform has no glibc ``malloc_trim``.
_MallocTrim = Callable[[int], int]


@lru_cache(maxsize=1)
def _resolved_malloc_trim() -> Optional[Tuple[ctypes.CDLL, _MallocTrim]]:
    """Resolve glibc's ``malloc_trim`` once per process.

    ``libc.so.6`` is the glibc soname; ``CDLL(None)`` is the fallback for a
    libc that only exports the symbol through the already-loaded process image.
    musl and non-Linux platforms resolve to ``None`` and stay a no-op.
    """

    if not sys.platform.startswith("linux"):
        return None
    for library in ("libc.so.6", None):
        try:
            libc = ctypes.CDLL(library)
        except OSError:
            continue
        try:
            trim = libc.malloc_trim
        except AttributeError:
            continue
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return libc, trim
    return None


def malloc_trim() -> Optional[int]:
    """Return free glibc arenas to the OS.

    Returns ``malloc_trim(0)``'s result (1 when memory was released) or
    ``None`` when the platform does not expose glibc ``malloc_trim``. Never
    raises: an unavailable symbol is a no-op, not a quantization failure.
    """

    resolved = _resolved_malloc_trim()
    if resolved is None:
        return None
    _, trim = resolved
    try:
        return int(trim(0))
    except Exception:  # pragma: no cover - defensive: exotic libc wrappers
        return None
