# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel-owned ``triton.jit`` for the package's own kernels.

Triton's compiler uses nanobind objects that are not free-thread safe. Two
threads that miss the same kernel cache entry compile together and libtriton
aborts the process ("attempted to delete an unknown instance", exit 134) under
``PYTHON_GIL=0``. GPT-QModel kernels are therefore built with :func:`jit`,
which returns a :class:`JITFunction` subclass whose ``_do_compile`` takes one
process-wide lock and rechecks the kernel cache, collapsing concurrent misses
for a key into a single compile.

Triton itself is never modified: kernels from other libraries keep Triton's
stock ``JITFunction``.
"""

from __future__ import annotations

import functools
import threading


__all__ = ["jit"]


# Process-wide because the unsafe state lives inside libtriton, not per kernel.
_TRITON_COMPILE_LOCK = threading.RLock()


def _single_flight_triton_compile(self, key, device, compile_fn):
    """Compile ``key`` once, serializing every GPT-QModel libtriton compile."""

    try:
        kernel_cache = self.device_caches[device][0]
    except (AttributeError, KeyError, IndexError, TypeError):
        kernel_cache = None

    with _TRITON_COMPILE_LOCK:
        if kernel_cache is not None:
            cached = kernel_cache.get(key)
            if cached is not None:
                return cached
        return compile_fn()


@functools.lru_cache(maxsize=1)
def jit_function_class():
    """Return ``GPTQModelJITFunction`` (built lazily: Triton is optional)."""

    from triton.runtime.jit import JITFunction

    class GPTQModelJITFunction(JITFunction):
        """``JITFunction`` whose cold compiles are single-flight and serialized."""

        def _do_compile(self, key, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and len(args) > 1:
                device = args[1]
            parent = super()._do_compile

            def compile_fn():
                return parent(key, *args, **kwargs)

            return _single_flight_triton_compile(self, key, device, compile_fn)

    GPTQModelJITFunction.__module__ = __name__
    GPTQModelJITFunction.__qualname__ = "GPTQModelJITFunction"
    return GPTQModelJITFunction


def _interpret_mode() -> bool:
    try:
        from triton import knobs
    except ImportError:  # Triton < 3.4 reads the environment directly
        import os

        return os.environ.get("TRITON_INTERPRET", "0").lower() in ("1", "true", "on", "yes")
    return bool(knobs.runtime.interpret)


def jit(fn=None, **kwargs):
    """Drop-in for ``triton.jit`` that builds a GPT-QModel-owned kernel object."""

    def decorator(func):
        if _interpret_mode():
            # The interpreter never reaches libtriton's compiler.
            import triton

            return triton.jit(func, **kwargs)
        return jit_function_class()(func, **kwargs)

    return decorator(fn) if fn is not None else decorator
