# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


"""Thread-safe utilities and module wrappers used across Transformers."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from functools import wraps
from types import ModuleType

import threadpoolctl as _threadpoolctl


class ThreadSafe(ModuleType):
    """Generic proxy that exposes a module through a shared (non-reentrant) lock."""

    def __init__(self, module: ModuleType):
        super().__init__(module.__name__)
        self._module = module
        self._lock = threading.Lock()
        self._callable_cache: dict[str, object] = {}
        # Keep core module metadata available so tools relying on attributes
        # like __doc__ or __spec__ see the original values.
        metadata = {}
        if module.__doc__ is not None:
            metadata["__doc__"] = module.__doc__
        if getattr(module, "__package__", None) is not None:
            metadata["__package__"] = module.__package__
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            metadata["__file__"] = module_file
        module_spec = getattr(module, "__spec__", None)
        if module_spec is not None:
            metadata["__spec__"] = module_spec
        self.__dict__.update(metadata)

    def __getattr__(self, name: str):
        attr = getattr(self._module, name)
        if callable(attr):
            cached = self._callable_cache.get(name)
            if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                return cached

            @wraps(attr)
            def locked(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)

            locked.__wrapped__ = attr
            self._callable_cache[name] = locked
            return locked
        return attr

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._module)))


class _ThreadSafeProxy:
    """Lightweight proxy that serializes access to an object with a shared lock."""

    def __init__(self, value, lock):
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_callable_cache", {})
        object.__setattr__(self, "__wrapped__", value)

    def __getattr__(self, name: str):
        attr = getattr(self._value, name)
        if callable(attr):
            cached = self._callable_cache.get(name)
            if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                return cached

            @wraps(attr)
            def locked(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)

            locked.__wrapped__ = attr
            self._callable_cache[name] = locked
            return locked
        return attr

    def __setattr__(self, name, value):
        setattr(self._value, name, value)

    def __dir__(self):
        return dir(self._value)

    def __repr__(self):
        return repr(self._value)

_THREADPOOL_LIMITS_LOCK = threading.RLock()
_THREADPOOL_LIMITS_DEPTH = 0
_THREADPOOL_LIMITS_SCOPE = None


def _restore_process_wide_limits(scope) -> None:
    """Restore only the process-wide (BLAS) controllers of a limiter scope.

    ``threadpoolctl`` restores every controller it snapshotted, including the
    per-thread OpenMP one. For the refcounted process-wide scope we only own the
    BLAS limits; the per-thread OpenMP scopes restore their own values. Fall
    back to a full restore if the (private) limiter internals are unavailable.
    """

    controller = getattr(scope, "_controller", None)
    originals = getattr(scope, "_original_info", None)
    if controller is None or originals is None:
        scope.__exit__(None, None, None)
        return

    for lib_controller, original in zip(controller.lib_controllers, originals):
        if lib_controller.user_api == "blas":
            lib_controller.set_num_threads(original["num_threads"])


@contextmanager
def _refcounted_process_wide_limits(limits, user_api, kwargs):
    """Reference-counted scope for the process-wide (BLAS) thread limits.

    The first entry records the original limits and applies ``limits``; later
    entries only increment a depth counter; the last exit restores the recorded
    BLAS limits. Enter/exit run under one lock so a concurrent entry can never
    observe or save a half-restored state.
    """

    global _THREADPOOL_LIMITS_DEPTH, _THREADPOOL_LIMITS_SCOPE

    with _THREADPOOL_LIMITS_LOCK:
        if _THREADPOOL_LIMITS_DEPTH == 0:
            scope = _threadpoolctl.threadpool_limits(
                limits=limits, user_api=user_api, **kwargs
            )
            scope.__enter__()
            _THREADPOOL_LIMITS_SCOPE = scope
        _THREADPOOL_LIMITS_DEPTH += 1

    try:
        yield
    finally:
        scope = None
        with _THREADPOOL_LIMITS_LOCK:
            _THREADPOOL_LIMITS_DEPTH -= 1
            if _THREADPOOL_LIMITS_DEPTH == 0:
                scope = _THREADPOOL_LIMITS_SCOPE
                _THREADPOOL_LIMITS_SCOPE = None
                if scope is not None:
                    _restore_process_wide_limits(scope)


@contextmanager
def threadpool_limits(limits=None, user_api=None, **kwargs):
    """Thread-safe, reference-counted ``threadpoolctl.threadpool_limits``.

    Native thread-pool limits have mixed scoping: BLAS limits (OpenBLAS, MKL,
    BLIS, ...) are process-wide, while OpenMP's limit is a per-thread internal
    control variable. A plain per-entry save/restore is therefore wrong for
    BLAS: when thread B enters while thread A is still inside, B records the
    already-limited value and restores it when B exits last, pinning the whole
    process to the reduced thread count (issue #4206).

    This guard refcounts the process-wide BLAS limits: the first entry records
    the original limits and applies ``limits``, later entries only increment a
    depth counter, and the last exit restores the original BLAS limits. The
    per-thread OpenMP limits are still applied and restored on every entry so
    concurrent workers keep their own limit, as before.

    The scope is re-entrant and thread-safe. The first entry's ``limits`` win
    until the last exit. ``user_api="openmp"`` keeps the plain per-entry
    behavior because that limit is already thread-local.
    """

    if user_api == "openmp":
        with _threadpoolctl.threadpool_limits(
            limits=limits, user_api=user_api, **kwargs
        ):
            yield
        return

    with _refcounted_process_wide_limits(limits, user_api or "blas", kwargs):
        if user_api is None and (limits is None or isinstance(limits, int)):
            with _threadpoolctl.threadpool_limits(
                limits=limits, user_api="openmp", **kwargs
            ):
                yield
        else:
            yield


THREADPOOLCTL = ThreadSafe(_threadpoolctl)
# Route the public attribute through the reference-counted guard above instead
# of the generic per-call lock wrapper.
THREADPOOLCTL.threadpool_limits = threadpool_limits
GC = ThreadSafe(gc)
__all__ = [
    "ThreadSafe",
    "THREADPOOLCTL",
    "threadpool_limits",
    "GC",
]
