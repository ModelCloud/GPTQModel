# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


"""Thread-safe utilities and module wrappers used across Transformers."""

from __future__ import annotations

import threading
from functools import wraps
from types import ModuleType

import torch

try:
    from triton.runtime import autotuner as _triton_autotuner
except Exception:  # pragma: no cover - Triton optional
    _triton_autotuner = None


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

    def __call__(self, *args, **kwargs):
        result = self.__getattr__("__call__")(*args, **kwargs)
        if callable(result):
            return _ThreadSafeProxy(result, self._lock)
        return result

    def __getitem__(self, item):
        result = self.__getattr__("__getitem__")(item)
        if callable(result):
            return _ThreadSafeProxy(result, self._lock)
        return result



TORCH_LINALG = ThreadSafe(torch.linalg)

if _triton_autotuner is not None:
    _orig_autotune = _triton_autotuner.autotune
    _orig_autotuner_cls = _triton_autotuner.Autotuner
    _autotune_lock = threading.Lock()

    class _ThreadSafeAutotuner(_orig_autotuner_cls):
        def run(self, *args, **kwargs):
            with _autotune_lock:
                return super().run(*args, **kwargs)

        def warmup(self, *args, **kwargs):
            with _autotune_lock:
                return super().warmup(*args, **kwargs)

    _triton_autotuner.Autotuner = _ThreadSafeAutotuner

    def _safe_triton_autotune(*args, **kwargs):
        return _orig_autotune(*args, **kwargs)

    _safe_triton_autotune.__wrapped__ = _orig_autotune
    _triton_autotuner.autotune = _safe_triton_autotune

__all__ = [
    "ThreadSafe",
    "TORCH_LINALG",
]
