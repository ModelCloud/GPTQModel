# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable, FrozenSet, Optional, Tuple


SupportedKwargInfo = Tuple[bool, Optional[FrozenSet[str]]]


def _get_supported_kwargs_uncached(callable_obj: Callable) -> SupportedKwargInfo:
    """Inspect one callable without retaining it in any global cache."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True, None

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True, None

    allowed = frozenset(
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return False, allowed


@lru_cache(maxsize=256)
def _get_supported_kwargs_cached(signature_target: Callable) -> SupportedKwargInfo:
    """Inspect a cache-safe callable identity.

    Only stable function objects should reach this helper. Bound methods and
    instance-bound builtin methods must be normalized or bypass this cache so
    the cache never keeps heavyweight objects alive.
    """
    return _get_supported_kwargs_uncached(signature_target)


def _is_cache_safe_builtin(callable_obj: Callable) -> bool:
    """Return True only for builtin callables that are safe to cache directly.

    Keep this branch conservative. Builtin methods such as ``[].append`` expose
    ``__self__`` as the owning instance; caching them directly would retain that
    instance. Module-level builtins like ``len`` instead point at the builtins
    module and are safe to reuse across the process.
    """
    if not inspect.isbuiltin(callable_obj):
        return False

    owner = getattr(callable_obj, "__self__", None)
    return owner is None or inspect.ismodule(owner)


def get_supported_kwargs(callable_obj: Callable) -> SupportedKwargInfo:
    """Return (accepts_var_kwargs, allowed_kwargs) for a callable.

    allowed_kwargs is None when the callable uses ``**kwargs`` or when inspection fails.

    Never cache ``callable_obj`` directly. Multi-device quantization passes
    bound ``nn.Module.forward`` methods here; caching those bound methods keeps
    the owning module replicas alive and pins their device tensors in memory.
    Instead, bound Python methods are normalized to ``__func__`` before hitting
    the internal cache, while less stable callables fall back to uncached
    inspection.
    """
    func = getattr(callable_obj, "__func__", None)
    if func is not None:
        # Bound Python methods differ per instance but share the same
        # underlying function object. Caching on ``__func__`` preserves reuse
        # without retaining the bound instance.
        return _get_supported_kwargs_cached(func)

    if inspect.isfunction(callable_obj):
        return _get_supported_kwargs_cached(callable_obj)

    if _is_cache_safe_builtin(callable_obj):
        return _get_supported_kwargs_cached(callable_obj)

    return _get_supported_kwargs_uncached(callable_obj)


def safe_kwargs_call(
    callable_obj: Callable,
    *args: Any,
    kwargs: Optional[dict] = None,
    on_removed: Optional[Callable[[list[str]], None]] = None,
):
    """Invoke ``callable_obj`` with kwargs filtered against its signature.

    Many third-party helpers (e.g., hub download utilities) have a strict
    keyword signature. This helper allows callers to gather keyword arguments
    from multiple sources, filter out unsupported ones via inspection, and
    invoke the callable safely without tripping ``TypeError``. When the
    callable accepts ``**kwargs`` or inspection fails, the original kwargs are
    forwarded unchanged.
    """

    kwargs = dict(kwargs or {})
    accepts_var_kw, allowed_kwargs = get_supported_kwargs(callable_obj)
    if accepts_var_kw or allowed_kwargs is None:
        return callable_obj(*args, **kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in allowed_kwargs}
    if on_removed is not None:
        removed = sorted(key for key in kwargs if key not in allowed_kwargs)
        if removed:
            on_removed(removed)
    return callable_obj(*args, **filtered)
