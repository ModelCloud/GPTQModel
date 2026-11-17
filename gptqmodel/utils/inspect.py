# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple


SupportedKwargInfo = Tuple[bool, Optional[FrozenSet[str]]]
PositionalBinding = Optional[Dict[str, Any]]


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


@lru_cache(maxsize=256)
def _get_signature_cached(signature_target: Callable) -> inspect.Signature:
    """Inspect one stable function object, cached like the kwarg table."""

    return inspect.signature(signature_target)


def _get_signature(callable_obj: Callable) -> Optional[inspect.Signature]:
    """Signature of a callable without retaining a bound method's instance.

    A bound Python method is cached through its underlying function, so the
    cached object never pins the module. That function's signature still starts
    with `self`; drop it here so binding the arguments a forward pre-hook
    received maps `args[0]` to `hidden_states`, not to `self`.
    """

    bound_instance = getattr(callable_obj, "__self__", None)
    func = getattr(callable_obj, "__func__", None)
    if func is not None:
        try:
            signature = _get_signature_cached(func)
        except (TypeError, ValueError):
            return None
        if bound_instance is None:
            return signature
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return signature.replace(parameters=parameters[1:])
        return signature

    if inspect.isfunction(callable_obj) or _is_cache_safe_builtin(callable_obj):
        try:
            return _get_signature_cached(callable_obj)
        except (TypeError, ValueError):
            return None

    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def bind_positional_arguments(
    callable_obj: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> PositionalBinding:
    """Bind *args to the callable's parameter names without calling it.

    `transformers` 5 passes the causal mask and other decoder inputs
    positionally: `GPT2Model.forward` hands `causal_mask` to every block as
    `args[2]`, so a forward hook that reads only `kwargs` observes no
    `attention_mask` at all. Return a mapping from parameter name to argument,
    or `None` when the signature cannot be inspected or the arguments do not
    bind. Signatures are cached on the underlying function object so a bound
    decoder block never pins its module.
    """

    if not args:
        return {}

    signature = _get_signature(callable_obj)
    if signature is None:
        return None

    try:
        bound = signature.bind_partial(*args, **(kwargs or {}))
    except TypeError:
        return None
    return dict(bound.arguments)


def bound_forward_argument(
    module: Any,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    name: str,
) -> Any:
    """Value a decoder block received for `name`, positional or keyword.

    `GPT2Model.forward` (transformers 5) calls every block as
    `block(hidden_states, past_key_values, causal_mask, encoder_hidden_states,
    ...)`, so `attention_mask` only exists as `args[2]`. A hook that reads
    `kwargs` alone captures `None` and replays the layer without causality.
    Bind the block's own signature and read the parameter by name; fall back to
    the keyword mapping when the signature cannot be inspected or does not
    bind.
    """

    callable_obj = getattr(module, "forward", module)
    signature = _get_signature(callable_obj)
    if signature is None:
        return (kwargs or {}).get(name)

    try:
        bound = signature.bind_partial(*args, **(kwargs or {}))
    except TypeError:
        return (kwargs or {}).get(name)

    if name in bound.arguments:
        return bound.arguments[name]
    # A block that declares `**kwargs` (GPT-2, OPT) receives `position_ids`
    # through it, so the keyword mapping is nested one level down.
    for parameter in signature.parameters.values():
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            continue
        extra = bound.arguments.get(parameter.name)
        if isinstance(extra, dict) and name in extra:
            return extra[name]
    return (kwargs or {}).get(name)


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
