# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable, FrozenSet, Optional, Tuple


SupportedKwargInfo = Tuple[bool, Optional[FrozenSet[str]]]


@lru_cache(maxsize=None)
def get_supported_kwargs(callable_obj: Callable) -> SupportedKwargInfo:
    """Return (accepts_var_kwargs, allowed_kwargs) for a callable.

    allowed_kwargs is None when the callable uses ``**kwargs`` or when inspection fails.
    """
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
