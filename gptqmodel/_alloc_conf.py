# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Default CUDA caching-allocator configuration.

``expandable_segments`` is what stops a long quantization run from shattering a
device into holes it can never satisfy again: without it the allocator fails a
single 6 GiB request while the driver still reports >19 GiB free, because no
*contiguous* block of that size survives.

The setting is consumed when the caching allocator is first initialised, so it
has to be in the environment *before* anything touches CUDA. This module
therefore imports nothing but :mod:`os` (and :mod:`torch` lazily) and is applied
from the top of ``gptqmodel/__init__.py`` -- setting it from
``gptqmodel/models/auto.py`` is too late, because importing ``gptqmodel.utils``
already initialises the allocator and the variable is then silently ignored.
"""

from __future__ import annotations

import os
import warnings


DEFAULT_ALLOC_CONF = (
    "expandable_segments:True,"
    "max_split_size_mb:256,"
    "garbage_collection_threshold:0.7"
)

# ``PYTORCH_CUDA_ALLOC_CONF`` is the historical name; ``PYTORCH_ALLOC_CONF`` is
# the generic alias understood by newer releases. Set both so the default holds
# across the Torch versions GPTQModel supports.
ALLOC_CONF_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")

#: Allocator backend that survives a wide-layer Hessian.
#:
#: A GPTQ Hessian for a 40960-wide projection is ``columns ** 2 * 4`` bytes
#: (6.25 GiB), and materialising it plus the Cholesky factor and cuSOLVER
#: workspace needs roughly three of those at once. The native caching allocator
#: fails such a single multi-GiB request on a busy multi-device run even when
#: the driver still reports >13 GiB free, because it cannot assemble a
#: contiguous block/arena for it. ``cudaMallocAsync`` has no such bookkeeping.
LARGE_HESSIAN_ALLOC_CONF = "backend:cudaMallocAsync"


def alloc_conf_configured() -> bool:
    """Whether the caller already chose an allocator configuration."""

    return any(os.environ.get(name) for name in ALLOC_CONF_ENV_VARS)


def _apply_runtime_alloc_conf() -> None:
    """Apply the setting when CUDA came up before this module ran.

    Torch parses the environment once, when the caching allocator is created.
    A caller that allocated on a device before importing GPTQModel therefore
    ignores the variable we just set, so mirror it through the runtime hook.
    """

    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        # ``expandable_segments`` is a native-caching-allocator knob. Torch
        # rejects an allocator-settings change once a different backend is
        # already parsed, so leave an explicit ``backend:cudaMallocAsync``
        # choice alone instead of tripping the internal assert.
        if torch.cuda.get_allocator_backend() != "native":
            return
        setter = getattr(getattr(torch.cuda, "memory", None), "_set_allocator_settings", None)
        if setter is None:
            return
        with warnings.catch_warnings():
            # Torch deprecates this spelling in favour of a private C++ hook
            # that is not exposed on every supported release.
            warnings.simplefilter("ignore")
            setter(DEFAULT_ALLOC_CONF)
    except Exception:  # pragma: no cover - best effort, never break the import
        pass


def apply_default_alloc_conf() -> str | None:
    """Install :data:`DEFAULT_ALLOC_CONF` unless the caller configured one.

    Returns the value that was installed, or ``None`` when the caller's own
    configuration was preserved. Safe to call more than once.
    """

    if alloc_conf_configured():
        return None
    for name in ALLOC_CONF_ENV_VARS:
        os.environ[name] = DEFAULT_ALLOC_CONF
    _apply_runtime_alloc_conf()
    return DEFAULT_ALLOC_CONF


def allocator_backend() -> str | None:
    """Name of the active CUDA allocator backend, or ``None`` without CUDA."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_allocator_backend()
    except Exception:  # pragma: no cover - very old Torch
        return None


def large_hessian_alloc_hint() -> str | None:
    """Actionable hint for an OOM that only the native allocator would hit."""

    if allocator_backend() in (None, "cudaMallocAsync"):
        return None
    return (
        "The native CUDA caching allocator could not serve a multi-GiB Hessian "
        f"allocation. Re-run with `PYTORCH_ALLOC_CONF={LARGE_HESSIAN_ALLOC_CONF}` "
        "(or `backend:cudaMallocAsync,expandable_segments:True`) to use the async "
        "allocator, which does not need a contiguous multi-GiB block."
    )


__all__ = [
    "ALLOC_CONF_ENV_VARS",
    "DEFAULT_ALLOC_CONF",
    "LARGE_HESSIAN_ALLOC_CONF",
    "allocator_backend",
    "alloc_conf_configured",
    "apply_default_alloc_conf",
    "large_hessian_alloc_hint",
]
