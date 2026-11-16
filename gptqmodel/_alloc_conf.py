# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Default accelerator caching-allocator configuration.

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

On ROCm 10.1 / gfx1201 ``expandable_segments:True`` combined with
``torch.cuda.empty_cache()`` corrupts ordinary Torch results (wrong rows, or
``hipErrorIllegalAddress``) while the VMM segments are released and remapped.
GPT-QModel empties the cache throughout quantization, so ROCm gets a default
without ``expandable_segments``; the other two knobs are unchanged.

On CUDA ``expandable_segments`` is the knob that removes the fragmentation
cliff, and it is the only one that survives: pinning ``max_split_size_mb`` to
256 MiB caps the block size the allocator may split, so a freed 256 MiB block
cannot serve the next layer's slightly smaller request and GPTQ's per-layer
Hessian/workspace churn maps new pages instead of reusing the cache. See
``DEFAULT_ALLOC_CONF`` for the measurements.
"""

from __future__ import annotations

import os
import warnings

from ._rocm_env import running_on_rocm


#: CUDA default: expandable segments alone.
#:
#: ``max_split_size_mb:256`` used to be pinned here too, on top of
#: ``expandable_segments:True``. That caps the largest block the allocator is
#: willing to split, so the per-layer Hessian/workspace churn during GPTQ keeps
#: mapping new pages instead of reusing the cache: *reserved* memory grows per
#: layer while *active* memory stays flat. Measured on an RTX 4090 (issue
#: #4389), Llama-3.2-1B GPTQ 4-bit, same calibration, peak allocated 1.23 GiB:
#:
#:   default (expandable + 256 MiB split cap + gc)  peak reserved 5.40 GiB
#:   expandable_segments:True only                  peak reserved 1.67 GiB
#:   no expandable segments                         peak reserved 2.55 GiB
#:
#: ``garbage_collection_threshold`` is dropped instead of paired with a memory
#: fraction: it is a ratio of the device's *total* memory, so 0.7 only starts a
#: collection above ~33 GiB on a 48 GiB card. A GPTQ run never approaches that,
#: and capping the process with a memory fraction would trade the reservation
#: for OOMs.
DEFAULT_ALLOC_CONF = "expandable_segments:True"

#: ROCm default: the historical fragmentation knobs, but never
#: ``expandable_segments``.
#:
#: The ROCm/Torch VMM path releases and remaps segments on ``empty_cache()``,
#: and a subsequent kernel can read stale/incorrect pages. Measured on a
#: R9700 (gfx1201, torch 2.16.0.dev+rocm10.1): a 2097152x16 matmul recomputed
#: with ``empty_cache()`` between reps returned up to 4.5M wrong rows with
#: ``expandable_segments:True`` and 0 with ``False``.
#:
#: Kept as-is for #4389: #4313's regression tests pin this exact value and the
#: ROCm path already mitigates the 256 MiB split cap with a per-layer
#: ``empty_cache()``. Changing it is a separate ROCm decision.
ROCM_DEFAULT_ALLOC_CONF = "max_split_size_mb:256,garbage_collection_threshold:0.7"

# ``PYTORCH_CUDA_ALLOC_CONF`` is the historical name; ``PYTORCH_ALLOC_CONF`` is
# the generic alias understood by newer releases. Set both so the default holds
# across the Torch versions GPTQModel supports.
ALLOC_CONF_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")

# HIP builds also read this alias, and it takes precedence over
# ``PYTORCH_ALLOC_CONF``. It is only consulted for user-provided detection:
# GPT-QModel writes the two generic names, which every supported build reads.
HIP_ALLOC_CONF_ENV_VAR = "PYTORCH_HIP_ALLOC_CONF"

#: Allocator backend that survives a wide-layer Hessian.
#:
#: A GPTQ Hessian for a 40960-wide projection is ``columns ** 2 * 4`` bytes
#: (6.25 GiB), and materialising it plus the Cholesky factor and cuSOLVER
#: workspace needs roughly three of those at once. The native caching allocator
#: fails such a single multi-GiB request on a busy multi-device run even when
#: the driver still reports >13 GiB free, because it cannot assemble a
#: contiguous block/arena for it. ``cudaMallocAsync`` has no such bookkeeping.
LARGE_HESSIAN_ALLOC_CONF = "backend:cudaMallocAsync"


def default_alloc_conf(backend: str | None = None) -> str:
    """Return the default allocator configuration for *backend*.

    ``backend`` is ``"cuda"`` or ``"rocm"``. ``None`` (the default) detects the
    running Torch build, which is what the import-time hook uses; passing a
    name makes the policy testable without a device.
    """

    if backend is None:
        backend = "rocm" if running_on_rocm() else "cuda"
    if backend == "rocm":
        return ROCM_DEFAULT_ALLOC_CONF
    if backend == "cuda":
        return DEFAULT_ALLOC_CONF
    raise ValueError(f"unknown allocator backend: {backend!r}")


def alloc_conf_env_vars() -> tuple[str, ...]:
    """Environment names that carry a caller configuration for this build."""

    if running_on_rocm():
        return (*ALLOC_CONF_ENV_VARS, HIP_ALLOC_CONF_ENV_VAR)
    return ALLOC_CONF_ENV_VARS


def alloc_conf_configured() -> bool:
    """Whether the caller already chose an allocator configuration."""

    return any(os.environ.get(name) for name in alloc_conf_env_vars())


def _apply_runtime_alloc_conf(conf: str) -> None:
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
            setter(conf)
    except Exception:  # pragma: no cover - best effort, never break the import
        pass


def apply_default_alloc_conf() -> str | None:
    """Install :func:`default_alloc_conf` unless the caller configured one.

    Returns the value that was installed, or ``None`` when the caller's own
    configuration was preserved. Safe to call more than once.
    """

    if alloc_conf_configured():
        return None
    conf = default_alloc_conf()
    for name in ALLOC_CONF_ENV_VARS:
        os.environ[name] = conf
    _apply_runtime_alloc_conf(conf)
    return conf


def allocator_backend() -> str | None:
    """Name of the active CUDA allocator backend, or ``None`` without CUDA."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_allocator_backend()
    except Exception:  # pragma: no cover - very old Torch
        return None


def _effective_alloc_conf() -> str | None:
    """The allocator settings Torch is actually running with, if known.

    Torch parses the settings once, when the caching allocator is created. The
    private ``_accelerator_getAllocatorSettings`` accessor returns that parsed
    value (including a runtime ``_set_allocator_settings`` call) on releases
    that expose it. Older releases only have the environment, so fall back to
    the same precedence Torch uses: the HIP-only alias first on ROCm, then
    ``PYTORCH_ALLOC_CONF`` before the legacy ``PYTORCH_CUDA_ALLOC_CONF``.
    """

    try:
        import torch

        getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
        if getter is not None:
            parsed = getter()
            if parsed:
                return parsed
    except Exception:  # pragma: no cover - private API may be absent or fail
        pass

    if running_on_rocm():
        value = os.environ.get(HIP_ALLOC_CONF_ENV_VAR)
        if value:
            return value
    for name in ALLOC_CONF_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def max_split_bytes_from_conf(conf: str | None) -> int | None:
    """Parse ``max_split_size_mb`` from an allocator settings string.

    ``None`` means unlimited (Torch's default): every block may be split.
    ``0`` is the documented "disable the cap" spelling, so it is unlimited too.
    """

    if not conf:
        return None
    for item in conf.split(","):
        key, separator, value = item.partition(":")
        if not separator or key.strip() != "max_split_size_mb":
            continue
        try:
            megabytes = int(value.strip())
        except ValueError:
            return None
        if megabytes <= 0:
            return None
        return megabytes << 20
    return None


def allocator_max_split_bytes() -> int | None:
    """Largest block size the active caching allocator will split, in bytes.

    A request below this size cannot be served from a free block at or above
    it, so a workspace sized just under the cap maps a new segment for every
    caller instead of reusing the cache. ``None`` means the allocator splits
    every block (the CUDA default with expandable segments).
    """

    return max_split_bytes_from_conf(_effective_alloc_conf())


def clamp_budget_below_allocator_split(
    budget: int,
    *,
    headroom: int = 2 << 20,
) -> int:
    """Keep a scratch-workspace budget below the allocator's split cap.

    The caching allocator rounds a segment up to a 2 MiB multiple and refuses
    to serve a request smaller than ``max_split_size`` from a free block at or
    above it. A workspace that lands in the last 2 MiB below the cap therefore
    allocates a fresh segment for every module: reserved VRAM grows with the
    module count while active memory stays flat. Sizing below ``cap - headroom``
    keeps the segment itself splittable, so the next module reuses it.

    This only bounds chunking; every candidate is still evaluated over the same
    axes, so clip-search results are unchanged. Without a configured split cap
    the budget is returned untouched.
    """

    cap = allocator_max_split_bytes()
    if cap is None:
        return budget
    return min(budget, max(1 << 20, cap - headroom))


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
    "HIP_ALLOC_CONF_ENV_VAR",
    "LARGE_HESSIAN_ALLOC_CONF",
    "ROCM_DEFAULT_ALLOC_CONF",
    "allocator_max_split_bytes",
    "allocator_backend",
    "alloc_conf_env_vars",
    "alloc_conf_configured",
    "apply_default_alloc_conf",
    "clamp_budget_below_allocator_split",
    "default_alloc_conf",
    "large_hessian_alloc_hint",
    "max_split_bytes_from_conf",
]
