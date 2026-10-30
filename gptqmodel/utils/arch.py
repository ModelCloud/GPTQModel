# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Device architecture classification shared by ROCm kernels and quantizers.

The repository used to gate the fused GPTQ ROCm kernels on a hard-coded
``gfx950`` string check.  RDNA4 (``gfx1200``/``gfx1201``/``gfx1250``) is wave32
rather than wave64, and its tuned launch constants differ from CDNA, so the
architecture decision is centralised here instead of being spelled out at each
call site.

Everything in this module is pure metadata lookup.  ``rocm_arch`` caches the
``gcnArchName`` string per device index because
``torch.cuda.get_device_properties`` is comparatively expensive and is called
from the forward hot path.

Never use ``torch.cuda.get_device_capability`` to classify AMD hardware: on
ROCm it returns an unrelated (and meaningless) numeric pair.
"""

from __future__ import annotations

import torch


__all__ = [
    "arch_class",
    "clear_arch_cache",
    "gate_limits",
    "gemm_fused_dtypes",
    "is_gfx950",
    "is_rdna4",
    "is_rocm_fused_arch",
    "normalize_arch",
    "rocm_arch",
    "wave_size",
]


# Conservative defaults for the fused small-M GPTQ kernel.  ``cdna4`` keeps the
# values that were measured on gfx950; RDNA4 inherits them provisionally (see
# ``gate_limits``) until the crossover is re-measured on gfx120*.
_CDNA4_GATE = (16, 5120)
# RDNA4 crossover measured on an AMD Radeon AI PRO R9700 (gfx1201, HIP
# 7.16.26385) with the (32, 32) tile: the fused small-M GEMM beat
# dequantize+matmul for every M in 2..16 across K in {896, 2048, 4096, 5120,
# 6144, 7168, 8192, 11008, 12288, 13824, 14336, 17408} (1.33x-3.06x).  CDNA's
# 5120 bound comes from a K-loop regression that does not reproduce on wave32,
# so RDNA4 keeps the full measured range while reusing the conservative
# ``small_m_max`` (M>16 was not uniformly a win).
_RDNA4_GATE = (16, 17408)
_DISABLED_GATE = (0, 0)

# Activation dtypes the fused small-M GEMM is *measurably* faster in.  CDNA has
# only been tuned/measured in FP16, and RDNA4 was measured in both:
#
#   gfx1201 (R9700, 4-bit gs=128, module forward, speedup = split / fused,
#   M in 2..16, shipped (32, 32) launch tile):
#     FP16    K=2048/4096/5120/8192: 1.11x-2.53x (median 1.64x)
#     BF16    same K:                0.43x-1.42x (median 0.85x, no uniform win)
#
# BF16 is therefore left on the dequantize+matmul path on RDNA4: the fused
# kernel is not a uniform win there, and it must never be slower than the
# fallback.  A per-tile sweep hints BF16 could be recovered by retuning the tile
# on RDNA4, but that sweep crashed the GPU and was not re-run.  FP16 and the
# CUDA/MPS Triton path (which does not use this gate) are unaffected.
# GEMV (M == 1) is a different kernel and *is* faster in both dtypes on RDNA4
# (FP16 1.64x-3.67x, BF16 1.40x-2.84x measured), so it is not gated here.
_CDNA4_GEMM_DTYPES = frozenset({"float16", "bfloat16"})
_RDNA4_GEMM_DTYPES = frozenset({"float16"})

_FUSED_ARCH_PREFIXES = ("gfx950", "gfx120", "gfx125")

_ARCH_CACHE: dict[tuple[int, bool], str | None] = {}


def clear_arch_cache() -> None:
    """Drop the cached architecture strings (used by tests and device hot-plug)."""
    _ARCH_CACHE.clear()


def normalize_arch(name) -> str:
    """Return a lower-case architecture id with feature suffixes removed.

    ``gcnArchName`` commonly carries target features, for example
    ``gfx1201:sramecc+:xnack-``.  Only the leading architecture token is
    meaningful for dispatch.
    """

    if name is None:
        return ""
    text = str(name).strip().lower()
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _read_raw_arch(index: int) -> str:
    props = torch.cuda.get_device_properties(index)
    return getattr(props, "gcnArchName", "") or ""


def rocm_arch(device: torch.device | str | int | None = None) -> str | None:
    """Return the normalized ROCm architecture for ``device`` (``gfx1201``).

    Returns ``None`` on non-ROCm hosts, when the device properties do not carry
    a ``gcnArchName``, or when the runtime cannot be queried.  Results are
    cached per ``(index, is_rocm)``.
    """

    is_rocm = torch.version.hip is not None
    if not is_rocm:
        return None

    try:
        index = _device_index(device)
    except Exception:
        return None

    key = (index, is_rocm)
    if key in _ARCH_CACHE:
        return _ARCH_CACHE[key]

    try:
        arch = normalize_arch(_read_raw_arch(index)) or None
    except Exception:
        # A device that cannot report properties is not a usable ROCm device;
        # cache the negative result so the forward path does not retry.
        arch = None
    _ARCH_CACHE[key] = arch
    return arch


def _resolve_arch(device_or_arch) -> str | None:
    """Accept either a device-like object or an architecture string."""

    if isinstance(device_or_arch, str) and device_or_arch.lower().startswith("gfx"):
        return normalize_arch(device_or_arch) or None
    return rocm_arch(device_or_arch)


def arch_class(device_or_arch=None) -> str:
    """Classify an architecture as ``cdna4``, ``rdna4``, ``rdna3``, or ``other``."""

    arch = _resolve_arch(device_or_arch)
    if not arch:
        return "other"
    if arch.startswith("gfx95"):
        return "cdna4"
    if arch.startswith("gfx120") or arch.startswith("gfx125"):
        return "rdna4"
    if arch.startswith("gfx11"):
        return "rdna3"
    return "other"


def is_gfx950(device_or_arch=None) -> bool:
    """True when the target is the tuned gfx950 (MI350X/MI355X) architecture."""

    arch = _resolve_arch(device_or_arch)
    return bool(arch) and arch.startswith("gfx950")


def is_rdna4(device_or_arch=None) -> bool:
    """True for the RDNA4 family: gfx1200, gfx1201, and gfx1250."""

    return arch_class(device_or_arch) == "rdna4"


def is_rocm_fused_arch(device_or_arch=None) -> bool:
    """True for architectures allowed to use the fused GPTQ ROCm kernels.

    Only tuned/registered families are enabled: gfx950 and the gfx120*/gfx125*
    RDNA4 family.  Unregistered architectures return ``False`` so callers keep
    their existing dequantize + matmul fallback.
    """

    arch = _resolve_arch(device_or_arch)
    if not arch:
        return False
    return arch.startswith(_FUSED_ARCH_PREFIXES)


def wave_size(device_or_arch=None) -> int:
    """Return the wavefront/warp width: 32 on RDNA, 64 on CDNA."""

    cls = arch_class(device_or_arch)
    if cls in ("rdna4", "rdna3"):
        return 32
    if cls == "cdna4":
        return 64
    if torch.version.hip is None:
        # NVIDIA warps are 32 lanes wide; MPS/CPU have no wavefront concept.
        return 32

    # Unregistered ROCm target: prefer the runtime's own value when available.
    try:
        index = _device_index(device_or_arch)
        return int(torch.cuda.get_device_properties(index).warp_size)
    except Exception:
        return 32


def gemm_fused_dtypes(device_or_arch=None) -> frozenset[str]:
    """Return the activation dtype names the fused small-M GEMM is enabled for.

    Values are the ``torch.dtype`` string names (``"float16"``/``"bfloat16"``)
    so this module needs no torch dtype table.  An empty set means the fused
    GEMM is disabled for that architecture (see :func:`gate_limits`).
    """

    cls = arch_class(device_or_arch)
    if cls == "cdna4":
        return _CDNA4_GEMM_DTYPES
    if cls == "rdna4":
        return _RDNA4_GEMM_DTYPES
    return frozenset()


def gate_limits(device_or_arch=None) -> tuple[int, int]:
    """Return ``(small_m_max, max_in_features)`` for the fused GPTQ kernels.

    ``cdna4`` keeps the measured gfx950 values (16, 5120).  ``rdna4`` uses the
    crossover measured on gfx1201 (16, 17408) -- see ``_RDNA4_GATE``.
    Unregistered architectures return ``(0, 0)``, which disables the shape gate.
    """

    cls = arch_class(device_or_arch)
    if cls == "cdna4":
        return _CDNA4_GATE
    if cls == "rdna4":
        return _RDNA4_GATE
    return _DISABLED_GATE
