# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Re-fusion contract: one container holds one quantization.

A checkpoint tensor that stores several logical projections at once (an
expert's ``gate_up_weight``, a fused ``qkv_proj``) describes exactly one
quantization: one format, one bit width, one block size, one storage dtype.
Rebuilding such a container from defused modules is therefore only sound while
every module that would be merged still carries that same quantization.

``contracts/model/refusion.md`` is the rule; this module is its enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn


#: Bit width and block size (values per scale) of each packed format.
_PACKED_BITS = {"nvfp4": 4, "mxfp8": 8}
_PACKED_BLOCK = {"nvfp4": 16, "mxfp8": 32}

#: Kernel-module attributes that decide whether two payloads can share a tensor.
_KERNEL_FIELDS = (
    "group_size",
    "sym",
    "desc_act",
    "pack_dtype",
    "scales_dtype",
    "qzero_dtype",
)


class RefusionContractError(RuntimeError):
    """Modules cannot be merged without changing one of their quantizations."""


@dataclass(frozen=True)
class QuantizationSignature:
    """Comparable identity of the quantization one module carries.

    Two modules may share one fused container only when their signatures are
    equal. Values that are per-expert (amax, global scale) are deliberately not
    part of the signature; their *kind* is.
    """

    kind: str  # "packed" | "quantized" | "dense" | "unknown"
    format: str
    bits: Optional[int] = None
    weight_dtype: Optional[str] = None
    scales_dtype: Optional[str] = None
    values_per_scale: Optional[int] = None
    payload: Tuple[Tuple[str, str], ...] = ()

    def describe(self) -> str:
        parts = [f"{self.kind}:{self.format}"]
        for label, value in (
            ("bits", self.bits),
            ("weight", self.weight_dtype),
            ("scales", self.scales_dtype),
            ("values_per_scale", self.values_per_scale),
        ):
            if value is not None:
                parts.append(f"{label}={value}")
        parts.extend(f"{key}={value}" for key, value in self.payload)
        return " ".join(parts)


def _dtype(value) -> Optional[str]:
    return str(value.dtype) if isinstance(value, torch.Tensor) else None


def _values_per_scale(fmt: str, in_features: int, scales) -> Optional[int]:
    """Block size a ``(rows, in_features / block)`` scale tensor implies.

    A degenerate shape (a meta placeholder, an intermediate dimension smaller
    than the block) falls back to the format's declared block size, which is
    what the decoder uses anyway.
    """

    declared = _PACKED_BLOCK.get(fmt)
    if not isinstance(scales, torch.Tensor) or scales.dim() != 2:
        return declared
    columns = int(scales.shape[-1])
    if columns <= 0 or in_features % columns:
        return declared
    return in_features // columns


def quantization_signature(module: nn.Module) -> QuantizationSignature:
    """Return the quantization identity of one module.

    ``packed`` is a block-scaled checkpoint payload that was never quantized,
    ``quantized`` is an installed kernel module, ``dense`` is an ordinary
    floating-point linear and ``unknown`` is anything else.
    """

    packed_weight = getattr(module, "packed_weight", None)
    if isinstance(packed_weight, torch.Tensor):
        fmt = str(getattr(module, "fmt", type(module).__name__))
        packed_scales = getattr(module, "packed_scales", None)
        return QuantizationSignature(
            kind="packed",
            format=fmt,
            bits=_PACKED_BITS.get(fmt),
            weight_dtype=_dtype(packed_weight),
            scales_dtype=_dtype(packed_scales),
            values_per_scale=_values_per_scale(fmt, int(getattr(module, "in_features", 0)), packed_scales),
            payload=(("global_scale", _dtype(getattr(module, "packed_global_scale", None))),),
        )

    bits = getattr(module, "bits", None)
    if isinstance(bits, int) and hasattr(module, "in_features"):
        return QuantizationSignature(
            kind="quantized",
            format=type(module).__name__,
            bits=bits,
            weight_dtype=_dtype(getattr(module, "qweight", None)),
            scales_dtype=_dtype(getattr(module, "scales", None)),
            payload=tuple((name, str(getattr(module, name, None))) for name in _KERNEL_FIELDS),
        )

    weight = getattr(module, "weight", None)
    if isinstance(module, nn.Linear) and isinstance(weight, torch.Tensor):
        return QuantizationSignature(kind="dense", format=str(weight.dtype), weight_dtype=str(weight.dtype))

    return QuantizationSignature(kind="unknown", format=type(module).__name__)


def assert_refusable(
    target: str,
    modules: Sequence[Tuple[str, nn.Module]],
    *,
    allow: Iterable[str] = ("packed",),
) -> QuantizationSignature:
    """Assert every module that would be merged carries one identical quantization.

    ``target`` names the container for the error message. ``allow`` lists the
    kinds that may be merged at all: a kernel module never can, because
    concatenating its payload with another's would change the checkpoint's
    format. Raises before the caller creates any tensor.
    """

    if not modules:
        raise RefusionContractError(f"{target}: nothing to re-fuse.")

    labelled = [(name, module, quantization_signature(module)) for name, module in modules]
    allowed = tuple(allow)
    reference_name, _, reference = labelled[0]

    for name, module, signature in labelled:
        if signature.kind not in allowed:
            reason = (
                "its storage is a different layout"
                if signature.kind == "quantized"
                else "it does not hold the container's packed payload"
            )
            raise RefusionContractError(
                f"{target} cannot be re-fused: {name} already holds "
                f"{type(module).__name__} ({signature.describe()}); {reason}. A container can "
                "only be rebuilt from its own packed payload, so quantized modules stay "
                "per-module (contracts/model/refusion.md R2)."
            )
        if signature != reference:
            raise RefusionContractError(
                f"{target} cannot be re-fused: its modules carry different quantization "
                "(contracts/model/refusion.md R1).\n"
                f"  {reference_name}: {reference.describe()}\n"
                f"  {name}: {signature.describe()}"
            )
    return reference


__all__ = [
    "QuantizationSignature",
    "RefusionContractError",
    "assert_refusable",
    "quantization_signature",
]
