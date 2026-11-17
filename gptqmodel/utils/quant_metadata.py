# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Classify quantization metadata this loader cannot execute.

The contract is weight-only loading: activations and the KV cache belong to the
serving runtime. A checkpoint that declares either must be refused with a
message that names the feature it actually asked for. Reporting a
``kv_cache_scheme`` as "activation quantization" sends the user looking in the
wrong part of their config, and ``num_bits`` thresholds have to agree between the
early config check and the ModelOpt runtime check or a 16-bit scheme is rejected
in one place and accepted in the other.
"""

from dataclasses import dataclass
from typing import Any, Optional


ACTIVATION_FIELDS = ("input_activations", "activation_quantization", "activations")
KV_CACHE_FIELD = "kv_cache_scheme"

# Bits at or above this are not a reduction, so a declared scheme that asks for
# them does not make the checkpoint unloadable.
_NO_REDUCTION_BITS = 16


@dataclass(frozen=True)
class UnsupportedMetadata:
    """One piece of metadata that this loader cannot honor."""

    field: str
    kind: str                       # "activation" | "kv_cache" | "checkpoint_scales"
    num_bits: Optional[int] = None


def _declared_bits(scheme: dict) -> Optional[int]:
    bits = scheme.get("num_bits")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        return None
    return int(bits)


def declares_reduction(scheme: Any) -> bool:
    """True when a declared scheme actually reduces precision.

    A scheme that says ``num_bits: 16`` is a no-op and must not block loading; a
    scheme that omits the width is assumed to quantize, because the safe reading
    of "there is a scheme here" is that bits were meant to be saved.
    """

    if not isinstance(scheme, dict) or not scheme:
        return False
    bits = _declared_bits(scheme)
    return bits is None or bits < _NO_REDUCTION_BITS


def find_unsupported_metadata(quantization_config: Any) -> Optional[UnsupportedMetadata]:
    """First unsupported feature in a quantization payload, or None."""

    if not isinstance(quantization_config, dict):
        return None

    config_groups = quantization_config.get("config_groups")
    if isinstance(config_groups, dict):
        for group_cfg in config_groups.values():
            if not isinstance(group_cfg, dict):
                continue
            for field in ACTIVATION_FIELDS:
                scheme = group_cfg.get(field)
                if declares_reduction(scheme):
                    return UnsupportedMetadata(field=field, kind="activation",
                                               num_bits=_declared_bits(scheme))

    scheme = quantization_config.get(KV_CACHE_FIELD)
    if declares_reduction(scheme):
        return UnsupportedMetadata(field=KV_CACHE_FIELD, kind="kv_cache",
                                   num_bits=_declared_bits(scheme))

    for field in ACTIVATION_FIELDS:
        scheme = quantization_config.get(field)
        if declares_reduction(scheme):
            return UnsupportedMetadata(field=field, kind="activation",
                                       num_bits=_declared_bits(scheme))
    return None


def describe_unsupported_metadata(found: UnsupportedMetadata) -> str:
    """A message that names the feature instead of the wrong one."""

    bits = f" ({found.num_bits} bits)" if found.num_bits is not None else ""
    if found.kind == "kv_cache":
        return (
            "GPT-QModel does not support loading of checkpoints that quantize the KV cache. "
            "The KV cache is allocated and dequantized by the serving runtime, so a declared "
            "kv_cache_scheme cannot be honored by this loader. "
            f"Detected unsupported metadata: {found.field}{bits}. "
            "Load a weight-only variant, or serve this checkpoint with a runtime that applies "
            "its KV-cache scheme."
        )
    if found.kind == "checkpoint_scales":
        return (
            "GPT-QModel does not support loading of checkpoints with pre-quantized activation "
            f"scales. Detected unsupported metadata: {found.field}."
        )
    return (
        "GPT-QModel does not support loading of activation quantized models. "
        f"Detected unsupported metadata: {found.field}{bits}."
    )
