# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""One normalized constructor contract shared by packing and checkpoint restore.

Packing resolves a module's effective settings once into a
``ModuleConstructorSpec``. Checkpoint serialization derives the same object
from the packed module and restore consumes it, so the two paths cannot drift
into separate field inventories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .config import (
    FORMAT,
    FORMAT_FIELD_CODE,
    _normalize_bitsandbytes_block_size,
    _normalize_bitsandbytes_format,
    _normalize_fp8_fmt,
    _normalize_fp8_scale_semantics,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
    _normalize_gguf_config_spec,
    _normalize_pack_dtype,
    _normalize_quant_bits,
    quant_bits_width,
    resolve_quant_format,
)


_PACK_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64})


@dataclass(frozen=True)
class ModuleConstructorSpec:
    """Validated, immutable constructor settings for one packed module.

    ``format`` is the ``FORMAT`` layout used for packing. Method-specific
    constructor arguments (FP8/BitsAndBytes subtypes, Paro rotation options)
    live in ``options`` so one object can describe every method without a
    universal quantization framework.
    """

    bits: Any
    group_size: Optional[int]
    desc_act: bool
    sym: bool
    pack_dtype: torch.dtype
    format: FORMAT
    options: Tuple[Tuple[str, Any], ...] = ()
    kind: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bits is None:
            raise ValueError("module constructor spec requires bits")
        if quant_bits_width(self.bits) <= 0:
            raise ValueError(f"module constructor spec has invalid bits: {self.bits!r}")
        if self.group_size is not None and (
            not isinstance(self.group_size, int) or self.group_size == 0
        ):
            raise ValueError(
                f"module constructor spec has invalid group_size: {self.group_size!r}"
            )
        if not isinstance(self.desc_act, bool):
            raise ValueError("module constructor spec requires a boolean desc_act")
        if not isinstance(self.sym, bool):
            raise ValueError("module constructor spec requires a boolean sym")
        if self.pack_dtype not in _PACK_DTYPES:
            raise ValueError(
                f"module constructor spec has invalid pack_dtype: {self.pack_dtype!r}"
            )
        if not isinstance(self.format, FORMAT):
            raise ValueError(
                f"module constructor spec has invalid format: {self.format!r}"
            )
        if any(not isinstance(key, str) for key, _ in self.options):
            raise ValueError("module constructor spec option keys must be strings")

    def options_dict(self) -> Dict[str, Any]:
        return dict(self.options)

    def constructor_kwargs(self) -> Dict[str, Any]:
        """Return the complete kwargs needed to rebuild this module."""
        kwargs = {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "pack_dtype": self.pack_dtype,
            "format": self.format,
        }
        # Method-specific options win: for FP8/BitsAndBytes `format` is the
        # storage subtype, not the top-level FORMAT enum.
        kwargs.update(self.options_dict())
        return kwargs


def resolve_module_constructor_spec(
    *,
    bits,
    group_size: Optional[int],
    desc_act: bool,
    sym: bool,
    pack_dtype: torch.dtype,
    format: FORMAT,
    init_kwargs: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[ModuleConstructorSpec]:
    """Apply one module's winning dynamic rule and return its constructor spec.

    ``overrides`` is the already-resolved rule payload (``None`` when no rule
    matched, ``False`` when the module is excluded). The function is the single
    place that normalizes per-module overrides for packing.
    """

    if overrides is False:
        return None

    tmp_bits = bits
    tmp_group_size = group_size
    tmp_desc_act = desc_act
    tmp_sym = sym
    tmp_pack_dtype = pack_dtype
    options = dict(init_kwargs or {})

    if overrides:
        tmp_bits = _normalize_quant_bits(
            overrides.get("bits", bits), format_value=format
        )
        tmp_group_size = overrides.get("group_size", group_size)
        tmp_desc_act = overrides.get("desc_act", desc_act)
        tmp_sym = overrides.get("sym", sym)
        tmp_pack_dtype = _normalize_pack_dtype(
            overrides.get("pack_dtype", pack_dtype)
        )

        if format == FORMAT.FP8:
            fp8_format_override = overrides.get(
                FORMAT_FIELD_CODE, overrides.get("fmt")
            )
            if fp8_format_override is not None:
                options["format"] = _normalize_fp8_fmt(fp8_format_override)
            block_size_override = overrides.get(
                "weight_block_size", options.get("weight_block_size")
            )
            normalized_block_size = _normalize_fp8_weight_block_size(
                block_size_override
            )
            if "weight_scale_method" in overrides or block_size_override is not None:
                options["weight_scale_method"] = _normalize_fp8_weight_scale_method(
                    overrides.get(
                        "weight_scale_method", options.get("weight_scale_method")
                    ),
                    weight_block_size=normalized_block_size,
                )
            if "weight_scale_semantics" in overrides:
                options["weight_scale_semantics"] = _normalize_fp8_scale_semantics(
                    overrides["weight_scale_semantics"]
                )
            if "weight_block_size" in overrides:
                options["weight_block_size"] = normalized_block_size
        elif format == FORMAT.BITSANDBYTES:
            raw_format = overrides.get(
                FORMAT_FIELD_CODE, overrides.get("bnb_quant_type")
            )
            if raw_format is not None:
                options["format"] = _normalize_bitsandbytes_format(
                    raw_format, bits=quant_bits_width(tmp_bits)
                )
            if "block_size" in overrides or "bnb_block_size" in overrides:
                options["block_size"] = _normalize_bitsandbytes_block_size(
                    overrides.get("block_size", overrides.get("bnb_block_size"))
                )
            if (
                "compress_statistics" in overrides
                or "bnb_compress_statistics" in overrides
            ):
                options["compress_statistics"] = bool(
                    overrides.get(
                        "compress_statistics",
                        overrides.get("bnb_compress_statistics"),
                    )
                )
        elif format == FORMAT.GGUF:
            # Resolve width and subtype independently, matching the config used
            # to quantize this module. Width-only overrides inherit the base
            # subtype; format-only overrides inherit only its width.
            base_width, base_format, _ = _normalize_gguf_config_spec(bits, None)
            _, _, tmp_bits = _normalize_gguf_config_spec(
                overrides.get("bits", base_width),
                overrides.get(FORMAT_FIELD_CODE, base_format),
            )

    return ModuleConstructorSpec(
        bits=tmp_bits,
        group_size=tmp_group_size,
        desc_act=bool(tmp_desc_act),
        sym=bool(tmp_sym),
        pack_dtype=tmp_pack_dtype,
        format=format,
        options=tuple(options.items()),
    )


def module_constructor_spec(
    module,
    config,
    *,
    module_name: Optional[str] = None,
) -> ModuleConstructorSpec:
    """Derive the spec from an already-packed module's resolved runtime state."""

    format_value = getattr(module, "format", None)
    if format_value is None and module_name is not None:
        format_value = config.resolve_module_config(module_name).get(
            "format", config.format
        )
    if format_value is None or format_value is False:
        format_value = config.format
    module_format = resolve_quant_format(format_value, config.method)

    options: Dict[str, Any] = {}
    kind: Optional[str] = None
    quant_type = getattr(module, "QUANT_TYPE", None)

    if quant_type == "awq_paroquant":
        kind = "paro"
        options = {
            key: getattr(module, key)
            for key in (
                "krot",
                "fp32_accum",
                "cache_runtime_dtype",
                "auto_cache_bf16_runtime_dtype",
                "cache_rotation_dtype",
                "auto_cache_bf16_rotation_dtype",
            )
        }
    if quant_type == "fp8":
        options = {
            "format": module.fp8_format,
            "weight_scale_method": module.weight_scale_method,
            "weight_block_size": (
                list(module.weight_block_size)
                if module.weight_block_size is not None
                else None
            ),
            "weight_scale_semantics": module.weight_scale_semantics,
        }
    elif quant_type == "bitsandbytes":
        options = {
            "format": module.bnb_format,
            "block_size": module.bnb_block_size,
            "compress_statistics": module.bnb_compress_statistics,
        }
    if quant_type == "qqq":
        kind = "qqq"

    return ModuleConstructorSpec(
        bits=getattr(module, "bits", config.runtime_bits),
        group_size=getattr(module, "group_size", config.group_size),
        desc_act=getattr(module, "desc_act", config.desc_act),
        sym=getattr(module, "sym", config.sym),
        pack_dtype=getattr(module, "pack_dtype", config.pack_dtype),
        format=module_format,
        options=tuple(options.items()),
        kind=kind,
    )
