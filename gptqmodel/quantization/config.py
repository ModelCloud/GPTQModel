# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import math
import os.path
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import total_ordering
from os.path import join
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pcre
import torch
from packaging import version

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger
from ..utils.random_str import get_random_string


log = setup_logger()

_DECODER_TARGET_DTYPE_MAP = {
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "format"
SYMMETRIC_FIELD_CODE = "sym"
# Deprecated JSON alias retained for backward compatibility.
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
# Hard-deprecated legacy alias. Presence should fail fast.
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
# Canonical method field; `quant_method` is a deprecated JSON alias.
METHOD_FIELD_CODE = "method"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]
# This is AwqBackendPackingMethod, not the GPT-QModel backend enum.
# It's used to distinguish between quantization by llm-awq and autoawq; llm-awq actually uses GEMV_FAST for packing.
AWQ_PACKING_BACKEND_FIELD = "backend"

MIN_VERSION_WITH_V2 = "0.9.0"

META_FIELD = "meta"
# quantizer is the tool that did the quantization
META_FIELD_QUANTIZER = "quantizer"

META_QUANTIZER_GPTQMODEL = "gptqmodel"

META_FIELD_URI = "uri"
META_VALUE_URI = "https://github.com/modelcloud/gptqmodel"

META_FIELD_DAMP_PERCENT = "damp_percent"
META_FIELD_DAMP_AUTO_INCREMENT = "damp_auto_increment"

META_FIELD_STATIC_GROUPS = "static_groups"
META_FIELD_TRUE_SEQUENTIAL = "true_sequential"

META_FIELD_MSE = "mse"
META_FIELD_ACT_GROUP_AWARE = "act_group_aware"

META_FIELD_GPTAQ_ENABLED = "gptaq"

META_FIELD_FOEM_ENABLED = "foem"

ADAPTER_FIELD = "adapter"

# saved formats
class FORMAT(str, Enum):
    """Checkpoint and runtime tensor layout identifiers."""

    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    QQQ = "qqq"
    EXL3 = "exl3"

    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    LLM_AWQ = "llm-awq"
    PAROQUANT = "paroquant"


# quant methods
class METHOD(str, Enum):
    """Supported quantization algorithms exposed by config payloads."""

    GPTQ = "gptq"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    QQQ = "qqq"
    AWQ = "awq"
    EXL3 = "exl3"
    PARO = "paroquant"


class VramStrategy(str, Enum):
    """Placement strategies shared by dense and MoE device pools."""

    EXCLUSIVE = "exclusive"
    BALANCED = "balanced"


class FallbackStrategy(str, Enum):
    """
    +-----------+----------------------+---------------------------+------------------------------+
    | strategy  | center               | scale                     | strengths / weaknesses       |
    +-----------+----------------------+---------------------------+------------------------------+
    | rtn       | min/max (quantizer)  | min/max (quantizer)        | simple, but outlier-driven   |
    | midpoint  | (min+max)/2          | (max-min)                  | symmetric, outlier-sensitive |
    | mean      | mean(w)              | 2*max(|w-mean|)            | stable for symmetric data    |
    | median    | median(w)            | 2*max(|w-median|)          | robust center vs outliers    |
    | stdclip   | mean(w)              | 2*sigma*std                | tames tails, may clip signal |
    +-----------+----------------------+---------------------------+------------------------------+
    """
    RTN = "rtn" # round to nearest
    MIDPOINT = "midpoint"
    MEAN = "mean"
    MEDIAN = "median"
    STDCLIP = "stdclip"


class WeightOnlyMethod(str, Enum):
    """Weight-only quantization backends available to fallback flows."""

    RTN = "rtn"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    NVFP4 = "nvfp4"


class PreProcessorCode(str, Enum):
    """Identifiers for preprocessing passes that run before quantization."""

    SMOOTHER = "smoother"
    AUTO_MODULE_DECODER = "auto_module_decoder"
    TENSOR_PARALLEL_PADDER = "tensor_parallel_padder"


_GGUF_BITS_ALIAS_INFO = {
    "q1_0": {"bits": 1, "version": "q", "variant": "0", "quality": None},
    "q1_0_g128": {"bits": 1, "version": "q", "variant": "0", "quality": "g128"},
    "q4_0": {"bits": 4, "version": "q", "variant": "0", "quality": None},
    "q8_0": {"bits": 8, "version": "q", "variant": "0", "quality": None},
    "q4_k": {"bits": 4, "version": "q", "variant": "k", "quality": None},
    "q4_k_s": {"bits": 4, "version": "q", "variant": "k", "quality": "s"},
    "q4_k_m": {"bits": 4, "version": "q", "variant": "k", "quality": "m"},
    "q5_k": {"bits": 5, "version": "q", "variant": "k", "quality": None},
    "q5_k_s": {"bits": 5, "version": "q", "variant": "k", "quality": "s"},
    "q5_k_m": {"bits": 5, "version": "q", "variant": "k", "quality": "m"},
    "q6_k": {"bits": 6, "version": "q", "variant": "k", "quality": None},
}
_GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH = {
    1: "q1_0",
    4: "q4_0",
    5: "q5_k_m",
    6: "q6_k",
    8: "q8_0",
}
_GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS = {
    "q1_0": 1.5,
    "q1_0_g128": 1.125,
    "q4_0": 4.5,
    "q8_0": 8.5,
    "q4_k": 4.5,
    "q4_k_s": 4.5,
    "q4_k_m": 4.5,
    "q5_k": 5.5,
    "q5_k_s": 5.0,
    "q5_k_m": 5.5,
    "q6_k": 6.0,
}


@total_ordering
class BaseComplexBits(ABC):
    """Comparable bit-spec base class for non-scalar bit encodings."""

    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> "BaseComplexBits":
        """Parse a serialized bit specification into an instance."""

        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        """Serialize the bit specification into its canonical string form."""

        raise NotImplementedError

    @property
    def width(self) -> int:
        """Return the integer width represented by this bit encoding."""

        return self.bits

    @property
    def name(self) -> str:
        """Return the canonical string name for this bit encoding."""

        return self.to_string()

    def _coerce_bits(self, other: Any) -> Any:
        """Convert compatible operands into raw bit widths for arithmetic."""

        if isinstance(other, BaseComplexBits):
            return other.bits
        if isinstance(other, int):
            return other
        if isinstance(other, str) and other.strip().isdigit():
            return int(other.strip())
        return NotImplemented

    def __str__(self) -> str:
        """Render the canonical string form for logging and serialization."""

        return self.to_string()

    def __hash__(self) -> int:
        """Hash bit encodings by their integer width."""

        return hash(self.bits)

    def __int__(self) -> int:
        """Expose the bit width as an integer."""

        return self.bits

    def __index__(self) -> int:
        """Allow the bit width to participate in index-style conversions."""

        return self.bits

    def __float__(self) -> float:
        """Expose the bit width as a float."""

        return float(self.bits)

    def __eq__(self, other: Any) -> bool:
        """Compare complex bit encodings against strings, ints, or peers."""

        if isinstance(other, BaseComplexBits):
            return self.to_string() == other.to_string()
        if isinstance(other, int):
            return self.bits == other
        if isinstance(other, str):
            normalized = other.strip().lower().replace("-", "_")
            if normalized.isdigit():
                return self.bits == int(normalized)
            return self.to_string() == normalized
        return False

    def __lt__(self, other: Any) -> bool:
        """Order bit encodings by their effective width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits < coerced

    def __add__(self, other: Any) -> int:
        """Add the effective bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits + coerced

    def __radd__(self, other: Any) -> int:
        """Support right-hand addition with scalar-like operands."""

        return self.__add__(other)

    def __sub__(self, other: Any) -> int:
        """Subtract another scalar-like operand from this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits - coerced

    def __rsub__(self, other: Any) -> int:
        """Support right-hand subtraction against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced - self.bits

    def __mul__(self, other: Any) -> int:
        """Multiply the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits * coerced

    def __rmul__(self, other: Any) -> int:
        """Support right-hand multiplication with scalar-like operands."""

        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> int:
        """Floor-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits // coerced

    def __rfloordiv__(self, other: Any) -> int:
        """Support right-hand floor division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced // self.bits

    def __truediv__(self, other: Any) -> float:
        """True-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits / coerced

    def __rtruediv__(self, other: Any) -> float:
        """Support right-hand true division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced / self.bits

    def __mod__(self, other: Any) -> int:
        """Take the modulo of the bit width with another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits % coerced

    def __rmod__(self, other: Any) -> int:
        """Support right-hand modulo against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced % self.bits

    def __pow__(self, other: Any) -> int:
        """Raise the bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits ** coerced

    def __rpow__(self, other: Any) -> int:
        """Support right-hand exponentiation against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced ** self.bits


@dataclass(frozen=True, eq=False)
class GGUFBits(BaseComplexBits):
    """Structured GGUF bit specification with version and variant tags."""

    bits: int
    version: str
    variant: str
    quality: Optional[str] = None

    def __post_init__(self):
        """Validate the GGUF bit-spec components after construction."""

        if self.bits <= 0:
            raise ValueError("GGUFBits: `bits` must be a positive integer.")
        if self.version not in {"q", "iq"}:
            raise ValueError("GGUFBits: `version` must be `q` or `iq`.")
        if self.variant not in {"0", "k"}:
            raise ValueError("GGUFBits: `variant` must be `0` or `k`.")
        if self.quality not in {None, "xs", "s", "m", "l", "g128"}:
            raise ValueError("GGUFBits: `quality` must be one of `[None, xs, s, m, l, g128]`.")

    @classmethod
    def from_string(cls, value: str) -> "GGUFBits":
        """Parse a GGUF alias such as ``q4_k_m`` into a typed bit spec."""

        normalized = str(value).strip().lower().replace("-", "_")
        info = _GGUF_BITS_ALIAS_INFO.get(normalized)
        if info is None:
            supported = ", ".join(sorted(_GGUF_BITS_ALIAS_INFO))
            raise ValueError(f"Unsupported GGUF bits `{value}`. Supported values: {supported}.")
        return cls(
            bits=info["bits"],
            version=info["version"],
            variant=info["variant"],
            quality=info["quality"],
        )

    def to_string(self) -> str:
        """Serialize this GGUF bit spec back to its alias form."""

        alias = f"{self.version}{self.bits}_{self.variant}"
        if self.quality is not None:
            alias = f"{alias}_{self.quality}"
        return alias

    @classmethod
    def from_alias(cls, value: str) -> "GGUFBits":
        """Backward-compatible alias parser for GGUF bit specs."""

        return cls.from_string(value)

    def serialize(self) -> str:
        """Return the canonical serialized form used in config payloads."""

        return self.to_string()

    def __repr__(self) -> str:
        """Return a debug-friendly constructor-style representation."""

        return f"GGUFBits({self.to_string()!r})"

    def to_public_format(self) -> str:
        """Return the GGUF public subtype string without the width prefix."""

        public_format = f"{self.version}_{self.variant}"
        if self.quality is not None:
            public_format = f"{public_format}_{self.quality}"
        return public_format


# Backward-compatible alias for the earlier wrapper-based refactor.
QuantBits = GGUFBits


_GGUF_PUBLIC_FORMAT_RE = pcre.compile(r"^(q|iq)_(0|k)(?:_(xs|s|m|l|g128))?$")


def _gguf_public_format_from_bits(bits: GGUFBits) -> str:
    """Project a full GGUF bit spec into its public subtype token."""

    return bits.to_public_format()


def _normalize_gguf_public_format(value: Any) -> Optional[str]:
    """Normalize GGUF subtype aliases into their public format string."""

    if value is None:
        return None

    if isinstance(value, GGUFBits):
        return _gguf_public_format_from_bits(value)

    if isinstance(value, FORMAT):
        value = value.value

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", FORMAT.GGUF.value}:
        return None
    if normalized in _GGUF_BITS_ALIAS_INFO:
        return _gguf_public_format_from_bits(GGUFBits.from_alias(normalized))
    if _GGUF_PUBLIC_FORMAT_RE.fullmatch(normalized):
        return normalized

    raise ValueError(
        "GGUFConfig: `format` must be a GGUF subtype like `q_0`, `q_k`, `q_k_s`, or `q_k_m`."
    )


def _default_gguf_public_format(bits: int) -> str:
    """Return the default GGUF subtype for a supported bit width."""

    alias = _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH.get(bits)
    if alias is None:
        raise ValueError(f"GGUFConfig: no default GGUF format exists for `{bits}`-bit quantization.")
    return _gguf_public_format_from_bits(GGUFBits.from_alias(alias))


def _gguf_bits_from_components(bits: int, public_format: str) -> GGUFBits:
    """Build a validated ``GGUFBits`` object from width and subtype parts."""

    match = _GGUF_PUBLIC_FORMAT_RE.fullmatch(public_format)
    if match is None:
        raise ValueError(
            "GGUFConfig: `format` must be a GGUF subtype like `q_0`, `q_k`, `q_k_s`, or `q_k_m`."
        )

    version_name, variant, quality = match.groups()
    bits_spec = GGUFBits(bits=bits, version=version_name, variant=variant, quality=quality)
    if bits_spec.to_string() not in _GGUF_BITS_ALIAS_INFO:
        raise ValueError(
            f"Unsupported GGUF combination: bits={bits}, format={public_format}."
        )
    return bits_spec


def _normalize_gguf_config_spec(
    bits: Union[int, str, GGUFBits],
    format_value: Optional[Union[str, FORMAT, GGUFBits]],
) -> Tuple[int, str, GGUFBits]:
    """Resolve GGUF bits and format inputs into a consistent typed triple."""

    bits_spec_from_bits: Optional[GGUFBits] = None
    normalized_bits = bits

    if isinstance(bits, GGUFBits):
        bits_spec_from_bits = bits
        normalized_bits = bits.bits
    elif isinstance(bits, str):
        raw_bits = bits.strip().lower().replace("-", "_")
        if raw_bits.isdigit():
            normalized_bits = int(raw_bits)
        else:
            bits_spec_from_bits = GGUFBits.from_alias(raw_bits)
            normalized_bits = bits_spec_from_bits.bits
    elif not isinstance(bits, int):
        raise ValueError(f"GGUFConfig: unsupported bits specification `{bits}`.")

    normalized_bits = int(normalized_bits)
    if normalized_bits not in [1, 2, 3, 4, 5, 6, 8]:
        raise ValueError("GGUFConfig: `bits` must resolve to one of `[1, 2, 3, 4, 5, 6, 8]`.")

    normalized_format = _normalize_gguf_public_format(format_value)
    if normalized_format is None:
        if bits_spec_from_bits is not None:
            bits_spec = bits_spec_from_bits
            normalized_format = _gguf_public_format_from_bits(bits_spec)
        else:
            normalized_format = _default_gguf_public_format(normalized_bits)
            bits_spec = _gguf_bits_from_components(normalized_bits, normalized_format)
    else:
        bits_spec = _gguf_bits_from_components(normalized_bits, normalized_format)
        if bits_spec_from_bits is not None and bits_spec_from_bits != bits_spec:
            raise ValueError(
                f"GGUFConfig: incompatible GGUF bits/format combination: bits={bits}, format={format_value}."
            )

    return normalized_bits, normalized_format, bits_spec


def _normalize_quant_bits(bits: Union[int, float, str, GGUFBits], format_value: Optional[Union[str, FORMAT]] = None) -> Union[int, GGUFBits]:
    """Normalize generic bit fields into ints or structured GGUF specs."""

    if isinstance(format_value, str):
        format_value = _normalize_format(format_value)

    if isinstance(bits, GGUFBits):
        normalized = bits
    elif isinstance(bits, float):
        if format_value == FORMAT.EXL3:
            normalized = bits
        elif bits.is_integer():
            normalized = int(bits)
        else:
            raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")
    elif isinstance(bits, int):
        normalized = bits
    elif isinstance(bits, str):
        raw = bits.strip().lower().replace("-", "_")
        normalized = int(raw) if raw.isdigit() else GGUFBits.from_alias(raw)
    else:
        raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")

    normalized_width = normalized.bits if isinstance(normalized, GGUFBits) else normalized
    valid_bit_widths = [1, 2, 3, 4, 5, 6, 8]
    if normalized_width not in valid_bit_widths:
        raise ValueError(f"QuantizeConfig: `bits` must resolve to one of `{valid_bit_widths}`.")

    if format_value == FORMAT.GGUF and not isinstance(normalized, GGUFBits):
        default_alias = _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH.get(normalized_width)
        if default_alias is None:
            raise ValueError(
                f"QuantizeConfig: no default GGUF bits alias exists for `{normalized_width}`-bit quantization."
            )
        normalized = GGUFBits.from_alias(default_alias)

    if isinstance(normalized, GGUFBits) and format_value is not None and format_value != FORMAT.GGUF:
        raise ValueError("QuantizeConfig: GGUF bit encodings require `format=gguf`.")

    return normalized


def resolve_quant_format(
    format_value: Optional[Union[str, FORMAT]],
    method: Optional[Union[str, METHOD]] = None,
    quant_method: Optional[Union[str, METHOD]] = None,
) -> FORMAT:
    """Infer the effective quantization format from method and format hints."""

    if method is None:
        method = quant_method

    if isinstance(method, str):
        method = _normalize_quant_method(method)

    if method == METHOD.GGUF:
        return FORMAT.GGUF
    if method == METHOD.FP8:
        return FORMAT.FP8
    if method == METHOD.BITSANDBYTES:
        return FORMAT.BITSANDBYTES
    if method == METHOD.EXL3:
        return FORMAT.EXL3
    if method == METHOD.PARO:
        return FORMAT.PAROQUANT

    if isinstance(format_value, FORMAT):
        return format_value

    try:
        if _normalize_gguf_public_format(format_value) is not None:
            return FORMAT.GGUF
    except ValueError:
        pass

    if _looks_like_fp8_fmt(format_value):
        return FORMAT.FP8
    if _looks_like_bitsandbytes_format(format_value):
        return FORMAT.BITSANDBYTES

    if format_value is None:
        return FORMAT.GPTQ

    return _normalize_format(format_value)


def _looks_like_gguf_bits(bits: Any) -> bool:
    """Return ``True`` when a value resembles a GGUF alias or bit spec."""

    if isinstance(bits, GGUFBits):
        return True
    if not isinstance(bits, str):
        return False
    normalized = bits.strip().lower().replace("-", "_")
    return normalized in _GGUF_BITS_ALIAS_INFO


def quant_bits_width(bits: Union[int, str, GGUFBits]) -> int:
    """Return the integer width represented by a quant bits field."""

    if isinstance(bits, float):
        if bits <= 0:
            raise ValueError("QuantizeConfig: EXL3 bits per weight must be greater than 0.")
        return max(1, int(math.floor(bits)))
    normalized = _normalize_quant_bits(bits)
    return normalized.bits if isinstance(normalized, GGUFBits) else normalized


def serialize_quant_bits(bits: Union[int, str, GGUFBits]) -> Union[int, str]:
    """Serialize a quant bits field for JSON-compatible output payloads."""

    if isinstance(bits, float):
        return float(bits)
    normalized = _normalize_quant_bits(bits)
    return normalized.serialize() if isinstance(normalized, GGUFBits) else normalized


def _normalize_exl3_bits(bits: Union[int, float, str]) -> float:
    """Normalize EXL3 fractional bits-per-weight values."""

    if isinstance(bits, str):
        bits = float(bits.strip())
    elif isinstance(bits, int):
        bits = float(bits)
    elif not isinstance(bits, float):
        raise ValueError(f"EXL3Config: unsupported bits specification `{bits}`.")

    if not math.isfinite(bits):
        raise ValueError("EXL3Config: `bits` must be finite.")
    if bits < 1.0 or bits > 8.0:
        raise ValueError("EXL3Config: `bits` must be between 1.0 and 8.0.")
    return float(bits)


# Canonical FP8 aliases are normalized here before validating torch runtime
# support so config payloads can use either shorthand or exact dtype names.
_FP8_FMT_ALIASES = {
    "e4m3": "float8_e4m3fn",
    "float8_e4m3": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "e4m3fnuz": "float8_e4m3fnuz",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
    "e5m2fnuz": "float8_e5m2fnuz",
    "float8_e5m2fnuz": "float8_e5m2fnuz",
    "e8m0": "float8_e8m0fnu",
    "e8m0fnu": "float8_e8m0fnu",
    "float8_e8m0": "float8_e8m0fnu",
    "float8_e8m0fnu": "float8_e8m0fnu",
}
_FP8_WEIGHT_SCALE_METHODS = {"tensor", "row", "block"}
_FP8_SCALE_SEMANTICS = {"inverse"}
_BITSANDBYTES_4BIT_FORMATS = {"fp4", "nf4"}
_BITSANDBYTES_8BIT_FORMATS = {"int8"}
_BITSANDBYTES_FORMATS = _BITSANDBYTES_4BIT_FORMATS | _BITSANDBYTES_8BIT_FORMATS
_BITSANDBYTES_BLOCK_SIZES = {32, 64, 128, 256, 512, 1024, 2048, 4096}


def _looks_like_fp8_fmt(value: Any) -> bool:
    """Return ``True`` when a value matches a supported FP8 format alias."""

    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in _FP8_FMT_ALIASES


def _normalize_fp8_fmt(value: Optional[str]) -> str:
    """Resolve FP8 format aliases to the canonical PyTorch dtype name."""

    if isinstance(value, FORMAT):
        if value != FORMAT.FP8:
            raise ValueError(f"FP8Config: unsupported `format` `{value}`.")
        value = None

    normalized = "float8_e4m3fn" if value is None else str(value).strip().lower()
    if normalized in {"", FORMAT.FP8.value}:
        normalized = "float8_e4m3fn"
    resolved = _FP8_FMT_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(_FP8_FMT_ALIASES))
        raise ValueError(f"FP8Config: unsupported `format` `{value}`. Supported values: {supported}.")
    if not hasattr(torch, resolved):
        raise ValueError(f"FP8Config: current PyTorch build does not provide `{resolved}`.")
    return resolved


def _normalize_fp8_weight_block_size(value: Optional[Union[List[int], Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Validate and normalize FP8 block-scale dimensions."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("FP8Config: `weight_block_size` must be a 2-item list/tuple or None.")
    rows, cols = int(value[0]), int(value[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("FP8Config: `weight_block_size` entries must be positive integers.")
    return rows, cols


def _normalize_fp8_weight_scale_method(
    value: Optional[str],
    *,
    weight_block_size: Optional[Tuple[int, int]],
) -> str:
    """Resolve the FP8 weight scaling strategy from config inputs."""

    normalized = "block" if weight_block_size is not None and value is None else (value or "row")
    normalized = str(normalized).strip().lower()
    if normalized not in _FP8_WEIGHT_SCALE_METHODS:
        supported = ", ".join(sorted(_FP8_WEIGHT_SCALE_METHODS))
        raise ValueError(
            f"FP8Config: `weight_scale_method` must be one of {{{supported}}}, got `{value}`."
        )
    if normalized == "block" and weight_block_size is None:
        raise ValueError("FP8Config: `weight_scale_method='block'` requires `weight_block_size`.")
    if normalized != "block" and weight_block_size is not None:
        raise ValueError(
            "FP8Config: `weight_block_size` is only valid when `weight_scale_method='block'`."
        )
    return normalized


def _normalize_fp8_scale_semantics(value: Optional[str]) -> str:
    """Normalize FP8 scale semantics to the supported enum-like string."""

    normalized = "inverse" if value is None else str(value).strip().lower()
    if normalized not in _FP8_SCALE_SEMANTICS:
        supported = ", ".join(sorted(_FP8_SCALE_SEMANTICS))
        raise ValueError(
            f"FP8Config: `weight_scale_semantics` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized


def _looks_like_bitsandbytes_format(value: Any) -> bool:
    """Return ``True`` when a value matches a bitsandbytes format alias."""

    if value is None:
        return False
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized in _BITSANDBYTES_FORMATS


def _normalize_bitsandbytes_format(value: Optional[str], *, bits: Optional[int] = None) -> str:
    """Normalize bitsandbytes format aliases for the requested bit width."""

    default_format = "int8" if bits == 8 else "fp4"
    normalized = default_format if value is None else str(value).strip().lower().replace("-", "_")
    if normalized in {"", FORMAT.BITSANDBYTES.value}:
        normalized = default_format

    if bits == 4:
        allowed_formats = _BITSANDBYTES_4BIT_FORMATS
    elif bits == 8:
        allowed_formats = _BITSANDBYTES_8BIT_FORMATS
    else:
        allowed_formats = _BITSANDBYTES_FORMATS

    if normalized not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        raise ValueError(
            f"BitsAndBytesConfig: `format` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized


def _normalize_bitsandbytes_quant_type(value: Optional[str]) -> str:
    """Normalize the legacy 4-bit bitsandbytes quant type field."""

    return _normalize_bitsandbytes_format(value, bits=4)


def _normalize_bitsandbytes_block_size(value: Optional[int]) -> int:
    """Validate and normalize the bitsandbytes block size setting."""

    normalized = 64 if value is None else int(value)
    if normalized not in _BITSANDBYTES_BLOCK_SIZES:
        supported = ", ".join(str(item) for item in sorted(_BITSANDBYTES_BLOCK_SIZES))
        raise ValueError(
            f"BitsAndBytesConfig: `block_size` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized

@dataclass
class SmoothMethod:
    """Base smoother descriptor shared by all smoothing strategies."""

    name: str
    # Apply the smoother only when group size >= this threshold.
    group_size_threshold: int = 128


@dataclass
class SmoothPercentile(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip(|w|) at p-th percentile             |
    | config         | SmoothPercentile(percentile=p)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile of |w| used as clip threshold |
    | effect         | higher p = less clipping                  |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0

    def __init__(self, percentile: float = 99.0, group_size_threshold: int = 128):
        """Configure percentile clipping with an optional group-size floor."""

        super().__init__(name="percentile", group_size_threshold=group_size_threshold)
        self.percentile = percentile


@dataclass
class SmoothPercentileAsymmetric(SmoothMethod):
    """
    +-------------------+-------------------------------------------+
    | math              | clip to [p_low, p_high] percentiles      |
    | config            | SmoothPercentileAsymmetric(low, high)    |
    +-------------------+-------------------------------------------+
    +-------------------+-------------------------------------------+
    | low/high          | percentile bounds on raw weights         |
    | effect            | asymmetric clipping of tails             |
    +-------------------+-------------------------------------------+
    """
    low: float = 0.5
    high: float = 99.5

    def __init__(self, low: float = 0.5, high: float = 99.5, group_size_threshold: int = 128):
        """Configure asymmetric percentile clipping bounds."""

        super().__init__(name="percentile_asym", group_size_threshold=group_size_threshold)
        self.low = low
        self.high = high


@dataclass
class SmoothMAD(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | median +/- K * MAD                        |
    | config         | SmoothMAD(k=K)                            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | width multiplier for MAD window           |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 2.75

    def __init__(self, k: float = 2.75, group_size_threshold: int = 128):
        """Configure MAD-based clipping width and activation threshold."""

        super().__init__(name="mad", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothMSE(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | grid-search shrink p in [1..maxshrink]    |
    | config         | SmoothMSE(steps=N, maxshrink=S)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | steps (N)      | number of shrink candidates               |
    | maxshrink (S)  | smallest range multiplier                 |
    | effect         | more steps = better fit, slower           |
    +----------------+-------------------------------------------+
    """
    steps: int = 32
    maxshrink: float = 0.8

    def __init__(self, steps: int = 32, maxshrink: float = 0.8, group_size_threshold: int = 128):
        """Configure search granularity for MSE-based shrinking."""

        super().__init__(name="mse", group_size_threshold=group_size_threshold)
        self.steps = steps
        self.maxshrink = maxshrink


@dataclass
class SmoothOutlier(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip by kth |w|, keep (100-pct)% mass     |
    | config         | SmoothOutlier(pct=p)                      |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | pct (p)        | top-pct of |w| treated as outliers        |
    | effect         | higher p = more clipping                  |
    +----------------+-------------------------------------------+
    """
    pct: float = 1.0

    def __init__(self, pct: float = 1.0, group_size_threshold: int = 128):
        """Configure top-percent outlier clipping behavior."""

        super().__init__(name="outlier", group_size_threshold=group_size_threshold)
        self.pct = pct


@dataclass
class SmoothSoftNorm(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | z=(w-mean)/rms, clip z to +/-K            |
    | config         | SmoothSoftNorm(k=K)                       |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | z-score clip limit                        |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 3.0

    def __init__(self, k: float = 3.0, group_size_threshold: int = 128):
        """Configure z-score clipping strength for soft normalization."""

        super().__init__(name="softnorm", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothLog(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | log1p(mu*|w|) percentile, invert to clip  |
    | config         | SmoothLog(percentile=p, mu=mu)            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile in log space for clip          |
    | mu             | log companding strength                   |
    | effect         | higher mu compresses outliers more        |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0
    mu: float = 8.0

    def __init__(self, percentile: float = 99.0, mu: float = 8.0, group_size_threshold: int = 128):
        """Configure log-domain smoothing with percentile and companding strength."""

        super().__init__(name="log", group_size_threshold=group_size_threshold)
        self.percentile = percentile
        self.mu = mu


@dataclass
class SmoothRowCol(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | divide by row/col RMS, re-scale after     |
    | config         | SmoothRowCol(axis="row"|"col")            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | axis           | apply RMS scale per "row" or "col"        |
    | effect         | normalizes dynamic range before quant     |
    +----------------+-------------------------------------------+
    """
    axis: str = "row"

    def __init__(self, axis: str = "row", group_size_threshold: int = 128):
        """Configure RMS normalization over rows or columns."""

        super().__init__(name="rowcol", group_size_threshold=group_size_threshold)
        self.axis = axis


class GcMode(str, Enum):
    """Policies for when staged garbage collection should run."""

    INTERVAL = "interval"
    ON_STAGE_END = "on_stage_end"


@dataclass
class Fallback:
    """Low-sample fallback strategy for modules with weak calibration coverage."""

    strategy: FallbackStrategy = FallbackStrategy.RTN # enable fallback by default due to moe routing behavior breaking calibration based quantization

    # int/float = if captured module fwd tokens is less than value, trigger strategy
    # string = if string is int/float followed by %, then if captured module fwd tokens is less than value in percentage relative to calibration, trigger strategy
    threshold: int | float | str = "0.5%" # if less than 0.5% of calibration reaches module (think moe) then we trigger per-module fallback quantization

    # Smoothers can help some low-sample fallback cases, but a static default can
    # hurt whole-model RTN quality. Leave smoothing opt-in.
    smooth: Optional[SmoothMethod] = None


@dataclass
class WeightOnlyConfig:
    """Configuration for weight-only fallback quantization flows."""

    method: WeightOnlyMethod = WeightOnlyMethod.RTN
    # Whole-model RTN is noticeably more stable without a smoother by default.
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
        """Normalize the weight-only method and optional smoother settings."""

        if isinstance(self.method, str):
            try:
                self.method = WeightOnlyMethod(self.method.lower())
            except ValueError as exc:
                raise ValueError(
                    f"WeightOnlyConfig: `method` must be one of {[v.value for v in WeightOnlyMethod]}."
                ) from exc
        elif not isinstance(self.method, WeightOnlyMethod):
            raise ValueError(
                f"WeightOnlyConfig: `method` must be one of {[v.value for v in WeightOnlyMethod]}."
            )

        self.smooth = _parse_smooth_method(self.smooth)


@dataclass
class BasePreProcessorConfig:
    """Base payload for preprocessing stages emitted into config JSON."""

    code: ClassVar[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the preprocessor config into a minimal dictionary."""

        return {"code": self.code}


@dataclass
class SmootherConfig(BasePreProcessorConfig):
    """Serialized wrapper for a configured smoothing preprocessor."""

    code: ClassVar[str] = PreProcessorCode.SMOOTHER.value
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
        """Normalize the smoother payload into a typed smoother instance."""

        self.smooth = _parse_smooth_method(self.smooth)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the smoother config, including the smoother payload."""

        payload = super().to_dict()
        payload["smooth"] = _serialize_smooth_method(self.smooth)
        return payload


@dataclass
class AutoModuleDecoderConfig(BasePreProcessorConfig):
    """Configure automatic module-local decode behavior for checkpoint dtypes such as FP8."""

    code: ClassVar[str] = PreProcessorCode.AUTO_MODULE_DECODER.value
    source_dtype: str = "auto"
    target_dtype: Union[str, torch.dtype] = torch.bfloat16

    def __post_init__(self):
        """Normalize the decoder payload into canonical string and dtype values."""

        source_dtype = str(self.source_dtype).strip().lower()
        if source_dtype != "auto":
            raise ValueError(
                f"AutoModuleDecoderConfig: unsupported `source_dtype` `{self.source_dtype}`."
            )
        self.source_dtype = source_dtype

        target_dtype = self.target_dtype
        if isinstance(target_dtype, torch.dtype):
            normalized_dtype = target_dtype
        else:
            normalized_dtype = _DECODER_TARGET_DTYPE_MAP.get(str(target_dtype).strip().lower())
        if normalized_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                "AutoModuleDecoderConfig: `target_dtype` must be `torch.float16` or `torch.bfloat16`."
            )
        self.target_dtype = normalized_dtype

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decoder config with a stable dtype string payload."""

        payload = super().to_dict()
        payload["source_dtype"] = self.source_dtype
        payload["target_dtype"] = str(self.target_dtype).split(".")[-1]
        return payload


@dataclass
class TensorParallelPadderConfig(BasePreProcessorConfig):
    """Configure tensor-parallel-safe column padding derived from module weight shapes."""

    code: ClassVar[str] = PreProcessorCode.TENSOR_PARALLEL_PADDER.value


@dataclass
class HessianConfig:
    """Controls for chunked Hessian accumulation during GPTQ calibration."""

    # Hessian accumulation controls (GPTQ only)
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Maximum rows per Hessian chunk"})
    chunk_bytes: Optional[int] = field(default=None, metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"})
    staging_dtype: Union[str, torch.dtype] = field(
        default=torch.float32,
        metadata={"help": "Stage Hessian chunks in a lower precision dtype when supported"},
    )

    def __post_init__(self):
        """Validate Hessian chunking and staging dtype settings."""

        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int):
                raise ValueError("HessianConfig: `chunk_size` must be an integer or None.")
            if self.chunk_size <= 0:
                raise ValueError("HessianConfig: `chunk_size` must be a positive integer.")

        if self.chunk_bytes is not None:
            if not isinstance(self.chunk_bytes, int):
                raise ValueError("HessianConfig: `chunk_bytes` must be an integer or None.")
            if self.chunk_bytes <= 0:
                raise ValueError("HessianConfig: `chunk_bytes` must be a positive integer amount of bytes.")

        if isinstance(self.staging_dtype, str):
            self.staging_dtype = self.staging_dtype.lower()
            if self.staging_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
            self.staging_dtype = getattr(torch, self.staging_dtype)
        elif isinstance(self.staging_dtype, torch.dtype):
            if self.staging_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
        else:
            raise ValueError("HessianConfig: `staging_dtype` must be a torch.dtype or string.")


@dataclass
class GPTAQConfig:
    alpha: float = field(default=0.25)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("GPTAQConfig: `alpha` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("GPTAQConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("GPTAQConfig: `device` must be a string or torch.device.")


@dataclass
class FOEMConfig:
    r"""Configuration parameters for the FOEM calibration process, including `alpha` and `beta`.

    The parameter `alpha` follows the same definition and role as in GPTAQ.
    Note: although GPTAQ does not explicitly mention this coefficient in the paper,
    its official implementation applies it to the rightmost term of Eq.18.

    The parameter `beta` is introduced by FOEM. Please refer to the paper for details:
    https://ojs.aaai.org/index.php/AAAI/article/view/40123.

    Special cases:
        - alpha = 0, beta = 0:
            Equivalent to GPTQ.
        - alpha > 0, beta = 0:
            Equivalent to GPTAQ. The recommended value for `alpha` is 0.25.
        - alpha = 0, beta > 0:
            Equivalent to FOEM. Empirically, setting `beta` in the range [0.1, 0.25] yields good performance.
        - alpha > 0, beta > 0:
            Equivalent to FOEM + GPTAQ. Using the default best settings
            (alpha = 0.25, beta = 0.2) generally produces strong results,
            although it is not consistently superior to using FOEM alone.

    Args:
        alpha (float, optional): Default is 0.
        beta (float, optional): Default is 0.2.
    """
    alpha: float = field(default=0)
    beta: float = field(default=0.2)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("FOEMConfig: `alpha` must be a numeric value.")
        if not isinstance(self.beta, (int, float)):
            raise ValueError("FOEMConfig: `beta` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("FOEMConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("FOEMConfig: `device` must be a string or torch.device.")


@dataclass
class BaseMoERouting:
    pass


MOE_ALL_EXPERTS = "all"


@dataclass
class ExpertsRoutingOverride(BaseMoERouting):
    num_experts_per_tok: Union[int, str] = MOE_ALL_EXPERTS

    def __post_init__(self):
        # Handle string values
        if isinstance(self.num_experts_per_tok, str):
            raw = self.num_experts_per_tok.strip()

            # Numeric string -> int (must be > 0)
            if raw.isdigit():
                value = int(raw)
                if value <= 0:
                    raise ValueError(
                        f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                        f"got '{self.num_experts_per_tok}'"
                    )
                self.num_experts_per_tok = value
                return

            # Normalize keyword string
            value = raw.lower()
            if value != MOE_ALL_EXPERTS:
                raise ValueError(
                    f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                    f"got '{self.num_experts_per_tok}'"
                )

            self.num_experts_per_tok = value
            return

        # Validate integer values
        if not isinstance(self.num_experts_per_tok, int) or self.num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                f"got {self.num_experts_per_tok}"
            )


# MoE quantization: forward whole calibration dataset to each expert instead of only routed data
# This ensures all experts receive sufficient calibration samples but increases quantization time
@dataclass
class ExpertsRoutingBypass(BaseMoERouting):
    # Number of modules to process in a single batch to reduce VRAM pressure during quantization
    # For example, with batch_size=10 and 20 expert modules (gate_proj + up_proj for 10 experts):
    # - First batch processes 10 modules (could be gate_proj for experts 0-9, or a mix depending on sorting)
    # - Second batch processes remaining 10 modules
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of modules to process in a single batch during MoE quantization"}
    )


@dataclass
class MoEConfig:
    routing: BaseMoERouting

    def __post_init__(self):
        if not isinstance(self.routing, BaseMoERouting):
            raise ValueError(
                f"routing must be an instance of BaseMoERouting, "
                f"got {type(self.routing).__name__}"
            )

    def routing_bypass(self) -> bool:
        return isinstance(self.routing, ExpertsRoutingBypass)

    def routing_override(self, num_experts: int) -> Union[int, None]:
        """
        Resolve MoE routing top-k override.

        Returns the effective number of experts per token if routing override
        is enabled, otherwise None.

        - "all" resolves to `num_experts`
        - integer value is returned directly
        """
        if isinstance(self.routing, ExpertsRoutingOverride):
            # Resolve "all" to full expert count
            if isinstance(self.routing.num_experts_per_tok, str) and self.routing.num_experts_per_tok.lower().strip() == MOE_ALL_EXPERTS:
                return num_experts

            assert isinstance(self.routing.num_experts_per_tok, int)
            top_k = self.routing.num_experts_per_tok

            # Clamp to valid range and warn user if needed
            if top_k > num_experts:
                log.info(f"MoEConfig: MoE routing override num_experts_per_tok ({top_k}) exceeds "
                    f"num_experts ({num_experts}); clamping to {num_experts}.",)
                top_k = num_experts

            return top_k

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": {
                "class": self.routing.__class__.__name__,
                **asdict(self.routing),
            }
        }


QUANT_METHOD_FORMAT_MAPPING = {
    METHOD.GPTQ: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    },
    METHOD.FP8: {
        FORMAT.FP8,
    },
    METHOD.BITSANDBYTES: {
        FORMAT.BITSANDBYTES,
    },
    METHOD.EXL3: {
        FORMAT.EXL3,
    },
    METHOD.GGUF: {
        FORMAT.GGUF,
    },
    METHOD.QQQ: {
        FORMAT.QQQ,
    },
    METHOD.AWQ: {
        FORMAT.GEMM,
        FORMAT.GEMV,
        FORMAT.GEMV_FAST,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
        FORMAT.LLM_AWQ,
    },
    METHOD.PARO: {
        FORMAT.PAROQUANT,
    },
}

GPTQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GPTQ,
    FORMAT.GPTQ_V2,
    FORMAT.MARLIN,
    FORMAT.BITBLAS,
)
AWQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GEMM,
    FORMAT.GEMV,
    FORMAT.GEMV_FAST,
    FORMAT.MARLIN,
    FORMAT.BITBLAS,
    FORMAT.LLM_AWQ,
)
PAROQUANT_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.PAROQUANT,
)
# Keep ParoQuant channel-scale clamps configurable so users can relax or
# tighten the safeguard without patching the optimizer code.
PAROQUANT_OPT_SCALE_CLAMP_MIN_DEFAULT = 1e-2
PAROQUANT_OPT_SCALE_CLAMP_MAX_DEFAULT = 1e2
QQQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.QQQ,
)
FP8_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.FP8,
)
BITSANDBYTES_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.BITSANDBYTES,
)
EXL3_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.EXL3,
)
RTN_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GPTQ,
    FORMAT.GPTQ_V2,
    FORMAT.GEMM,
    FORMAT.GEMV,
    FORMAT.GEMV_FAST,
    FORMAT.LLM_AWQ,
)
GGUF_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GGUF,
)

_UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT = {
    FORMAT.GPTQ: METHOD.GPTQ,
    FORMAT.GPTQ_V2: METHOD.GPTQ,
    FORMAT.FP8: METHOD.FP8,
    FORMAT.BITSANDBYTES: METHOD.BITSANDBYTES,
    FORMAT.EXL3: METHOD.EXL3,
    FORMAT.GGUF: METHOD.GGUF,
    FORMAT.BITBLAS: METHOD.GPTQ,
    FORMAT.GEMM: METHOD.AWQ,
    FORMAT.GEMV: METHOD.AWQ,
    FORMAT.GEMV_FAST: METHOD.AWQ,
    FORMAT.LLM_AWQ: METHOD.AWQ,
    FORMAT.PAROQUANT: METHOD.PARO,
    FORMAT.QQQ: METHOD.QQQ,
}

# inference only methods should go here
QUANTIZE_BLACK_LIST = {}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": BITS_FIELD_CODE,

    # QQQ compat
    "wbits": BITS_FIELD_CODE,
    "q_group_size": GROUP_SIZE_FIELD_CODE,

    # AWQ compat
    "version" : FORMAT_FIELD_CODE,

    # map deprecated aliases to canonical fields
    FORMAT_FIELD_CHECKPOINT: FORMAT_FIELD_CODE,
    QUANT_METHOD_FIELD: METHOD_FIELD_CODE,
    "bnb_quant_type": FORMAT_FIELD_CODE,
    "bnb_block_size": "block_size",
    "bnb_compress_statistics": "compress_statistics",
}

# compat (values are negated)
QUANT_CONFIG_ARG_SYNONYMS_NEGATED = {
    # AWQ compat
    "zero_point": SYMMETRIC_FIELD_CODE,
}
DYNAMIC_FIELD_SYNONYMS = {}

def dict_scale_dtype_to_str(d: Dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *scale_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("scale_dtype", None) is not None and not isinstance(d["scale_dtype"], str):
        d["scale_dtype"] = str(d["scale_dtype"]).split(".")[1]
    for value in d.values():
        if isinstance(value, dict):
            dict_scale_dtype_to_str(value)


def _build_smooth_method_from_dict(payload: Dict[str, Any]) -> Optional[SmoothMethod]:
    method_type = payload.get("type") or payload.get("name")
    if not method_type:
        return None
    method_type = str(method_type).strip().lower()
    group_size_threshold_raw = payload.get("group_size_threshold", 128)
    group_size_threshold = int(group_size_threshold_raw) if group_size_threshold_raw is not None else 128
    if method_type == "percentile":
        return SmoothPercentile(
            percentile=float(payload.get("percentile", 99.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type in ("percentile_asym", "percentile_asymmetric"):
        return SmoothPercentileAsymmetric(
            low=float(payload.get("low", 0.5)),
            high=float(payload.get("high", 99.5)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mad":
        return SmoothMAD(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mse":
        return SmoothMSE(
            steps=int(payload.get("steps", 32)),
            maxshrink=float(payload.get("maxshrink", 0.8)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "outlier":
        return SmoothOutlier(
            pct=float(payload.get("pct", 1.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "softnorm":
        return SmoothSoftNorm(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "log":
        return SmoothLog(
            percentile=float(payload.get("percentile", 99.0)),
            mu=float(payload.get("mu", 8.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "rowcol":
        return SmoothRowCol(
            axis=str(payload.get("axis", "row")),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "none":
        return None
    raise ValueError(f"QuantizeConfig: Unknown smooth type `{method_type}`.")


def _parse_smooth_method(setting: Any) -> Optional[SmoothMethod]:
    if setting is None:
        return None
    if isinstance(setting, SmoothMethod):
        return setting
    if isinstance(setting, str):
        return _build_smooth_method_from_dict({"type": setting})
    if isinstance(setting, dict):
        return _build_smooth_method_from_dict(setting)
    raise ValueError("QuantizeConfig: `fallback.smooth` must be a SmoothMethod, string, or dict.")


def _serialize_smooth_method(method: Optional[SmoothMethod]) -> Optional[Dict[str, Any]]:
    if method is None:
        return None

    payload = {"type": method.name, "group_size_threshold": method.group_size_threshold}
    if isinstance(method, SmoothPercentile):
        payload["percentile"] = method.percentile
    elif isinstance(method, SmoothPercentileAsymmetric):
        payload["low"] = method.low
        payload["high"] = method.high
    elif isinstance(method, SmoothMAD):
        payload["k"] = method.k
    elif isinstance(method, SmoothMSE):
        payload["steps"] = method.steps
        payload["maxshrink"] = method.maxshrink
    elif isinstance(method, SmoothOutlier):
        payload["pct"] = method.pct
    elif isinstance(method, SmoothSoftNorm):
        payload["k"] = method.k
    elif isinstance(method, SmoothLog):
        payload["percentile"] = method.percentile
        payload["mu"] = method.mu
    elif isinstance(method, SmoothRowCol):
        payload["axis"] = method.axis
    return payload


def _normalize_smoother_config(
    payload: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]]
) -> Optional[SmootherConfig]:
    if payload is None:
        return None
    if isinstance(payload, SmootherConfig):
        return payload
    if isinstance(payload, dict) and "smooth" in payload and "type" not in payload:
        return SmootherConfig(smooth=payload.get("smooth"))
    return SmootherConfig(smooth=payload)


def _normalize_preprocessor_config(payload: Any) -> BasePreProcessorConfig:
    if isinstance(payload, BasePreProcessorConfig):
        return payload
    if isinstance(payload, SmoothMethod):
        return SmootherConfig(smooth=payload)
    if isinstance(payload, str):
        normalized = payload.strip().lower()
        if normalized == PreProcessorCode.SMOOTHER.value:
            return SmootherConfig(smooth=None)
        if normalized == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig()
        if normalized == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        return SmootherConfig(smooth=payload)
    if isinstance(payload, dict):
        code = str(payload.get("code", "")).strip().lower()
        if code == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig(
                source_dtype=payload.get("source_dtype", "auto"),
                target_dtype=payload.get("target_dtype", torch.bfloat16),
            )
        if code == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        if code and code != PreProcessorCode.SMOOTHER.value:
            raise ValueError(f"QuantizeConfig: unsupported preprocessor code `{code}`.")
        if "smooth" in payload:
            return SmootherConfig(smooth=payload.get("smooth"))
        if "type" in payload:
            return SmootherConfig(smooth=payload)
        return SmootherConfig(smooth=None)
    raise ValueError("QuantizeConfig: `preprocessors` entries must be preprocessor configs, smooth configs, dicts, or strings.")


def _normalize_preprocessors(payload: Optional[List[Any]]) -> List[BasePreProcessorConfig]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("QuantizeConfig: `preprocessors` must be a list or None.")
    return [_normalize_preprocessor_config(item) for item in payload]


def _validate_unique_preprocessors(preprocessors: List[BasePreProcessorConfig]) -> None:
    codes_seen = set()
    for preprocessor in preprocessors:
        if preprocessor.code in codes_seen:
            raise ValueError(f"QuantizeConfig: duplicate preprocessor `{preprocessor.code}` is not allowed.")
        codes_seen.add(preprocessor.code)


def dynamic_get(dynamic: Dict[str, Dict[str, Union[int, bool]]], module_name: str, key: str = None,
                default: Union[int, bool] = None, sub_key: str = None) -> Union[Dict, int, bool]:

    if dynamic is None:
        return default

    for pattern, overrides in dynamic.items():
        if pattern.startswith("-:"):
            if pcre.compile(pattern.removeprefix("-:")).match(module_name):
                return False
        elif pcre.compile(pattern.removeprefix("+:")).match(module_name):
            if key is None:
                return overrides
            else:
                # subkey example: Lora override format: `{ "adapter": { "rank": 512 } }`
                if sub_key:
                    sub_value = overrides.get(key, None)
                    if sub_value is None and key in DYNAMIC_FIELD_SYNONYMS:
                        for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
                            if legacy_key in overrides:
                                sub_value = overrides[legacy_key]
                                break
                    if isinstance(sub_value, Dict):
                        return sub_value.get(sub_key, default)
                    else:
                        log.info(f"QuantConfig: Dynamic `sub_key`: `{sub_key}` failed extraction from  `sub_value`: `{sub_value}`")
                else:
                    if key in overrides:
                        return overrides[key]
                    if key in DYNAMIC_FIELD_SYNONYMS:
                        for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
                            if legacy_key in overrides:
                                return overrides[legacy_key]
                    return default
    return default

def _normalize_quant_method(value: Union[str, METHOD]) -> METHOD:
    if isinstance(value, str):
        value = value.lower()
        if value == FORMAT.MARLIN:
            return METHOD.GPTQ
        if value == FORMAT.BITBLAS:
            return METHOD.GPTQ
        if value == FORMAT.FP8:
            return METHOD.FP8
        if value == FORMAT.BITSANDBYTES:
            return METHOD.BITSANDBYTES
        if value == FORMAT.EXL3:
            return METHOD.EXL3
        if value == FORMAT.PAROQUANT:
            return METHOD.PARO
        try:
            return METHOD(value)
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization method: `{value}`.") from exc
    if not isinstance(value, METHOD):
        raise ValueError(f"QuantizeConfig: Unsupported `method`: {value}")
    return value


def _normalize_format(value: Union[str, FORMAT]) -> FORMAT:
    if isinstance(value, str):
        try:
            return FORMAT(value.lower())
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.") from exc
    if not isinstance(value, FORMAT):
        raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.")
    return value


def _normalize_pack_dtype(pack_dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if pack_dtype is None:
        return torch.int32
    if isinstance(pack_dtype, str):
        pack_dtype = pack_dtype.lower()
        if pack_dtype not in ["int64", "int32", "int16", "int8"]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return getattr(torch, pack_dtype)
    if isinstance(pack_dtype, torch.dtype):
        if pack_dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return pack_dtype
    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")


def _normalize_paroquant_best_state_dtype(best_state_dtype: Optional[Union[str, torch.dtype]]) -> str:
    """Canonicalize the ParoQuant best-state snapshot dtype into a serialized string."""
    if best_state_dtype is None:
        return "fp32"
    if isinstance(best_state_dtype, str):
        normalized = best_state_dtype.strip().lower()
        if normalized in {"fp16", "float16"}:
            return "fp16"
        if normalized in {"bf16", "bfloat16"}:
            return "bf16"
        if normalized in {"fp32", "float32"}:
            return "fp32"
    elif isinstance(best_state_dtype, torch.dtype):
        if best_state_dtype == torch.float16:
            return "fp16"
        if best_state_dtype == torch.bfloat16:
            return "bf16"
        if best_state_dtype == torch.float32:
            return "fp32"
    raise ValueError(
        "ParoConfig: `opt_best_state_dtype` must be one of {'fp16', 'bf16', 'fp32'} "
        "or torch.float16/torch.bfloat16/torch.float32."
    )


def _normalize_fallback(fallback: Optional[Union[Fallback, Dict[str, Any], str, int, float]]) -> Optional[Fallback]:
    if fallback is None:
        return None
    if isinstance(fallback, dict):
        strategy = fallback.get("strategy", FallbackStrategy.RTN)
        threshold = fallback.get("threshold", "1.0%")
        smooth = fallback.get("smooth")
        if smooth is None:
            smooth = fallback.get("smooth_method")
        if smooth is None and "clip_method" in fallback:
            smooth = fallback.get("clip_method")
        smooth = _parse_smooth_method(smooth)
        if smooth is None:
            if "smooth_percentile" in fallback:
                smooth = SmoothPercentile(percentile=float(fallback.get("smooth_percentile", 99.0)))
            elif "smooth_mad_k" in fallback:
                smooth = SmoothMAD(k=float(fallback.get("smooth_mad_k", 3.0)))
            elif "smooth_mse_steps" in fallback or "smooth_mse_maxshrink" in fallback:
                smooth = SmoothMSE(
                    steps=int(fallback.get("smooth_mse_steps", 32)),
                    maxshrink=float(fallback.get("smooth_mse_maxshrink", 0.8)),
                )
            elif "smooth_outlier_pct" in fallback:
                smooth = SmoothOutlier(pct=float(fallback.get("smooth_outlier_pct", 1.0)))
            elif "smooth_rms_k" in fallback:
                smooth = SmoothSoftNorm(k=float(fallback.get("smooth_rms_k", 3.0)))
            elif "smooth_log_mu" in fallback:
                smooth = SmoothLog(
                    percentile=float(fallback.get("smooth_percentile", 99.0)),
                    mu=float(fallback.get("smooth_log_mu", 8.0)),
                )
            elif "smooth_axis" in fallback:
                smooth = SmoothRowCol(axis=str(fallback.get("smooth_axis", "row")))
        fallback = Fallback(strategy=strategy, threshold=threshold, smooth=smooth)
    elif isinstance(fallback, (str, int, float)):
        fallback = Fallback(strategy=FallbackStrategy.RTN, threshold=fallback)
    elif not isinstance(fallback, Fallback):
        raise ValueError("QuantizeConfig: `fallback` must be a Fallback config, dict, string, int, float, or None.")

    if isinstance(fallback.strategy, str):
        try:
            fallback.strategy = FallbackStrategy(fallback.strategy.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
            ) from exc
    elif not isinstance(fallback.strategy, FallbackStrategy):
        raise ValueError(
            f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
        )

    fallback.smooth = _parse_smooth_method(fallback.smooth)
    return fallback


def _normalize_weight_only(
    weight_only: Optional[Union[WeightOnlyConfig, Dict[str, Any], str]]
) -> Optional[WeightOnlyConfig]:
    if weight_only is None:
        return None
    if isinstance(weight_only, dict):
        method = weight_only.get("method", WeightOnlyMethod.RTN)
        smooth = weight_only.get("smooth")
        if smooth is None:
            smooth = weight_only.get("smooth_method")
        return WeightOnlyConfig(method=method, smooth=smooth)
    if isinstance(weight_only, str):
        return WeightOnlyConfig(method=weight_only)
    if not isinstance(weight_only, WeightOnlyConfig):
        raise ValueError(
            "QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None."
        )
    return weight_only


def _normalize_hessian(hessian: Optional[Union[HessianConfig, Dict[str, Any]]]) -> HessianConfig:
    if hessian is None:
        return HessianConfig()
    if isinstance(hessian, dict):
        return HessianConfig(**hessian)
    if not isinstance(hessian, HessianConfig):
        raise ValueError("QuantizeConfig: `hessian` must be a HessianConfig, dict, or None.")
    return hessian


def _normalize_gptaq(gptaq: Optional[Union[GPTAQConfig, Dict[str, Any]]]) -> Optional[GPTAQConfig]:
    if gptaq is None:
        return None
    if isinstance(gptaq, dict):
        return GPTAQConfig(**gptaq)
    if not isinstance(gptaq, GPTAQConfig):
        raise ValueError("QuantizeConfig: `gptaq` must be a GPTAQConfig, dict, or None.")
    return gptaq


def _normalize_foem(foem: Optional[Union[FOEMConfig, Dict[str, Any]]]) -> Optional[FOEMConfig]:
    if foem is None:
        return None
    if isinstance(foem, dict):
        return FOEMConfig(**foem)
    if not isinstance(foem, FOEMConfig):
        raise ValueError("QuantizeConfig: `foem` must be a FOEMConfig, dict, or None.")
    return foem


def _normalize_dense_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied dense-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_moe_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied MoE expert-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_strategy_devices(
    value: Optional[List[Union[str, torch.device]]],
    *,
    field_name: str,
) -> Optional[List[str]]:
    """Normalize one user-facing strategy device pool to stable device strings."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"QuantizeConfig: `{field_name}` must be a list of device strings or torch.device values.")
    if not value:
        raise ValueError(f"QuantizeConfig: `{field_name}` must not be empty when provided.")

    # Import lazily to keep config parsing light and avoid depending on looper
    # modules unless the caller actually configures explicit device pools.
    from ..utils.looper_helpers import normalize_device_like

    normalized_devices: List[str] = []
    seen = set()
    for raw_device in value:
        normalized = normalize_device_like(raw_device)
        if normalized is None:
            raise ValueError(f"QuantizeConfig: `{field_name}` contains an unsupported device value: {raw_device!r}.")
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_devices.append(key)
    return normalized_devices


def _normalize_gc_mode(value: Union[str, GcMode]) -> GcMode:
    if isinstance(value, str):
        try:
            return GcMode(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
            ) from exc
    if not isinstance(value, GcMode):
        raise ValueError(
            f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
        )
    return value


def _normalize_moe_config(value: Optional[Union[MoEConfig, Dict[str, Any]]]) -> Optional[MoEConfig]:
    if value is None:
        return None
    if isinstance(value, MoEConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("QuantizeConfig: `moe` must be a MoEConfig, dict, or None.")

    routing = value.get("routing")
    if isinstance(routing, BaseMoERouting):
        return MoEConfig(routing=routing)
    if not isinstance(routing, dict):
        raise ValueError("QuantizeConfig: `moe.routing` must be a BaseMoERouting, dict, or None.")

    routing_class = routing.get("class")
    if routing_class == ExpertsRoutingOverride.__name__:
        routing_obj = ExpertsRoutingOverride(
            num_experts_per_tok=routing.get("num_experts_per_tok", MOE_ALL_EXPERTS)
        )
    elif routing_class == ExpertsRoutingBypass.__name__:
        routing_obj = ExpertsRoutingBypass(batch_size=routing.get("batch_size"))
    else:
        raise ValueError(f"QuantizeConfig: Unknown `moe.routing.class`: `{routing_class}`.")

    return MoEConfig(routing=routing_obj)


def _resolve_dynamic_group_size_error() -> str:
    return "QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`."


def _default_damp_percent(method: METHOD) -> float:
    return 0.005 if method == METHOD.QQQ else 0.05


def _default_damp_auto_increment(method: METHOD) -> float:
    return 0.001 if method == METHOD.QQQ else 0.01


def _peek_weight_only_method(payload: Any) -> Optional[WeightOnlyMethod]:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return payload.method
    if isinstance(payload, str):
        try:
            return WeightOnlyMethod(payload.lower())
        except ValueError:
            return None
    if isinstance(payload, dict):
        method = payload.get("method", WeightOnlyMethod.RTN)
        try:
            return WeightOnlyMethod(str(method).lower())
        except ValueError:
            return None
    return None


def _extract_weight_only_smooth(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return payload.smooth
    if isinstance(payload, dict):
        smooth = payload.get("smooth")
        if smooth is None:
            smooth = payload.get("smooth_method")
        return smooth
    if isinstance(payload, str):
        return None
    raise ValueError("QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None.")


def _extract_weight_only_legacy_gguf_bits(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return getattr(payload, "gguf_qtype", None)
    if isinstance(payload, dict):
        return payload.get("gguf_qtype")
    if isinstance(payload, str):
        return None
    raise ValueError("QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None.")


def _normalize_rtn_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    legacy_gguf_bits = normalized.pop("gguf_qtype", None)
    weight_only = normalized.pop("weight_only", None)
    weight_only_method = _peek_weight_only_method(weight_only)

    # `weight_only.method="gguf"` is a backward-compatible shorthand for the direct GGUF weight-only lifecycle.
    if weight_only_method == WeightOnlyMethod.GGUF and FORMAT_FIELD_CODE not in normalized:
        normalized[FORMAT_FIELD_CODE] = FORMAT.GGUF

    if "smooth" not in normalized:
        normalized["smooth"] = _extract_weight_only_smooth(weight_only)
    if legacy_gguf_bits is None:
        legacy_gguf_bits = _extract_weight_only_legacy_gguf_bits(weight_only)
    if legacy_gguf_bits is not None and BITS_FIELD_CODE not in normalized:
        normalized[BITS_FIELD_CODE] = legacy_gguf_bits
    return normalized


def _normalize_gguf_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    legacy_gguf_bits = normalized.pop("gguf_qtype", None)
    weight_only = normalized.pop("weight_only", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)
    if legacy_gguf_bits is None:
        legacy_gguf_bits = _extract_weight_only_legacy_gguf_bits(weight_only)
    if legacy_gguf_bits is not None and BITS_FIELD_CODE not in normalized:
        normalized[BITS_FIELD_CODE] = legacy_gguf_bits
    normalized[BITS_FIELD_CODE], normalized[FORMAT_FIELD_CODE], _ = _normalize_gguf_config_spec(
        normalized.get(BITS_FIELD_CODE, 4),
        normalized.get(FORMAT_FIELD_CODE),
    )
    return normalized


def _normalize_fp8_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    weight_only = normalized.pop("weight_only", None)
    legacy_fmt = normalized.pop("fmt", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)

    normalized[FORMAT_FIELD_CODE] = _normalize_fp8_fmt(
        normalized.get(FORMAT_FIELD_CODE, legacy_fmt)
    )

    weight_block_size = _normalize_fp8_weight_block_size(normalized.get("weight_block_size"))
    normalized["weight_block_size"] = list(weight_block_size) if weight_block_size is not None else None

    normalized["weight_scale_method"] = _normalize_fp8_weight_scale_method(
        normalized.get("weight_scale_method"),
        weight_block_size=weight_block_size,
    )
    return normalized


def _normalize_bitsandbytes_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    weight_only = normalized.pop("weight_only", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)

    legacy_format = normalized.pop("bnb_quant_type", None)
    legacy_block_size = normalized.pop("bnb_block_size", None)
    legacy_compress_statistics = normalized.pop("bnb_compress_statistics", None)

    normalized[FORMAT_FIELD_CODE] = _normalize_bitsandbytes_format(
        normalized.get(FORMAT_FIELD_CODE, legacy_format),
        bits=int(normalized.get(BITS_FIELD_CODE, 4)),
    )
    normalized["block_size"] = _normalize_bitsandbytes_block_size(
        normalized.get("block_size", legacy_block_size)
    )
    normalized["compress_statistics"] = bool(
        normalized.get("compress_statistics", legacy_compress_statistics if legacy_compress_statistics is not None else True)
    )
    return normalized


def _resolve_export_quant_method(format_value: FORMAT, fallback_method: Optional[METHOD] = None) -> METHOD:
    if format_value == FORMAT.MARLIN:
        if fallback_method is None:
            raise ValueError("QuantizeConfig: FORMAT.MARLIN requires an explicit quantization method family.")
        return fallback_method

    method = _UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT.get(format_value)
    if method is None:
        if fallback_method is not None:
            return fallback_method
        raise ValueError(f"QuantizeConfig: Unable to resolve export method for format `{format_value}`.")
    return method


def _normalize_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    if target_cls is AWQConfig:
        expected_method = METHOD.AWQ
    elif target_cls is FP8Config:
        expected_method = METHOD.FP8
    elif target_cls is BitsAndBytesConfig:
        expected_method = METHOD.BITSANDBYTES
    elif target_cls is EXL3Config:
        expected_method = METHOD.EXL3
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.EXL3:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.EXL3}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.EXL3
    elif target_cls is ParoConfig:
        expected_method = METHOD.PARO
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.PAROQUANT:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.PAROQUANT}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.PAROQUANT
    elif target_cls is GGUFConfig:
        expected_method = METHOD.GGUF
    elif target_cls is QQQConfig:
        expected_method = METHOD.QQQ
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.QQQ:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QQQ}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.QQQ
    else:
        expected_method = METHOD.GPTQ

    method = normalized.get(METHOD_FIELD_CODE)
    normalized_method = None
    if method is not None:
        try:
            normalized_method = _normalize_quant_method(method)
            normalized[METHOD_FIELD_CODE] = normalized_method
        except ValueError:
            normalized_method = None

    if normalized_method is not None and normalized_method != expected_method:
        if target_cls is GGUFConfig and normalized_method == METHOD.GPTQ:
            pass
        else:
            log.warn(
                f"QuantizeConfig: `{METHOD_FIELD_CODE}`=`{normalized_method}` is incompatible with `{target_cls.__name__}`. "
                f"Auto-fix method to `{expected_method}`."
            )
        normalized[METHOD_FIELD_CODE] = expected_method

    return normalized


def _filter_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    target_field_names = {field.name for field in fields(target_cls) if field.init}
    return {key: value for key, value in payload.items() if key in target_field_names}


def _prepare_target_quantize_config_kwargs(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, payload)
    if target_cls is RTNConfig:
        normalized = _normalize_rtn_kwargs(normalized)
    elif target_cls is GGUFConfig:
        normalized = _normalize_gguf_kwargs(normalized)
    elif target_cls is FP8Config:
        normalized = _normalize_fp8_kwargs(normalized)
    elif target_cls is BitsAndBytesConfig:
        normalized = _normalize_bitsandbytes_kwargs(normalized)
    return _filter_quantize_config_payload_for_target_cls(target_cls, normalized)


class QuantizeConfigMeta(type):
    def __instancecheck__(cls, instance):
        if cls is QuantizeConfig:
            return isinstance(instance, BaseQuantizeConfig)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is QuantizeConfig:
            try:
                return issubclass(subclass, BaseQuantizeConfig)
            except TypeError:
                return False
        return super().__subclasscheck__(subclass)

    def __call__(cls, *args, **kwargs):
        kwargs = _normalize_quantize_config_constructor_kwargs(kwargs)
        if cls is QuantizeConfig:
            target_cls = _resolve_quantize_config_class(kwargs)
            target_kwargs = _prepare_target_quantize_config_kwargs(target_cls, kwargs)
            return type.__call__(target_cls, *args, **target_kwargs)
        return super().__call__(*args, **kwargs)


def _normalize_quantize_config_constructor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return kwargs

    normalized = dict(kwargs)
    if FORMAT_FIELD_COMPAT_MARLIN in normalized:
        raise ValueError(
            "QuantizeConfig: `is_marlin_format` has been removed. Use `format=\"marlin\"` only for legacy checkpoint inspection, "
            "or `format=\"gptq\"` for new GPTQ quantization."
        )
    if METHOD_FIELD_CODE not in normalized and QUANT_METHOD_FIELD in normalized:
        normalized[METHOD_FIELD_CODE] = normalized[QUANT_METHOD_FIELD]
    normalized.pop(QUANT_METHOD_FIELD, None)

    if FORMAT_FIELD_CODE not in normalized and FORMAT_FIELD_CHECKPOINT in normalized:
        normalized[FORMAT_FIELD_CODE] = normalized[FORMAT_FIELD_CHECKPOINT]
    normalized.pop(FORMAT_FIELD_CHECKPOINT, None)
    return normalized


@dataclass
class BaseQuantizeConfig(metaclass=QuantizeConfigMeta):
    bits: Union[int, str, GGUFBits] = field(default=4, metadata={"choices": [2, 3, 4, 5, 6, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, str, bool, GGUFBits]]]] = field(default=None)

    # 128 offers a good balance between inference speed, VRAM usage, and quality.
    group_size: int = field(default=128)

    desc_act: Optional[bool] = field(default=None)

    # symmetric quantization toggle (True=symmetric, False=asymmetric).
    sym: bool = field(default=True)

    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    method: METHOD = field(default=METHOD.GPTQ)

    # Serialized/exported checkpoint layout. This is the authoritative post-quantization format.
    format: FORMAT = field(default=FORMAT.GPTQ)

    # properties that do not directly contribute to quantization or inference should be placed in meta
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # packing implementation hint (`original` = legacy CPU pack, `gpu` enables CUDA pack, `cpu` forces block CPU pack).
    pack_impl: str = field(default="cpu")

    adapter: Optional[Union[Dict[str, Any], Lora]] = field(default=None)

    # controls cpu memory saving by offloading layers/modules to disk in the slow quantization process
    offload_to_disk: bool = field(
        default=True,
        metadata={"help": "Offload completed module memory to disk during quantization loop"},
    )
    offload_to_disk_path: str = field(
        default=None,
        metadata={"help": "Offload disk path. Only applicable if Offload to disk is enabled"},
    )

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    # if calibration is insufficient, fallback to a simple quantization strategy
    fallback: Optional[Fallback] = field(default_factory=Fallback)

    # Callback function to filter devices for compute-intensive stages (quantization and forwarding)
    compute_device_filter: Optional[callable] = field(
        default=None,
        metadata={"help": "Callback function to filter devices for compute-intensive stages. Function signature: fn(devices: List) -> List. "
                  "Example to exclude device 0: compute_device_filter=lambda devices: [d for d in devices if d.index != 0]"}
    )

    # Device for storing calibration data during input capture
    calibration_data_device: Optional[Union[str, torch.device]] = field(
        default=None,
        metadata={"help": "Device for storing calibration data. 'balanced' = round-robin across GPUs, or specify device like 'cuda:1'."}
    )

    auto_forward_data_parallel: bool = field(
        default=True,
        metadata={"help": "When multi-gpu is detected, we may data clone modules to each gpu for data parallelism "
        "to speed up quantization forwarding. This causes extra time spent (especially for MoE layers) and vram pressure, "
        "leading in some cases to slower forwarding or vram OOM"}
    )

    # User-facing dense-pool strategy. The dense pool owns the serial path:
    # qkv, z, out_proj, norms, router, shared expert, and dense MLP modules.
    dense_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "Dense pool placement strategy. The dense pool owns qkv, z, out_proj, norms, router, shared expert, and dense MLP modules."},
    )
    # Optional dense-pool device list, relative to CUDA_VISIBLE_DEVICES. In
    # BALANCED mode, model-tree calculation groups stay together, so qkv is not split.
    dense_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for dense modules. In dense BALANCED mode, modules are assigned by calculation groups, so qkv stays co-located."},
    )
    # User-facing expert-pool strategy. Expert families are placed as whole
    # units so gate/up/down for one expert stay on the same device.
    moe_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "MoE expert-pool placement strategy. Expert families stay co-located and can be balanced across this pool."},
    )
    # Optional expert-pool device list, relative to CUDA_VISIBLE_DEVICES.
    moe_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for MoE expert modules. Each expert family (gate/up/down) stays on one device."},
    )

    gc_mode: GcMode = field(
        default=GcMode.INTERVAL,
        metadata={"help": "Garbage collection mode: 'interval' for regular GC or 'on_stage_end' for GC after stage end (after forward pass, quantize, layer finilization)."}
    )

    wait_for_submodule_finalizers: bool = field(
        default=False,
        metadata={"help": "Wait for all layer finalization tasks (packing, offloading to disk, etc) to complete before proceeding to next layer. May reduce vram pressure for some env."}
    )

    moe: Optional[MoEConfig] = field(
        default=None,
        metadata={"help": "Mixture-of-Experts (MoE) configuration for routing strategy and expert batching. "
                  "Requires import: from gptqmodel.quantization.config import MoEConfig, ExpertsRoutingBypass, ExpertsRoutingOverride. "
                  "Example with bypass routing (forward all data to each expert): "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass()) - processes all experts in one batch (default). "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=4)) - processes 4 modules at a time to reduce VRAM pressure. "
                  "Example with routing override (limit experts per token): "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=2)). "
                  "Example to forward to all experts: "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok='all'))"}
    )

    @property
    def quant_method(self) -> METHOD:
        return self.method

    @quant_method.setter
    def quant_method(self, value: Union[str, METHOD]) -> None:
        self.method = value

    @property
    def checkpoint_format(self):
        return self.format

    @checkpoint_format.setter
    def checkpoint_format(self, value) -> None:
        self.format = value

    @property
    def runtime_bits(self):
        return self.bits

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_format(self.format)
        return self.format

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        return _normalize_quant_bits(bits_value, format_value=checkpoint_format)

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        for key, value in layer_dict.items():
            if key == "bits":
                normalized_bits = self._normalize_bits_field(value, checkpoint_format=checkpoint_format)
                layer_dict[key] = normalized_bits
                if quant_bits_width(normalized_bits) not in valid_bit_widths:
                    raise ValueError(
                        f"QuantizeConfig: Layer `{layer_name}` only support quantization of `{valid_bit_widths}` bits."
                    )
            if key == "group_size" and value != -1 and value <= 0:
                raise ValueError(_resolve_dynamic_group_size_error())

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return tuple(METHOD)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `method`: {self.method}")
        return tuple(valid_formats)

    def export_quant_method(self) -> METHOD:
        return _resolve_export_quant_method(resolve_quant_format(self.format, self.method), fallback_method=self.method)

    def default_desc_act(self) -> bool:
        return True

    def __post_init__(self):
        fields_info = fields(self)

        self.method = _normalize_quant_method(self.method)
        format_family = self._resolve_checkpoint_format()
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = self._normalize_bits_field(self.bits, checkpoint_format=format_family)

        allowed_methods = self.allowed_quant_methods()
        if allowed_methods and self.method not in allowed_methods:
            raise ValueError(
                f"{self.__class__.__name__}: `method` must be one of {[v.value for v in allowed_methods]}."
            )

        # TODO FIXME awq compat which didn't have checkpoint_format before merging to gptqmodel
        if self.quant_method == METHOD.AWQ and self.format not in [FORMAT.MARLIN, FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.GEMM, FORMAT.BITBLAS, FORMAT.LLM_AWQ]:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.GEMM}`")
            self.format = FORMAT.GEMM
            format_family = self._resolve_checkpoint_format()

        valid_formats = self.supported_export_formats()
        if format_family not in valid_formats:
            raise ValueError(
                f"{self.__class__.__name__}: unsupported export `format` `{format_family}`."
            )

        self.fallback = _normalize_fallback(self.fallback)

        valid_bit_widths = fields_info[0].metadata["choices"]
        if quant_bits_width(self.bits) not in valid_bit_widths:
            raise ValueError(f"QuantizeConfig: `bits` must be in the set of `{fields_info[0].metadata['choices']}`.")

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }

            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=valid_bit_widths,
                    checkpoint_format=format_family,
                )

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError(_resolve_dynamic_group_size_error())

        if self.desc_act is None:
            self.desc_act = self.default_desc_act()
        elif not isinstance(self.desc_act, bool):
            self.desc_act = bool(self.desc_act)

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key in self.meta:
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        self.adapter = normalize_adapter(self.adapter)

        if self.offload_to_disk and not self.offload_to_disk_path:
            path_key = f"{get_random_string()}-{get_random_string()}"
            self.offload_to_disk_path = f"./gptqmodel_offload/{path_key}/"
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to `{self.offload_to_disk_path}`")

        self.dense_vram_strategy = _normalize_dense_vram_strategy(self.dense_vram_strategy)
        self.dense_vram_strategy_devices = _normalize_strategy_devices(
            self.dense_vram_strategy_devices,
            field_name="dense_vram_strategy_devices",
        )
        self.moe_vram_strategy = _normalize_moe_vram_strategy(self.moe_vram_strategy)
        self.moe_vram_strategy_devices = _normalize_strategy_devices(
            self.moe_vram_strategy_devices,
            field_name="moe_vram_strategy_devices",
        )
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)

        # Normalize calibration_data_device to canonical form if it's a specific device (not "balanced")
        if self.calibration_data_device is not None:
            if isinstance(self.calibration_data_device, str):
                if self.calibration_data_device.lower() == "balanced":
                    self.calibration_data_device = "balanced"
                else:
                    # Import here to avoid circular import
                    from ..utils.looper_helpers import _canonical_device
                    self.calibration_data_device = _canonical_device(torch.device(self.calibration_data_device))
            elif isinstance(self.calibration_data_device, torch.device):
                # Also normalize when passed as torch.device object
                from ..utils.looper_helpers import _canonical_device
                self.calibration_data_device = _canonical_device(self.calibration_data_device)

    def extension_set(self, key: str, value: Any):
        if self.adapter is None:
            self.adapter = {}
        self.adapter[key.lower()] = value

    def extension_get(self, key: str) -> Any:
        return self.adapter.get(key.lower()) if self.adapter else None

    def meta_set(self, key: str, value: Any):
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        return self.meta.get(key)

    def dynamic_get(
        self,
        layer_name: str,
        key: str = None,
        default: Union[int, bool, float] = None,
        sub_key: str = None,
    ) -> Union[Dict, int, bool, float]:
        return dynamic_get(self.dynamic, layer_name, key, default, sub_key)

    def meta_set_versionable(self, key: str, value: List[str]):
        self.meta_set(key, value)

    def meta_get_versionable(self, key: str) -> List[Tuple[str, str]]:
        values = self.meta_get(key)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for val in values:
            parts = val.split(":")
            if len(parts) >= 2:
                result.append((parts[0].lower(), parts[1].lower()))
        return result

    def is_quantized_by_gptaq(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def is_quantized_by_foem(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def extract_adapter_rank_patterns(self) -> Optional[Dict[str, int]]:
        adapter_rank_patterns = {}
        if not self.dynamic or not self.adapter:
            return adapter_rank_patterns

        for k, v in self.dynamic.items():
            adapter_override = v.get("adapter", None)
            if adapter_override and isinstance(adapter_override, Dict):
                rank = adapter_override.get("rank", None)
                if rank and isinstance(rank, int):
                    adapter_rank_patterns[k.lstrip("+:")] = rank

        return adapter_rank_patterns

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            payload = self.to_dict()
            json_str = json.dumps(payload, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    def from_quant_config(cls, quantize_cfg, format: str = None):
        valid_formats = set(FORMAT)
        format_auto_inferred = False
        checkpoint_format_hint = quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) if isinstance(quantize_cfg, dict) else None
        serialized_format = quantize_cfg.get(FORMAT_FIELD_CODE) if isinstance(quantize_cfg, dict) else None
        if format:
            if _looks_like_fp8_fmt(format):
                format = _normalize_fp8_fmt(format)
            elif _looks_like_bitsandbytes_format(format):
                format = _normalize_bitsandbytes_format(format)
            else:
                format = _normalize_format(format)
                if format not in valid_formats:
                    raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if checkpoint_format_hint is not None or serialized_format is not None:
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        elif checkpoint_format_hint is None and serialized_format is None:
            format_auto_inferred = True

        field_names = _known_quantize_config_field_names()

        normalized = {
            METHOD_FIELD_CODE: METHOD.GPTQ,
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }
        format_field_present = format is not None
        legacy_checkpoint_format = None

        for key, val in quantize_cfg.items():
            key = key.lower()

            if key == FORMAT_FIELD_COMPAT_MARLIN:
                raise ValueError(
                    "QuantizeConfig: `is_marlin_format` is no longer supported. Replace it with an explicit `format` field."
                )

            if key == FORMAT_FIELD_CHECKPOINT:
                if _looks_like_fp8_fmt(val):
                    legacy_checkpoint_format = _normalize_fp8_fmt(val)
                elif _looks_like_bitsandbytes_format(val):
                    legacy_checkpoint_format = _normalize_bitsandbytes_format(val)
                else:
                    try:
                        legacy_checkpoint_format = _normalize_gguf_public_format(val)
                    except ValueError:
                        legacy_checkpoint_format = None
                    if legacy_checkpoint_format is None:
                        legacy_checkpoint_format = _normalize_format(val)
                if legacy_checkpoint_format is not None:
                    checkpoint_format_hint = legacy_checkpoint_format
                continue

            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]
            elif key in QUANT_CONFIG_ARG_SYNONYMS_NEGATED and QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key]
                val = not bool(val)

            if key == METHOD_FIELD_CODE:
                if isinstance(val, str) and val.lower() == FORMAT.MARLIN:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.MARLIN
                elif isinstance(val, str) and val.lower() == FORMAT.BITBLAS:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.BITBLAS
                else:
                    normalized[METHOD_FIELD_CODE] = _normalize_quant_method(val)
            elif key == FORMAT_FIELD_CODE:
                format_field_present = True
                serialized_format_hint = None
                try:
                    serialized_format_hint = resolve_quant_format(
                        val,
                        normalized.get(METHOD_FIELD_CODE),
                    )
                except ValueError:
                    serialized_format_hint = None

                format_hint = format or legacy_checkpoint_format or checkpoint_format_hint
                if format_hint is not None:
                    try:
                        format_hint = resolve_quant_format(
                            format_hint,
                            normalized.get(METHOD_FIELD_CODE),
                        )
                    except ValueError:
                        format_hint = None
                if serialized_format_hint in {FORMAT.GGUF, FORMAT.FP8, FORMAT.BITSANDBYTES} or format_hint in {
                    FORMAT.GGUF,
                    FORMAT.FP8,
                    FORMAT.BITSANDBYTES,
                }:
                    normalized[key] = val
                else:
                    normalized[key] = _normalize_format(val)
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if not format_field_present and legacy_checkpoint_format is not None:
            normalized[FORMAT_FIELD_CODE] = legacy_checkpoint_format

        if quantize_cfg.get(AWQ_PACKING_BACKEND_FIELD) == "llm-awq":
            normalized[METHOD_FIELD_CODE] = METHOD.AWQ
            normalized[FORMAT_FIELD_CODE] = FORMAT.LLM_AWQ
            normalized[PACK_DTYPE_FIELD] = torch.int16
            log.info("Detected llm-awq quantization format; FORMAT automatically set to FORMAT.LLM_AWQ.")

        meta_payload = normalized.get(META_FIELD)
        meta_field_map = {
            "fallback": "fallback",
            "hessian": "hessian",
            "gptaq": "gptaq",
            "foem": "foem",
            "weight_only": "weight_only",
            "preprocessors": "preprocessors",
            "gc_mode": "gc_mode",
            "wait_for_submodule_finalizers": "wait_for_submodule_finalizers",
            "auto_forward_data_parallel": "auto_forward_data_parallel",
            "dense_vram_strategy": "dense_vram_strategy",
            "dense_vram_strategy_devices": "dense_vram_strategy_devices",
            "moe_vram_strategy": "moe_vram_strategy",
            "moe_vram_strategy_devices": "moe_vram_strategy_devices",
            "moe": "moe",
            "offload_to_disk": "offload_to_disk",
            "offload_to_disk_path": "offload_to_disk_path",
            "pack_impl": "pack_impl",
            "mse": "mse",
            "mock_quantization": "mock_quantization",
            "act_group_aware": "act_group_aware",
            "true_sequential": "true_sequential",
            "damp_percent": "damp_percent",
            "damp_auto_increment": "damp_auto_increment",
            "opt_rotation_epochs": "opt_rotation_epochs",
            "opt_finetune_epochs": "opt_finetune_epochs",
            "opt_train_samples": "opt_train_samples",
            "opt_validation_samples": "opt_validation_samples",
            "opt_batch_size": "opt_batch_size",
            "opt_rotation_lr": "opt_rotation_lr",
            "opt_weight_lr": "opt_weight_lr",
            "opt_quantizer_lr": "opt_quantizer_lr",
            "opt_pair_ratio": "opt_pair_ratio",
            "opt_seed": "opt_seed",
            "opt_optimizer": "opt_optimizer",
            "opt_weight_decay": "opt_weight_decay",
            "opt_betas": "opt_betas",
            "opt_eps": "opt_eps",
            "opt_amsgrad": "opt_amsgrad",
            "opt_sgd_momentum": "opt_sgd_momentum",
            "opt_sgd_dampening": "opt_sgd_dampening",
            "opt_sgd_nesterov": "opt_sgd_nesterov",
            "opt_fused_rotation": "opt_fused_rotation",
            "opt_gradient_checkpointing": "opt_gradient_checkpointing",
            "opt_stage_cudagraph": "opt_stage_cudagraph",
            "opt_best_state_dtype": "opt_best_state_dtype",
            "opt_train_on_noisy_inputs": "opt_train_on_noisy_inputs",
            "opt_scope": "opt_scope",
            "opt_stage_impl": "opt_stage_impl",
            "opt_pair_impl": "opt_pair_impl",
            "opt_quantizer_impl": "opt_quantizer_impl",
            "opt_channel_scale_clamp_min": "opt_channel_scale_clamp_min",
            "opt_channel_scale_clamp_max": "opt_channel_scale_clamp_max",
        }
        if isinstance(meta_payload, dict):
            for normalized_key, meta_key in meta_field_map.items():
                if normalized_key not in normalized and meta_key in meta_payload:
                    normalized[normalized_key] = meta_payload.get(meta_key)

        target_cls = cls if cls not in {BaseQuantizeConfig, QuantizeConfig} else _resolve_quantize_config_class(normalized)
        normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, normalized)
        if target_cls is RTNConfig:
            normalized = _normalize_rtn_kwargs(normalized)
        elif target_cls is GGUFConfig:
            normalized = _normalize_gguf_kwargs(normalized)
        elif target_cls is FP8Config:
            normalized = _normalize_fp8_kwargs(normalized)
        elif target_cls is BitsAndBytesConfig:
            normalized = _normalize_bitsandbytes_kwargs(normalized)

        if format_auto_inferred:
            log.info(
                f"QuantizeConfig: `{FORMAT_FIELD_CODE}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}"
            )

        resolved_format_family = resolve_quant_format(
            normalized[FORMAT_FIELD_CODE],
            normalized.get(METHOD_FIELD_CODE),
        )
        if resolved_format_family in {FORMAT.BITBLAS, FORMAT.BITSANDBYTES}:
            normalized["desc_act"] = False

        if "sym" not in normalized and target_cls not in {GGUFConfig, FP8Config, BitsAndBytesConfig, EXL3Config}:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )
        return target_cls(**_filter_quantize_config_payload_for_target_cls(target_cls, normalized))

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        format = kwargs.pop("format", None)

        transformers_config = False
        resolved_config_file = None
        for quantize_config_filename in QUANT_CONFIG_FILENAME_COMPAT:
            resolved_config_file = join(save_dir, quantize_config_filename)
            if os.path.exists(resolved_config_file):
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "QuantizeConfig: No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)
            if transformers_config:
                args_from_json = args_from_json["quantization_config"]
            return cls.from_quant_config(args_from_json, format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        return None

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        return None

    def to_dict(self):
        smooth = _serialize_smooth_method(self.fallback.smooth if self.fallback is not None else None)

        meta_payload = dict(self.meta) if self.meta else {}
        if self.moe:
            meta_payload["moe"] = self.moe.to_dict()

        if self.fallback is None:
            meta_payload["fallback"] = None
        else:
            meta_payload["fallback"] = {
                "strategy": (
                    self.fallback.strategy.value
                    if isinstance(self.fallback.strategy, FallbackStrategy)
                    else self.fallback.strategy
                ),
                "threshold": self.fallback.threshold,
                "smooth": smooth,
            }

        meta_payload["offload_to_disk"] = self.offload_to_disk
        meta_payload["offload_to_disk_path"] = self.offload_to_disk_path
        meta_payload["pack_impl"] = self.pack_impl
        meta_payload["gc_mode"] = self.gc_mode.value if isinstance(self.gc_mode, GcMode) else self.gc_mode
        meta_payload["wait_for_submodule_finalizers"] = self.wait_for_submodule_finalizers
        meta_payload["auto_forward_data_parallel"] = self.auto_forward_data_parallel
        meta_payload["dense_vram_strategy"] = (
            self.dense_vram_strategy.value
            if isinstance(self.dense_vram_strategy, VramStrategy)
            else self.dense_vram_strategy
        )
        meta_payload["dense_vram_strategy_devices"] = self.dense_vram_strategy_devices
        meta_payload["moe_vram_strategy"] = (
            self.moe_vram_strategy.value
            if isinstance(self.moe_vram_strategy, VramStrategy)
            else self.moe_vram_strategy
        )
        meta_payload["moe_vram_strategy_devices"] = self.moe_vram_strategy_devices
        self._update_meta_payload(meta_payload)

        out = {
            "bits": serialize_quant_bits(self.bits),
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "lm_head": self.lm_head,
            METHOD_FIELD_CODE: self.method,
            QUANT_METHOD_FIELD: self.method,
            FORMAT_FIELD_CODE: self.format,
            FORMAT_FIELD_CHECKPOINT: self.format,
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: meta_payload,
        }
        self._update_output_payload(out)

        dynamic = out["dynamic"]
        if dynamic:
            for _, v in dynamic.items():
                v.pop("adapter", None)
                if "bits" in v:
                    v["bits"] = serialize_quant_bits(v["bits"])

        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}
        dict_scale_dtype_to_str(out)
        return out

    def calculate_bits_per_weight(self):
        bit_width = quant_bits_width(self.bits)
        if self.group_size != -1:
            per_group_bits = self.group_size * bit_width
            per_group_bits += 16
            per_group_bits += bit_width
            per_group_bits += 4
            bpw = per_group_bits / self.group_size
            bpw += 0.1
        else:
            bpw = bit_width
        log.info(f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, group_size: {self.group_size}]")

    def moe_routing_override(self, num_experts: int) -> Union[int, None]:
        if self.moe is None:
            return None
        return self.moe.routing_override(num_experts)

    def moe_routing_bypass(self) -> bool:
        if self.moe is None:
            return False
        return self.moe.routing_bypass()

    def uses_weight_only_lifecycle(self) -> bool:
        return False

    def requires_calibration_dataset(self) -> bool:
        return not self.uses_weight_only_lifecycle()

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {}


@dataclass
class PreProcessorConfig(BaseQuantizeConfig):
    preprocessors: Optional[List[Union[BasePreProcessorConfig, Dict[str, Any], str]]] = field(default_factory=list)
    smoother: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]] = field(default=None)
    # Backward-compatible alias. New code should use `smoother`.
    smooth: Optional[Union[SmoothMethod, Dict[str, Any], str]] = field(default=None, repr=False)

    def _normalize_preprocessor_state(self) -> None:
        self.preprocessors = _normalize_preprocessors(self.preprocessors)

        smoother_payload = self.smoother if self.smoother is not None else self.smooth
        self.smoother = _normalize_smoother_config(smoother_payload)

        if self.smoother is None:
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, SmootherConfig):
                    self.smoother = preprocessor
                    break

        non_smoother_preprocessors = [
            preprocessor for preprocessor in self.preprocessors if not isinstance(preprocessor, SmootherConfig)
        ]
        if self.smoother is not None:
            non_smoother_preprocessors.append(self.smoother)
        self.preprocessors = non_smoother_preprocessors
        _validate_unique_preprocessors(self.preprocessors)
        self.smooth = self.resolve_smooth_method()

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

    def resolve_smooth_method(self) -> Optional[SmoothMethod]:
        if self.smoother is None:
            return None
        return self.smoother.smooth

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.preprocessors:
            meta_payload["preprocessors"] = [preprocessor.to_dict() for preprocessor in self.preprocessors]


@dataclass
class QuantizeConfig(BaseQuantizeConfig, metaclass=QuantizeConfigMeta):
    """Backward-compatible quantization config factory.

    Direct construction dispatches to a concrete method-specific config class.
    """


@dataclass
class GPTQConfig(PreProcessorConfig):
    damp_percent: Optional[float] = field(default=None)
    damp_auto_increment: Optional[float] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)
    mse: float = field(default=0.0)
    gptaq: Optional[GPTAQConfig] = field(default=None)
    foem: Optional[FOEMConfig] = field(default=None)
    mock_quantization: bool = field(
        default=False,
        metadata={"help": "Skip heavy computations for fast model loading validation"},
    )
    hessian: Optional[HessianConfig] = field(default_factory=HessianConfig)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GPTQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return GPTQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        desc_act_user_value = self.desc_act
        act_group_aware_user_value = self.act_group_aware
        super().__post_init__()

        if self.damp_percent is None:
            self.damp_percent = _default_damp_percent(self.method)
        if self.damp_auto_increment is None:
            self.damp_auto_increment = _default_damp_auto_increment(self.method)
        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")
        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

        self.hessian = _normalize_hessian(self.hessian)
        self.gptaq = _normalize_gptaq(self.gptaq)
        self.foem = _normalize_foem(self.foem)

        if act_group_aware_user_value is None:
            self.act_group_aware = self.method == METHOD.GPTQ
        elif not isinstance(act_group_aware_user_value, bool):
            self.act_group_aware = bool(act_group_aware_user_value)

        self._resolve_activation_ordering(desc_act_user_value, act_group_aware_user_value)
        if self.act_group_aware and self.desc_act:
            raise ValueError("QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False`.")

    def _resolve_activation_ordering(
        self,
        desc_act_user_value: Optional[bool],
        act_group_aware_user_value: Optional[bool],
    ) -> None:
        desc_act_enabled_by_user = bool(desc_act_user_value) if desc_act_user_value is not None else False
        act_group_aware_enabled_by_user = (
            bool(act_group_aware_user_value) if act_group_aware_user_value is not None else False
        )

        if desc_act_enabled_by_user and act_group_aware_user_value is not None and act_group_aware_enabled_by_user:
            raise ValueError(
                "QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False` when both are explicitly set."
            )

        if desc_act_enabled_by_user and act_group_aware_user_value is None and self.act_group_aware:
            log.warn(
                "QuantizeConfig: `desc_act=True` automatically disables `act_group_aware`. "
                "Set `act_group_aware=False` explicitly to silence this warning."
            )
            self.act_group_aware = False

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.gptaq is None:
            meta_payload["gptaq"] = None
        elif self.foem is None:
            device = self.gptaq.device
            meta_payload["gptaq"] = {
                "alpha": self.gptaq.alpha,
                "device": device if isinstance(device, str) else str(device),
            }
        else:
            device = self.foem.device
            meta_payload["foem"] = {
                "alpha": self.foem.alpha,
                "beta": self.foem.beta,
                "device": device if isinstance(device, str) else str(device),
            }

        meta_payload["mse"] = self.mse
        meta_payload["mock_quantization"] = self.mock_quantization
        meta_payload["act_group_aware"] = self.act_group_aware
        meta_payload["hessian"] = {
            "chunk_size": self.hessian.chunk_size,
            "chunk_bytes": self.hessian.chunk_bytes,
            "staging_dtype": str(self.hessian.staging_dtype).split(".")[-1],
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["sym"] = self.sym
        out[FORMAT_FIELD_CODE] = self.format


@dataclass
class AWQConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.AWQ)
    format: FORMAT = field(default=FORMAT.GEMM)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.AWQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return AWQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        # AWQ runtimes do not use GPTQ-style activation reordering unless the
        # checkpoint explicitly asks for it.
        return False

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        if self.format not in self.supported_export_formats():
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.GEMM}`")
            self.format = FORMAT.GEMM
        super().__post_init__()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["zero_point"] = not self.sym
        out["version"] = self.format
        out[FORMAT_FIELD_CODE] = self.format


@dataclass
class ParoConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.PARO)
    format: FORMAT = field(default=FORMAT.PAROQUANT)
    krot: int = field(default=8)
    opt_rotation_epochs: int = field(default=10)
    opt_finetune_epochs: int = field(default=10)
    opt_train_samples: int = field(default=2048)
    opt_validation_samples: int = field(default=64)
    opt_batch_size: int = field(default=64)
    opt_rotation_lr: float = field(default=0.05)
    opt_weight_lr: float = field(default=1e-5)
    opt_quantizer_lr: float = field(default=1e-6)
    opt_pair_ratio: float = field(default=0.5)
    opt_seed: int = field(default=0)
    opt_optimizer: str = field(default="adamw")
    opt_weight_decay: float = field(default=0.01)
    opt_betas: Tuple[float, float] = field(default=(0.9, 0.95))
    opt_eps: float = field(default=1e-10)
    opt_amsgrad: bool = field(default=False)
    opt_sgd_momentum: float = field(default=0.0)
    opt_sgd_dampening: float = field(default=0.0)
    opt_sgd_nesterov: bool = field(default=False)
    opt_fused_rotation: bool = field(default=True)
    opt_gradient_checkpointing: Optional[bool] = field(default=None)
    opt_stage_cudagraph: bool = field(default=True)
    opt_best_state_dtype: Union[str, torch.dtype] = field(default="fp32")
    opt_train_on_noisy_inputs: bool = field(default=False)
    opt_scope: str = field(default="module")
    opt_stage_impl: str = field(default="fast")
    opt_pair_impl: str = field(default="fast")
    opt_quantizer_impl: str = field(default="reference")
    opt_channel_scale_clamp_min: float = field(default=PAROQUANT_OPT_SCALE_CLAMP_MIN_DEFAULT)
    opt_channel_scale_clamp_max: float = field(default=PAROQUANT_OPT_SCALE_CLAMP_MAX_DEFAULT)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.PARO,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return PAROQUANT_EXPORT_FORMATS

    @staticmethod
    def default_opt_gradient_checkpointing_for_scope(opt_scope: str) -> bool:
        """Enable activation checkpointing by default only for whole-layer optimization."""
        return str(opt_scope).strip().lower() == "layer"

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        if self.format != FORMAT.PAROQUANT:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.PAROQUANT}`")
            self.format = FORMAT.PAROQUANT
        super().__post_init__()
        self.krot = int(self.krot)
        if self.krot <= 0:
            raise ValueError("ParoConfig: `krot` must be a positive integer.")
        self.opt_rotation_epochs = int(self.opt_rotation_epochs)
        self.opt_finetune_epochs = int(self.opt_finetune_epochs)
        self.opt_train_samples = int(self.opt_train_samples)
        self.opt_validation_samples = int(self.opt_validation_samples)
        self.opt_batch_size = int(self.opt_batch_size)
        self.opt_rotation_lr = float(self.opt_rotation_lr)
        self.opt_weight_lr = float(self.opt_weight_lr)
        self.opt_quantizer_lr = float(self.opt_quantizer_lr)
        self.opt_pair_ratio = float(self.opt_pair_ratio)
        self.opt_seed = int(self.opt_seed)
        self.opt_optimizer = str(self.opt_optimizer).strip().lower()
        self.opt_weight_decay = float(self.opt_weight_decay)
        if not isinstance(self.opt_betas, (list, tuple)) or len(self.opt_betas) != 2:
            raise ValueError("ParoConfig: `opt_betas` must be a 2-tuple/list of floats.")
        self.opt_betas = (float(self.opt_betas[0]), float(self.opt_betas[1]))
        self.opt_eps = float(self.opt_eps)
        self.opt_amsgrad = bool(self.opt_amsgrad)
        self.opt_sgd_momentum = float(self.opt_sgd_momentum)
        self.opt_sgd_dampening = float(self.opt_sgd_dampening)
        self.opt_sgd_nesterov = bool(self.opt_sgd_nesterov)
        self.opt_fused_rotation = bool(self.opt_fused_rotation)
        self.opt_scope = str(self.opt_scope).strip().lower()
        checkpointing = self.opt_gradient_checkpointing
        if isinstance(checkpointing, str):
            normalized_checkpointing = checkpointing.strip().lower()
            if normalized_checkpointing in {"1", "true", "yes", "on", "y", "t"}:
                checkpointing = True
            elif normalized_checkpointing in {"0", "false", "no", "off", "n", "f"}:
                checkpointing = False
            else:
                raise ValueError(
                    "ParoConfig: `opt_gradient_checkpointing` string values must be one of "
                    "{'1','0','true','false','yes','no','on','off','y','n','t','f'}."
                )
        if checkpointing is None:
            checkpointing = self.default_opt_gradient_checkpointing_for_scope(self.opt_scope)
        self.opt_gradient_checkpointing = bool(checkpointing)
        self.opt_stage_cudagraph = bool(self.opt_stage_cudagraph)
        self.opt_best_state_dtype = _normalize_paroquant_best_state_dtype(self.opt_best_state_dtype)
        self.opt_train_on_noisy_inputs = bool(self.opt_train_on_noisy_inputs)
        self.opt_stage_impl = str(self.opt_stage_impl).strip().lower()
        self.opt_pair_impl = str(self.opt_pair_impl).strip().lower()
        self.opt_quantizer_impl = str(self.opt_quantizer_impl).strip().lower()
        self.opt_channel_scale_clamp_min = float(self.opt_channel_scale_clamp_min)
        self.opt_channel_scale_clamp_max = float(self.opt_channel_scale_clamp_max)
        if self.opt_rotation_epochs < 0 or self.opt_finetune_epochs < 0:
            raise ValueError("ParoConfig: optimization epochs must be non-negative.")
        if self.opt_train_samples <= 0 or self.opt_validation_samples <= 0:
            raise ValueError("ParoConfig: optimization sample counts must be positive.")
        if self.opt_batch_size <= 0:
            raise ValueError("ParoConfig: `opt_batch_size` must be positive.")
        if self.opt_rotation_lr <= 0 or self.opt_weight_lr <= 0 or self.opt_quantizer_lr <= 0:
            raise ValueError("ParoConfig: optimization learning rates must be positive.")
        if not (0.0 < self.opt_pair_ratio <= 0.5):
            raise ValueError("ParoConfig: `opt_pair_ratio` must be in the interval (0, 0.5].")
        if self.opt_optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError("ParoConfig: `opt_optimizer` must be one of {'adamw', 'adam', 'sgd'}.")
        if self.opt_weight_decay < 0:
            raise ValueError("ParoConfig: `opt_weight_decay` must be non-negative.")
        if self.opt_eps <= 0:
            raise ValueError("ParoConfig: `opt_eps` must be positive.")
        if not all(0.0 <= beta < 1.0 for beta in self.opt_betas):
            raise ValueError("ParoConfig: `opt_betas` values must be in the interval [0, 1).")
        if self.opt_sgd_momentum < 0:
            raise ValueError("ParoConfig: `opt_sgd_momentum` must be non-negative.")
        if self.opt_sgd_dampening < 0:
            raise ValueError("ParoConfig: `opt_sgd_dampening` must be non-negative.")
        if self.opt_sgd_nesterov and self.opt_sgd_momentum <= 0:
            raise ValueError("ParoConfig: `opt_sgd_nesterov=True` requires `opt_sgd_momentum > 0`.")
        if self.opt_sgd_nesterov and self.opt_sgd_dampening != 0:
            raise ValueError("ParoConfig: `opt_sgd_nesterov=True` requires `opt_sgd_dampening == 0`.")
        if self.opt_scope not in {"module", "compute_block", "layer"}:
            raise ValueError("ParoConfig: `opt_scope` must be one of {'module', 'compute_block', 'layer'}.")
        if self.opt_stage_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_stage_impl` must be one of {'fast', 'reference'}.")
        if self.opt_pair_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_pair_impl` must be one of {'fast', 'reference'}.")
        if self.opt_quantizer_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_quantizer_impl` must be one of {'fast', 'reference'}.")
        if self.opt_channel_scale_clamp_min <= 0 or self.opt_channel_scale_clamp_max <= 0:
            raise ValueError("ParoConfig: scale clamp bounds must be positive.")
        if self.opt_channel_scale_clamp_min >= self.opt_channel_scale_clamp_max:
            raise ValueError(
                "ParoConfig: `opt_channel_scale_clamp_min` must be smaller than "
                "`opt_channel_scale_clamp_max`."
            )

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "krot": self.krot,
        }

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        meta_payload["opt_rotation_epochs"] = self.opt_rotation_epochs
        meta_payload["opt_finetune_epochs"] = self.opt_finetune_epochs
        meta_payload["opt_train_samples"] = self.opt_train_samples
        meta_payload["opt_validation_samples"] = self.opt_validation_samples
        meta_payload["opt_batch_size"] = self.opt_batch_size
        meta_payload["opt_rotation_lr"] = self.opt_rotation_lr
        meta_payload["opt_weight_lr"] = self.opt_weight_lr
        meta_payload["opt_quantizer_lr"] = self.opt_quantizer_lr
        meta_payload["opt_pair_ratio"] = self.opt_pair_ratio
        meta_payload["opt_seed"] = self.opt_seed
        meta_payload["opt_optimizer"] = self.opt_optimizer
        meta_payload["opt_weight_decay"] = self.opt_weight_decay
        meta_payload["opt_betas"] = list(self.opt_betas)
        meta_payload["opt_eps"] = self.opt_eps
        meta_payload["opt_amsgrad"] = self.opt_amsgrad
        meta_payload["opt_sgd_momentum"] = self.opt_sgd_momentum
        meta_payload["opt_sgd_dampening"] = self.opt_sgd_dampening
        meta_payload["opt_sgd_nesterov"] = self.opt_sgd_nesterov
        meta_payload["opt_fused_rotation"] = self.opt_fused_rotation
        meta_payload["opt_gradient_checkpointing"] = self.opt_gradient_checkpointing
        meta_payload["opt_stage_cudagraph"] = self.opt_stage_cudagraph
        meta_payload["opt_best_state_dtype"] = self.opt_best_state_dtype
        meta_payload["opt_train_on_noisy_inputs"] = self.opt_train_on_noisy_inputs
        meta_payload["opt_scope"] = self.opt_scope
        meta_payload["opt_stage_impl"] = self.opt_stage_impl
        meta_payload["opt_pair_impl"] = self.opt_pair_impl
        meta_payload["opt_quantizer_impl"] = self.opt_quantizer_impl
        meta_payload["opt_channel_scale_clamp_min"] = self.opt_channel_scale_clamp_min
        meta_payload["opt_channel_scale_clamp_max"] = self.opt_channel_scale_clamp_max

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["zero_point"] = not self.sym
        out["krot"] = self.krot
        out[FORMAT_FIELD_CODE] = self.format


@dataclass
class QQQConfig(GPTQConfig):
    method: METHOD = field(default=METHOD.QQQ)
    format: FORMAT = field(default=FORMAT.QQQ)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.QQQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return QQQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return True


@dataclass
class FP8Config(PreProcessorConfig):
    bits: int = field(default=8, metadata={"choices": [8]})
    method: METHOD = field(default=METHOD.FP8)
    format: Optional[str] = field(default="float8_e4m3fn")
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    weight_scale_method: str = field(default="row")
    weight_block_size: Optional[Union[List[int], Tuple[int, int]]] = field(default=None)
    weight_scale_semantics: str = field(default="inverse")

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_fp8_fmt(self.format)
        return FORMAT.FP8

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.FP8,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return FP8_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

        if self.bits != 8:
            raise ValueError("FP8Config: `bits` must be `8`.")

        if self.method != METHOD.FP8:
            raise ValueError("FP8Config: `method` must be `fp8`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.format = _normalize_fp8_fmt(self.format)
        block_size = _normalize_fp8_weight_block_size(self.weight_block_size)
        self.weight_scale_method = _normalize_fp8_weight_scale_method(
            self.weight_scale_method,
            weight_block_size=block_size,
        )
        self.weight_block_size = list(block_size) if block_size is not None else None
        self.weight_scale_semantics = _normalize_fp8_scale_semantics(self.weight_scale_semantics)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[8],
                    checkpoint_format=FORMAT.FP8,
                )

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        del valid_bit_widths, checkpoint_format
        if "bits" in layer_dict and int(layer_dict["bits"]) != 8:
            raise ValueError(f"FP8Config: layer `{layer_name}` only supports 8-bit FP8 weights.")
        if "group_size" in layer_dict and layer_dict["group_size"] not in (-1, None):
            raise ValueError("FP8Config: `group_size` is not used; keep it at `-1`.")

        block_size = _normalize_fp8_weight_block_size(layer_dict.get("weight_block_size"))
        raw_format = layer_dict.get(FORMAT_FIELD_CODE, layer_dict.get("fmt"))
        if raw_format is not None:
            layer_dict[FORMAT_FIELD_CODE] = _normalize_fp8_fmt(raw_format)
        layer_dict.pop("fmt", None)
        if "weight_scale_method" in layer_dict or block_size is not None:
            layer_dict["weight_scale_method"] = _normalize_fp8_weight_scale_method(
                layer_dict.get("weight_scale_method"),
                weight_block_size=block_size,
            )
        if "weight_scale_semantics" in layer_dict:
            layer_dict["weight_scale_semantics"] = _normalize_fp8_scale_semantics(
                layer_dict["weight_scale_semantics"]
            )
        if "weight_block_size" in layer_dict:
            layer_dict["weight_block_size"] = list(block_size) if block_size is not None else None

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "weight_scale_method": self.weight_scale_method,
            "weight_block_size": self.weight_block_size,
            "weight_scale_semantics": self.weight_scale_semantics,
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format
        out["weight_scale_method"] = self.weight_scale_method
        out["weight_block_size"] = self.weight_block_size
        out["weight_scale_semantics"] = self.weight_scale_semantics

    def uses_weight_only_lifecycle(self) -> bool:
        return True

@dataclass
class BitsAndBytesConfig(PreProcessorConfig):
    bits: int = field(default=4, metadata={"choices": [4, 8]})
    method: METHOD = field(default=METHOD.BITSANDBYTES)
    format: Optional[str] = field(default=None)
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    block_size: int = field(default=64)
    compress_statistics: bool = field(default=True)

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_bitsandbytes_format(self.format, bits=int(self.bits))
        return FORMAT.BITSANDBYTES

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.BITSANDBYTES,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return BITSANDBYTES_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

        if self.bits not in {4, 8}:
            raise ValueError("BitsAndBytesConfig: `bits` must be `4` or `8`.")
        if self.method != METHOD.BITSANDBYTES:
            raise ValueError("BitsAndBytesConfig: `method` must be `bitsandbytes`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.format = _normalize_bitsandbytes_format(self.format, bits=int(self.bits))
        self.block_size = _normalize_bitsandbytes_block_size(self.block_size)
        self.compress_statistics = bool(self.compress_statistics)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[4, 8],
                    checkpoint_format=FORMAT.BITSANDBYTES,
                )

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        del valid_bit_widths, checkpoint_format
        if "bits" in layer_dict and int(layer_dict["bits"]) not in {4, 8}:
            raise ValueError(f"BitsAndBytesConfig: layer `{layer_name}` only supports 4-bit or 8-bit weights.")
        if "group_size" in layer_dict and layer_dict["group_size"] not in (-1, None):
            raise ValueError("BitsAndBytesConfig: `group_size` is not used; keep it at `-1`.")
        if "desc_act" in layer_dict and bool(layer_dict["desc_act"]):
            raise ValueError("BitsAndBytesConfig: `desc_act` is not supported.")
        if "sym" in layer_dict and layer_dict["sym"] is not True:
            raise ValueError("BitsAndBytesConfig: `sym` must stay `True`.")
        dynamic_bits = int(layer_dict.get("bits", self.bits))
        raw_format = layer_dict.get(FORMAT_FIELD_CODE, layer_dict.get("bnb_quant_type"))
        if raw_format is not None:
            layer_dict[FORMAT_FIELD_CODE] = _normalize_bitsandbytes_format(raw_format, bits=dynamic_bits)
        if "block_size" in layer_dict or "bnb_block_size" in layer_dict:
            layer_dict["block_size"] = _normalize_bitsandbytes_block_size(
                layer_dict.get("block_size", layer_dict.get("bnb_block_size"))
            )
        layer_dict.pop("bnb_block_size", None)
        if "compress_statistics" in layer_dict or "bnb_compress_statistics" in layer_dict:
            layer_dict["compress_statistics"] = bool(
                layer_dict.get("compress_statistics", layer_dict.get("bnb_compress_statistics"))
            )
        layer_dict.pop("bnb_compress_statistics", None)

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "block_size": self.block_size,
            "compress_statistics": self.compress_statistics,
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format
        out["block_size"] = self.block_size
        out["compress_statistics"] = self.compress_statistics

    def uses_weight_only_lifecycle(self) -> bool:
        return True

    @property
    def bnb_quant_type(self) -> str:
        return self.format

    @bnb_quant_type.setter
    def bnb_quant_type(self, value: str) -> None:
        self.format = _normalize_bitsandbytes_format(value, bits=int(self.bits))

    @property
    def bnb_block_size(self) -> int:
        return self.block_size

    @bnb_block_size.setter
    def bnb_block_size(self, value: int) -> None:
        self.block_size = _normalize_bitsandbytes_block_size(value)

    @property
    def bnb_compress_statistics(self) -> bool:
        return self.compress_statistics

    @bnb_compress_statistics.setter
    def bnb_compress_statistics(self, value: bool) -> None:
        self.compress_statistics = bool(value)

@dataclass
class EXL3Config(BaseQuantizeConfig):
    bits: float = field(default=3.0)
    method: METHOD = field(default=METHOD.EXL3)
    format: FORMAT = field(default=FORMAT.EXL3)
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    head_bits: Optional[float] = field(default=None)
    out_scales: Optional[str] = field(default="auto")
    codebook: str = field(default="mcg")
    tensor_storage: Optional[Dict[str, Any]] = field(default=None)
    calibration: Optional[Dict[str, int]] = field(default=None)

    @property
    def runtime_bits(self) -> int:
        return quant_bits_width(self.bits)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.EXL3,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return EXL3_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        return _normalize_exl3_bits(bits_value)

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        del valid_bit_widths, checkpoint_format
        for key, value in layer_dict.items():
            if key == "bits":
                layer_dict[key] = _normalize_exl3_bits(value)
            elif key == "head_bits":
                layer_dict[key] = None if value is None else _normalize_exl3_bits(value)
            elif key == "group_size" and value not in (-1, None):
                raise ValueError("EXL3Config: `group_size` is not used; keep it at `-1`.")

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = _normalize_exl3_bits(self.bits)
        self.head_bits = None if self.head_bits is None else _normalize_exl3_bits(self.head_bits)

        if self.method != METHOD.EXL3:
            raise ValueError("EXL3Config: `method` must be `exl3`.")
        if self.format != FORMAT.EXL3:
            raise ValueError("EXL3Config: `format` must be `exl3`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.fallback = _normalize_fallback(self.fallback)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[],
                    checkpoint_format=FORMAT.EXL3,
                )

        if self.out_scales is not None:
            normalized_out_scales = str(self.out_scales).strip().lower()
            out_scale_aliases = {
                "always": "always",
                "true": "always",
                "never": "never",
                "false": "never",
                "auto": "auto",
                "none": "auto",
            }
            if normalized_out_scales not in out_scale_aliases:
                raise ValueError("EXL3Config: `out_scales` must be one of `always`, `never`, or `auto`.")
            self.out_scales = out_scale_aliases[normalized_out_scales]

        self.codebook = str(self.codebook).strip().lower()
        if self.codebook not in {"mcg", "mul1", "3inst"}:
            raise ValueError("EXL3Config: `codebook` must be one of `mcg`, `mul1`, or `3inst`.")

        if self.tensor_storage is not None and not isinstance(self.tensor_storage, dict):
            raise ValueError("EXL3Config: `tensor_storage` must be a dictionary when provided.")
        if self.calibration is not None:
            if not isinstance(self.calibration, dict):
                raise ValueError("EXL3Config: `calibration` must be a dictionary when provided.")
            self.calibration = {
                str(key): int(value)
                for key, value in self.calibration.items()
            }

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key in self.meta:
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        self.adapter = normalize_adapter(self.adapter)

        if self.offload_to_disk and not self.offload_to_disk_path:
            path_key = f"{get_random_string()}-{get_random_string()}"
            self.offload_to_disk_path = f"./gptqmodel_offload/{path_key}/"
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to `{self.offload_to_disk_path}`")

        self.dense_vram_strategy = _normalize_dense_vram_strategy(self.dense_vram_strategy)
        self.dense_vram_strategy_devices = _normalize_strategy_devices(
            self.dense_vram_strategy_devices,
            field_name="dense_vram_strategy_devices",
        )
        self.moe_vram_strategy = _normalize_moe_vram_strategy(self.moe_vram_strategy)
        self.moe_vram_strategy_devices = _normalize_strategy_devices(
            self.moe_vram_strategy_devices,
            field_name="moe_vram_strategy_devices",
        )
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["bits"] = float(self.bits)
        out["head_bits"] = None if self.head_bits is None else float(self.head_bits)
        out["out_scales"] = self.out_scales
        out["codebook"] = self.codebook
        out["tensor_storage"] = self.tensor_storage
        out["calibration"] = self.calibration

    def calculate_bits_per_weight(self):
        head_bits = self.head_bits if self.head_bits is not None else self.bits
        log.info(
            "Estimated Quantization BPW (bits per weight): %s bpw, based on [bits: %s, head_bits: %s]",
            self.bits,
            self.bits,
            head_bits,
        )

@dataclass
class RTNConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.GPTQ)
    format: FORMAT = field(default=FORMAT.GPTQ)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GPTQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return RTN_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        super().__post_init__()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["sym"] = self.sym
        out[FORMAT_FIELD_CODE] = self.format

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)
        meta_payload["weight_only"] = {
            "smooth": _serialize_smooth_method(self.smooth),
        }

    def uses_weight_only_lifecycle(self) -> bool:
        return True


@dataclass
class GGUFConfig(PreProcessorConfig):
    bits: Union[int, str, GGUFBits] = field(default=4, metadata={"choices": [1, 2, 3, 4, 5, 6, 8]})
    format: Optional[str] = field(default=None)
    method: METHOD = field(default=METHOD.GGUF, init=False)
    group_size: int = field(default=-1, init=False, repr=False)
    desc_act: Optional[bool] = field(default=False, init=False, repr=False)
    sym: bool = field(default=True, init=False, repr=False)
    _gguf_bits: GGUFBits = field(init=False, repr=False, compare=False)

    @property
    def runtime_bits(self) -> GGUFBits:
        return self._gguf_bits

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GGUF,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return (FORMAT.GGUF,)

    def default_desc_act(self) -> bool:
        return False

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.bits, self.format, self._gguf_bits = _normalize_gguf_config_spec(self.bits, self.format)
        return FORMAT.GGUF

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        normalized = _normalize_quant_bits(bits_value, format_value=FORMAT.GGUF)
        return normalized.bits if isinstance(normalized, GGUFBits) else normalized

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        bits_override_present = "bits" in layer_dict
        format_override_present = FORMAT_FIELD_CODE in layer_dict

        if bits_override_present or format_override_present:
            raw_bits = layer_dict.get("bits", self.bits)
            raw_format = layer_dict.get(FORMAT_FIELD_CODE, self.format)
            normalized_bits, normalized_format, normalized_runtime_bits = _normalize_gguf_config_spec(raw_bits, raw_format)

            layer_dict["bits"] = normalized_bits

            bits_implied_format = (
                isinstance(raw_bits, GGUFBits)
                or (isinstance(raw_bits, str) and not raw_bits.strip().isdigit())
            )
            if format_override_present or bits_implied_format:
                layer_dict[FORMAT_FIELD_CODE] = normalized_format

            if quant_bits_width(normalized_runtime_bits) not in valid_bit_widths:
                raise ValueError(
                    f"QuantizeConfig: Layer `{layer_name}` only support quantization of `{valid_bit_widths}` bits."
                )

        if "group_size" in layer_dict and layer_dict["group_size"] != -1 and layer_dict["group_size"] <= 0:
            raise ValueError(_resolve_dynamic_group_size_error())

    def __post_init__(self):
        self._normalize_preprocessor_state()
        # GGUFConfig already normalized preprocessors above; skip the parent hook to
        # avoid running that normalization twice.
        BaseQuantizeConfig.__post_init__(self)
        self._gguf_bits = _gguf_bits_from_components(self.bits, self.format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format

    def to_dict(self):
        out = super().to_dict()
        out.pop(GROUP_SIZE_FIELD_CODE, None)
        out.pop("desc_act", None)
        out.pop(PACK_DTYPE_FIELD, None)

        meta_payload = out.get(META_FIELD)
        if isinstance(meta_payload, dict):
            for key in (
                "fallback",
                "offload_to_disk",
                "offload_to_disk_path",
                "pack_impl",
                "gc_mode",
                "wait_for_submodule_finalizers",
                "auto_forward_data_parallel",
                "dense_vram_strategy",
                "dense_vram_strategy_devices",
                "moe_vram_strategy",
                "moe_vram_strategy_devices",
                "weight_only",
            ):
                meta_payload.pop(key, None)
            if not meta_payload:
                out.pop(META_FIELD, None)

        return out

    def calculate_bits_per_weight(self):
        bits_name = self.runtime_bits.to_string()
        bpw = _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS.get(bits_name, float(quant_bits_width(self.runtime_bits)))
        log.info(
            f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, format: {self.format}]"
        )

    def uses_weight_only_lifecycle(self) -> bool:
        return True

def clone_weight_only_config_for_module(
    qcfg: Union[RTNConfig, GGUFConfig, FP8Config, BitsAndBytesConfig],
    module_full_name: str,
) -> Optional[Union[RTNConfig, GGUFConfig, FP8Config, BitsAndBytesConfig]]:
    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        smooth_override = qcfg.dynamic_get(module_full_name, "smoother", None)
        if smooth_override is None:
            smooth_override = qcfg.dynamic_get(module_full_name, "smooth", None)
        if smooth_override is not None:
            qcfg_clone.smoother = _normalize_smoother_config(smooth_override)
            qcfg_clone.smooth = qcfg_clone.resolve_smooth_method()

        if isinstance(qcfg_clone, GGUFConfig):
            dynamic_bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
            dynamic_format = qcfg.dynamic_get(module_full_name, FORMAT_FIELD_CODE, qcfg_clone.format)
            qcfg_clone.bits, qcfg_clone.format, qcfg_clone._gguf_bits = _normalize_gguf_config_spec(
                dynamic_bits,
                dynamic_format,
            )
        elif isinstance(qcfg_clone, FP8Config):
            dynamic_format = qcfg.dynamic_get(module_full_name, FORMAT_FIELD_CODE, None)
            if dynamic_format is None:
                dynamic_format = qcfg.dynamic_get(module_full_name, "fmt", qcfg_clone.format)
            dynamic_block_size = qcfg.dynamic_get(
                module_full_name,
                "weight_block_size",
                qcfg_clone.weight_block_size,
            )
            block_size = _normalize_fp8_weight_block_size(dynamic_block_size)
            qcfg_clone.format = _normalize_fp8_fmt(dynamic_format)
            qcfg_clone.weight_scale_method = _normalize_fp8_weight_scale_method(
                qcfg.dynamic_get(
                    module_full_name,
                    "weight_scale_method",
                    qcfg_clone.weight_scale_method,
                ),
                weight_block_size=block_size,
            )
            qcfg_clone.weight_block_size = list(block_size) if block_size is not None else None
            qcfg_clone.weight_scale_semantics = _normalize_fp8_scale_semantics(
                qcfg.dynamic_get(
                    module_full_name,
                    "weight_scale_semantics",
                    qcfg_clone.weight_scale_semantics,
                )
            )
        elif isinstance(qcfg_clone, BitsAndBytesConfig):
            qcfg_clone.bits = _normalize_quant_bits(
                qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits),
                format_value=FORMAT.BITSANDBYTES,
            )
            qcfg_clone.format = _normalize_bitsandbytes_format(
                qcfg.dynamic_get(
                    module_full_name,
                    FORMAT_FIELD_CODE,
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_quant_type",
                        qcfg_clone.format,
                    ),
                ),
                bits=int(qcfg_clone.bits),
            )
            qcfg_clone.block_size = _normalize_bitsandbytes_block_size(
                qcfg.dynamic_get(
                    module_full_name,
                    "block_size",
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_block_size",
                        qcfg_clone.block_size,
                    ),
                )
            )
            qcfg_clone.compress_statistics = bool(
                qcfg.dynamic_get(
                    module_full_name,
                    "compress_statistics",
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_compress_statistics",
                        qcfg_clone.compress_statistics,
                    ),
                )
            )
        else:
            qcfg_clone.bits = _normalize_quant_bits(
                qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits),
                format_value=resolve_quant_format(qcfg_clone.format, qcfg_clone.method),
            )

        if isinstance(qcfg_clone, RTNConfig):
            qcfg_clone.sym = qcfg.dynamic_get(module_full_name, "sym", qcfg_clone.sym)
            qcfg_clone.group_size = qcfg.dynamic_get(module_full_name, "group_size", qcfg_clone.group_size)

            desc_act_override = qcfg.dynamic_get(module_full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override

    return qcfg_clone


clone_rtn_config_for_module = clone_weight_only_config_for_module


def _resolve_quantize_config_class(payload: Dict[str, Any]) -> type[BaseQuantizeConfig]:
    method = payload.get(METHOD_FIELD_CODE, payload.get(QUANT_METHOD_FIELD, METHOD.GPTQ))
    raw_format_value = payload.get(FORMAT_FIELD_CODE, payload.get(FORMAT_FIELD_CHECKPOINT, FORMAT.GPTQ))
    weight_only = payload.get("weight_only")
    bits = payload.get(BITS_FIELD_CODE)
    gguf_public_format = payload.get(FORMAT_FIELD_CODE)

    try:
        method = _normalize_quant_method(method)
    except Exception:
        method = METHOD.GPTQ

    if _looks_like_fp8_fmt(raw_format_value):
        format_value = FORMAT.FP8
    else:
        try:
            format_value = _normalize_format(raw_format_value)
        except Exception:
            try:
                gguf_public_format = _normalize_gguf_public_format(raw_format_value)
            except ValueError:
                gguf_public_format = payload.get(FORMAT_FIELD_CODE)
            format_value = FORMAT.GPTQ

    gguf_format_detected = False
    if gguf_public_format is not None:
        try:
            gguf_format_detected = _normalize_gguf_public_format(gguf_public_format) is not None
        except ValueError:
            gguf_format_detected = False

    weight_only_method = _peek_weight_only_method(weight_only)
    fp8_storage_fmt = payload.get(FORMAT_FIELD_CODE, payload.get("fmt"))
    if weight_only is not None and weight_only_method not in {
        None,
        WeightOnlyMethod.RTN,
        WeightOnlyMethod.GGUF,
        WeightOnlyMethod.FP8,
        WeightOnlyMethod.BITSANDBYTES,
    }:
        raise ValueError(
            "QuantizeConfig: unsupported weight-only config. Weight-only export currently supports "
            "`rtn`, `gguf`, `fp8`, and `bitsandbytes`."
        )
    if (
        format_value == FORMAT.GGUF
        or weight_only_method == WeightOnlyMethod.GGUF
        or _looks_like_gguf_bits(bits)
        or gguf_format_detected
    ):
        return GGUFConfig
    if weight_only_method == WeightOnlyMethod.FP8:
        return FP8Config
    if weight_only_method == WeightOnlyMethod.BITSANDBYTES:
        return BitsAndBytesConfig
    if weight_only_method == WeightOnlyMethod.RTN:
        return RTNConfig
    if weight_only is not None:
        return RTNConfig
    if method == METHOD.FP8 or format_value == FORMAT.FP8 or _looks_like_fp8_fmt(fp8_storage_fmt):
        return FP8Config
    if method == METHOD.BITSANDBYTES or format_value == FORMAT.BITSANDBYTES or _looks_like_bitsandbytes_format(raw_format_value):
        return BitsAndBytesConfig
    if method == METHOD.EXL3 or format_value == FORMAT.EXL3:
        return EXL3Config
    if method == METHOD.PARO or format_value == FORMAT.PAROQUANT:
        return ParoConfig
    if method == METHOD.QQQ or format_value == FORMAT.QQQ:
        return QQQConfig
    if method == METHOD.AWQ:
        return AWQConfig
    if format_value in {FORMAT.GEMM, FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ}:
        return AWQConfig
    if format_value == FORMAT.MARLIN:
        return AWQConfig if method == METHOD.AWQ else GPTQConfig
    return GPTQConfig


def _known_quantize_config_field_names() -> set[str]:
    field_names: set[str] = set()
    for cls in (
        BaseQuantizeConfig,
        PreProcessorConfig,
        QuantizeConfig,
        GPTQConfig,
        AWQConfig,
        ParoConfig,
        QQQConfig,
        FP8Config,
        BitsAndBytesConfig,
        EXL3Config,
        RTNConfig,
        GGUFConfig,
    ):
        field_names.update(field.name for field in fields(cls))
    return field_names
