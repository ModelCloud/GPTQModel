# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import os.path
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import total_ordering
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import pcre as re
import torch
from packaging import version

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger
from ..utils.random_str import get_random_string


log = setup_logger()

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "format"
SYMMETRIC_FIELD_CODE = "sym"
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]
# This is AwqBackendPackingMethod, not GPTQModel.BACKEND.
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

ADAPTER_FIELD = "adapter"

# saved formats
class FORMAT(str, Enum):
    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"
    GGUF = "gguf"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    QQQ = "qqq"

    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    LLM_AWQ = "llm-awq"


# quant methods
class METHOD(str, Enum):
    GPTQ = "gptq"
    GGUF = "gguf"
    QQQ = "qqq"
    AWQ = "awq"


class VramStrategy(str, Enum):
    EXCLUSIVE = "exclusive"
    BALANCED = "balanced"


class FailSafeStrategy(str, Enum):
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
    RTN = "rtn"
    GGUF = "gguf"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


class PreFilterCode(str, Enum):
    SMOOTHER = "smoother"


_GGUF_BITS_ALIAS_INFO = {
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
    4: "q4_0",
    5: "q5_k_m",
    6: "q6_k",
    8: "q8_0",
}
_GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS = {
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
    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> "BaseComplexBits":
        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @property
    def width(self) -> int:
        return self.bits

    @property
    def name(self) -> str:
        return self.to_string()

    def _coerce_bits(self, other: Any) -> Any:
        if isinstance(other, BaseComplexBits):
            return other.bits
        if isinstance(other, int):
            return other
        if isinstance(other, str) and other.strip().isdigit():
            return int(other.strip())
        return NotImplemented

    def __str__(self) -> str:
        return self.to_string()

    def __hash__(self) -> int:
        return hash(self.bits)

    def __int__(self) -> int:
        return self.bits

    def __index__(self) -> int:
        return self.bits

    def __float__(self) -> float:
        return float(self.bits)

    def __eq__(self, other: Any) -> bool:
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
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits < coerced

    def __add__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits + coerced

    def __radd__(self, other: Any) -> int:
        return self.__add__(other)

    def __sub__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits - coerced

    def __rsub__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced - self.bits

    def __mul__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits * coerced

    def __rmul__(self, other: Any) -> int:
        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits // coerced

    def __rfloordiv__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced // self.bits

    def __truediv__(self, other: Any) -> float:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits / coerced

    def __rtruediv__(self, other: Any) -> float:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced / self.bits

    def __mod__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits % coerced

    def __rmod__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced % self.bits

    def __pow__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits ** coerced

    def __rpow__(self, other: Any) -> int:
        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced ** self.bits


@dataclass(frozen=True, eq=False)
class GGUFBits(BaseComplexBits):
    bits: int
    version: str
    variant: str
    quality: Optional[str] = None

    def __post_init__(self):
        if self.bits <= 0:
            raise ValueError("GGUFBits: `bits` must be a positive integer.")
        if self.version not in {"q", "iq"}:
            raise ValueError("GGUFBits: `version` must be `q` or `iq`.")
        if self.variant not in {"0", "k"}:
            raise ValueError("GGUFBits: `variant` must be `0` or `k`.")
        if self.quality not in {None, "xs", "s", "m", "l"}:
            raise ValueError("GGUFBits: `quality` must be one of `[None, xs, s, m, l]`.")

    @classmethod
    def from_string(cls, value: str) -> "GGUFBits":
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
        alias = f"{self.version}{self.bits}_{self.variant}"
        if self.quality is not None:
            alias = f"{alias}_{self.quality}"
        return alias

    @classmethod
    def from_alias(cls, value: str) -> "GGUFBits":
        return cls.from_string(value)

    def serialize(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"GGUFBits({self.to_string()!r})"

    def to_public_format(self) -> str:
        public_format = f"{self.version}_{self.variant}"
        if self.quality is not None:
            public_format = f"{public_format}_{self.quality}"
        return public_format


# Backward-compatible alias for the earlier wrapper-based refactor.
QuantBits = GGUFBits


_GGUF_PUBLIC_FORMAT_RE = re.compile(r"^(q|iq)_(0|k)(?:_(xs|s|m|l))?$")


def _gguf_public_format_from_bits(bits: GGUFBits) -> str:
    return bits.to_public_format()


def _normalize_gguf_public_format(value: Any) -> Optional[str]:
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
    alias = _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH.get(bits)
    if alias is None:
        raise ValueError(f"GGUFConfig: no default GGUF format exists for `{bits}`-bit quantization.")
    return _gguf_public_format_from_bits(GGUFBits.from_alias(alias))


def _gguf_bits_from_components(bits: int, public_format: str) -> GGUFBits:
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
    if normalized_bits not in [2, 3, 4, 5, 6, 8]:
        raise ValueError("GGUFConfig: `bits` must resolve to one of `[2, 3, 4, 5, 6, 8]`.")

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


def _normalize_quant_bits(bits: Union[int, str, GGUFBits], format_value: Optional[Union[str, FORMAT]] = None) -> Union[int, GGUFBits]:
    if isinstance(format_value, str):
        format_value = _normalize_format(format_value)

    if isinstance(bits, GGUFBits):
        normalized = bits
    elif isinstance(bits, int):
        normalized = bits
    elif isinstance(bits, str):
        raw = bits.strip().lower().replace("-", "_")
        normalized = int(raw) if raw.isdigit() else GGUFBits.from_alias(raw)
    else:
        raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")

    normalized_width = normalized.bits if isinstance(normalized, GGUFBits) else normalized
    if normalized_width not in [2, 3, 4, 5, 6, 8]:
        raise ValueError("QuantizeConfig: `bits` must resolve to one of `[2, 3, 4, 5, 6, 8]`.")

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


def _looks_like_gguf_bits(bits: Any) -> bool:
    if isinstance(bits, GGUFBits):
        return True
    if not isinstance(bits, str):
        return False
    normalized = bits.strip().lower().replace("-", "_")
    return normalized in _GGUF_BITS_ALIAS_INFO


def quant_bits_width(bits: Union[int, str, GGUFBits]) -> int:
    normalized = _normalize_quant_bits(bits)
    return normalized.bits if isinstance(normalized, GGUFBits) else normalized


def serialize_quant_bits(bits: Union[int, str, GGUFBits]) -> Union[int, str]:
    normalized = _normalize_quant_bits(bits)
    return normalized.serialize() if isinstance(normalized, GGUFBits) else normalized

@dataclass
class SmoothMethod:
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
        super().__init__(name="rowcol", group_size_threshold=group_size_threshold)
        self.axis = axis


class GcMode(str, Enum):
    INTERVAL = "interval"
    ON_STAGE_END = "on_stage_end"


@dataclass
class FailSafe:
    strategy: FailSafeStrategy = FailSafeStrategy.RTN # enable failsafe by default due to moe routing behavior breaking calibration based quantization

    # int/float = if captured module fwd tokens is less than value, trigger strategy
    # string = if string is int/float followed by %, then if captured module fwd tokens is less than value in percentage relative to calibration, trigger strategy
    threshold: int | float | str = "0.5%" # if less than 0.5% of calibration reaches module (think moe) then we trigger per-module failsafe quantization

    # naive quantization methods used in failsafe has issue with very small/large outliers that can severely degrade the quantization quality
    # use smoothers to normalize these outliers so they do not dominate the scale/zero calculation
    smooth: Optional[SmoothMethod] = field(default_factory=SmoothMAD)


@dataclass
class WeightOnlyConfig:
    method: WeightOnlyMethod = WeightOnlyMethod.RTN
    # Whole-model RTN is noticeably more stable without a smoother by default.
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
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
class BasePreFilterConfig:
    code: str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code}


@dataclass
class SmootherConfig(BasePreFilterConfig):
    smooth: Optional[SmoothMethod] = None
    code: str = field(default=PreFilterCode.SMOOTHER.value, init=False)

    def __post_init__(self):
        self.smooth = _parse_smooth_method(self.smooth)

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["smooth"] = _serialize_smooth_method(self.smooth)
        return payload


@dataclass
class HessianConfig:
    # Hessian accumulation controls (GPTQ only)
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Maximum rows per Hessian chunk"})
    chunk_bytes: Optional[int] = field(default=None, metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"})
    staging_dtype: Union[str, torch.dtype] = field(
        default=torch.float32,
        metadata={"help": "Stage Hessian chunks in a lower precision dtype when supported"},
    )

    def __post_init__(self):
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
        FORMAT.LLM_AWQ,
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
    FORMAT.LLM_AWQ,
)
QQQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.QQQ,
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
    FORMAT.GGUF: METHOD.GGUF,
    FORMAT.BITBLAS: METHOD.GPTQ,
    FORMAT.GEMM: METHOD.AWQ,
    FORMAT.GEMV: METHOD.AWQ,
    FORMAT.GEMV_FAST: METHOD.AWQ,
    FORMAT.LLM_AWQ: METHOD.AWQ,
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

    # map format field (checkpoint_format) to class/code (format)
    FORMAT_FIELD_CHECKPOINT: FORMAT_FIELD_CODE,
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
    raise ValueError("QuantizeConfig: `failsafe.smooth` must be a SmoothMethod, string, or dict.")


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


def _normalize_pre_filter_config(payload: Any) -> BasePreFilterConfig:
    if isinstance(payload, BasePreFilterConfig):
        return payload
    if isinstance(payload, SmoothMethod):
        return SmootherConfig(smooth=payload)
    if isinstance(payload, str):
        normalized = payload.strip().lower()
        if normalized == PreFilterCode.SMOOTHER.value:
            return SmootherConfig(smooth=None)
        return SmootherConfig(smooth=payload)
    if isinstance(payload, dict):
        code = str(payload.get("code", "")).strip().lower()
        if code and code != PreFilterCode.SMOOTHER.value:
            raise ValueError(f"QuantizeConfig: unsupported pre-filter code `{code}`.")
        if "smooth" in payload:
            return SmootherConfig(smooth=payload.get("smooth"))
        if "type" in payload:
            return SmootherConfig(smooth=payload)
        return SmootherConfig(smooth=None)
    raise ValueError("QuantizeConfig: `pre_filters` entries must be pre-filter configs, smooth configs, dicts, or strings.")


def _normalize_pre_filters(payload: Optional[List[Any]]) -> List[BasePreFilterConfig]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("QuantizeConfig: `pre_filters` must be a list or None.")
    return [_normalize_pre_filter_config(item) for item in payload]


def dynamic_get(dynamic: Dict[str, Dict[str, Union[int, bool]]], module_name: str, key: str = None,
                default: Union[int, bool] = None, sub_key: str = None) -> Union[Dict, int, bool]:

    if dynamic is None:
        return default

    for pattern, overrides in dynamic.items():
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), module_name):
                return False
        elif re.match(pattern.removeprefix("+:"), module_name):
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
        try:
            return METHOD(value)
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization method: `{value}`.") from exc
    if not isinstance(value, METHOD):
        raise ValueError(f"QuantizeConfig: Unsupported `quant_method`: {value}")
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


def _normalize_failsafe(failsafe: Optional[Union[FailSafe, Dict[str, Any], str, int, float]]) -> Optional[FailSafe]:
    if failsafe is None:
        return None
    if isinstance(failsafe, dict):
        strategy = failsafe.get("strategy", FailSafeStrategy.RTN)
        threshold = failsafe.get("threshold", "1.0%")
        smooth = failsafe.get("smooth")
        if smooth is None:
            smooth = failsafe.get("smooth_method")
        if smooth is None and "clip_method" in failsafe:
            smooth = failsafe.get("clip_method")
        smooth = _parse_smooth_method(smooth)
        if smooth is None:
            if "smooth_percentile" in failsafe:
                smooth = SmoothPercentile(percentile=float(failsafe.get("smooth_percentile", 99.0)))
            elif "smooth_mad_k" in failsafe:
                smooth = SmoothMAD(k=float(failsafe.get("smooth_mad_k", 3.0)))
            elif "smooth_mse_steps" in failsafe or "smooth_mse_maxshrink" in failsafe:
                smooth = SmoothMSE(
                    steps=int(failsafe.get("smooth_mse_steps", 32)),
                    maxshrink=float(failsafe.get("smooth_mse_maxshrink", 0.8)),
                )
            elif "smooth_outlier_pct" in failsafe:
                smooth = SmoothOutlier(pct=float(failsafe.get("smooth_outlier_pct", 1.0)))
            elif "smooth_rms_k" in failsafe:
                smooth = SmoothSoftNorm(k=float(failsafe.get("smooth_rms_k", 3.0)))
            elif "smooth_log_mu" in failsafe:
                smooth = SmoothLog(
                    percentile=float(failsafe.get("smooth_percentile", 99.0)),
                    mu=float(failsafe.get("smooth_log_mu", 8.0)),
                )
            elif "smooth_axis" in failsafe:
                smooth = SmoothRowCol(axis=str(failsafe.get("smooth_axis", "row")))
        failsafe = FailSafe(strategy=strategy, threshold=threshold, smooth=smooth)
    elif isinstance(failsafe, (str, int, float)):
        failsafe = FailSafe(strategy=FailSafeStrategy.RTN, threshold=failsafe)
    elif not isinstance(failsafe, FailSafe):
        raise ValueError("QuantizeConfig: `failsafe` must be a FailSafe config, dict, string, int, float, or None.")

    if isinstance(failsafe.strategy, str):
        try:
            failsafe.strategy = FailSafeStrategy(failsafe.strategy.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `failsafe.strategy` must be one of {[v.value for v in FailSafeStrategy]}."
            ) from exc
    elif not isinstance(failsafe.strategy, FailSafeStrategy):
        raise ValueError(
            f"QuantizeConfig: `failsafe.strategy` must be one of {[v.value for v in FailSafeStrategy]}."
        )

    failsafe.smooth = _parse_smooth_method(failsafe.smooth)
    return failsafe


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


def _normalize_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


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
    normalized[FORMAT_FIELD_CHECKPOINT] = FORMAT.GGUF
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

    if target_cls is AWQQuantizeConfig:
        expected_method = METHOD.AWQ
    elif target_cls is GGUFConfig:
        expected_method = METHOD.GGUF
    elif target_cls is QQQQuantizeConfig:
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

    method = normalized.get(QUANT_METHOD_FIELD)
    normalized_method = None
    if method is not None:
        try:
            normalized_method = _normalize_quant_method(method)
            normalized[QUANT_METHOD_FIELD] = normalized_method
        except ValueError:
            normalized_method = None

    if normalized_method is not None and normalized_method != expected_method:
        if target_cls is GGUFConfig and normalized_method == METHOD.GPTQ:
            pass
        else:
            log.warn(
                f"QuantizeConfig: `{QUANT_METHOD_FIELD}`=`{normalized_method}` is incompatible with `{target_cls.__name__}`. "
                f"Auto-fix method to `{expected_method}`."
            )
        normalized[QUANT_METHOD_FIELD] = expected_method

    return normalized


def _filter_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    target_field_names = {field.name for field in fields(target_cls) if field.init}
    return {key: value for key, value in payload.items() if key in target_field_names}


def _prepare_target_quantize_config_kwargs(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, payload)
    if target_cls is RTNQuantizeConfig:
        normalized = _normalize_rtn_kwargs(normalized)
    elif target_cls is GGUFConfig:
        normalized = _normalize_gguf_kwargs(normalized)
    return _filter_quantize_config_payload_for_target_cls(target_cls, normalized)


class QuantizeConfigMeta(type):
    def __call__(cls, *args, **kwargs):
        if cls is QuantizeConfig:
            target_cls = _resolve_quantize_config_class(kwargs)
            target_kwargs = _prepare_target_quantize_config_kwargs(target_cls, kwargs)
            return type.__call__(target_cls, *args, **target_kwargs)
        return super().__call__(*args, **kwargs)


@dataclass
class BaseQuantizeConfig:
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

    quant_method: METHOD = field(default=METHOD.GPTQ)

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
    failsafe: Optional[FailSafe] = field(default_factory=FailSafe)

    # deprecated: only used for compat when reading legacy configs
    is_marlin_format: bool = False

    # Callback function to filter devices for compute-intensive stages (quantization and forwarding)
    compute_device_filter: Optional[callable] = field(
        default=None,
        metadata={"help": "Callback function to filter devices for compute-intensive stages. Function signature: fn(devices: List) -> List. "
                  "Example to exclude device 0: compute_device_filter=lambda devices: [d for d in devices if d.index != 0]"}
    )

    auto_forward_data_parallel: bool = field(
        default=True,
        metadata={"help": "When multi-gpu is detected, we may data clone modules to each gpu for data parallelism "
        "to speed up quantization forwarding. This causes extra time spent (especially for MoE layers) and vram pressure, "
        "leading in some cases to slower forwarding or vram OOM"}
    )

    vram_strategy: VramStrategy = field(default=VramStrategy.EXCLUSIVE)

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
                  "Example with bypass routing (forward all data to each expert): "
                  "moe={'routing': {'class': 'ExpertsRoutingBypass', 'batch_size': None}} - processes all experts in one batch (default). "
                  "moe={'routing': {'class': 'ExpertsRoutingBypass', 'batch_size': 4}} - processes 4 experts at a time to reduce VRAM pressure. "
                  "Example with routing override (limit experts per token): "
                  "moe={'routing': {'class': 'ExpertsRoutingOverride', 'num_experts_per_tok': 2}}. "
                  "Example to forward to all experts: "
                  "moe={'routing': {'class': 'ExpertsRoutingOverride', 'num_experts_per_tok': 'all'}}"}
    )

    @property
    def checkpoint_format(self) -> FORMAT:
        return self.format if isinstance(self.format, FORMAT) else _normalize_format(self.format)

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
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.quant_method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `quant_method`: {self.quant_method}")
        return tuple(valid_formats)

    def export_quant_method(self) -> METHOD:
        return _resolve_export_quant_method(self.checkpoint_format, fallback_method=self.quant_method)

    def default_desc_act(self) -> bool:
        return True

    def __post_init__(self):
        fields_info = fields(self)

        self.quant_method = _normalize_quant_method(self.quant_method)
        checkpoint_format = self._resolve_checkpoint_format()
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = self._normalize_bits_field(self.bits, checkpoint_format=checkpoint_format)

        allowed_methods = self.allowed_quant_methods()
        if allowed_methods and self.quant_method not in allowed_methods:
            raise ValueError(
                f"{self.__class__.__name__}: `quant_method` must be one of {[v.value for v in allowed_methods]}."
            )

        valid_formats = self.supported_export_formats()
        if checkpoint_format not in valid_formats:
            raise ValueError(
                f"{self.__class__.__name__}: unsupported export `format` `{checkpoint_format}`."
            )

        self.failsafe = _normalize_failsafe(self.failsafe)

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
                    checkpoint_format=checkpoint_format,
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

        self.vram_strategy = _normalize_vram_strategy(self.vram_strategy)
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)

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
        if format:
            format = _normalize_format(format)
            if format not in valid_formats:
                raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if quantize_cfg.get(FORMAT_FIELD_CHECKPOINT):
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        elif quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) is None:
            format_auto_inferred = True

        field_names = _known_quantize_config_field_names()

        normalized = {
            QUANT_METHOD_FIELD: METHOD.GPTQ,
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }

        for key, val in quantize_cfg.items():
            key = key.lower()

            if key == FORMAT_FIELD_CHECKPOINT:
                normalized[key] = _normalize_format(val)
                continue

            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]
            elif key in QUANT_CONFIG_ARG_SYNONYMS_NEGATED and QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key]
                val = not bool(val)

            if key == QUANT_METHOD_FIELD:
                if isinstance(val, str) and val.lower() == FORMAT.MARLIN:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.MARLIN
                elif isinstance(val, str) and val.lower() == FORMAT.BITBLAS:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.BITBLAS
                else:
                    normalized[QUANT_METHOD_FIELD] = _normalize_quant_method(val)
            elif key == FORMAT_FIELD_CODE:
                gguf_checkpoint_hint = format or checkpoint_format_hint
                if gguf_checkpoint_hint is not None:
                    try:
                        gguf_checkpoint_hint = _normalize_format(gguf_checkpoint_hint)
                    except ValueError:
                        gguf_checkpoint_hint = None
                if gguf_checkpoint_hint == FORMAT.GGUF:
                    normalized[key] = val
                else:
                    normalized[key] = _normalize_format(val)
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if quantize_cfg.get(AWQ_PACKING_BACKEND_FIELD) == "llm-awq":
            normalized[QUANT_METHOD_FIELD] = METHOD.AWQ
            normalized[FORMAT_FIELD_CODE] = FORMAT.LLM_AWQ
            normalized[PACK_DTYPE_FIELD] = torch.int16
            log.info("Detected llm-awq quantization format; FORMAT automatically set to FORMAT.LLM_AWQ.")

        meta_payload = normalized.get(META_FIELD)
        meta_field_map = {
            "failsafe": "failsafe",
            "hessian": "hessian",
            "gptaq": "gptaq",
            "weight_only": "weight_only",
            "pre_filters": "pre_filters",
            "gc_mode": "gc_mode",
            "wait_for_submodule_finalizers": "wait_for_submodule_finalizers",
            "auto_forward_data_parallel": "auto_forward_data_parallel",
            "vram_strategy": "vram_strategy",
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
        }
        if isinstance(meta_payload, dict):
            for normalized_key, meta_key in meta_field_map.items():
                if normalized_key not in normalized and meta_key in meta_payload:
                    normalized[normalized_key] = meta_payload.get(meta_key)

        target_cls = cls if cls not in {BaseQuantizeConfig, QuantizeConfig} else _resolve_quantize_config_class(normalized)
        normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, normalized)
        if target_cls is RTNQuantizeConfig:
            normalized = _normalize_rtn_kwargs(normalized)
        elif target_cls is GGUFConfig:
            normalized = _normalize_gguf_kwargs(normalized)

        if format_auto_inferred:
            log.info(
                f"QuantizeConfig: `{FORMAT_FIELD_CHECKPOINT}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}"
            )

        if normalized[FORMAT_FIELD_CODE] in {FORMAT.BITBLAS}:
            normalized["desc_act"] = False

        if "sym" not in normalized and target_cls is not GGUFConfig:
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
        smooth = _serialize_smooth_method(self.failsafe.smooth if self.failsafe is not None else None)

        meta_payload = dict(self.meta) if self.meta else {}
        if self.moe:
            meta_payload["moe"] = self.moe.to_dict()

        if self.failsafe is None:
            meta_payload["failsafe"] = None
        else:
            meta_payload["failsafe"] = {
                "strategy": (
                    self.failsafe.strategy.value
                    if isinstance(self.failsafe.strategy, FailSafeStrategy)
                    else self.failsafe.strategy
                ),
                "threshold": self.failsafe.threshold,
                "smooth": smooth,
            }

        meta_payload["offload_to_disk"] = self.offload_to_disk
        meta_payload["offload_to_disk_path"] = self.offload_to_disk_path
        meta_payload["pack_impl"] = self.pack_impl
        meta_payload["gc_mode"] = self.gc_mode.value if isinstance(self.gc_mode, GcMode) else self.gc_mode
        meta_payload["wait_for_submodule_finalizers"] = self.wait_for_submodule_finalizers
        meta_payload["auto_forward_data_parallel"] = self.auto_forward_data_parallel
        meta_payload["vram_strategy"] = (
            self.vram_strategy.value if isinstance(self.vram_strategy, VramStrategy) else self.vram_strategy
        )
        self._update_meta_payload(meta_payload)

        out = {
            "bits": serialize_quant_bits(self.bits),
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "lm_head": self.lm_head,
            QUANT_METHOD_FIELD: self.quant_method,
            FORMAT_FIELD_CHECKPOINT: self.checkpoint_format,
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


@dataclass
class PreFilterQuantizeConfig(BaseQuantizeConfig):
    pre_filters: Optional[List[Union[BasePreFilterConfig, Dict[str, Any], str]]] = field(default=None)
    smoother: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]] = field(default=None)
    # Backward-compatible alias. New code should use `smoother`.
    smooth: Optional[Union[SmoothMethod, Dict[str, Any], str]] = field(default=None, repr=False)

    def _normalize_prefilter_state(self) -> None:
        self.pre_filters = _normalize_pre_filters(self.pre_filters)

        smoother_payload = self.smoother if self.smoother is not None else self.smooth
        self.smoother = _normalize_smoother_config(smoother_payload)

        if self.smoother is None:
            for pre_filter in self.pre_filters:
                if isinstance(pre_filter, SmootherConfig):
                    self.smoother = pre_filter
                    break

        non_smoother_filters = [pre_filter for pre_filter in self.pre_filters if not isinstance(pre_filter, SmootherConfig)]
        if self.smoother is not None:
            non_smoother_filters.append(self.smoother)
        self.pre_filters = non_smoother_filters
        self.smooth = self.resolve_smooth_method()

    def __post_init__(self):
        self._normalize_prefilter_state()
        super().__post_init__()

    def resolve_smooth_method(self) -> Optional[SmoothMethod]:
        if self.smoother is None:
            return None
        return self.smoother.smooth

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.pre_filters:
            meta_payload["pre_filters"] = [pre_filter.to_dict() for pre_filter in self.pre_filters]


@dataclass
class QuantizeConfig(BaseQuantizeConfig, metaclass=QuantizeConfigMeta):
    """Backward-compatible quantization config factory.

    Direct construction dispatches to a concrete method-specific config class.
    """


@dataclass
class GPTQQuantizeConfig(QuantizeConfig):
    damp_percent: Optional[float] = field(default=None)
    damp_auto_increment: Optional[float] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)
    mse: float = field(default=0.0)
    gptaq: Optional[GPTAQConfig] = field(default=None)
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
            self.damp_percent = _default_damp_percent(self.quant_method)
        if self.damp_auto_increment is None:
            self.damp_auto_increment = _default_damp_auto_increment(self.quant_method)
        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")
        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

        self.hessian = _normalize_hessian(self.hessian)
        self.gptaq = _normalize_gptaq(self.gptaq)

        if act_group_aware_user_value is None:
            self.act_group_aware = self.quant_method == METHOD.GPTQ
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
        else:
            device = self.gptaq.device
            meta_payload["gptaq"] = {
                "alpha": self.gptaq.alpha,
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
class AWQQuantizeConfig(QuantizeConfig):
    quant_method: METHOD = field(default=METHOD.AWQ)
    format: FORMAT = field(default=FORMAT.GEMM)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.AWQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return AWQ_EXPORT_FORMATS

    def __post_init__(self):
        self.quant_method = _normalize_quant_method(self.quant_method)
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
class QQQQuantizeConfig(GPTQQuantizeConfig):
    quant_method: METHOD = field(default=METHOD.QQQ)
    format: FORMAT = field(default=FORMAT.QQQ)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.QQQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return QQQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return True


@dataclass
class RTNQuantizeConfig(PreFilterQuantizeConfig):
    quant_method: METHOD = field(default=METHOD.GPTQ)
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
class GGUFConfig(PreFilterQuantizeConfig):
    format: Optional[str] = field(default=None)
    quant_method: METHOD = field(default=METHOD.GGUF, init=False)
    group_size: int = field(default=-1, init=False, repr=False)
    desc_act: Optional[bool] = field(default=False, init=False, repr=False)
    sym: bool = field(default=True, init=False, repr=False)
    _gguf_bits: GGUFBits = field(init=False, repr=False, compare=False)

    @property
    def checkpoint_format(self) -> FORMAT:
        return FORMAT.GGUF

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
        normalized = _normalize_quant_bits(bits_value, format_value=None)
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
        self._normalize_prefilter_state()
        super(PreFilterQuantizeConfig, self).__post_init__()
        self._gguf_bits = _gguf_bits_from_components(self.bits, self.format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format

    def to_dict(self):
        out = super().to_dict()
        out.pop(GROUP_SIZE_FIELD_CODE, None)
        out.pop("desc_act", None)
        out.pop(QUANT_METHOD_FIELD, None)
        out.pop(PACK_DTYPE_FIELD, None)

        meta_payload = out.get(META_FIELD)
        if isinstance(meta_payload, dict):
            for key in (
                "failsafe",
                "offload_to_disk",
                "offload_to_disk_path",
                "pack_impl",
                "gc_mode",
                "wait_for_submodule_finalizers",
                "auto_forward_data_parallel",
                "vram_strategy",
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


GGUFQuantizeConfig = GGUFConfig


def clone_weight_only_config_for_module(
    qcfg: Union[RTNQuantizeConfig, GGUFConfig],
    module_full_name: str,
) -> Optional[Union[RTNQuantizeConfig, GGUFConfig]]:
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
        else:
            qcfg_clone.bits = _normalize_quant_bits(
                qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits),
                format_value=qcfg_clone.checkpoint_format,
            )

        if isinstance(qcfg_clone, RTNQuantizeConfig):
            qcfg_clone.sym = qcfg.dynamic_get(module_full_name, "sym", qcfg_clone.sym)
            qcfg_clone.group_size = qcfg.dynamic_get(module_full_name, "group_size", qcfg_clone.group_size)

            desc_act_override = qcfg.dynamic_get(module_full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override

    return qcfg_clone


clone_rtn_config_for_module = clone_weight_only_config_for_module


def _resolve_quantize_config_class(payload: Dict[str, Any]) -> type[BaseQuantizeConfig]:
    method = payload.get(QUANT_METHOD_FIELD, METHOD.GPTQ)
    checkpoint_format_value = payload.get(FORMAT_FIELD_CHECKPOINT, payload.get(FORMAT_FIELD_CODE, FORMAT.GPTQ))
    weight_only = payload.get("weight_only")
    bits = payload.get(BITS_FIELD_CODE)
    gguf_public_format = payload.get(FORMAT_FIELD_CODE)

    try:
        method = _normalize_quant_method(method)
    except Exception:
        method = METHOD.GPTQ

    try:
        checkpoint_format_value = _normalize_format(checkpoint_format_value)
    except Exception:
        checkpoint_format_value = FORMAT.GPTQ
    format_value = checkpoint_format_value

    gguf_format_detected = False
    if gguf_public_format is not None:
        try:
            gguf_format_detected = _normalize_gguf_public_format(gguf_public_format) is not None
        except ValueError:
            gguf_format_detected = False

    weight_only_method = _peek_weight_only_method(weight_only)
    if weight_only is not None and weight_only_method not in {None, WeightOnlyMethod.RTN, WeightOnlyMethod.GGUF}:
        raise ValueError(
            "QuantizeConfig: unsupported weight-only config. Weight-only export currently supports `rtn` and `gguf`."
        )
    if (
        checkpoint_format_value == FORMAT.GGUF
        or weight_only_method == WeightOnlyMethod.GGUF
        or _looks_like_gguf_bits(bits)
        or gguf_format_detected
    ):
        return GGUFConfig
    if weight_only_method == WeightOnlyMethod.RTN:
        return RTNQuantizeConfig
    if weight_only is not None:
        return RTNQuantizeConfig
    if method == METHOD.QQQ or format_value == FORMAT.QQQ:
        return QQQQuantizeConfig
    if method == METHOD.AWQ:
        return AWQQuantizeConfig
    if format_value in {FORMAT.GEMM, FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ}:
        return AWQQuantizeConfig
    if format_value == FORMAT.MARLIN:
        return AWQQuantizeConfig if method == METHOD.AWQ else GPTQQuantizeConfig
    return GPTQQuantizeConfig


def _known_quantize_config_field_names() -> set[str]:
    field_names: set[str] = set()
    for cls in (
        BaseQuantizeConfig,
        PreFilterQuantizeConfig,
        QuantizeConfig,
        GPTQQuantizeConfig,
        AWQQuantizeConfig,
        QQQQuantizeConfig,
        RTNQuantizeConfig,
        GGUFConfig,
    ):
        field_names.update(field.name for field in fields(cls))
    return field_names
