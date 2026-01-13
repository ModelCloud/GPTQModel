# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os.path
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
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
    pass


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
    METHOD.QQQ: {
        FORMAT.QQQ,
    },
    METHOD.AWQ: {
        FORMAT.GEMM,
        FORMAT.GEMV,
        FORMAT.GEMV_FAST,
        FORMAT.MARLIN,
    },
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

@dataclass
class QuantizeConfig():
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, bool]]]] = field(default=None)

    # GPTQ only
    # 128 offer good balance between inference speed, vram usage (bpw), and quality
    # use 32 for highest quality with slower inference and higher vram usage
    group_size: int = field(default=128)

    # increase damp if NaN is encountered during `.quantize()` and/or increase calib dataset size
    damp_percent: float = field(default=None)
    damp_auto_increment: float = field(default=None)

    desc_act: Optional[bool] = field(default=None)

    # GPTQ only
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)

    # symmetric quantization toggle (True=symmetric, False=asymmetric).
    sym: bool = field(default=True)

    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    quant_method: METHOD = field(default=METHOD.GPTQ)

    # default to gptq v1 format for maximum compat with 3rd party inference libs with minimal loss vs v2
    # if you inference with gptqmodel, save to gptq_v2 format for best result
    format: FORMAT = field(default=FORMAT.GPTQ)

    # quantization_order: str = "activate",
    # quantization_scale: str = "mse", # or absmax
    # is_distributed: bool = False,
    # tied_gptq_handle: Optional["GPTQ"] = None

    # GPTQ only
    # mean square error calculation: may reduce error loss for some models
    mse: float = field(default=0.0)

    # properties that do not directly contributes to quantization or quant inference should be placed in meta
    # i.e. quantizer tool (producer) + version, timestamp, entity who made the quant, etc
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # packing implementation hinpt (`original` = legacy CPU pack, `gpu` enables CUDA pack, `cpu` forces block CPU pack).
    pack_impl: str = field(default="cpu")

    # pending used field
    adapter: Optional[Union[Dict[str, Any], Lora]] = field(default=None)

    # quantization only:
    # controls cpu memory saving by offloading layers/modules to disk in the slow quantization process
    # default to true as the benefit of ~73.5% cpu memory saving is tremendous
    offload_to_disk: bool = field(default=True, metadata={"help": "Offload completed module memory to disk during quantization loop"})
    offload_to_disk_path: str = field(default=None, metadata={"help": "Offload disk path. Only applicable if Offload to disk is enabled"})

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    # GPTQ only
    # deprecated: only used for compat
    is_marlin_format: bool = False

    # gptq only:
    # if calibration is insufficient, fallback to a simple quantization strategy; encapsulated in FailSafe config
    failsafe: Optional[FailSafe] = field(default_factory=FailSafe)

    # GPTQ only
    # gptaq only:
    gptaq: Optional[GPTAQConfig] = field(default=None)

    # gptq only:
    # skip all heavy computations for testing model loading
    mock_quantization: bool = field(default=False, metadata={"help": "Skip heavy computations for fast model loading validation"})

    # GPTQ only
    # Hessian accumulation controls (GPTQ only)
    hessian: Optional[HessianConfig] = field(default_factory=HessianConfig)

    # Callback function to filter devices for compute-intensive stages (quantization and forwarding)
    # Takes a list of devices and returns either the original list or a filtered subset
    compute_device_filter: Optional[callable] = field(
        default=None,
        metadata={"help": "Callback function to filter devices for compute-intensive stages. Function signature: fn(devices: List) -> List. "
                  "Example to exclude device 0: compute_device_filter=lambda devices: [d for d in devices if d.index != 0]"}
    )

    # Works faster than data parallel with some configurations
    auto_forward_data_parallel: bool = field(
        default=True,
        metadata={"help": "When multi-gpu is detected, we may data clone modules to each gpu for data parallelism "
        "to speed up quantization forwarding. This causes extra time spent (especially for MoE layers) and vram pressure, "
        "leading in some cases to slower forwarding or vram OOM"}
    )

    # VRAM allocation strategy for MoE-heavy subsets
    vram_strategy: VramStrategy = field(default=VramStrategy.EXCLUSIVE)

    gc_mode: GcMode = field(
        default=GcMode.INTERVAL,
        metadata={"help": "Garbage collection mode: 'interval' for regular GC or 'on_stage_end' for GC after stage end (after forward pass, quantize, layer finilization)."}
    )

    # Control whether to wait for layer finalization (packing, writing) before proceeding to next layer
    # Default False preserves current behavior (async finalization in background while next layer starts)
    wait_for_submodule_finalizers: bool = field(
        default=False,
        metadata={"help": "Wait for all layer finalization tasks (packing, offloading to disk, etc) to complete before proceeding to next layer. May reduce vram pressure for some env."}
    )

    moe: MoEConfig = field(
        default=None,
        metadata={"help": "Mixture-of-Experts (MoE) configuration, including routing strategy and related overrides."}
    )

    def __post_init__(self):
        fields_info = fields(self)

        # validate/normalizes pack_dtype from string and dtype to valid dtype
        if self.pack_dtype is None:
            self.pack_dtype = torch.int32
        else:
            if isinstance(self.pack_dtype, str):
                self.pack_dtype = self.pack_dtype.lower()
                if self.pack_dtype not in ["int64", "int32", "int16", "int8"]:
                    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")
                self.pack_dtype = getattr(torch, self.pack_dtype)
            elif isinstance(self.pack_dtype, torch.dtype):
                if self.pack_dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
                    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")
            else:
                raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")

        # validate quant method and format is matched
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.quant_method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `quant_method`: {self.quant_method}")

        # If the user does not pass it, the default value will be set according to quant_method
        if self.damp_percent is None:
            if self.quant_method == METHOD.QQQ:
                self.damp_percent = 0.005
            else:
                self.damp_percent = 0.05
        if self.damp_auto_increment is None:
            if self.quant_method == METHOD.QQQ:
                self.damp_auto_increment = 0.001
            else:
                self.damp_auto_increment = 0.01

        # TODO FIXME awq compat which didn't have checkpoint_format before merging to gptqmodel
        if self.quant_method == METHOD.AWQ and self.format not in [FORMAT.MARLIN, FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.GEMM]:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.GEMM}`")
            self.format = FORMAT.GEMM

        if self.format not in valid_formats:
            raise ValueError(
                f"QuantizeConfig: checkpoint `format` used is {self.format}, and the quantization method is {self.quant_method}. "
            )

        # normalize failsafe config
        if self.failsafe is None:
            pass
        elif isinstance(self.failsafe, dict):
            strategy = self.failsafe.get("strategy", FailSafeStrategy.RTN)
            threshold = self.failsafe.get("threshold", "1.0%")
            smooth = self.failsafe.get("smooth")
            if smooth is None:
                smooth = self.failsafe.get("smooth_method")
            if smooth is None and "clip_method" in self.failsafe:
                smooth = self.failsafe.get("clip_method")
            smooth = _parse_smooth_method(smooth)
            if smooth is None:
                if "smooth_percentile" in self.failsafe:
                    smooth = SmoothPercentile(
                        percentile=float(self.failsafe.get("smooth_percentile", 99.0))
                    )
                elif "smooth_mad_k" in self.failsafe:
                    smooth = SmoothMAD(k=float(self.failsafe.get("smooth_mad_k", 3.0)))
                elif "smooth_mse_steps" in self.failsafe or "smooth_mse_maxshrink" in self.failsafe:
                    smooth = SmoothMSE(
                        steps=int(self.failsafe.get("smooth_mse_steps", 32)),
                        maxshrink=float(self.failsafe.get("smooth_mse_maxshrink", 0.8)),
                    )
                elif "smooth_outlier_pct" in self.failsafe:
                    smooth = SmoothOutlier(pct=float(self.failsafe.get("smooth_outlier_pct", 1.0)))
                elif "smooth_rms_k" in self.failsafe:
                    smooth = SmoothSoftNorm(k=float(self.failsafe.get("smooth_rms_k", 3.0)))
                elif "smooth_log_mu" in self.failsafe:
                    smooth = SmoothLog(
                        percentile=float(self.failsafe.get("smooth_percentile", 99.0)),
                        mu=float(self.failsafe.get("smooth_log_mu", 8.0)),
                    )
                elif "smooth_axis" in self.failsafe:
                    smooth = SmoothRowCol(axis=str(self.failsafe.get("smooth_axis", "row")))
            self.failsafe = FailSafe(
                strategy=strategy,
                threshold=threshold,
                smooth=smooth,
            )
        elif isinstance(self.failsafe, (str, int, float)):
            self.failsafe = FailSafe(strategy=FailSafeStrategy.RTN, threshold=self.failsafe)
        elif not isinstance(self.failsafe, FailSafe):
            raise ValueError("QuantizeConfig: `failsafe` must be a FailSafe config, dict, string, int, float, or None.")

        if self.failsafe is not None:
            if isinstance(self.failsafe.strategy, str):
                try:
                    self.failsafe.strategy = FailSafeStrategy(self.failsafe.strategy.lower())
                except ValueError as exc:
                    raise ValueError(
                        f"QuantizeConfig: `failsafe.strategy` must be one of {[v.value for v in FailSafeStrategy]}."
                    ) from exc
            elif not isinstance(self.failsafe.strategy, FailSafeStrategy):
                raise ValueError(
                    f"QuantizeConfig: `failsafe.strategy` must be one of {[v.value for v in FailSafeStrategy]}."
                )

            self.failsafe.smooth = _parse_smooth_method(self.failsafe.smooth)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"QuantizeConfig: `bits` must be in the set of `{fields_info[0].metadata['choices']}`.")

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},  # 先添加以 "-" 开头的键
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')}  # 然后添加其他键
            }

            for layer, layer_dict in self.dynamic.items():
                for key, value in layer_dict.items():
                    if key == "bits" and value not in fields_info[0].metadata["choices"]:
                        raise ValueError(f"QuantizeConfig: Layer `{layer}` only support quantization of  `{fields_info[0].metadata['choices']}` bits.")
                    elif key == "group_size" and value != -1 and value <= 0:
                        raise ValueError("QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`.")

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`.")

        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")

        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

        if self.hessian is None:
            self.hessian = HessianConfig()
        elif isinstance(self.hessian, dict):
            self.hessian = HessianConfig(**self.hessian)
        elif not isinstance(self.hessian, HessianConfig):
            raise ValueError("QuantizeConfig: `hessian` must be a HessianConfig, dict, or None.")

        if self.gptaq is None:
            pass
        elif isinstance(self.gptaq, dict):
            self.gptaq = GPTAQConfig(**self.gptaq)
        elif not isinstance(self.gptaq, GPTAQConfig):
            raise ValueError("QuantizeConfig: `gptaq` must be a GPTAQConfig, dict, or None.")

        # resolve activation ordering compatibility and defaults
        desc_act_user_value = self.desc_act
        act_group_aware_user_value = self.act_group_aware

        if desc_act_user_value is None:
            # GPTQ defaults to higher quality ordering disabled, others retain legacy default
            self.desc_act = False if self.quant_method == METHOD.GPTQ else True
        elif isinstance(desc_act_user_value, bool):
            self.desc_act = desc_act_user_value
        else:
            self.desc_act = bool(desc_act_user_value)

        if act_group_aware_user_value is None:
            # auto-enable for GPTQ unless user explicitly disables it
            self.act_group_aware = self.quant_method == METHOD.GPTQ
        elif isinstance(act_group_aware_user_value, bool):
            self.act_group_aware = act_group_aware_user_value
        else:
            self.act_group_aware = bool(act_group_aware_user_value)

        self._resolve_activation_ordering(desc_act_user_value, act_group_aware_user_value)

        # validate hybrid act order
        if self.act_group_aware and self.desc_act:
            raise ValueError("QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False`.")

        # validate meta
        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key, value in self.meta.items():
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        # adapter normalize
        self.adapter = normalize_adapter(self.adapter)

        #print(f"adapter: {self.adapter}")

        if self.offload_to_disk and not self.offload_to_disk_path:
            path_key = f"{get_random_string()}-{get_random_string()}"
            self.offload_to_disk_path = f"./gptqmodel_offload/{path_key}/"
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to `{self.offload_to_disk_path}`")

        if isinstance(self.vram_strategy, str):
            try:
                self.vram_strategy = VramStrategy(self.vram_strategy.lower())
            except ValueError as exc:
                raise ValueError(
                    f"QuantizeConfig: `vram_strategy` must be one of {[v.value for v in VramStrategy]}."
                ) from exc
        elif not isinstance(self.vram_strategy, VramStrategy):
            raise ValueError(
                f"QuantizeConfig: `vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            )

        if isinstance(self.gc_mode, str):
            try:
                self.gc_mode = GcMode(self.gc_mode.lower())
            except ValueError as exc:
                raise ValueError(
                    f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
                ) from exc
        elif not isinstance(self.gc_mode, GcMode):
            raise ValueError(
                f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
            )

    def extension_set(self, key: str, value: Any):
        if self.adapter is None:
            self.adapter = {}

        self.adapter[key.lower()] = value

    def _resolve_activation_ordering(
        self,
        desc_act_user_value: Optional[bool],
        act_group_aware_user_value: Optional[bool],
    ) -> None:
        """Normalize defaults and enforce compatibility between desc_act and act_group_aware."""

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

    def extension_get(self, key: str) -> Any:
            return self.adapter.get(key.lower()) if self.adapter else None

    def meta_set(self, key: str, value: Any):
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        return self.meta.get(key)

    def dynamic_get(self, layer_name: str, key: str = None, default: Union[int, bool, float] = None, sub_key: str = None
                    ) -> Union[Dict, int, bool, float]:
        return dynamic_get(self.dynamic, layer_name, key, default, sub_key)

    # versionable is a meta.property that pairs value with version i.e "value:1.0.0"
    def meta_set_versionable(self, key: str, value: List[str]):
        self.meta_set(key, value)

    # versionable is a meta.property that pairs value with version i.e "value:1.0.0"
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

    # is quantized model quantized or packed by gptqmodel version with gptaq format code
    def is_quantized_by_gptaq(self) -> bool:
        # check meta.quantizer
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)

        return False

    def extract_adapter_rank_patterns(self) -> Optional[Dict[str, int]]:
        adapter_rank_patterns = {}

        # no rank can be had if there is no dynamic or adapter
        if not self.dynamic or not self.adapter:
            return adapter_rank_patterns

        # override format: `{ "adapter": { "rank": 512 } }`
        for k, v in self.dynamic.items():
            adapter_override = v.get("adapter", None) # TODO use const, not str
            if adapter_override and isinstance(adapter_override, Dict):
                rank = adapter_override.get("rank", None)
                if rank and isinstance(rank, int):
                    # need to strip `+:` positive prefix
                    adapter_rank_patterns[k.lstrip("+:")] = rank  # TODO use const, not str

        return adapter_rank_patterns

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            d = self.to_dict()
            json_str = json.dumps(d, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    # normalize quant config for compat and also performs validation
    def from_quant_config(cls, quantize_cfg, format: str = None):
        valid_formats = {FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.MARLIN, FORMAT.BITBLAS}
        format_auto_inferred = False
        # compat: format can be passed in via from_quantized() if field missing from json
        if format:
            if format not in valid_formats:
                raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if quantize_cfg.get(FORMAT_FIELD_CHECKPOINT):
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        # compat: warn if checkpoint_format is missing
        elif quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) is None:
            format_auto_inferred = True

        field_names = [field.name for field in fields(cls)]

        # FIXME convert awg quantize_config to gptq quantize_config
        normalized = {
            QUANT_METHOD_FIELD: METHOD.GPTQ,
            # compat: default to gptq(v1) when loading models
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }

        for key, val in quantize_cfg.items():
            key = key.lower()

            # remap keys according to compat map
            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]
            elif key in QUANT_CONFIG_ARG_SYNONYMS_NEGATED and QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key]
                val = not bool(val)

            if key == FORMAT_FIELD_CHECKPOINT:
                val = val.lower()

                if val in {FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.MARLIN, FORMAT.BITBLAS}:
                    normalized[key] = val
                else:
                    raise ValueError(f"QuantizeConfig: Unknown quantization format: `{val}`.")
            elif key == QUANT_METHOD_FIELD:
                val = val.lower()
                # compat: some hf models use quant_method=marlin or bitblas
                if val == FORMAT.MARLIN:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.MARLIN
                elif val == FORMAT.BITBLAS:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.BITBLAS
                elif val not in {METHOD.GPTQ, METHOD.QQQ, METHOD.AWQ}:
                    raise ValueError(f"QuantizeConfig: Unknown quantization method: `{val}`.")
                else:
                    normalized[QUANT_METHOD_FIELD] = val
            elif key == FORMAT_FIELD_CODE:
                normalized[key] = val.lower() if isinstance(val, str) else val
            elif key == "failsafe":
                normalized[key] = val
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        # fix method if format is not allowed for the method
        fmt = normalized.get(FORMAT_FIELD_CODE)
        method = normalized.get(QUANT_METHOD_FIELD)

        # TODO FIXME qqq compat which didn't have checkpoint_format before merging to gptqmodel
        if method == METHOD.QQQ and fmt != FORMAT.QQQ:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QQQ}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.QQQ
            fmt = FORMAT.QQQ

        if fmt is not None:
            allowed_methods = [m for m, fmts in QUANT_METHOD_FORMAT_MAPPING.items() if fmt in fmts]
            if method not in allowed_methods:
                if fmt in {FORMAT.GEMM, FORMAT.GEMV, FORMAT.GEMV_FAST}:
                    new_method = METHOD.AWQ
                elif fmt in {FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.BITBLAS}:
                    new_method = METHOD.GPTQ
                elif fmt == FORMAT.QQQ:
                    new_method = METHOD.QQQ
                elif fmt == FORMAT.MARLIN:
                    new_method = method if method in {METHOD.GPTQ, METHOD.AWQ} else METHOD.GPTQ
                else:
                    new_method = allowed_methods[0] if allowed_methods else METHOD.GPTQ
                if new_method != method:
                    log.warn(
                        f"QuantizeConfig: `{FORMAT_FIELD_CODE}`=`{fmt}` is incompatible with `{QUANT_METHOD_FIELD}`=`{method}`. Auto-fix method to `{new_method}`.")
                    normalized[QUANT_METHOD_FIELD] = new_method

        if format_auto_inferred:
            log.info(f"QuantizeConfig: `{FORMAT_FIELD_CHECKPOINT}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}")

        if normalized[FORMAT_FIELD_CODE] in {FORMAT.BITBLAS}:
            # AWQ and Marlin do not reorder the rows.
            normalized["desc_act"] = False

        if "sym" not in normalized:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )

        meta_payload = normalized.get(META_FIELD)
        if "failsafe" not in normalized and isinstance(meta_payload, dict) and "failsafe" in meta_payload:
            normalized["failsafe"] = meta_payload.get("failsafe")
        if "hessian" not in normalized and isinstance(meta_payload, dict) and "hessian" in meta_payload:
            normalized["hessian"] = meta_payload.get("hessian")
        if "gptaq" not in normalized and isinstance(meta_payload, dict) and "gptaq" in meta_payload:
            normalized["gptaq"] = meta_payload.get("gptaq")
        if "gc_mode" not in normalized and isinstance(meta_payload, dict) and "gc_mode" in meta_payload:
            normalized["gc_mode"] = meta_payload.get("gc_mode")
        if (
            "wait_for_submodule_finalizers" not in normalized
            and isinstance(meta_payload, dict)
            and "wait_for_submodule_finalizers" in meta_payload
        ):
            normalized["wait_for_submodule_finalizers"] = meta_payload.get("wait_for_submodule_finalizers")
        if (
            "auto_forward_data_parallel" not in normalized
            and isinstance(meta_payload, dict)
            and "auto_forward_data_parallel" in meta_payload
        ):
            normalized["auto_forward_data_parallel"] = meta_payload.get("auto_forward_data_parallel")

        cfg = cls(**normalized)

        if quantize_cfg.get(AWQ_PACKING_BACKEND_FIELD) and quantize_cfg[AWQ_PACKING_BACKEND_FIELD] == "llm-awq":
            cfg.quant_method = METHOD.AWQ
            cfg.format = FORMAT.LLM_AWQ
            cfg.pack_dtype = torch.int16
            log.info(
                "Detected llm-awq quantization format; FORMAT automatically set to FORMAT.LLM_AWQ."
            )

        return cfg

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

    def to_dict(self):
        smooth = None
        if self.failsafe is not None and self.failsafe.smooth is not None:
            payload = {"type": self.failsafe.smooth.name}
            payload["group_size_threshold"] = self.failsafe.smooth.group_size_threshold
            if isinstance(self.failsafe.smooth, SmoothPercentile):
                payload["percentile"] = self.failsafe.smooth.percentile
            elif isinstance(self.failsafe.smooth, SmoothPercentileAsymmetric):
                payload["low"] = self.failsafe.smooth.low
                payload["high"] = self.failsafe.smooth.high
            elif isinstance(self.failsafe.smooth, SmoothMAD):
                payload["k"] = self.failsafe.smooth.k
            elif isinstance(self.failsafe.smooth, SmoothMSE):
                payload["steps"] = self.failsafe.smooth.steps
                payload["maxshrink"] = self.failsafe.smooth.maxshrink
            elif isinstance(self.failsafe.smooth, SmoothOutlier):
                payload["pct"] = self.failsafe.smooth.pct
            elif isinstance(self.failsafe.smooth, SmoothSoftNorm):
                payload["k"] = self.failsafe.smooth.k
            elif isinstance(self.failsafe.smooth, SmoothLog):
                payload["percentile"] = self.failsafe.smooth.percentile
                payload["mu"] = self.failsafe.smooth.mu
            elif isinstance(self.failsafe.smooth, SmoothRowCol):
                payload["axis"] = self.failsafe.smooth.axis
            smooth = payload

        meta_payload = dict(self.meta) if self.meta else {}
        if self.moe:
            meta_payload["moe"] = self.moe.to_dict()

        if self.failsafe is None:
            meta_payload["failsafe"] = None
        else:
            meta_payload["failsafe"] = {
                "strategy": self.failsafe.strategy.value if isinstance(self.failsafe.strategy, FailSafeStrategy) else self.failsafe.strategy,
                "threshold": self.failsafe.threshold,
                "smooth": smooth,
            }

        if self.gptaq is None:
            meta_payload["gptaq"] = None
        else:
            device = self.gptaq.device
            device_value = device if isinstance(device, str) else str(device)
            meta_payload["gptaq"] = {
                "alpha": self.gptaq.alpha,
                "device": device_value,
            }
        meta_payload["offload_to_disk"] = self.offload_to_disk
        meta_payload["offload_to_disk_path"] = self.offload_to_disk_path
        meta_payload["pack_impl"] = self.pack_impl
        meta_payload["mse"] = self.mse
        meta_payload["mock_quantization"] = self.mock_quantization
        meta_payload["act_group_aware"] = self.act_group_aware
        meta_payload["gc_mode"] = self.gc_mode
        meta_payload["wait_for_submodule_finalizers"] = self.wait_for_submodule_finalizers
        meta_payload["auto_forward_data_parallel"] = self.auto_forward_data_parallel
        meta_payload["hessian"] = {
            "chunk_size": self.hessian.chunk_size,
            "chunk_bytes": self.hessian.chunk_bytes,
            "staging_dtype": str(self.hessian.staging_dtype).split(".")[-1],
        }
        meta_payload["vram_strategy"] = (
            self.vram_strategy.value if isinstance(self.vram_strategy, VramStrategy) else self.vram_strategy
        )

        out = {
            "bits": self.bits,
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "lm_head": self.lm_head,
            QUANT_METHOD_FIELD:self.quant_method,
            FORMAT_FIELD_CHECKPOINT: self.format,
            # torch.dtype convert to string
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: meta_payload,
            # DO NOT EXPORT Adapter to config/json since adapter can be swapped out/in
            # ADAPTER_FIELD: self.adapter.to_dict() if self.adapter else None,
            # DO NOT EXPORT compute_device_filter since functions are not serializable
        }

        if self.quant_method == METHOD.AWQ:
            out["zero_point"] = not self.sym
            # awq compat with vllm/sglang/transformers loaders
            out["version"] = self.format
            out[FORMAT_FIELD_CODE] = self.format
        else:
            out["sym"] = self.sym
        if self.quant_method == METHOD.GPTQ:
            out[FORMAT_FIELD_CODE] = self.format

        dynamic = out["dynamic"]
        if dynamic:
            # dynamic adapter config is only used in the quantize phase and is deleted when saving.
            for _, v in dynamic.items():
                v.pop("adapter", None)

        # simplify: clean keys where the value is None or empty [list, dict]
        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}

        dict_scale_dtype_to_str(out)
        return out

     # TODO FIX ME, g_idx int32 per infeature but infeature count is per module
    def calculate_bits_per_weight(self):
        if self.group_size != -1:
            # naive bits is
            #mlp.down_proj.g_idx: I32
            #mlp.down_proj.qweight: I32
            #mlp.down_proj.qzeros: I32
            #mlp.down_proj.scales: F16
            per_group_bits = self.group_size * self.bits # qweight: packed by group_size
            per_group_bits += 16 # scales fp16: one per group
            per_group_bits += self.bits # qzeros: one per group
            # FIX ME: g_idx is I32, one per infeature
            per_group_bits += 4  # ESTIMATE for g_idx int32: one per features/group_size item
            bpw = per_group_bits / self.group_size

            # normally g_idx (int32 allocated one per in_feature) is allocated in device memory
            # but each module may have different infeatures we don't have enouch ctx here, use estimated `0.1` for now
            bpw += 0.1
        else:
            # there is only one scale int32 + one qzero int32 per entire module so overall it contributes to close to 0 bpw
            bpw = self.bits
        log.info(f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, group_size: {self.group_size}]")

    def moe_routing_override(self, num_experts: int) -> Union[int, None]:
        if self.moe is None:
            return None
        return self.moe.routing_override(num_experts)

    def moe_routing_bypass(self) -> bool:
        if self.moe is None:
            return False
        return self.moe.routing_bypass()

# deprecated: will be removed in future update
@dataclass
class BaseQuantizeConfig(QuantizeConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warn("QuantizeConfig: BaseQuantizeConfig is re-named and pending deprecation. Please use `QuantizeConfig` instead.")
