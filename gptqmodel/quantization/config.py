# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os.path
from dataclasses import dataclass, field, fields
from enum import Enum
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import pcre as re
import torch
from packaging import version
from random_word import random_word

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger


log = setup_logger()

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "format"
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]

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

META_FIELD_V2_ENABLED = "v2"
META_FIELD_V2_ALPHA = "v2_alpha"
META_FIELD_V2_MEMORY_DEVICE = "v2_memory_device"

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


# quant methods
class METHOD(str, Enum):
    GPTQ = "gptq"
    QQQ = "qqq"
    AWQ = "awq"


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
                    if isinstance(sub_value, Dict):
                        return sub_value.get(sub_key, default)
                    else:
                        log.info(f"QuantConfig: Dynamic `sub_key`: `{sub_key}` failed extraction from  `sub_value`: `{sub_value}`")
                else:
                    return overrides.get(key, default)
    return default

@dataclass
class QuantizeConfig():
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, bool]]]] = field(default=None)

    # 128 offer good balance between inference speed, vram usage (bpw), and quality
    # use 32 for highest quality with slower inference and higher vram usage
    group_size: int = field(default=128)

    # increase damp if NaN is encountered during `.quantize()` and/or increase calib dataset size
    damp_percent: float = field(default=None)
    damp_auto_increment: float = field(default=None)

    desc_act: Optional[bool] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)
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

    # deprecated: only used for compat
    is_marlin_format: bool = False

    # use mock quantization to quantize module so the gptq process can continue and not fail
    fail_safe: bool = field(default=False)

    # gptq v2* only:
    v2: bool = field(default=False)
    v2_alpha: float = field(default=0.25)
    v2_memory_device: str = field(default="auto")

    # awq only:
    zero_point: bool = field(default=True)

    # gptq only:
    # skip all heavy computations for testing model loading
    mock_quantization: bool = field(default=False, metadata={"help": "Skip heavy computations for fast model loading validation"})

    # Hessian accumulation controls (GPTQ only)
    hessian_chunk_size: Optional[int] = field(default=None, metadata={"help": "Maximum rows per Hessian chunk"})
    hessian_chunk_bytes: Optional[int] = field(default=None, metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"})
    hessian_use_bfloat16_staging: bool = field(default=False, metadata={"help": "Stage Hessian chunks in bfloat16 when supported"})

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

        # TODO FIXME qqq compat which didn't have checkpoint_format before merging to gptqmodel
        if self.quant_method == METHOD.QQQ and self.format != FORMAT.QQQ:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QQQ}`")
            self.format = FORMAT.QQQ

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

        if self.hessian_chunk_size is not None:
            if not isinstance(self.hessian_chunk_size, int):
                raise ValueError("QuantizeConfig: `hessian_chunk_size` must be an integer or None.")
            if self.hessian_chunk_size <= 0:
                raise ValueError("QuantizeConfig: `hessian_chunk_size` must be a positive integer.")

        if self.hessian_chunk_bytes is not None:
            if not isinstance(self.hessian_chunk_bytes, int):
                raise ValueError("QuantizeConfig: `hessian_chunk_bytes` must be an integer or None.")
            if self.hessian_chunk_bytes <= 0:
                raise ValueError("QuantizeConfig: `hessian_chunk_bytes` must be a positive integer amount of bytes.")

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
            randWords = random_word.RandomWords()
            path_key = f"{randWords.get_random_word()}-{randWords.get_random_word()}"
            self.offload_to_disk_path = f"./gptqmodel_offload/{path_key}/"
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to `{self.offload_to_disk_path}`")

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

    # is quantized model quantized or packed by gptqmodel version with v2 format code
    def is_quantized_by_v2(self) -> bool:
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
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if format_auto_inferred:
            log.info(f"QuantizeConfig: `{FORMAT_FIELD_CHECKPOINT}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}")

        if normalized[FORMAT_FIELD_CODE] in {FORMAT.BITBLAS}:
            # AWQ and Marlin do not reorder the rows.
            normalized["desc_act"] = False

        if "sym" not in normalized:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )

        return cls(**normalized)

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
        out = {
            "bits": self.bits,
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "lm_head": self.lm_head,
            QUANT_METHOD_FIELD:self.quant_method,
            FORMAT_FIELD_CHECKPOINT: self.format,
            # torch.dtype convert to string
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: self.meta,
            # DO NOT EXPORT Adapter to config/json since adapter can be swapped out/in
            # ADAPTER_FIELD: self.adapter.to_dict() if self.adapter else None,
        }

        if getattr(self, "pack_impl", "original") != "original":
            out["pack_impl"] = self.pack_impl

        # TODO FIXME: upstream gpt-qmodel config for awq recognition to transformers/sglang/vllm
        if self.quant_method == METHOD.AWQ:
            out["zero_point"] = self.zero_point
            # awq compat with vllm/sglang/transformers loaders
            out["version"] = self.format

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

# deprecated: will be removed in future update
@dataclass
class BaseQuantizeConfig(QuantizeConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warn("QuantizeConfig: BaseQuantizeConfig is re-named and pending deprecation. Please use `QuantizeConfig` instead.")
