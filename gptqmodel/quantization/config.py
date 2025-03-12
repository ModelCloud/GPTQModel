# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os.path
import re
from dataclasses import dataclass, field, fields
from importlib.metadata import version as pkg_version
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger

log = setup_logger()

FORMAT_FIELD_CODE = "format"
FORMAT_FIELD_JSON = "checkpoint_format"
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

ADAPTER_FIELD = "adapter"

# pkg names
PKG_AUTO_ROUND = "auto-round"

# saved formats
class FORMAT:
    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    IPEX = "ipex"
    QQQ = "qqq"


# quant methods
class QUANT_METHOD:
    GPTQ = "gptq"
    AUTO_ROUND = "auto_round"
    QQQ = "qqq"


QUANT_METHOD_FORMAT_MAPPING = {
    QUANT_METHOD.GPTQ: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
        FORMAT.IPEX,
    },
    QUANT_METHOD.AUTO_ROUND: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    },
    QUANT_METHOD.QQQ: {
        FORMAT.QQQ,
    },
}

# inference only methods should go here
QUANTIZE_BLACK_LIST = {}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": "bits",
    "q_group_size": "group_size",
    # map format field (checkpoint_format) to class/code (format)
    FORMAT_FIELD_JSON: FORMAT_FIELD_CODE,
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

    # increase damp if NaN is encountred during `.quantize()` and/or increase calib dataset size
    damp_percent: float = field(default=0.01)
    damp_auto_increment: float = field(default=0.0025)

    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    quant_method: QUANT_METHOD = field(default=QUANT_METHOD.GPTQ)

    # default to gptq v1 format for maximum compat with 3rd party inference libs with minimal loss vs v2
    # if you inference with gptqmodel, save to gptq_v2 format for best result
    format: FORMAT = field(default=FORMAT.GPTQ)

    # mean square error calculation: may reduce error loss for some models
    mse: float = field(default=0.0)

    # parallel packing will make ~40% speedup for many models, but may cause OOM in some large models
    # if OOM, can set to False
    parallel_packing: bool = field(default=True)

    # properties that do not directly contributes to quantization or quant inference should be placed in meta
    # i.e. quantizer tool (producer) + version, timestamp, entity who made the quant, etc
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # pending used field
    adapter: Optional[Union[Dict[str, Any], Lora]] = field(default=None)

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    is_marlin_format: bool = False

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
        if self.quant_method == QUANT_METHOD.QQQ and self.format != FORMAT.QQQ:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QQQ}`")
            self.format = FORMAT.QQQ

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
                        raise ValueError("QuantizeConfig: `group_size` must in the value set of `[-1, 16, 32, 64, 128]`.")

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("QuantizeConfig: `group_size` must in the value set of `[-1, 16, 32, 64, 128]`.")

        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")

        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

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
        valid_formats = {FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.MARLIN, FORMAT.BITBLAS, FORMAT.IPEX}
        format_auto_inferred = False
        # compat: format can be passed in via from_quantized() if field missing from json
        if format:
            if format not in valid_formats:
                raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if quantize_cfg.get(FORMAT_FIELD_JSON):
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        # compat: warn if checkpoint_format is missing
        elif quantize_cfg.get(FORMAT_FIELD_JSON) is None:
            format_auto_inferred = True

        field_names = [field.name for field in fields(cls)]

        normalized = {
            QUANT_METHOD_FIELD: QUANT_METHOD.GPTQ,
            # compat: default to gptq(v1) when loading models
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }
        for key, val in quantize_cfg.items():
            key = key.lower()

            # remap keys according to compat map
            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]

            if key == FORMAT_FIELD_JSON:
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
                elif val not in {QUANT_METHOD.GPTQ, QUANT_METHOD.AUTO_ROUND, QUANT_METHOD.QQQ}:
                    raise ValueError(f"QuantizeConfig: Unknown quantization method: `{val}`.")
                else:
                    normalized[QUANT_METHOD_FIELD] = val
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if format_auto_inferred:
            log.info(f"QuantizeConfig: `{FORMAT_FIELD_JSON}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}")

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
            FORMAT_FIELD_JSON: self.format,
            # torch.dtype convert to string
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: self.meta,
            # DO NOT EXPORT Adapter to config/json since adapter can be swapped out/in
            # ADAPTER_FIELD: self.adapter.to_dict() if self.adapter else None,
        }

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

@dataclass
class AutoRoundQuantizeConfig(QuantizeConfig):
    layer_config: dict = field(default_factory=dict)
    enable_full_range: bool = False  ##for symmetric, TODO support later
    batch_size: int = 1
    amp: bool = True
    lr_scheduler = None
    enable_quanted_input: bool = True
    enable_minmax_tuning: bool = True
    lr: float = None
    minmax_lr: float = None
    low_gpu_mem_usage: bool = False
    iters: int = 200
    sampler: str = "rand"
    seed: int = 42
    nblocks: int = 1
    gradient_accumulate_steps: int = 1
    not_use_best_mse: bool = False
    dynamic_max_gap: int = -1
    data_type: str = "int"  ##only support int for now
    scale_dtype: str = "fp16"
    quant_method: str = QUANT_METHOD.AUTO_ROUND

    def to_dict(self):
        # inject auto-round specific meta data
        self.meta_set("auto_round", pkg_version(PKG_AUTO_ROUND))
        self.meta_set("enable_full_range", self.enable_full_range)
        layer_config = copy.deepcopy(self.layer_config)
        for key in layer_config:
            info = layer_config[key]
            # layer_config will store "scale" and "zp" Tensors, which cannot be serialized by json.
            # see AutoRound.dump_qinfo_to_layer_config()
            info.pop("scale", None)
            info.pop("zp", None)
        self.meta_set("layer_config", layer_config)
        self.meta_set("batch_size", self.batch_size)
        self.meta_set("amp", self.amp)
        self.meta_set("lr_scheduler", self.lr_scheduler)
        self.meta_set("enable_quanted_input", self.enable_quanted_input)
        self.meta_set("enable_minmax_tuning", self.enable_minmax_tuning)
        self.meta_set("lr", self.lr)
        self.meta_set("minmax_lr", self.minmax_lr)
        self.meta_set("low_gpu_mem_usage", self.low_gpu_mem_usage)
        self.meta_set("iters", self.iters)
        self.meta_set("sampler", self.sampler)
        self.meta_set("seed", self.seed)
        self.meta_set("nblocks", self.nblocks)
        self.meta_set("gradient_accumulate_steps", self.gradient_accumulate_steps)
        self.meta_set("not_use_best_mse", self.not_use_best_mse)
        self.meta_set("dynamic_max_gap", self.dynamic_max_gap)
        self.meta_set("data_type", self.data_type)
        self.meta_set("scale_dtype", self.scale_dtype)

        r = super().to_dict()

        # override quant_method from AUTO_ROUND to GPTQ in json output for max compat with vllm/sglang
        r[QUANT_METHOD_FIELD] = QUANT_METHOD.GPTQ
        return r

# deprecated: will be removed in future update
@dataclass
class BaseQuantizeConfig(QuantizeConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warn("QuantizeConfig: BaseQuantizeConfig is re-named and pending deprecation. Please use `QuantizeConfig` instead.")
