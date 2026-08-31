# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import collections
import functools
import json
import math
import operator
import os
import shutil
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import accelerate
import pcre
import torch
import torch.nn as nn
import transformers
from packaging import version
from safetensors import safe_open
from torch.nn.modules.conv import _ConvNd
from transformers import PretrainedConfig
from transformers.pytorch_utils import id_tensor_storage
from transformers.utils.hub import cached_file

from ..adapter.adapter import Adapter
from ..looper.named_module import NamedModule
from ..models._const import (
    CPU,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    SUPPORTS_MODULE_TYPES,
)
from ..nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from ..nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from ..nn_modules.qlinear.torch import TorchQuantEmbeddings
from ..quantization import FORMAT, QuantizeConfig
from ..quantization.config import (
    FORMAT_FIELD_CODE,
    METHOD,
    _normalize_bitsandbytes_block_size,
    _normalize_bitsandbytes_format,
    _normalize_fp8_fmt,
    _normalize_fp8_scale_semantics,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
    _normalize_quant_bits,
    dynamic_get,
    quant_bits_width,
    resolve_quant_format,
)
from . import has_gil_disabled
from .backend import BACKEND, normalize_backend
from .ctx import ctx
from .device import get_device
from .hf import get_hf_config_dtype
from .hub import hf_hub_download, model_info
from .importer import select_quant_linear
from .logger import log_time_block, setup_logger
from .model_dequant import _correct_gptq_v1_qzeros, _revert_gptq_v1_qzeros_correction
from .torch import HAS_CUDA, torch_empty_cache


log = setup_logger()
_REQUIRES_VERSION_RE = pcre.compile(r"(<=|>=|==|<|>)\s*([\d\.]+)")


_DTYPE_SAFE_MAP = {
    torch.float32: ("F32", 4),
    torch.float16: ("F16", 2),
    torch.float64: ("F64", 8),
    torch.bfloat16: ("BF16", 2),
    torch.int64: ("I64", 8),
    torch.int32: ("I32", 4),
    torch.int16: ("I16", 2),
    torch.int8: ("I8", 1),
    torch.uint8: ("U8", 1),
    torch.bool: ("BOOL", 1),
}

if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_SAFE_MAP[torch.float8_e4m3fn] = ("F8_E4M3", 1)
if hasattr(torch, "float8_e5m2"):
    _DTYPE_SAFE_MAP[torch.float8_e5m2] = ("F8_E5M2", 1)

_FLOAT8_DTYPE_NAMES = tuple(
    name
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)
_FLOAT4_PACKED_DTYPE_NAMES = tuple(
    name for name in ("float4_e2m1fn_x2",) if hasattr(torch, name)
)

# Byte-size fallbacks keep ancillary metadata math working for torch floatx
# dtypes even when the current safetensors header schema cannot serialize them.
_DTYPE_NUM_BYTES = dict.fromkeys((*[getattr(torch, name) for name in _FLOAT8_DTYPE_NAMES], *[getattr(torch, name) for name in _FLOAT4_PACKED_DTYPE_NAMES]), 1)


_DTYPE_STR_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "float64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int": torch.int32,
    "int16": torch.int16,
    "short": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

for name in _FLOAT8_DTYPE_NAMES:
    dtype = getattr(torch, name)
    _DTYPE_STR_MAP[name] = dtype
    _DTYPE_STR_MAP[f"f8_{name.removeprefix('float8_')}"] = dtype

if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_STR_MAP["f8_e4m3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _DTYPE_STR_MAP["f8_e5m2"] = torch.float8_e5m2
if hasattr(torch, "float8_e8m0fnu"):
    _DTYPE_STR_MAP["float8_e8m0"] = torch.float8_e8m0fnu
    _DTYPE_STR_MAP["f8_e8m0"] = torch.float8_e8m0fnu

for name in _FLOAT4_PACKED_DTYPE_NAMES:
    dtype = getattr(torch, name)
    _DTYPE_STR_MAP[name] = dtype
    _DTYPE_STR_MAP[f"f4_{name.removeprefix('float4_')}"] = dtype

MoETopKState = List[Tuple[nn.Module, str, int]]

MOE_TOPK_FIELD_NAMES = [
    "top_k",
    "moe_k", # ernie4_5_vl_moe
]

MOE_NUM_EXPERTS_FIELD_NAMES = [
    "num_experts",
    "moe_num_experts",  # ernie4_5_vl_moe
]


def _torch_dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_SAFE_MAP:
        return _DTYPE_SAFE_MAP[dtype][1]
    if dtype in _DTYPE_NUM_BYTES:
        return _DTYPE_NUM_BYTES[dtype]
    raise NotImplementedError(f"Unsupported dtype for safetensors export: {dtype}")


def _torch_dtype_to_safetensors(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_SAFE_MAP:
        raise NotImplementedError(f"Unsupported dtype for safetensors export: {dtype}")
    return _DTYPE_SAFE_MAP[dtype][0]


def _dtype_string_to_torch(dtype_str: Optional[str], fallback: torch.dtype) -> torch.dtype:
    if dtype_str is None:
        return fallback
    key = dtype_str.lower()
    return _DTYPE_STR_MAP.get(key, fallback)


@dataclass(frozen=True)
class OffloadTensorRef:
    path: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    format: str  # 'dat' or 'safetensors'
    weight_name: Optional[str] = None
    data_offsets: Optional[Tuple[int, int]] = None

    @property
    def num_bytes(self) -> int:
        return _torch_dtype_num_bytes(self.torch_dtype) * math.prod(self.shape or (1,))


@dataclass
class TensorSource:
    name: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    source: Union[torch.Tensor, OffloadTensorRef]

    @property
    def num_bytes(self) -> int:
        return _torch_dtype_num_bytes(self.torch_dtype) * math.prod(self.shape or (1,))

def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def _module_has_meta_tensors(module: nn.Module) -> bool:
    for param in module.parameters(recurse=True):
        if getattr(param, "is_meta", False) or param.device.type == "meta":
            return True
    for buf in module.buffers(recurse=True):
        if getattr(buf, "is_meta", False) or buf.device.type == "meta":
            return True
    return False


def move_to(obj: torch.Tensor | nn.Module, device: torch.device, dtype: torch.dtype = None):
    if isinstance(obj, nn.Module) and _module_has_meta_tensors(obj):
        if not accelerate.utils.has_offloaded_params(obj):
            raise NotImplementedError(
                "Cannot move a module that still contains meta tensors without offload hooks. "
                "Materialize it first before calling move_to()."
            )

        # Accelerate disk-offloaded modules keep meta placeholders until they are
        # explicitly restored, so materialize those leaves before the device move.
        from .offload import undo_offload_to_disk

        return undo_offload_to_disk(obj, device=device, dtype=dtype)

    if get_device(obj) != device or dtype is not None:
        obj = obj.to(device=device, dtype=dtype, non_blocking=False)

    return obj


def nested_move_to(v, device, dtype: torch.dtype = None):
    if isinstance(v, torch.Tensor):
        return move_to(v, device=device, dtype=dtype)

    elif isinstance(v, dict):
        return {
            k: nested_move_to(val, device=device, dtype=dtype)
            for k, val in v.items()
        }

    elif isinstance(v, (list, tuple)):
        return type(v)(
            nested_move_to(e, device=device, dtype=dtype)
            for e in v
        )

    else:
        return v


def find_modules(module: nn.Module, layers=None, name: str="") -> Dict[str, nn.Module]:
    if not layers:
        layers = SUPPORTS_MODULE_TYPES

    if isinstance(module, tuple(layers)):
       return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_by_name(module, child_name):
    # get the child module by its name relative to the module
    for name, m in module.named_modules():
        if name == child_name:
            return m
    raise ValueError(f"Cannot find child_name {child_name} in module {module}")


def get_module_by_name_prefix(model, module_name: Union[List[str], str]):
    module_name_list = module_name if isinstance(module_name, list) else [module_name]
    for name, module in model.named_modules():
        for prefix in module_name_list:
            if name.startswith(prefix):
                return module, prefix

    return None, ""


def get_layers_with_prefixes(model, module_name: Union[List[str], str]):
    """Resolve one or more layer containers into a flat layer list plus per-layer names.

    Existing model definitions often expose a single `layers` ModuleList, but some
    architectures split the decoder into multiple stacks that still need to be
    quantized as one ordered sequence. This helper flattens every matching layer
    container while preserving the source prefix for each layer.
    """

    module_name_list = module_name if isinstance(module_name, list) else [module_name]

    layers = []
    layer_names = []
    seen_container_ids = set()

    for prefix in module_name_list:
        if not prefix:
            continue
        try:
            module = get_module_by_name(model, prefix)
        except ValueError:
            module = None
        if module is None:
            continue

        container_id = id(module)
        if container_id in seen_container_ids:
            continue
        seen_container_ids.add(container_id)

        if isinstance(module, (nn.ModuleList, list, tuple)):
            for local_index, layer in enumerate(module):
                if isinstance(layer, nn.Module):
                    layers.append(layer)
                    layer_names.append(f"{prefix}.{local_index}")
        elif isinstance(module, nn.Module):
            layers.append(module)
            layer_names.append(prefix)

    if layers:
        return layers, layer_names

    module, prefix = get_module_by_name_prefix(model, module_name_list)
    if module is None:
        return None, []
    if isinstance(module, (nn.ModuleList, list, tuple)):
        return list(module), [f"{prefix}.{index}" for index in range(len(module))]
    return [module], [prefix]


def get_layer_name(layer_names: Union[List[str], str, None], layer_index: int) -> str:
    """Return the concrete full layer path for one flattened layer index."""

    if not layer_names:
        return ""
    if isinstance(layer_names, str):
        return layer_names
    if layer_index < len(layer_names):
        return layer_names[layer_index]
    return layer_names[-1]


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module

def make_quant(
    module,
    qcfg: QuantizeConfig,
    quant_result: Dict[str, Dict[str, Any]],
    backend: BACKEND,
    lm_head_name: str,
    pack: bool = False,
    device: DEVICE = None,
    from_quantized: bool = False,
    dtype: Optional[torch.dtype] = None,
    is_sharded: bool = False,
) -> Type[BaseQuantLinear]:

    bits = qcfg.runtime_bits
    group_size =qcfg.group_size
    extension = qcfg.adapter
    format = resolve_quant_format(qcfg.format, qcfg.method)
    desc_act = qcfg.desc_act
    sym = qcfg.sym
    dynamic = qcfg.dynamic
    pack_dtype = qcfg.pack_dtype
    init_kwargs = qcfg.quant_linear_init_kwargs()

    export_quant_method = qcfg.export_quant_method()
    backend = normalize_backend(backend, quant_method=export_quant_method)

    # BitBLAS-native checkpoints can load directly. Other formats need a compatible preload kernel first.
    if not pack and backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
        if format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
            backend = BACKEND.GPTQ_TORCH
        elif qcfg.quant_method == METHOD.AWQ and format == FORMAT.GEMM:
            backend = BACKEND.AWQ_TORCH

    # returns multiple validated kernels
    quant_linear_candidates = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        quant_method=export_quant_method,
        pack=pack,
        dynamic=dynamic,
        device=device,
        pack_dtype=pack_dtype,
        dtype=dtype,
        multi_select=True,
        adapter=extension,
        is_sharded=is_sharded,
    )

    log.info(f"Kernel: candidates -> `[{', '.join(cls.__name__ for cls in quant_linear_candidates)}]`")

    # Per-module kernel selection: each submodule picks the first candidate that
    # validates against its effective (possibly dynamic-overridden) quant config.
    linear_cls = create_quant_layer(
        linear_candidates=quant_linear_candidates,
        bits=bits,
        desc_act=desc_act,
        dynamic=dynamic,
        group_size=group_size,
        module=module,
        sym=sym,
        device=device,
        quant_result=quant_result,
        lm_head_name=lm_head_name,
        pack_dtype=pack_dtype,
        backend=backend,
        adapter=qcfg.adapter,
        format=format,
        init_kwargs=init_kwargs,
        dtype=dtype,
    )
    log.info(f"Kernel: selected -> `{linear_cls.__name__}`.")
    return linear_cls

def create_quant_module(
    name: str,
    linear_cls: Type[BaseQuantLinear],
    bits,
    desc_act: bool,
    dynamic,
    group_size: int,
    module: nn.Module,
    submodule: nn.Module,
    sym: bool,
    device: DEVICE,
    lm_head_name: str,
    pack_dtype: torch.dtype,
    format: FORMAT = FORMAT.GPTQ,
    backend: BACKEND = BACKEND.AUTO,
    register_buffers: bool = True,
    adapter: Optional[Adapter] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,

):
    # unwrap named module
    if isinstance(submodule, NamedModule):
        # print(f"offloading named module: {module.full_name}")
        submodule = submodule.module

    if isinstance(submodule, nn.Embedding):
        linear_cls = TorchQuantEmbeddings

    # submodule may be BaseQuantLinear, and the next QuantLinear is selected because of in_features/out_features
    # mismatch and other reasons.
    # In this case, need to call list_buffer() to get the device.
    if not isinstance(submodule, BaseQuantLinear):
        ori_layer_device = next(submodule.parameters()).device
    else:
        ori_layer_device = submodule.list_buffers()[0].device

    if ori_layer_device.type != CPU.type:
        raise AssertionError(
            f"Expected `{name}` to reside on CPU during quant module creation, "
            f"but found tensors on `{ori_layer_device}`."
        )

    if isinstance(submodule, NamedModule):
        in_features = submodule.state.get("in_features")
        out_features = submodule.state.get("out_features")
    elif isinstance(submodule, nn.Linear):
        in_features = submodule.in_features
        out_features = submodule.out_features
    elif isinstance(submodule, nn.Embedding):
        in_features = submodule.num_embeddings
        out_features = submodule.embedding_dim
    elif isinstance(submodule, _ConvNd):
        in_features = submodule.in_channels
        out_features = submodule.out_channels
    elif isinstance(submodule, transformers.Conv1D):
        in_features = submodule.weight.shape[0]
        out_features = submodule.weight.shape[1]
    elif isinstance(submodule, BaseQuantLinear):
        # if submodule is already a quant layer, we need to get in_features and out_features from the submodule
        in_features = submodule.in_features
        out_features = submodule.out_features
    else:
        raise NotImplementedError(f"Unsupported module {submodule}")

    bias = submodule.bias is not None if hasattr(submodule, "bias") else False

    # need copies as dynamic config may override these in for loop
    tmp_bits = _normalize_quant_bits(bits, format_value=format)
    tmp_group_size = group_size
    tmp_desc_act = desc_act
    tmp_sym = sym
    tmp_pack_dtype = pack_dtype
    tmp_init_kwargs = dict(init_kwargs or {})

    # dynamic bits, group_size, sym, pack_dtype for each layer/module
    if dynamic is not None:
        overrides = dynamic_get(dynamic=dynamic, module_name=name)
        # negative module match, skip this module
        if overrides == False:  # noqa: E712
            return

        # positive module match
        if overrides:
            # override base QuantizeConfig for every quant config key/value
            tmp_bits = _normalize_quant_bits(overrides.get("bits", bits), format_value=format)
            tmp_group_size = overrides.get("group_size", group_size)
            tmp_desc_act = overrides.get("desc_act", desc_act)
            tmp_sym = overrides.get("sym", sym)
            tmp_pack_dtype = overrides.get("pack_dtype", pack_dtype)

            if format == FORMAT.FP8:
                fp8_format_override = overrides.get(FORMAT_FIELD_CODE, overrides.get("fmt"))
                if fp8_format_override is not None:
                    tmp_init_kwargs["format"] = _normalize_fp8_fmt(fp8_format_override)
                block_size_override = overrides.get(
                    "weight_block_size",
                    tmp_init_kwargs.get("weight_block_size"),
                )
                normalized_block_size = _normalize_fp8_weight_block_size(block_size_override)
                if "weight_scale_method" in overrides or block_size_override is not None:
                    tmp_init_kwargs["weight_scale_method"] = _normalize_fp8_weight_scale_method(
                        overrides.get(
                            "weight_scale_method",
                            tmp_init_kwargs.get("weight_scale_method"),
                        ),
                        weight_block_size=normalized_block_size,
                    )
                if "weight_scale_semantics" in overrides:
                    tmp_init_kwargs["weight_scale_semantics"] = _normalize_fp8_scale_semantics(
                        overrides["weight_scale_semantics"]
                    )
                if "weight_block_size" in overrides:
                    tmp_init_kwargs["weight_block_size"] = normalized_block_size
            elif format == FORMAT.BITSANDBYTES:
                raw_format = overrides.get(FORMAT_FIELD_CODE, overrides.get("bnb_quant_type"))
                if raw_format is not None:
                    tmp_init_kwargs["format"] = _normalize_bitsandbytes_format(
                        raw_format,
                        bits=quant_bits_width(tmp_bits),
                    )
                if "block_size" in overrides or "bnb_block_size" in overrides:
                    tmp_init_kwargs["block_size"] = _normalize_bitsandbytes_block_size(
                        overrides.get("block_size", overrides.get("bnb_block_size"))
                    )
                if "compress_statistics" in overrides or "bnb_compress_statistics" in overrides:
                    tmp_init_kwargs["compress_statistics"] = bool(
                        overrides.get("compress_statistics", overrides.get("bnb_compress_statistics"))
                    )

    validate_bits = quant_bits_width(tmp_bits)
    constructor_bits = tmp_bits if getattr(linear_cls, "QUANT_TYPE", None) == "gguf" else validate_bits

    # GPTQ modules need the checkpoint format to select between the continuous
    # (gptq/gptq_v2) and planar (gptq_p) packed layouts.
    if issubclass(linear_cls, GPTQQuantLinear):
        tmp_init_kwargs.setdefault("format", format)

    # when loading a quantized model, device is the target passed through the GPT-QModel load path
    # check in_features and out_features validate
    _, err = linear_cls.validate(
        bits=validate_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        pack_dtype=tmp_pack_dtype,
        dtype=dtype,
        in_features=in_features,
        out_features=out_features,
        device=DEVICE(device) if isinstance(device, str) else device,
        adapter=adapter, # TODO FIX ME..need to pass Eora if loaded
    )
    if err is not None:
        raise err

    new_layer = linear_cls(
        bits=constructor_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=tmp_pack_dtype,
        bias=bias,
        dtype=dtype,
        #weight_dtype=submodule.qweight.dtype if isinstance(submodule, BaseQuantLinear) else submodule.weight.dtype,
        name=name,
        lm_head_name=lm_head_name,
        backend=backend,
        register_buffers=register_buffers,
        adapter=adapter,
        **tmp_init_kwargs,
    )
    new_layer.device = ori_layer_device
    recurse_setattr(module, name, new_layer.to(ori_layer_device))

def create_quant_layer(
        linear_candidates: List[Type[BaseQuantLinear]],
        bits,
        desc_act: bool,
        dynamic,
        group_size: int,
        quant_result: Dict[str, Dict[str, Any]],
        module,
        sym: bool,
        device: DEVICE,
        lm_head_name: str,
        pack_dtype: torch.dtype,
        backend: BACKEND,
        adapter: Optional[Adapter] = None,
        format: FORMAT = FORMAT.GPTQ,
        init_kwargs: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,

) -> Type[BaseQuantLinear]:
    if any(isinstance(module, candidate) for candidate in linear_candidates):
        return type(module)

    selected_counts = dict.fromkeys(linear_candidates, 0)
    selected_counts[TorchQuantEmbeddings] = 0
    for name, submodule in module.named_modules():
        # skip non-quantized modules
        if name not in quant_result:
            continue

        candidates = [TorchQuantEmbeddings] if isinstance(submodule, nn.Embedding) else linear_candidates
        last_error = None
        for qlinear_cls in candidates:
            try:
                create_quant_module(
                    name=name,
                    linear_cls=qlinear_cls,
                    bits=bits,
                    desc_act=desc_act,
                    dynamic=dynamic,
                    group_size=group_size,
                    module=module,
                    submodule=submodule,
                    sym=sym,
                    device=device,
                    lm_head_name=lm_head_name,
                    pack_dtype=pack_dtype,
                    format=format,
                    backend=backend,
                    adapter=adapter,
                    init_kwargs=init_kwargs,
                    dtype=dtype,
                )
            except NotImplementedError as error:
                last_error = error
                if backend not in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
                    raise
                continue

            selected_counts[qlinear_cls] = selected_counts.get(qlinear_cls, 0) + 1
            break
        else:
            if last_error is not None:
                raise last_error
            raise ValueError(f"No compatible quant linear was found for module `{name}`.")

    selected_counts = {candidate: count for candidate, count in selected_counts.items() if count}
    if not selected_counts:
        raise ValueError(f"No compatible quant linear was found for this module: {module.__class__.__name__}")

    summary = ", ".join(f"{candidate.__name__}={count}" for candidate, count in selected_counts.items())
    log.info(f"Kernel: per-module selections -> `[{summary}]`")
    non_embedding_counts = {
        candidate: count
        for candidate, count in selected_counts.items()
        if candidate is not TorchQuantEmbeddings
    }
    return max(non_embedding_counts or selected_counts, key=(non_embedding_counts or selected_counts).get)

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v1_to_v2_format(
    model: nn.Module,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    if checkpoint_format == "gptq":
        # Skip v1 to v2 conversion when no loaded quant module requires v2.
        if not any(
            isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
            for m in model.modules()
        ):
            return model, False

        cfg = QuantizeConfig(bits=bits)
        return convert_gptq_v1_to_v2_format(model, cfg, qlinear_kernel), True
    else:
        return model, False

def convert_gptq_v1_to_v2_format_module(module: BaseQuantLinear, bits: int, pack_dtype: torch.dtype) -> nn.Module:
    assert isinstance(module, BaseQuantLinear)

    log.info.once("Format: Converting GPTQ v1 to v2")

    # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
    # additions here do not overflow.
    # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
    # overflow ~<=13% based on testing
    if bits == 2:
        if pack_dtype == torch.int64:
            module.qzeros.data += 0b0101010101010101010101010101010101010101010101010101010101010101
        elif pack_dtype == torch.int32:
            module.qzeros.data += 0b01010101010101010101010101010101
        elif pack_dtype == torch.int16:
            module.qzeros.data += 0b0101010101010101
        elif pack_dtype == torch.int8:
            module.qzeros.data += 0b01010101
    elif bits == 3:
        if pack_dtype == torch.int32:
            # GPTQ INT3 spills some zero-point bits across adjacent packed words.
            # Reuse the logical field shift so module conversion matches the
            # safetensor dequant path and the canonical INT3 pack layout.
            module.qzeros.data.copy_(
                _correct_gptq_v1_qzeros(module.qzeros.data, bits, planar=getattr(module, "planar", False))
            )
        else:
            # Only int32 packing words are used for GPTQ INT3 in actual checkpoints.
            # Keep the legacy constant-offset path for synthetic smaller word sizes.
            # range 0 offset
            if pack_dtype == torch.int64:
                offset = 0b0010010010010010010010010010010000100100100100100100100100100100
            elif pack_dtype == torch.int16:
                offset = 0b0010010010010010
            elif pack_dtype == torch.int8:
                offset = 0b00100100

            module.qzeros.data[:, range(0, module.qzeros.data.shape[1], 3)] += (
                offset
            )

            # range 1 offset
            if pack_dtype == torch.int64:
                offset = 0b1001001001001001001001001001001010010010010010010010010010010010
            elif pack_dtype == torch.int16:
                offset = 0b1001001001001001
            elif pack_dtype == torch.int8:
                offset = 0b10010010

            module.qzeros.data[:, range(1, module.qzeros.data.shape[1], 3)] += (
                offset
            )

            # range 2 offset
            if pack_dtype == torch.int64:
                offset = 0b0100100100100100100100100100100101001001001001001001001001001001
            elif pack_dtype == torch.int16:
                offset = 0b0100100100100100
            elif pack_dtype == torch.int8:
                offset = 0b01001001

            module.qzeros.data[:, range(2, module.qzeros.data.shape[1], 3)] += (
                offset
            )
    elif bits == 4:
        if pack_dtype == torch.int64:
            module.qzeros.data += 0b0001000100010001000100010001000100010001000100010001000100010001
        elif pack_dtype == torch.int32:
            module.qzeros.data += 0b00010001000100010001000100010001
        elif pack_dtype == torch.int16:
            module.qzeros.data += 0b0001000100010001
        elif pack_dtype == torch.int8:
            module.qzeros.data += 0b00010001
    elif bits == 8:
        if pack_dtype == torch.int64:
            module.qzeros.data += 0b0000000100000001000000010000000100000001000000010000000100000001
        elif pack_dtype == torch.int32:
            module.qzeros.data += 0b00000001000000010000000100000001
        elif pack_dtype == torch.int16:
            module.qzeros.data += 0b0000000100000001
        elif pack_dtype == torch.int8:
            module.qzeros.data += 0b00000001
    elif bits in (5, 6, 7):
        if pack_dtype != torch.int32:
            raise NotImplementedError(
                f"Planar {bits}-bit GPTQ only supports 32-bit packing words, got pack_dtype={pack_dtype}."
            )
        # Planar layouts spread each zero-point across bit planes, so shift the
        # decoded logical values instead of adding a packed-word constant.
        module.qzeros.data.copy_(_correct_gptq_v1_qzeros(module.qzeros.data, bits, planar=True))
    else:
        raise NotImplementedError("Only 2,3,4,5,6,7,8 bits are supported.")

    # change format id
    module.qzero_format(format=2)

# Optionally convert weight from gptq_v1 to v2 format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v1_to_v2_format(
    model,
    cfg: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):
    # skip v2 to v1 conversion when no loaded quant module requires v2
    if cfg.export_quant_method() == METHOD.GPTQ and not any(
        isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
        for m in model.modules()
    ):
        log.info(
            f"Format: Skipped v1 to v2 conversion; no selected kernel requires v2 (`{qlinear_kernel}`).")
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    time.time()
    log.info(
        f"Format: Converting `{FORMAT_FIELD_CODE}` from `{FORMAT.GPTQ}` to internal `{FORMAT.GPTQ_V2}`.")

    for _, submodule in model.named_modules():
        # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
        # additions here do not overflow.
        # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
        # overflow ~<=13% based on testing
        if isinstance(submodule, BaseQuantLinear) and getattr(submodule, "REQUIRES_FORMAT_V2", False):
            convert_gptq_v1_to_v2_format_module(
                module=submodule,
                bits=getattr(submodule, "bits", cfg.bits),
                pack_dtype=getattr(submodule, "pack_dtype", cfg.pack_dtype),
            )

        #log.info(f"Format: Conversion complete: {time.time() - t}s")

    return model

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v2_to_v1_format(
    model: nn.Module,
    sym: bool,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    # note: sym=False is valid for gptq_v2 for all gptqmodel and gptq(v1) for gptqmodel >= `0.9.0`
    if sym and checkpoint_format == "gptq_v2":
        quantize_config = QuantizeConfig(bits=bits)
        return convert_gptq_v2_to_v1_format(model, quantize_config, qlinear_kernel), True
    else:
        return model, False

def convert_gptq_v2_to_v1_format_module(
    module: BaseQuantLinear,
    quantize_config: QuantizeConfig,
):
    assert isinstance(module, BaseQuantLinear)

    log.info.once("Format: Converting GPTQ v2 to v1")

    bits = getattr(module, "bits", quantize_config.bits)
    pack_dtype = getattr(module, "pack_dtype", quantize_config.pack_dtype)
    if bits == 2:
        module.qzeros.data -= 0b01010101010101010101010101010101
    elif bits == 3:
        if pack_dtype == torch.int32:
            # Keep INT3 export symmetric with the load-side logical correction.
            module.qzeros.data.copy_(
                _revert_gptq_v1_qzeros_correction(module.qzeros.data, bits, planar=getattr(module, "planar", False))
            )
        else:
            module.qzeros.data[:, range(0, module.qzeros.data.shape[1], 3)] -= (
                0b00100100100100100100100100100100
            )
            module.qzeros.data[:, range(1, module.qzeros.data.shape[1], 3)] -= (
                0b10010010010010010010010010010010
            )
            module.qzeros.data[:, range(2, module.qzeros.data.shape[1], 3)] -= (
                0b01001001001001001001001001001001
            )
    elif bits == 4:
        module.qzeros.data -= 0b00010001000100010001000100010001
    elif bits == 8:
        module.qzeros.data -= 0b00000001000000010000000100000001
    elif bits in (5, 6, 7):
        if pack_dtype != torch.int32:
            raise NotImplementedError(
                f"Planar {bits}-bit GPTQ only supports 32-bit packing words, got pack_dtype={pack_dtype}."
            )
        module.qzeros.data.copy_(
            _revert_gptq_v1_qzeros_correction(module.qzeros.data, bits, planar=True)
        )
    else:
        raise NotImplementedError("Only 2,3,4,5,6,7,8 bits are supported.")

    module.qzero_format(format=1)

# Optionally convert weight from gptq_v2 to v1 export format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):

    # skip v2 to v1 conversion when no loaded quant module requires v2
    if quantize_config.export_quant_method() == METHOD.GPTQ and not any(
        isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
        for m in model.modules()
    ):
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    for _, submodule in model.named_modules():
        # sym=False has underflow probability of ~<=13% during testing. No underflow possible for sym=True.
        if isinstance(submodule, BaseQuantLinear) and getattr(submodule, "REQUIRES_FORMAT_V2", False):
            convert_gptq_v2_to_v1_format_module(module=submodule, quantize_config=quantize_config)

    return model

@torch.inference_mode()
def pack_module(
    name,
    qModules,
    q_scales,
    q_zeros,
    q_g_idx,
    layers,
    quant_linear_cls,
    lock: threading.Lock,
    q_scales_extra=None,
    quantize_config: Optional[QuantizeConfig] = None,
    quant_result: Optional[Dict[str, Any]] = None,
):
    # Limit pack() thread usage to avoid auto-parallizataion regression
    # with ctx(tctl.threadpool_limits(limits=1), lock):
    layer = layers[name]
    module = qModules[name]

    assert get_device(module) == CPU
    assert get_device(layer) == CPU
    assert get_device(q_scales) == CPU
    assert get_device(q_zeros) == CPU

    # module = module.to(CPU)
    # layer = layer.to(CPU)
    # q_scales = q_scales.to(CPU)
    # q_zeros = q_zeros.to(CPU)

    if q_g_idx is not None:
        assert get_device(q_g_idx) == CPU
        #q_g_idx = q_g_idx.to(CPU)

    pack_impl = "original"
    target_device = None
    if quantize_config is not None:
        pack_impl = getattr(quantize_config, "pack_impl", "original") or "original"
        cfg_device = getattr(quantize_config, "device", None)
        if isinstance(cfg_device, DEVICE):
            target_device = cfg_device.to_torch_device()
        elif isinstance(cfg_device, torch.device):
            target_device = cfg_device
        elif isinstance(cfg_device, str):
            try:
                target_device = torch.device(cfg_device)
            except (RuntimeError, ValueError):
                log.warning(f"pack_module: unable to parse target device `{cfg_device}`; defaulting to CUDA auto-select.")

    packer_label = None

    if lock is not None:
        with lock:
            layers[name] = layer
            qModules[name] = module
    else:
        layers[name] = layer
        qModules[name] = module

    # Use the module's actual class when it is a real BaseQuantLinear; otherwise
    # fall back to the caller-supplied representative class (e.g. unit-test mocks).
    module_cls = type(module) if isinstance(module, BaseQuantLinear) else quant_linear_cls

    # TODO FIX ME..remove hard coded qqq pack
    if module_cls.QUANT_TYPE == "qqq":
        if q_scales_extra is not None:
            q_scales_extra = q_scales_extra.to(CPU)
        packer_label = "module.pack"
        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            module.pack(linear=layer, scales=q_scales, s_extra=q_scales_extra)
    elif module_cls.QUANT_TYPE.startswith("awq_") or module_cls.QUANT_TYPE == "llm-awq":
        packer_label = "module.pack"
        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            module.pack(
                linear=layer,
                scales=q_scales,
                zeros=q_zeros,
                g_idx=q_g_idx,
            )
    else:
        effective_impl = (pack_impl or "original").lower()

        if effective_impl in {"cpu", "block", "pack_block"}:
            effective_impl = "block"
        elif effective_impl in {"original", "pack_original"}:
            effective_impl = "original"
        elif effective_impl == "gpu":
            if not HAS_CUDA:
                log.warning("pack_module: GPU packing requested but CUDA is unavailable; falling back to original pack.")
                effective_impl = "original"
            elif not hasattr(module, "pack_gpu"):
                log.warning("pack_module: GPU packing requested but module lacks pack_gpu; falling back to original pack.")
                effective_impl = "original"
        elif effective_impl != "original":
            log.warning(
                "pack_module: Unknown pack_impl `%s`; defaulting to original pack.",
                pack_impl,
            )
            effective_impl = "original"

        label_map = {
            "gpu": "module.pack_gpu",
            "block": "module.pack_block",
            "original": "module.pack_original",
        }

        packer_label = label_map[effective_impl]

        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            if effective_impl == "gpu":
                try:
                    module.pack_gpu(
                        linear=layer,
                        scales=q_scales,
                        zeros=q_zeros,
                        g_idx=q_g_idx,
                        device=target_device,
                    )
                except ValueError:
                    module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)
            elif effective_impl == "block":
                try:
                    module.pack_block(
                        linear=layer,
                        scales=q_scales,
                        zeros=q_zeros,
                        g_idx=q_g_idx,
                    )
                except ValueError:
                    module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)
            else:
                module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)

        if (
            quantize_config is not None
            and quantize_config.export_quant_method() == METHOD.GPTQ
            and resolve_quant_format(quantize_config.format, quantize_config.method) == FORMAT.GPTQ
            and getattr(module_cls, "REQUIRES_FORMAT_V2", False)
        ):
            with log_time_block(
                "convert_v2_to_v1",
                logger=log,
                module_name=name,
            ):
                convert_gptq_v2_to_v1_format_module(
                    module=module,
                    quantize_config=quantize_config,
                )

        # TODO: why move it back to gpu?
        # start = time.time()
        # qModules[name].to(layer_device)
        # log.info(f"Pack: moving module back to `{layer_device}` cost = {time.time()-start} seconds")

    return packer_label

def pack_model(
    model,
    quant_result: Dict[str, Dict[str, Any]],
    bits,
    group_size,
    backend: BACKEND,
    format: str | FORMAT,
    quant_method: str | METHOD,
    lm_head_name: str,
    desc_act=False,
    sym: bool = True,
    dynamic=None,
    pack_dtype: torch.dtype = None,
):
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        format=format,
        quant_method=quant_method,
        desc_act=desc_act,
        sym=sym,
        dynamic=dynamic,
        pack_dtype=pack_dtype,
    )

    model.to(CPU)

    log.info("Packing model...")

    modules = find_modules(model)

    modules = {n: modules[n] for n in quant_result}
    quant_linear_cls = make_quant(
        model,
        qcfg=qcfg,
        quant_result=quant_result,
        backend=backend,
        lm_head_name=lm_head_name,
        pack=True,
        device=DEVICE.CPU,
    )

    qModules = {
        name: quant_module
        for name, quant_module in find_modules(model, [BaseQuantLinear]).items()
        if name in quant_result
    }

    assert len(qModules) > 0, "No quantized modules found in the model."

    names = list(qModules.keys())
    lock = threading.Lock()

    if has_gil_disabled():
        from device_smi import Device
        cpu = Device("cpu")
        default_max_packers = min(8, max(2, cpu.count * cpu.cores))
    else:
        default_max_packers = 1 # due to gil, there is no point packing with more than 1 thread

    env_max_packers = os.getenv("GPTQMODEL_MAX_PACKERS")
    if env_max_packers is not None:
        try:
            max_packers = max(1, int(env_max_packers))
        except ValueError:
            max_packers = default_max_packers
    else:
        max_packers = default_max_packers

    with ctx(ThreadPoolExecutor(max_workers=max_packers), log.pb(names).manual()) as (executor, pb):
        def wrapper(name):
            # TODO FIX, thread pool executor does not advance iterator
            pb.next()
            pb.title(f"Packing {name}").draw()
            pack_module(
                name=name,
                qModules=qModules,
                quant_result=quant_result,
                layers=modules,
                quant_linear_cls=quant_linear_cls,
                lock=lock,
                quantize_config=qcfg,
            )

        for _ in executor.map(wrapper, names):
            pass

    log.info("Model packed.")
    return quant_linear_cls


def no_placement_module_names(model: nn.Module) -> set[str]:
    """Resolve Transformers no-placement parameter patterns to leaf modules."""

    patterns = getattr(model, "_no_placement_params", ()) or ()
    patterns = tuple(pattern for pattern in patterns if isinstance(pattern, str) and pattern)
    if not patterns:
        return set()

    names = set()
    tensors = (*model.named_parameters(), *model.named_buffers())
    for tensor_name, _ in tensors:
        if any(tensor_name == pattern or tensor_name.endswith(f".{pattern}") for pattern in patterns):
            names.add(tensor_name.rsplit(".", 1)[0])
    return names


def _remove_redundant_device_map_children(device_map: Dict[str, Union[str, int]]) -> None:
    """Remove child entries already covered by a same-device parent."""

    for name in sorted(device_map, key=lambda item: item.count(".")):
        if name not in device_map:
            continue
        prefix = f"{name}." if name else ""
        for child_name in list(device_map):
            if child_name != name and child_name.startswith(prefix) and device_map[child_name] == device_map[name]:
                device_map.pop(child_name)


def _split_device_map_around_module(
    model: nn.Module,
    device_map: Dict[str, Union[str, int]],
    ancestor_name: str,
    target_name: str,
    ancestor_device: Union[str, int],
) -> None:
    """Replace one parent mapping with non-overlapping branches around a target."""

    current_name = ancestor_name
    while current_name != target_name:
        current_module = model if not current_name else model.get_submodule(current_name)
        relative_target = target_name if not current_name else target_name[len(current_name) + 1:]
        path_child = relative_target.split(".", 1)[0]

        # Preserve direct tensors and sibling branches on the parent's device;
        # descend only through the branch containing the excluded module.
        for param_name, _ in (*current_module.named_parameters(recurse=False), *current_module.named_buffers(recurse=False)):
            full_name = f"{current_name}.{param_name}" if current_name else param_name
            device_map.setdefault(full_name, ancestor_device)
        for child_name, _ in current_module.named_children():
            full_name = f"{current_name}.{child_name}" if current_name else child_name
            if child_name != path_child:
                device_map.setdefault(full_name, ancestor_device)

        current_name = f"{current_name}.{path_child}" if current_name else path_child


def apply_no_placement_to_device_map(model: nn.Module, device_map: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
    """Build a non-overlapping load map with excluded leaf modules on CPU."""

    result = dict(device_map)
    _remove_redundant_device_map_children(result)
    for module_name in no_placement_module_names(model):
        ancestors = [
            name
            for name in result
            if name != module_name and (not name or module_name.startswith(f"{name}."))
        ]
        for ancestor_name in sorted(ancestors, key=lambda item: item.count("."), reverse=True):
            ancestor_device = result.pop(ancestor_name)
            _split_device_map_around_module(
                model,
                result,
                ancestor_name,
                module_name,
                ancestor_device,
            )
        # The checkpoint preloader expands parent and child entries independently;
        # keep this map non-overlapping so the same tensor is not read on both devices.
        result[module_name] = "cpu"
    _remove_redundant_device_map_children(result)
    return result


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    device_map = dict(device_map)
    single_root = "" in device_map and len(device_map) == 1
    all_single_cpu_or_mps = all(
        d in ("cpu", "mps") for d in device_map.values()
    )
    # CPU offload is unnecessary for all-CPU/MPS device maps and must be skipped.
    if single_root or all_single_cpu_or_mps:
        d = next(iter(device_map.values()))
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    # Avoid eagerly materialising the full model on a single device when the map
    # provides finer-grained placements; rely on accelerate hooks instead.
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    # These modules perform their own CPU lookup and must remain resident;
    # a normal CPU-offload hook would move their full weights back to the GPU.
    resident_cpu_modules = no_placement_module_names(model)
    module_names = dict(model.named_modules())
    cpu_offload_group = [
        (n, d)
        for n, d in device_map.items()
        if d == "cpu" and n not in resident_cpu_modules and n in module_names
    ]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        if n == "":
            continue
        m = module_names.get(n)
        if m is None:
            # Fine-grained maps can contain direct parameter entries.
            continue
        if d != "cpu":
            d = torch.device(d)
            has_other_device_child = any(
                child_name.startswith(f"{n}.") and child_device != device_map[n]
                for child_name, child_device in device_map.items()
            )
            # A mixed-device child means the parent hook may move inputs, but not descendants.
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=not has_other_device_child)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)

    model.hf_device_map = device_map

    return model


# public/stable api exposed to transformer/optimum
def hf_gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None,
                        max_input_length: Optional[int] = None):
    return gptqmodel_post_init(model, use_act_order, quantize_config, max_input_length)


def gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None,
                        max_input_length: Optional[int] = None):
    """
    Initialize model-persistent backend scratch buffers after quantized weights are loaded.
    """
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaV2Linear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))
        elif isinstance(submodule, AwqExllamaV2Linear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed(
                max_input_len=max_input_length or 2048,
                max_batch_size=int(os.getenv("AWQ_BATCH_SIZE", 1))
            )
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from ..utils.exllamav2 import ScratchSpace

        # we allocate a model-persistent scratch space for each device
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ScratchSpace(scratch_bytes=scratch_bytes, dev=device)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

    # The continuous and planar (gptq_p) 3-bit word layouts are not
    # interchangeable, and a 3-bit module is only planar when its constructor
    # received `format=gptq_p`. Fail loudly if any construction site dropped
    # the format instead of silently decoding the wrong layout.
    if quantize_config is not None:
        expect_planar3 = resolve_quant_format(quantize_config.format, quantize_config.method) == FORMAT.GPTQ_P
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, GPTQQuantLinear)
                and submodule.bits == 3
                and submodule.planar != expect_planar3
            ):
                raise ValueError(
                    f"`{name}`: 3-bit quant module was constructed with planar={submodule.planar} "
                    f"but the checkpoint format is `{quantize_config.format}`; the continuous and "
                    f"planar 3-bit layouts are not interchangeable. Pass `format=` when "
                    "constructing the module."
                )

    # The buffers need to have been initialized first before calling make_q4.
    for _, submodule in model.named_modules():
        if isinstance(submodule, (ExllamaV2Linear, AwqExllamaV2Linear)):
            device = submodule.qweight.device
            submodule.post_init(scratch_space=model.device_tensors[device])
        elif isinstance(submodule, BaseQuantLinear):
            submodule.post_init()

    torch_empty_cache()

    return model


def get_checkpoints(model_id_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_id_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_id_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    # The model is sharded over several checkpoints.
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_id_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                shard_index = cached_file(
                    model_id_or_path,
                    shard_index_name,
                    **cached_file_kwargs,
                )
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    # The model is sharded over several checkpoints.
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        # Download the shards from the index.json.
                        shards = list(set(index_json["weight_map"].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(
                                model_id_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_id_or_path,
                        possible_model_basename + ext,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename

    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find a model in {model_id_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
        )

    return False, resolved_archive_file, true_model_basename


# return the most stable tensor dtype for quantization while minimizing vram
def auto_dtype(config: PretrainedConfig,
               device: DEVICE,
               quant_inference: bool = False) -> torch.dtype:

    assert isinstance(device, DEVICE)

    # TODO: both MPS and XPU are locked to float16
    # XPU stack is missing bfloat16 (hardware supports it)
    # MPS stack has bfloat16 bugs in pytorch
    if device in [DEVICE.MPS, DEVICE.XPU]:
        log.info("Loader: Auto dtype (MPS or XPU): `torch.float16`")
        return torch.float16

    # TODO: need to verify this
    # Torch 2.8 fused kernel for CPU is optimized for bfloat16
    if device in [DEVICE.CPU]:
        log.info("Loader: Auto dtype (CPU + Torch Fused): `torch.bfloat16`")
        return torch.bfloat16

    # Update: latest kernel accuracies have shown, with multiple ranges of shapes
    # There are no accuracy issues with bf16 vs fp16. The Marlin reduce-accumulation
    # path can use fp16, but it is disabled by default via
    # GPTQMODEL_MARLIN_USE_FP32=True.
    # # for inference, always use FP16 for max accuracy
    # # check test_kernel_outputs for validation between fp16 and b16 in terms of kernel accuracy
    # if quant_inference:
    #     log.info("Loader: Auto dtype: `torch.float16` due to inference mode. If you wish to use `bfloat16`, please pass in `dtype` arg to `loader()`.")
    #     return torch.float16

    # get dtype from config
    dtype = get_hf_config_dtype(config)
    if dtype and not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype in config must be a torch.dtype, but got {dtype}")

    if dtype in [torch.float32, torch.float64]:
        log.info("Loader: Auto dtype (float32 down-cast): `torch.bfloat16`")
        return torch.bfloat16
    elif dtype == torch.float16:
        log.info("Loader: Auto dtype (native float16): `torch.float16`")
        return torch.float16
    elif dtype == torch.bfloat16:
        log.info("Loader: Auto dtype (native bfloat16): `torch.bfloat16`")
        return torch.bfloat16
    else:
        # TODO: extract weights from model file to check their original type, instead of forcing bfloat16
        # up/down-cast everything else to bfloat16 if not already in bfloat16
        log.info(f"Loader: Auto dtype (native = `{dtype}`): `torch.bfloat16`")
        return torch.bfloat16


# generate layer modules for moe models with experts
def get_moe_layer_modules(layer_modules: List, num_experts: int) -> List:
    new_inside_layer_modules = []
    for names in layer_modules:
        new_inside_layer_modules.append([])
        for n in names:
            if EXPERT_INDEX_PLACEHOLDER in n:
                for index in range(num_experts):
                    new_inside_layer_modules[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
            else:
                new_inside_layer_modules[-1].append(n)

    return new_inside_layer_modules


def check_to_quantized(config):
    if isinstance(config, dict):
        if config["bits"] > 8 or "fp" in config["data_type"] or "float" in config["data_type"]:
            return False
        return True
    else:
        if config.bits > 8 or "fp" in config.data_type or "float" in config.data_type:
            return False
        return True


def copy_py_files(save_dir, file_extension=".py", model_id_or_path=""):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(model_id_or_path):
        py_files = [f for f in os.listdir(model_id_or_path) if f.endswith('.py')]
        for file in py_files:
            shutil.copy2(os.path.join(model_id_or_path, file), save_dir)
    else:
        remote_model_info = model_info(model_id_or_path)
        for file in remote_model_info.siblings:
            if file.rfilename.endswith(file_extension):
                _ = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=file.rfilename,
                    local_dir=save_dir,
                )


def get_model_files_size(pre_quantized_model_path, file_extension=['.bin', '.safetensors', '.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx']):
    if os.path.isdir(pre_quantized_model_path):
        pre_quantized_size_bytes = sum(
            os.path.getsize(os.path.join(pre_quantized_model_path, f))
            for f in os.listdir(pre_quantized_model_path)
            if os.path.isfile(os.path.join(pre_quantized_model_path, f)) and os.path.splitext(f)[
                1] in file_extension
        )
    else:
        remote_model_info = model_info(pre_quantized_model_path, files_metadata=True)
        pre_quantized_size_bytes = sum(
            (file_data.size or 0)
            for file_data in remote_model_info.siblings
            if any(file_data.rfilename.endswith(ext) for ext in file_extension)
        )
    pre_quantized_size_mb = pre_quantized_size_bytes / (1024 * 1024)
    return pre_quantized_size_mb

def check_requires_version(requires_version, current_version):
    OPERATOR_MAP = {
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
    }
    match = _REQUIRES_VERSION_RE.match(requires_version)
    if match:
        op_symbol, required_version = match.groups()
        current_version = version.parse(current_version)
        required_version = version.parse(required_version)
        return OPERATOR_MAP[op_symbol](current_version, required_version)
    else:
        return None


class MODALITY(str, Enum):
    TEXT = "text"
    IMAGE_TO_TEXT = "image_to_text"
    # TEXT_TO_IMAGE = "text_to_image"


def _split_parameter_path(full_name: str) -> Tuple[str, str]:
    if "." in full_name:
        module_path, leaf = full_name.rsplit(".", 1)
    else:
        module_path, leaf = "", full_name
    return module_path, leaf


def _generate_offload_search_paths(offload_root, module_path):
    """
    Generate fallback paths:
    'a.b.c' -> ['gptqmodel_offload/a.b.c', 'gptqmodel_offload/a.b', 'gptqmodel_offload/a', 'gptqmodel_offload/']
    """
    if not module_path:
        return [offload_root]

    parts = module_path.split(".")
    paths = []

    for i in range(len(parts), 0, -1):
        sub = ".".join(parts[:i])
        paths.append(os.path.join(offload_root, sub))

    paths.append(os.path.join(offload_root, ""))

    return paths


def _generate_entry_keys(module_path, leaf):
    """
        Generate fallback entry keys for lookup inside `index`.

        Given:
            module_path = "code2wav.upsample.0.1"
            leaf = "gamma"

        This function produces a list of possible lookup keys, ordered from
        least specific to most specific, so that the caller can try them in order.

        Example output:
            [
                "gamma",
                "1.gamma",
                "0.1.gamma",
                "upsample.0.1.gamma",
                "code2wav.upsample.0.1.gamma",
            ]
    """
    keys = [leaf]
    if module_path:
        parts = module_path.split(".")
        for i in range(len(parts)-1, -1, -1):
            suffix = ".".join(parts[i:])
            keys.append(f"{suffix}.{leaf}")
    return keys


def _resolve_offload_entry(
    offload_root: str,
    module_path: str,
    leaf: str,
    dtype: torch.dtype,
    shape_hint: Tuple[int, ...],
    index_cache: Dict[str, Optional[Dict]],
) -> Optional[OffloadTensorRef]:
    if not offload_root:
        return None

    search_paths = _generate_offload_search_paths(offload_root, module_path)
    index = None
    for module_dir in search_paths:
        index = index_cache.get(module_dir)
        if index is not None:
            break

        index_path = os.path.join(module_dir, "index.json")
        if not os.path.isfile(index_path):
            index_cache[module_dir] = None
            continue

        # load index.json
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)

        index_cache[module_dir] = index
        break

    if not index:
        return None

    keys = _generate_entry_keys(module_path, leaf)
    entry = None
    for key in keys:
        entry = index.get(key)
        if entry is not None:
            break
    if entry is None:
        return None

    resolved_dtype = _dtype_string_to_torch(entry.get("dtype"), dtype)
    if "shape" in entry:
        shape = tuple(entry["shape"])
    else:
        shape = shape_hint

    safetensors_file = entry.get("safetensors_file")
    if safetensors_file:
        path = safetensors_file
        if not os.path.isabs(path):
            path = os.path.join(module_dir, path)
        offsets = entry.get("data_offsets")
        if offsets is not None:
            offsets = tuple(int(x) for x in offsets)
        return OffloadTensorRef(
            path=os.path.abspath(path),
            torch_dtype=resolved_dtype,
            shape=shape,
            format="safetensors",
            weight_name=entry.get("weight_name", leaf),
            data_offsets=offsets,
        )

    filename = entry.get("filename")
    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(module_dir, filename)
        start = int(entry.get("offset", 0))
        end = start + (_torch_dtype_num_bytes(resolved_dtype) * math.prod(shape or (1,)))
        return OffloadTensorRef(
            path=os.path.abspath(path),
            dtype=resolved_dtype,
            shape=shape,
            format="dat",
            weight_name=None,
            data_offsets=(start, end),
        )

    data_path = os.path.join(module_dir, f"{leaf}.dat")
    if not os.path.isfile(data_path):
        return None

    return OffloadTensorRef(
        path=os.path.abspath(data_path),
        torch_dtype=resolved_dtype,
        shape=shape,
        format="dat",
        weight_name=None,
        data_offsets=None,
    )


def _collect_state_dict_with_offload(model: nn.Module, offload_root: str) -> Dict[str, TensorSource]:
    state_dict: Dict[str, TensorSource] = collections.OrderedDict()
    index_cache: Dict[str, Optional[Dict]] = {}

    for name, param in model.named_parameters():
        module_path, leaf = _split_parameter_path(name)
        source = None
        if getattr(param, "is_meta", False) or param.device.type == "meta":
            source = _resolve_offload_entry(
                offload_root,
                module_path,
                leaf,
                param.dtype,
                tuple(param.shape),
                index_cache,
            )
            if source is None:
                raise FileNotFoundError(
                    f"Offloaded tensor '{name}' not found in offload directory '{offload_root}'."
                )
        else:
            source = param
        state_dict[name] = TensorSource(name=name, torch_dtype=param.dtype, shape=tuple(param.shape), source=source)

    # Collect persistent buffers in a single module-tree walk: each module owns
    # its own buffers via `named_buffers(recurse=False)`, and non-persistent ones
    # are skipped inline against `_non_persistent_buffers_set`.
    for module_name, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", ())
        for buffer_name, buf in module.named_buffers(recurse=False):
            if buffer_name in non_persistent:
                continue
            name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if name in state_dict:
                continue
            module_path, leaf = _split_parameter_path(name)
            if getattr(buf, "is_meta", False) or buf.device.type == "meta":
                source = _resolve_offload_entry(
                    offload_root,
                    module_path,
                    leaf,
                    buf.dtype,
                    tuple(buf.shape),
                    index_cache,
                )
                if source is None:
                    raise FileNotFoundError(
                        f"Offloaded buffer '{name}' not found in offload directory '{offload_root}'."
                    )
            else:
                source = buf
            state_dict[name] = TensorSource(name=name, torch_dtype=buf.dtype, shape=tuple(buf.shape), source=source)

    return state_dict


def get_state_dict_for_save(model: nn.Module, offload_root: Optional[str] = None) -> Dict[str, TensorSource]:
    """
    Filter weight-sharing tensors.
    Referenced from transformers.modeling_utils.PreTrainedModel.save_pretrained.

    See https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L2369
    """
    if offload_root:
        state_dict = _collect_state_dict_with_offload(model, offload_root)
    else:
        state_dict = collections.OrderedDict()
        for name, param in model.named_parameters():
            state_dict[name] = TensorSource(name=name, torch_dtype=param.dtype, shape=tuple(param.shape), source=param)
        # Collect persistent buffers in a single module-tree walk, skipping
        # non-persistent buffers inline via `_non_persistent_buffers_set`.
        for module_name, module in model.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", ())
            for buffer_name, buf in module.named_buffers(recurse=False):
                if buffer_name in non_persistent:
                    continue
                name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                if name in state_dict:
                    continue
                state_dict[name] = TensorSource(name=name, torch_dtype=buf.dtype, shape=tuple(buf.shape), source=buf)

    ptrs = collections.defaultdict(list)
    for name, entry in state_dict.items():
        source = entry.source
        if isinstance(source, OffloadTensorRef):
            key = ("offload", source.path, source.weight_name or name, source.data_offsets)
        elif isinstance(source, torch.Tensor):
            tensor = source
            if getattr(tensor, "is_meta", False) or tensor.device.type == "meta":
                key = ("meta", id(tensor))
            else:
                try:
                    key = ("storage", id_tensor_storage(tensor))
                except Exception:
                    key = ("tensor", id(tensor))
        else:
            key = ("other", id(source))
        ptrs[key].append(name)

    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        # Removing the keys which are declared as known duplicates on
        # load. This allows to make sure the name which is kept is consistent.
        if model._tied_weights_keys is not None:
            found = 0
            for name in sorted(names):
                matches_pattern = any(pcre.compile(pat).search(name) for pat in model._tied_weights_keys)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        del state_dict[name]

        # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
        # If the link between tensors was done at runtime then `from_pretrained` will not get
        # the key back leading to random tensor. A proper warning will be shown
        # during reload (if applicable), but since the file is not necessarily compatible with
        # the config, better show a proper warning.
        found = 0
        for name in names:
            if name in state_dict:
                found += 1
                if found > 1:
                    del state_dict[name]
                    warn_names.add(name)
    if len(warn_names) > 0:
        log.warn.once(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    return state_dict


def _checkpoint_tensor_keys(checkpoint: str | os.PathLike) -> Optional[set[str]]:
    # accelerate.load_checkpoint_in_model() does not return the checkpoint key
    # set. Read only metadata/index keys here so tie_weights() can distinguish
    # tensors that were truly absent from tensors that were loaded separately.
    checkpoint = os.fspath(checkpoint)

    if os.path.isfile(checkpoint):
        if checkpoint.endswith(".json"):
            with open(checkpoint, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", index)
            if isinstance(weight_map, dict):
                return set(weight_map)
            return None

        if checkpoint.endswith(".safetensors"):
            with safe_open(checkpoint, framework="pt", device="cpu") as handler:
                return set(handler.keys())

        return None

    if not os.path.isdir(checkpoint):
        return None

    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    if os.path.isfile(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as handler:
            return set(handler.keys())

    index_files = [name for name in os.listdir(checkpoint) if name.endswith(".index.json")]
    if len(index_files) != 1:
        return None

    with open(os.path.join(checkpoint, index_files[0]), encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", index)
    if isinstance(weight_map, dict):
        return set(weight_map)
    return None


def _tie_weights_after_checkpoint_load(model, checkpoint: str | os.PathLike | None) -> None:
    # Match transformers.from_pretrained(): when both sides of a tied-weight
    # pair are present in the checkpoint, tie_weights(missing_keys=...) checks
    # whether their loaded values are equal and skips tying if they differ.
    # This preserves checkpoints whose config incorrectly advertises tied
    # embeddings while storing a distinct lm_head.
    missing_keys = None
    if checkpoint is not None:
        checkpoint_keys = _checkpoint_tensor_keys(checkpoint)
        if checkpoint_keys is not None:
            missing_keys = set(model.state_dict().keys()) - checkpoint_keys

    if missing_keys is None:
        model.tie_weights()
        return

    try:
        model.tie_weights(missing_keys=missing_keys)
    except TypeError:
        model.tie_weights()


# Call tied_weights() after load_checkpoint_in_model() to have the weights tied correctly.
def load_checkpoint_in_model_then_tie_weights(model, *args, **kwargs):
    checkpoint = kwargs.get("checkpoint")
    if checkpoint is None and args:
        checkpoint = args[0]
    accelerate.load_checkpoint_in_model(model, *args, **kwargs)
    _tie_weights_after_checkpoint_load(model, checkpoint)


# 32MB read/write i/o buffer
_STREAM_BUFFER_SIZE = 32 * 1024 * 1024
_STREAM_BUFFER = memoryview(bytearray(_STREAM_BUFFER_SIZE))
_STREAM_BUFFER_LOCK = threading.Lock()

def _copy_file_stream(src_path: str, dst_fh, length: int, *, offset: int = 0) -> None:
    with ctx(open(src_path, "rb", buffering=0), _STREAM_BUFFER_LOCK) as (src, _):
        if offset:
            src.seek(offset)
        remaining = length
        while remaining > 0:
            chunk_size = min(_STREAM_BUFFER_SIZE, remaining)
            read = src.readinto(_STREAM_BUFFER[:chunk_size])
            if not read:
                raise IOError(f"Unexpected EOF while copying from {src_path}")
            dst_fh.write(_STREAM_BUFFER[:read])
            remaining -= read


def _write_tensor_bytes(out, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    tensor = tensor.detach().to("cpu").contiguous()
    if dtype is torch.bfloat16:
        view = tensor.view(torch.int16)
        out.write(view.numpy().tobytes())
        return

    try:
        out.write(tensor.numpy().tobytes())
    except TypeError:
        # PyTorch float8 dtypes and some future storage dtypes may not expose a NumPy bridge.
        # Fall back to the raw byte view so safetensors still receives the exact tensor payload.
        out.write(tensor.view(torch.uint8).numpy().tobytes())


def _write_shard_file(path: str, entries: List[TensorSource], metadata: Dict[str, str]) -> int:
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": _torch_dtype_to_safetensors(entry.torch_dtype),
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.num_bytes],
        }
        offset += entry.num_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Safetensors pads the JSON header to an 8-byte boundary.
    # Without that padding, some readers reject the file as malformed.
    header_padding = (-len(header_bytes)) % 8
    if header_padding:
        header_bytes += b" " * header_padding

    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)

        for entry in entries:
            source = entry.source
            if isinstance(source, OffloadTensorRef):
                if source.format == "dat":
                    # print("offload tensor io buffered transfer DAT")
                    start = 0
                    if source.data_offsets is not None:
                        start = source.data_offsets[0]
                    _copy_file_stream(source.path, out, entry.num_bytes, offset=start)
                elif source.format == "safetensors" and source.data_offsets is not None:
                    # print("offload tensor io buffered transfer SAFETENSOR stream")
                    start, end = source.data_offsets
                    _copy_file_stream(source.path, out, end - start, offset=start)
                else:
                    # print("offload tensor slow tensor read")
                    with safe_open(source.path, framework="pt", device="cpu") as handler:
                        tensor = handler.get_tensor(source.weight_name or entry.name)
                    tensor = tensor.to(source.torch_dtype)
                    _write_tensor_bytes(out, tensor, source.torch_dtype)
            else:
                tensor = source.detach()
                _write_tensor_bytes(out, tensor, entry.torch_dtype)
                del tensor

        file_size = out.tell()

    return file_size


def _plan_shards(entries: List[TensorSource], max_shard_size: Optional[int]) -> List[List[TensorSource]]:
    if not max_shard_size or max_shard_size <= 0:
        return [entries]

    shards: List[List[TensorSource]] = []
    current: List[TensorSource] = []
    current_size = 0

    for entry in entries:
        size = entry.num_bytes
        if size > max_shard_size:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([entry])
            continue
        if current_size + size > max_shard_size and current:
            shards.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += size

    if current:
        shards.append(current)

    return shards


def streaming_state_dict_to_shards(
    state_dict: Dict[str, TensorSource],
    save_dir: str,
    model_base_name: str,
    single_file_name: str,
    metadata: Dict[str, str],
    max_shard_size: Optional[int],
) -> Tuple[List[str], Dict[str, str], int]:
    entries = list(state_dict.values())
    shards = _plan_shards(entries, max_shard_size)
    num_shards = len(shards)
    filenames: List[str] = []
    tensor_to_filename: Dict[str, str] = {}
    total_size = 0

    for idx, shard_entries in enumerate(shards, start=1):
        if num_shards == 1:
            filename = single_file_name
        else:
            filename = f"{model_base_name}-{idx:05d}-of-{num_shards:05d}.safetensors"

        path = os.path.join(save_dir, filename)
        size = _write_shard_file(path, shard_entries, metadata)
        total_size += size
        filenames.append(filename)
        for entry in shard_entries:
            tensor_to_filename[entry.name] = filename

    return filenames, tensor_to_filename, total_size


def find_config_seq_len(config_dict, target_keys):
    for k, v in config_dict.items():
        if k in target_keys:
            return v
        if isinstance(v, dict):
            found = find_config_seq_len(v, target_keys)
            if found is not None:
                return found
    return None


def get_module_name(module: nn.Module, child_module: nn.Module) -> str:
    for name, candidate in module.named_modules():
        if candidate is child_module:
            return name
    raise ValueError(f"Cannot find child_module {child_module} in module {module}")


def check_module_quantized_in_keys(keys, module_name: str) -> bool:
    return any(
        key.startswith(module_name + ".")
        and (".qweight" in key or ".qzeros" in key or ".scales" in key)
        for key in keys
    )


def is_embeddings_module_quantized(
    model_dir: str,
    input_embed_name: Optional[str],
    output_embed_name: Optional[str],
) -> Tuple[bool, bool]:
    input_quantized = False
    output_quantized = False

    def inspect_keys(keys) -> Tuple[bool, bool]:
        return (
            bool(input_embed_name and check_module_quantized_in_keys(keys, input_embed_name)),
            bool(output_embed_name and check_module_quantized_in_keys(keys, output_embed_name)),
        )

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return inspect_keys(index.get("weight_map", {}).keys())

    safetensor_files = [
        os.path.join(model_dir, filename)
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    ]
    for safefile in safetensor_files:
        try:
            with safe_open(safefile, framework="pt") as handle:
                found_input, found_output = inspect_keys(handle.keys())
                input_quantized = input_quantized or found_input
                output_quantized = output_quantized or found_output
                if input_quantized and output_quantized:
                    break
        except Exception as exc:
            log.warn(f"Failed to inspect {safefile}: {exc}")

    return input_quantized, output_quantized


def has_any_attr(obj, names):
    return any(hasattr(obj, name) for name in names)


def find_moe_routing_modules(model):
    modules = []
    for module in model.modules():
        if has_any_attr(module, MOE_TOPK_FIELD_NAMES) and \
                has_any_attr(module, MOE_NUM_EXPERTS_FIELD_NAMES):
            modules.append(module)
    return modules


def set_moe_topk(model: nn.Module, new_topk: int) -> MoETopKState:
    routers = find_moe_routing_modules(model)
    state: MoETopKState = []
    for r in routers:
        for name in MOE_TOPK_FIELD_NAMES:
            if hasattr(r, name):
                old = getattr(r, name)
                assert isinstance(old, int)
                state.append((r, name, old))
                setattr(r, name, new_topk)
                break
    return state


def restore_moe_topk(state: MoETopKState):
    for module, name, old in state:
        if hasattr(module, name):
            setattr(module, name, old)
