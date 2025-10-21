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
import pcre as re
import torch
import torch.nn as nn
import transformers
from huggingface_hub import HfApi, hf_hub_download
from packaging import version
from safetensors import safe_open
from torch.nn.modules.conv import _ConvNd
from transformers import PretrainedConfig
from transformers.pytorch_utils import id_tensor_storage
from transformers.utils.hub import cached_file

from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear

from ..adapter.adapter import Adapter
from ..looper.named_module import NamedModule
from ..models._const import (
    CPU,
    DEVICE,
    EXLLAMA_DEFAULT_MAX_INPUT_LENGTH,
    EXPERT_INDEX_PLACEHOLDER,
    SUPPORTS_MODULE_TYPES,
)
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.awq_exllamav2 import AwqExllamaV2QuantLinear
from ..nn_modules.qlinear.exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..quantization import FORMAT, QuantizeConfig
from ..quantization.config import FORMAT_FIELD_CHECKPOINT, METHOD, dynamic_get
from . import has_gil_disabled
from .backend import BACKEND
from .ctx import ctx
from .device import get_device
from .importer import select_quant_linear
from .logger import log_time_block, setup_logger
from .torch import HAS_CUDA, torch_empty_cache


log = setup_logger()


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


def _torch_dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_SAFE_MAP:
        raise NotImplementedError(f"Unsupported dtype for safetensors export: {dtype}")
    return _DTYPE_SAFE_MAP[dtype][1]


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


def move_to(obj: torch.Tensor | nn.Module, device: torch.device, dtype: torch.dtype = None):
    if get_device(obj) != device or dtype is not None:
        obj = obj.to(device=device, dtype=dtype, non_blocking=False)

    return obj


def nested_move_to(v, device, dtype: torch.dtype = None):
    if isinstance(v, torch.Tensor):
        return move_to(v, device=device, dtype=dtype)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to(e, device=device, dtype=dtype) for e in v])
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


def get_module_by_name_prefix(model, module_name: Union[List[str], str]):
    module_name_list = module_name if isinstance(module_name, list) else [module_name]
    for name, module in model.named_modules():
        for prefix in module_name_list:
            if name.startswith(prefix):
                return module, prefix

    return None, ""


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
) -> Type[BaseQuantLinear]:

    bits = qcfg.bits
    group_size =qcfg.group_size
    extension = qcfg.adapter
    format = qcfg.format
    desc_act = qcfg.desc_act
    sym = qcfg.sym
    dynamic = qcfg.dynamic
    pack_dtype = qcfg.pack_dtype

    # Bitblas needs to be loaded as gptq's quant linear first, and then converted to bitblas format.
    if not pack and format == FORMAT.GPTQ and backend == BACKEND.BITBLAS:
        backend = BACKEND.TORCH

    # returns multiple validated kernels
    quant_linear_candidates = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        quant_method=qcfg.quant_method,
        pack=pack,
        dynamic=dynamic,
        device=device,
        pack_dtype=pack_dtype,
        multi_select=True,
        adapter=extension,
    )

    log.info(f"Kernel: candidates -> `[{', '.join(cls.__name__ for cls in quant_linear_candidates)}]`")

    # loop over actual QLinear init, catch errors and use fallbacks if applicable
    for cls in quant_linear_candidates:
        try:
            # if linear is not selectedQLinear:
            #     logger.info(f"make_quant: Faild linear: `{selectedQLinear}` failed, trying to use fallback: `{linear}`")
            # else:
            #     logger.info("make_quant: Testing linear: {linear}")

            linear_cls = create_quant_layer(
                linear_cls=cls,
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
            )
            log.info(f"Kernel: selected -> `{linear_cls.__name__}`.")
            return linear_cls
        except NotImplementedError as e:
            log.info(f"Kernel: skipped -> `{cls}`.")

            # only fallback to other quant linears when backend is auto.
            if backend not in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
                raise e

    raise ValueError(f"No compatible quant linear was found for this module: {module.__class__.__name__}")

def create_quant_module(
    name: str,
    linear_cls: Type[BaseQuantLinear],
    bits: int,
    desc_act: bool,
    dynamic,
    group_size: int,
    module: nn.Module,
    submodule: nn.Module,
    sym: bool,
    device: DEVICE,
    lm_head_name: str,
    pack_dtype: torch.dtype,
    backend: BACKEND = BACKEND.AUTO,
    register_buffers: bool = True,
    adapter: Optional[Adapter] = None,
):
    # unwrap named module
    if isinstance(submodule, NamedModule):
        # print(f"offloading named module: {module.full_name}")
        submodule = submodule.module

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

    bias = submodule.bias is not None

    # need copies as dynamic config may override these in for loop
    tmp_bits = bits
    tmp_group_size = group_size
    tmp_desc_act = desc_act
    tmp_sym = sym
    tmp_pack_dtype = pack_dtype

    # dynamic bits, group_size, sym, pack_dtype for each layer/module
    if dynamic is not None:
        overrides = dynamic_get(dynamic=dynamic, module_name=name)
        # negative module match, skip this module
        if overrides == False:  # noqa: E712
            return

        # positive module match
        if overrides:
            # override base QuantizeConfig for every quant config key/value
            tmp_bits = overrides.get("bits", bits)
            tmp_group_size = overrides.get("group_size", group_size)
            tmp_desc_act = overrides.get("desc_act", desc_act)
            tmp_sym = overrides.get("sym", sym)
            tmp_pack_dtype = overrides.get("pack_dtype", pack_dtype)

    # when loading a quantized model, device is target device passed in GPTQModel.load()
    # check in_features and out_features validate
    _, err = linear_cls.validate(
        bits=tmp_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        pack_dtype=tmp_pack_dtype,
        in_features=in_features,
        out_features=out_features,
        device=device,
        adapter=adapter, # TODO FIX ME..need to pass Eora if loaded
    )
    if err is not None:
        raise err

    new_layer = linear_cls(
        bits=tmp_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=tmp_pack_dtype,
        bias=bias,
        #weight_dtype=submodule.qweight.dtype if isinstance(submodule, BaseQuantLinear) else submodule.weight.dtype,
        name=name,
        lm_head_name=lm_head_name,
        backend=backend,
        register_buffers=register_buffers,
        adapter=adapter,
    )
    new_layer.device = ori_layer_device
    recurse_setattr(module, name, new_layer.to(ori_layer_device))

def create_quant_layer(
        linear_cls: Type[BaseQuantLinear],
        bits: int,
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
) -> Type[BaseQuantLinear]:
    if isinstance(module, linear_cls):
        return linear_cls
    for name, submodule in module.named_modules():
        # skip non-quantized modules
        if name not in quant_result:
            continue

        create_quant_module(
            name=name,
            linear_cls=linear_cls,
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
            backend=backend,
            adapter=adapter,
        )

    return linear_cls

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v1_to_v2_format(
    model: nn.Module,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    if checkpoint_format == "gptq":
        # skip v1 to v2 conversion for kernels that can only operate on sym=True (gptq_v1)
        if qlinear_kernel in [MarlinQuantLinear, ExllamaEoraQuantLinear]:
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
        # range 0 offset
        if pack_dtype == torch.int64:
            offset = 0b0010010010010010010010010010010000100100100100100100100100100100
        elif pack_dtype == torch.int32:
            offset = 0b00100100100100100100100100100100
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
        elif pack_dtype == torch.int32:
            offset = 0b10010010010010010010010010010010
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
        elif pack_dtype == torch.int32:
            offset = 0b01001001001001001001001001001001
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
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    # change format id
    module.qzero_format(format=2)

# Optionally convert weight from gptq_v1 to v2 format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v1_to_v2_format(
    model,
    cfg: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):
    # skip v2 to v1 conversion for gptq_v1 kernels
    if cfg.quant_method in [METHOD.GPTQ] and not qlinear_kernel.REQUIRES_FORMAT_V2:
        log.info(
            f"Format: Skipped v1 to v2 conversion due to Kernel  `{qlinear_kernel}`.")
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    time.time()
    log.info(
        f"Format: Converting `{FORMAT_FIELD_CHECKPOINT}` from `{FORMAT.GPTQ}` to internal `{FORMAT.GPTQ_V2}`.")

    for _, submodule in model.named_modules():
        # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
        # additions here do not overflow.
        # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
        # overflow ~<=13% based on testing
        if isinstance(submodule, qlinear_kernel):
            convert_gptq_v1_to_v2_format_module(module=submodule, bits=cfg.bits, pack_dtype=cfg.pack_dtype)

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

    if quantize_config.bits == 2:
        module.qzeros.data -= 0b01010101010101010101010101010101
    elif quantize_config.bits == 3:
        module.qzeros.data[:, range(0, module.qzeros.data.shape[1], 3)] -= (
            0b00100100100100100100100100100100
        )
        module.qzeros.data[:, range(1, module.qzeros.data.shape[1], 3)] -= (
            0b10010010010010010010010010010010
        )
        module.qzeros.data[:, range(2, module.qzeros.data.shape[1], 3)] -= (
            0b01001001001001001001001001001001
        )
    elif quantize_config.bits == 4:
        module.qzeros.data -= 0b00010001000100010001000100010001
    elif quantize_config.bits == 8:
        module.qzeros.data -= 0b00000001000000010000000100000001
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    module.qzero_format(format=1)

# Optionally convert weight from gptq_v2 to v1 export format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):

    # skip v2 to v1 conversion for gptq_v1 kernels
    if quantize_config.quant_method in [METHOD.GPTQ] and not qlinear_kernel.REQUIRES_FORMAT_V2:
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    for _, submodule in model.named_modules():
        # sym=False has underflow probability of ~<=13% during testing. No underflow possible for sym=True.
        if isinstance(submodule, qlinear_kernel):
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

    # TODO FIX ME..remove hard coded qqq pack
    if quant_linear_cls.QUANT_TYPE == "qqq":
        if q_scales_extra is not None:
            q_scales_extra = q_scales_extra.to(CPU)
        packer_label = "module.pack"
        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            module.pack(linear=layer, scales=q_scales, s_extra=q_scales_extra)
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
                module.pack_gpu(
                    linear=layer,
                    scales=q_scales,
                    zeros=q_zeros,
                    g_idx=q_g_idx,
                    device=target_device,
                )
            elif effective_impl == "block":
                module.pack_block(
                    linear=layer,
                    scales=q_scales,
                    zeros=q_zeros,
                    g_idx=q_g_idx,
                )
            else:
                module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)

        if (
            quantize_config is not None
            and quantize_config.quant_method == METHOD.GPTQ
            and quantize_config.format == FORMAT.GPTQ
            and getattr(quant_linear_cls, "REQUIRES_FORMAT_V2", False)
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
        backend=backend,
        lm_head_name=lm_head_name,
        pack=True,
    )

    qModules = find_modules(model, [quant_linear_cls])

    assert len(qModules) > 0, f"No quantizeed modules[{quant_linear_cls}] found in the model."

    names = list(qModules.keys())
    lock = threading.Lock()

    if has_gil_disabled():
        from device_smi import Device
        cpu = Device("cpu")
        max_packers = cpu.count * cpu.cores
    else:
        max_packers = 1 # due to gil, there is no point packing with more than 1 thread

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

def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    device_map = dict(device_map)
    if "" in device_map and len(device_map) == 1:
        d = device_map[""]
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

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
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
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
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
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    # post init for bitblas backend.
    device_to_buffers_size = {}
    # exllama
    model_uses_exllama = False

    # exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaV2QuantLinear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))
        elif isinstance(submodule, AwqExllamaV2QuantLinear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed(
                max_input_len=max_input_length or 2048,
                max_batch_size=int(os.getenv("AWQ_BATCH_SIZE", 1))
            )
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))
        elif isinstance(submodule, ExllamaQuantLinear):
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1,
                }
            submodule._use_act_order = True if use_act_order else False

            # Disable this heuristic for detecting act_order, but it could be used instead of the config.
            """
            if submodule.g_idx is None:
                submodule.act_order = False
            elif submodule.g_idx is not None and ((submodule.g_idx == 0).all() or torch.equal(submodule.g_idx.cpu(), torch.tensor([i // submodule.group_size for i in range(submodule.g_idx.shape[0])], dtype=torch.int32))):
                submodule.g_idx = None
                submodule.act_order = False
            else:
                submodule.act_order = True
            """

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(
                device_to_buffers_size[device]["max_dq_buffer_size"],
                submodule.qweight.numel() * 8,
                )

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(
                    device_to_buffers_size[device]["max_inner_outer_dim"],
                    submodule.in_features,
                    submodule.out_features,
                )

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        from gptqmodel_exllama_kernels import prepare_buffers, set_tuning_params

        device_to_buffers = {}

        if use_act_order:
            if max_input_length is None:
                max_input_len = EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_len = max_input_length
        else:
            if max_input_length is not None:
                log.info(
                    "Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored."
                )
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros(
                    (max_input_len, buffers_size["max_inner_outer_dim"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "temp_dq": torch.zeros(
                    (1, buffers_size["max_dq_buffer_size"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }

        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers

        for device, buffers in model.device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 16
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    if model_uses_exllamav2:
        from ..utils.exllamav2 import ScratchSpace

        # we allocate a model-persistent scratch space for each device
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ScratchSpace(scratch_bytes=scratch_bytes, dev=device)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

    # The buffers need to have been initialized first before calling make_q4.
    for _, submodule in model.named_modules():
        if isinstance(submodule, (ExllamaV2QuantLinear, AwqExllamaV2QuantLinear)):
            device = submodule.qweight.device
            submodule.post_init(scratch_space=model.device_tensors[device])
        elif isinstance(submodule, BaseQuantLinear):
            submodule.post_init()

    torch_empty_cache()

    # if use_act_order and max_input_length and isinstance(submodule, ExllamaQuantLinear):
    #     model = exllama_set_max_input_length(model, max_input_length)

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
    # There are no accuracy issues with bf16 vs fp16. The only kernel with severe
    # regression in bf16 is MARLIN_FP16 (reduce math in fp16) which is not auto-selectable
    # # for inference, always use FP16 for max accuracy
    # # check test_kernel_outputs for validation between fp16 and b16 in terms of kernel accuracy
    # if quant_inference:
    #     log.info("Loader: Auto dtype: `torch.float16` due to inference mode. If you wish to use `bfloat16`, please pass in `dtype` arg to `loader()`.")
    #     return torch.float16

    # get dtype from config
    dtype = getattr(config, "dtype") if hasattr(config, "dtype") else getattr(config, "torch_dtype")
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
        api = HfApi()
        model_info = api.model_info(model_id_or_path)
        for file in model_info.siblings:
            if file.rfilename.endswith(file_extension):
                _ = hf_hub_download(repo_id=model_id_or_path, filename=file.rfilename,
                                                  local_dir=save_dir)

def get_model_files_size(pre_quantized_model_path, file_extension=['.bin', '.safetensors', '.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx']):
    if os.path.isdir(pre_quantized_model_path):
        pre_quantized_size_bytes = sum(
            os.path.getsize(os.path.join(pre_quantized_model_path, f))
            for f in os.listdir(pre_quantized_model_path)
            if os.path.isfile(os.path.join(pre_quantized_model_path, f)) and os.path.splitext(f)[
                1] in file_extension
        )
    else:
        api = HfApi()
        files_data = api.list_repo_files(pre_quantized_model_path)
        pre_quantized_size_bytes = 0
        for file_info in files_data:
            if any(file_info.endswith(ext) for ext in file_extension):
                file_metadata = api.model_info(pre_quantized_model_path, files_metadata=True)
                for file_data in file_metadata.siblings:
                    if file_data.rfilename == file_info:
                        pre_quantized_size_bytes += file_data.size
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
    match = re.match(r"(<=|>=|==|<|>)\s*([\d\.]+)", requires_version)
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

    module_dir = os.path.join(offload_root, module_path) if module_path else offload_root
    index = index_cache.get(module_dir)
    if index is None:
        index_path = os.path.join(module_dir, "index.json")
        if not os.path.isfile(index_path):
            index_cache[module_dir] = None
            return None
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        index_cache[module_dir] = index

    if not index:
        return None

    entry = index.get(leaf) or index.get(f"{module_path}.{leaf}")
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

    for name, buf in model.named_buffers():
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
        for name, buf in model.named_buffers():
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
                matches_pattern = any(re.search(pat, name) for pat in model._tied_weights_keys)
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

# Call tied_weights() after load_checkpoint_in_model() to have the weights tied correctly.
def load_checkpoint_in_model_then_tie_weights(model, *args, **kwargs):
    accelerate.load_checkpoint_in_model(model, *args, **kwargs)
    model.tie_weights()


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
    else:
        out.write(tensor.numpy().tobytes())


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
