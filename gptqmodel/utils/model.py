# Copyright 2025 ModelCloud
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

from __future__ import annotations

import collections
import functools
import hashlib
import json
import operator
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import accelerate
import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from huggingface_hub import HfApi, hf_hub_download
from packaging import version
from transformers import AutoConfig, PretrainedConfig
from transformers.pytorch_utils import id_tensor_storage
from transformers.utils.hub import cached_file

from ..models._const import (CPU, DEVICE, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH,
                             EXPERT_INDEX_PLACEHOLDER, SUPPORTED_MODELS, SUPPORTS_MODULE_TYPES)
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.ipex import IPEXQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import FORMAT, QuantizeConfig
from ..quantization.config import dynamic_get
from .backend import BACKEND
from .importer import select_quant_linear
from .logger import setup_logger
from .progress import ProgressBar
from .torch import torch_empty_cache

logger = setup_logger()


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


def get_device(obj: torch.Tensor | nn.Module):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to(obj: torch.Tensor | nn.Module, device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


def nested_move_to(v, device):
    if isinstance(v, torch.Tensor):
        return move_to(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to(e, device) for e in v])
    else:
        return v


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = SUPPORTS_MODULE_TYPES

    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_by_name_prefix(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


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
    names,
    bits: int,
    group_size: int,
    backend: BACKEND,
    format: str | FORMAT,
    lm_head_name: str,
    desc_act: bool = False,
    sym: bool = True,
    pack: bool = False,
    dynamic=None,
    device: DEVICE = None,
    from_quantized: bool = False,
) -> BaseQuantLinear:
    QuantLinear = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        pack=pack,
        dynamic=dynamic,
        device=device,
    )

    if pack:
        reserve_quant_linear = ExllamaQuantLinear
    else:
        reserve_quant_linear = ExllamaV2QuantLinear

    # TODO, we need fix here. if select other linears
    for linear in list(dict.fromkeys([QuantLinear, reserve_quant_linear, TorchQuantLinear])):
        try:
            if linear is not QuantLinear:
                logger.info(f"Use {QuantLinear} failed, try to use {linear} instead.")

            result = create_quant_layer(linear, bits, desc_act, dynamic, group_size, module, names, sym, device
                                        , lm_head_name)
            return result
        except NotImplementedError as e:
            # only fallback to other quant linears when backend is auto.
            if backend not in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
                raise e

    raise ValueError("no support quant linear was found for this module.")


def create_quant_layer(QuantLinear, bits, desc_act, dynamic, group_size, module, names, sym, device, lm_head_name: str
                       ) -> BaseQuantLinear:
    if isinstance(module, QuantLinear):
        return QuantLinear
    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device
            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            elif isinstance(submodule, BaseQuantLinear):
                # if submodule is already a quant layer, we need to get in_features and out_features from the submodule
                in_features = submodule.infeatures
                out_features = submodule.outfeatures
            else:
                raise NotImplementedError(f"Unsupported module {submodule}")

            # when load a quantized model, device is target device passed in GPTQModel.load()
            # check in_features and out_features validate
            _, err = QuantLinear.validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, infeatures=in_features, outfeatures=out_features, device=device)
            if err is not None:
                raise err

            bias = submodule.bias is not None

            d_bits = bits
            d_group_size = group_size
            d_sym = sym
            # dynamic bits, group_size, sym for each layer/module
            if dynamic is not None:
                if dynamic_get(dynamic=dynamic, layer_name=name) == False:  # noqa: E712
                    # skip create this quant linear
                    continue

                for pattern, pattern_dict in dynamic.items():
                    if re.match(pattern, name):
                        d_bits = pattern_dict.get("bits", bits)
                        d_group_size = pattern_dict.get("group_size", group_size)
                        d_sym = pattern_dict.get("sym", sym)
                        break
            new_layer = QuantLinear(
                bits=d_bits,
                group_size=d_group_size,
                desc_act=desc_act,
                sym=d_sym,
                infeatures=in_features,
                outfeatures=out_features,
                bias=bias,
                weight_dtype=submodule.qweight.dtype if isinstance(submodule, BaseQuantLinear) else submodule.weight.dtype,
                name=name,
                lm_head_name=lm_head_name,
            )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))
    return QuantLinear

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v1_to_v2_format(
    model: nn.Module,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    if checkpoint_format == "gptq":
        quantize_config = QuantizeConfig(bits=bits)
        return convert_gptq_v1_to_v2_format(model, quantize_config, qlinear_kernel), True
    else:
        return model, False


def convert_gptq_v1_to_v2_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):
    # skip v1 to v2 conversion for ipex
    if qlinear_kernel == IPEXQuantLinear:
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        for _, submodule in model.named_modules():
            # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
            # additions here do not overflow.
            # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
            # overflow ~<=13% based on testing
            if isinstance(submodule, qlinear_kernel):
                if quantize_config.bits == 2:
                    submodule.qzeros.data += 0b01010101010101010101010101010101
                elif quantize_config.bits == 3:
                    submodule.qzeros.data[:, range(0, submodule.qzeros.data.shape[1], 3)] += (
                        0b00100100100100100100100100100100
                    )
                    submodule.qzeros.data[:, range(1, submodule.qzeros.data.shape[1], 3)] += (
                        0b10010010010010010010010010010010
                    )
                    submodule.qzeros.data[:, range(2, submodule.qzeros.data.shape[1], 3)] += (
                        0b01001001001001001001001001001001
                    )
                elif quantize_config.bits == 4:
                    submodule.qzeros.data += 0b00010001000100010001000100010001
                elif quantize_config.bits == 8:
                    submodule.qzeros.data += 0b00000001000000010000000100000001
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")

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


def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):
    # skip v2 to v1 conversion for ipex
    if qlinear_kernel == IPEXQuantLinear:
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        for _, submodule in model.named_modules():
            # sym=False has underflow probability of ~<=13% during testing. No underflow possible for sym=True.
            if isinstance(submodule, qlinear_kernel):
                if quantize_config.bits == 2:
                    submodule.qzeros.data -= 0b01010101010101010101010101010101
                elif quantize_config.bits == 3:
                    submodule.qzeros.data[:, range(0, submodule.qzeros.data.shape[1], 3)] -= (
                        0b00100100100100100100100100100100
                    )
                    submodule.qzeros.data[:, range(1, submodule.qzeros.data.shape[1], 3)] -= (
                        0b10010010010010010010010010010010
                    )
                    submodule.qzeros.data[:, range(2, submodule.qzeros.data.shape[1], 3)] -= (
                        0b01001001001001001001001001001001
                    )
                elif quantize_config.bits == 4:
                    submodule.qzeros.data -= 0b00010001000100010001000100010001
                elif quantize_config.bits == 8:
                    submodule.qzeros.data -= 0b00000001000000010000000100000001
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    return model


def pack_layer(name, qlayers, quantizers, layers, QuantLinear, pbar):
    # Limit pack() thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        pbar.set_description(f"Packing {name}")
        quantizers[name], scale, zero, g_idx = quantizers[name]
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx = (
            layers[name].to(CPU),
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU) if g_idx is not None else None,
        )
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)
        pbar.progress()


def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    backend: BACKEND,
    format: str | FORMAT,
    lm_head_name: str,
    desc_act=False,
    sym: bool = True,
    dynamic=None,
    parallel_packing: bool = True,
):
    QuantLinear = select_quant_linear(
        bits=bits,
        dynamic=dynamic,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        pack=True,
    )

    model.to(CPU)

    logger.info("Packing model...")

    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
        backend=backend,
        format=format,
        lm_head_name=lm_head_name,
        desc_act=desc_act,
        pack=True,
        dynamic=dynamic,
    )
    qlayers = find_layers(model, [QuantLinear])
    names = list(qlayers.keys())

    if parallel_packing:
        max_workers = 2
    else:
        max_workers = 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with ProgressBar(total=len(names)) as pbar:
            def wrapper(name):
                pack_layer(name, qlayers, quantizers, layers, QuantLinear, pbar)

            for _ in executor.map(wrapper, names):
                pass

    logger.info("Model packed.")
    return QuantLinear


def verify_model_hash(file_path: str, verify_hash: str):
    if not isinstance(verify_hash, str):
        raise ValueError("model verify_hash must be a string")
    if ':' not in verify_hash:
        raise ValueError("verify_hash must be in the format 'hash_type:hash_value'")
    hash_type, hash_value = verify_hash.split(':', 1)
    hash_func = getattr(hashlib, hash_type, None)
    if not hash_func:
        raise ValueError(f"No hash function found for type: {hash_type}")
    with open(file_path, "rb") as f:
        file_hash = hash_func(f.read()).hexdigest()
    return file_hash == hash_value


def verify_sharded_model_hashes(jsonPath: str, verify_hash: List[str]):
    if not isinstance(verify_hash, list):
        raise ValueError("sharded model verify_hash must be a list")

    with open(jsonPath, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data['weight_map']
    shard_files = set(weight_map.values())
    if len(shard_files) != len(verify_hash):
        raise ValueError("Number of shards and number of hash values do not match.")

    for shard_file, expected_hash in zip(shard_files, verify_hash):
        if not verify_model_hash(shard_file, expected_hash):
            logger.info(f"Hash verification failed for {shard_file}")
            return False
    return True


def check_and_get_model_type(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


def simple_dispatch_model(model, device_map):

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model
    else:
        raise ValueError("internal device_map must contain an empty string")

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
                    submodule.infeatures,
                    submodule.outfeatures,
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
                logger.info(
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
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    if model_uses_exllamav2:
        from ..nn_modules.qlinear.exllamav2 import ExLlamaV2DeviceTensors

        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

    # The buffers need to have been initialized first before calling make_q4.
    for _, submodule in model.named_modules():
        if isinstance(submodule, ExllamaV2QuantLinear):
            device = submodule.qweight.device
            submodule.post_init(temp_dq=model.device_tensors[device])
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

    # for inference, DynamicCuda, Exllama, Triton, and Marlin are all fp16 kernels
    if quant_inference and device != DEVICE.CPU:
        return torch.float16

    # TODO: both MPS and XPU are locked to float16
    # XPU stack is missing bfloat16 (hardware supports it)
    # MPS stack has bfloat16 bugs in pytorch
    if device in [DEVICE.MPS, DEVICE.XPU]:
        return torch.float16

    # get dtype from config
    dtype = getattr(config, "torch_dtype")
    if dtype and not isinstance(dtype, torch.dtype):
        raise ValueError(f"torch_dtype in config must be a torch.dtype, but got {dtype}")

    if dtype == torch.float32:
        return torch.bfloat16
    elif dtype == torch.float16:
        return torch.float16
    else:
        # TODO: extract weights from model file to check their original type, instead of forcing bfloat16
        # up/down-cast everything else to bfloat16 if not already in bfloat16
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

def normalize_tokenizer(config, tokenizer):
    pad_token_id = config.pad_token_id
    if not pad_token_id:
        if tokenizer:
            vocab = tokenizer.get_vocab()

            # auto select the best pad token to use
            for token in ["<|finetune_right_pad_id|>", "<|pad|>", "<pad>", "<|unk|>", "<unk>"]:
                token_id = vocab.get(token)
                if token_id is not None:
                    pad_token_id = token_id
                    break
        else:
            logger.warning(
                "Model config does not have pad token mapped. Please pass in tokenizer to `quantize()` so GPTQModel can auto-select the best pad token.")

        if not pad_token_id and isinstance(config.eos_token_id,
                                           list):  # Llama-3.1-8B-Instruct's eos_token_id is a list
            pad_token_id = config.eos_token_id[0]
        elif not pad_token_id:
            pad_token_id = config.eos_token_id

    if pad_token_id is None:
        raise ValueError(
            "Calibration data requires model's `pad_token_id` or `eos_token_id` to be set: actual = `None`.")

    tokenizer.pad_token_id = pad_token_id

    return tokenizer

class MODALITY(str, Enum):
    TEXT = "text"
    IMAGE_TO_TEXT = "image_to_text"
    # TEXT_TO_IMAGE = "text_to_image"


def get_state_dict_for_save(model: nn.Module) -> Dict:
    """
    Filter weight-sharing tensors.
    Referenced from transformers.modeling_utils.PreTrainedModel.save_pretrained.

    See https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L2369
    """

    state_dict = model.state_dict()

    # Safetensors does not allow tensor aliasing.
    # We're going to remove aliases before saving
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        # Sometimes in the state_dict we have non-tensor objects.
        # e.g. in bitsandbytes we have some `str` objects in the state_dict
        if isinstance(tensor, torch.Tensor):
            ptrs[id_tensor_storage(tensor)].append(name)
        else:
            # In the non-tensor case, fall back to the pointer of the object itself
            ptrs[id(tensor)].append(name)

    # These are all the pointers of shared tensors.
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
        logger.warning_once(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    return state_dict

# Call tied_weights() after load_checkpoint_in_model() to have the weights tied correctly.
def load_checkpoint_in_model_then_tie_weights(model, *args, **kwargs):
    accelerate.load_checkpoint_in_model(model, *args, **kwargs)
    model.tie_weights()
