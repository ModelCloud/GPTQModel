from __future__ import annotations

import functools
import hashlib
import json
import operator
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import accelerate
import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from huggingface_hub import HfApi, hf_hub_download
from packaging import version
from transformers import AutoConfig, PretrainedConfig
from transformers.utils.hub import cached_file

from .backend import BACKEND
from .importer import select_quant_linear
from .logger import setup_logger
from .progress import ProgressBar
from ..models._const import CPU, EXPERT_INDEX_PLACEHOLDER, SUPPORTED_MODELS
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_marlin_inference import MarlinInferenceQuantLinear
from ..quantization import FORMAT, QuantizeConfig

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
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
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


def make_quant(
    module,
    names,
    bits: int,
    group_size: int,
    backend: BACKEND,
    format: str | FORMAT,
    desc_act: bool = False,
    sym: bool = True,
    pack: bool = False,
    dynamic=None,
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
    )

    try:
        result = create_quant_layer(QuantLinear, bits, desc_act, dynamic, group_size, module, names, sym)
    except NotImplementedError as e:
        if QuantLinear == MarlinInferenceQuantLinear:
            # If create MarlinInferenceQuantLinear fails, we try to convert to MarlinQuantLinear.
            # First use ExllamaV2QuantLinear to preload, then call convert_to_marlin().
            result = create_quant_layer(ExllamaV2QuantLinear, bits, desc_act, dynamic, group_size, module, names, sym)
        else:
            raise e

    return result


def create_quant_layer(QuantLinear, bits, desc_act, dynamic, group_size, module, names, sym) -> BaseQuantLinear:
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
            else:
                raise NotImplementedError(f"Unsupported module {submodule}")

            bias = submodule.bias is not None

            d_bits = bits
            d_group_size = group_size
            d_sym = sym
            # dynamic bits, group_size, sym for each layer/module
            if dynamic is not None:
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
                weight_dtype=submodule.weight.dtype,
            )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))
    return QuantLinear


def convert_gptq_v1_to_v2_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: nn.Module,
):
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


def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: nn.Module,
):
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
        if QuantLinear is MarlinQuantLinear:
            qlayers[name].pack(layers[name], scale)
        else:
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
    desc_act=False,
    sym: bool = True,
    force_layer_back_to_cpu: bool = False,
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

    if force_layer_back_to_cpu:
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
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

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
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None,
                        max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    # post init for bitblas backend.

    # exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaV2QuantLinear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from ..nn_modules.qlinear.qlinear_exllamav2 import ExLlamaV2DeviceTensors

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

    torch.cuda.empty_cache()

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
def auto_dtype_from_config(config: PretrainedConfig, quant_inference: bool = False) -> torch.dtype:
    # all the gptq inference kernels are float16 only
    if quant_inference:
        return torch.float16

    dtype = getattr(config, "torch_dtype")
    if not dtype or not isinstance(dtype, torch.dtype):
        raise ValueError("Your model config.json does not have torch_dtype set. Please check for model " "corruption.")

    if dtype == torch.float32:
        return torch.bfloat16
    elif dtype == torch.float16:
        return torch.float16
    else:
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
