# Adapted from https://github.com/huggingface/accelerate

import functools
import gc
import json
import logging
import os
import shutil
import tempfile
from logging import getLogger
from typing import Dict, List, Optional, Union

import accelerate
import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from accelerate.utils import (check_tied_parameters_in_config, check_tied_parameters_on_same_device,
                              find_tied_parameters, load_offloaded_weights, load_state_dict, offload_weight,
                              retie_parameters, save_offload_index, set_module_tensor_to_device)
from accelerate.utils.constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig
from transformers.utils.hub import cached_file

from ..models._const import CPU, CUDA_0, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH, EXPERT_INDEX_PLACEHOLDER, SUPPORTED_MODELS
from ..nn_modules.qlinear import BaseQuantLinear
from ..quantization import QuantizeConfig
from .importer import select_quant_linear

logger = getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    bits,
    group_size,
    use_triton: bool = False,
    use_marlin: bool = False,
    disable_exllama: bool = False,
    disable_exllamav2: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
):
    QuantLinear = select_quant_linear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        use_marlin=use_marlin,
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
    )

    if isinstance(module, QuantLinear):
        return

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
            bias = submodule.bias is not None
            if (not (desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    use_cuda_fp16=use_cuda_fp16,
                    weight_dtype=submodule.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=submodule.weight.dtype,
                )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))


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


def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    warmup_triton: bool = False,
    force_layer_back_to_cpu: bool = False,
    use_marlin: bool = False,
):
    QuantLinear = select_quant_linear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_exllama=False,
        disable_exllamav2=True,
        use_marlin=use_marlin,
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
        use_triton=use_triton,
        use_cuda_fp16=use_cuda_fp16,
        desc_act=desc_act,
        disable_exllama=False,
        disable_exllamav2=True,
        use_marlin=use_marlin,
    )
    qlayers = find_layers(model, [QuantLinear])

    # Limit pack() thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        pbar = tqdm(qlayers.keys(), leave=True)
        for name in pbar:
            pbar.set_description(f"Packing {name}")

            quantizers[name], scale, zero, g_idx = quantizers[name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to(CPU)
            layers[name], scale, zero, g_idx = (
                layers[name].to(CPU),
                scale.to(CPU),
                zero.to(CPU),
                g_idx.to(CPU),
            )
            if QuantLinear.QUANT_TYPE == "marlin":
                qlayers[name].pack(layers[name], scale)
            else:
                qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)

    logger.info("Model packed.")

    if use_triton and warmup_triton:
        logger.warning(
            "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
        )
        QuantLinear.warmup(model.to(CUDA_0), seqlen=model.seqlen)
    return QuantLinear


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


def gptqmodel_post_init(model, use_act_order: bool, max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    for name, submodule in model.named_modules():
        if isinstance(submodule, BaseQuantLinear) and submodule.QUANT_TYPE == "exllama":
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1,
                }

            if not use_act_order:
                submodule._use_act_order = False
            else:
                submodule._use_act_order = True

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
        try:
            from exllama_kernels import prepare_buffers, set_tuning_params
        except ImportError as e:
            raise ImportError(
                f"Could not import exllama backend dependencies prepare_buffers, set_tuning_params with the following error: {e}"
            )

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

        # The buffers need to have been initialized first before calling make_q4.
        for name, submodule in model.named_modules():
            if isinstance(submodule, BaseQuantLinear) and submodule.QUANT_TYPE == "exllama":
                submodule.post_init()

    # exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if isinstance(submodule, BaseQuantLinear) and submodule.QUANT_TYPE == "exllamav2":
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

        for _, submodule in model.named_modules():
            if isinstance(submodule, BaseQuantLinear) and submodule.QUANT_TYPE == "exllamav2":
                device = submodule.qweight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model


def get_checkpoints(
    model_name_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs
):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_name_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_name_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    # The model is sharded over several checkpoints.
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_name_or_path, possible_model_basename)
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
                    model_name_or_path,
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
                                model_name_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_name_or_path,
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
            f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
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

# This code is sourced from https://github.com/huggingface/accelerate/pull/2880.
# Until the PR is merged, we need to incorporate this code into our codebase.
# Once the PR is merged, we can remove the code, relying instead on accelerate.utils.modeling.load_checkpoint_in_model.
def load_checkpoint_in_model(
        model: nn.Module,
        checkpoint: Union[str, os.PathLike],
        device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        offload_state_dict: bool = False,
        offload_buffers: bool = False,
        keep_in_fp32_modules: List[str] = None,
        offload_8bit_bnb: bool = False,
        strict: bool = False,
        ignore_unexpected_keys: bool = False,
):
    """
        Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
        loaded.

        <Tip warning={true}>

        Once loaded across devices, you still need to call [`dispatch_model`] on your model to make it able to run. To
        group the checkpoint loading and dispatch in one single call, use [`load_checkpoint_and_dispatch`].

        </Tip>

        Args:
            model (`torch.nn.Module`):
                The model in which we want to load a checkpoint.
            checkpoint (`str` or `os.PathLike`):
                The folder checkpoint to load. It can be:
                - a path to a file containing a whole model state dict
                - a path to a `.json` file containing the index to a sharded checkpoint
                - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
                - a path to a folder containing a unique pytorch_model.bin or a model.safetensors file.
            device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
                name, once a given module name is inside, every submodule of it will be sent to the same device.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            dtype (`str` or `torch.dtype`, *optional*):
                If provided, the weights will be converted to that type when loaded.
            offload_state_dict (`bool`, *optional*, defaults to `False`):
                If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
                the weight of the CPU state dict + the biggest shard does not fit.
            offload_buffers (`bool`, *optional*, defaults to `False`):
                Whether or not to include the buffers in the weights offloaded to disk.
            keep_in_fp32_modules(`List[str]`, *optional*):
                A list of the modules that we keep in `torch.float32` dtype.
            offload_8bit_bnb (`bool`, *optional*):
                Whether or not to enable offload of 8-bit modules on cpu/disk.
            strict (`bool`, *optional*, defaults to `False`):
                Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
                state_dict.
            ignore_unexpected_keys (`bool`, *optional*, defaults to `False`):
                Whether to log warning for unexpected keys in checkpoint state_dict. Applicable for quantized models.

        """
    if offload_8bit_bnb:
        from accelerate.utils.bnb import quantize_and_offload_8bit

    tied_params = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_params) == 0:
        logger.warn(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )
    if device_map is not None:
        check_tied_parameters_on_same_device(tied_params, device_map)

    if offload_folder is None and device_map is not None and "disk" in device_map.values():
        raise ValueError(
            "At least one of the model submodule will be offloaded to disk, please pass along an `offload_folder`."
        )
    elif offload_folder is not None and device_map is not None and "disk" in device_map.values():
        os.makedirs(offload_folder, exist_ok=True)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        # check if the whole state dict is present
        potential_state_bin = [f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME]
        potential_state_safetensor = [f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME]
        if len(potential_state_bin) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
        elif len(potential_state_safetensor) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
        else:
            # otherwise check for sharded checkpoints
            potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
            if len(potential_index) == 0:
                raise ValueError(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                )
            elif len(potential_index) == 1:
                index_filename = os.path.join(checkpoint, potential_index[0])
            else:
                raise ValueError(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename) as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.

    offload_index = {}
    if offload_state_dict:
        state_dict_folder = tempfile.mkdtemp()
        state_dict_index = {}

    unexpected_keys = set()
    model_keys = set(model.state_dict().keys())
    buffer_names = [name for name, _ in model.named_buffers()]
    for checkpoint_file in checkpoint_files:
        loaded_checkpoint = load_state_dict(checkpoint_file, device_map=device_map)
        if device_map is None:
            model.load_state_dict(loaded_checkpoint, strict=strict)
            unexpected_keys.update(set(loaded_checkpoint.keys()) - model_keys)
        else:
            for param_name, param in loaded_checkpoint.items():
                # skip SCB parameter (for 8-bit serialization)
                if "SCB" in param_name:
                    continue

                if param_name not in model_keys:
                    unexpected_keys.add(param_name)
                    if not strict:
                        continue  # Skip loading this parameter.

                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]
                new_dtype = dtype
                if dtype is not None and torch.is_floating_point(param):
                    if keep_in_fp32_modules is not None and dtype == torch.float16:
                        proceed = False
                        for key in keep_in_fp32_modules:
                            if ((key in param_name) and (key + "." in param_name)) or key == param_name:
                                proceed = True
                                break
                        if proceed:
                            new_dtype = torch.float32

                if "weight" in param_name and param_name.replace("weight", "SCB") in loaded_checkpoint.keys():
                    if param.dtype == torch.int8:
                        fp16_statistics = loaded_checkpoint[param_name.replace("weight", "SCB")]
                else:
                    fp16_statistics = None

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        if new_dtype is None:
                            new_dtype = param.dtype
                        if offload_8bit_bnb:
                            quantize_and_offload_8bit(
                                model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics
                            )
                            continue
                        else:
                            set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    if new_dtype is None:
                        new_dtype = param.dtype
                    if offload_8bit_bnb:
                        quantize_and_offload_8bit(
                            model, param, param_name, new_dtype, state_dict_folder, state_dict_index, fp16_statistics
                        )
                    else:
                        set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(
                        model,
                        param_name,
                        param_device,
                        value=param,
                        dtype=new_dtype,
                        fp16_statistics=fp16_statistics,
                    )

        # Force Python to clean up.
        del loaded_checkpoint
        gc.collect()

    if not strict and not ignore_unexpected_keys and len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {checkpoint} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}. This may or may not be an issue - make sure that the checkpoint does not have unnecessary parameters, or that the model definition correctly corresponds to the checkpoint."
        )

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)
