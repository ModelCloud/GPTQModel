import copy
import json
import logging
import os
import re
from os.path import isfile, join
from typing import Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate.hooks import remove_hook_from_module
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights, shard_checkpoint
from transformers.utils.generic import ContextManagers

from ..quantization import GPTQ, QuantizeConfig
from ..quantization.config import (FORMAT, FORMAT_FIELD_JSON, META_FIELD_QUANTIZER,
                                   META_QUANTIZER_GPTQMODEL, MIN_VERSION_WITH_V2, QUANTIZE_BLACK_LIST)
from ..utils.bitblas import convert_to_bitblas, prepare_model_for_bitblas_load
from ..utils.data import collate_data
from ..utils.importer import select_quant_linear
from ..utils.marlin import (_validate_marlin_compatibility,
                            _validate_marlin_device_support, prepare_model_for_marlin_load)
from ..utils.model import (auto_dtype_from_config, convert_gptq_v1_to_v2_format, convert_gptq_v2_to_v1_format,
                           find_layers, get_checkpoints, get_device, get_module_by_name_prefix,
                           get_module_by_name_suffix, get_moe_layer_modules, gptqmodel_post_init, make_quant,
                           move_to, nested_move_to, pack_model, simple_dispatch_model, verify_model_hash,
                           verify_sharded_model_hashes)
from ..version import __version__
from ._const import CPU, CUDA_0, SUPPORTED_MODELS

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class BaseGPTQModel(nn.Module):
    # these modules are non-repeating and at the root level
    # does not include the node which holds all the repeating layers
    base_modules: List[str] = None

    # name of lm_head
    lm_head: str = "lm_head"

    # repeating layers
    # node holding all the repeating layers
    layers_node: str = None
    # repeating layer type
    layer_type: str = None
    # for each repeating layer there are multiple modules within each layer
    layer_modules: List[List[str]] = None

    # some models require trust_remove_code = True (dbrx_converted)
    require_trust_remote_code = None

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        # TODO: remove is_triton_backend arg..why? doesn't pass smell test @ZX-ModelCloud
        is_triton_backend: bool = False,
        qlinear_kernel: nn.Module = None,
    ):
        super().__init__()

        self.model = model
        self.model_type = self.model.config.model_type
        self._quantized = quantized
        self.quantize_config = quantize_config
        self.config = self.model.config

        self.is_triton_backend = is_triton_backend

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        self.qlinear_kernel = qlinear_kernel

    @property
    def quantized(self):
        return self._quantized

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)

    def _prepare_dataset_for_quantization(
            self,
            calibration_dataset: List[Dict[str, Union[List[int], torch.LongTensor]]],
            batch_size: int = 1,
    ):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_calibration_dataset = []
        for example in calibration_dataset:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            new_calibration_dataset.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError("Calibration data requires model's `pad_token_id` or `eos_token_id` to be set: actual = `None`.")

        new_calibration_dataset_batched = [
            collate_data(new_calibration_dataset[start: start + batch_size], pad_token_id)
            for start in range(0, len(new_calibration_dataset), batch_size)
        ]

        for new_example in new_calibration_dataset_batched:
            del new_example["labels"]

        return new_calibration_dataset_batched

    @torch.inference_mode()
    def quantize(
        self,
        calibration_dataset: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,

        # TODO: remove use_triton and use_cuda_fp16 arg..why? doesn't pass smell test @ZX-ModelCloud
        use_triton: bool = False,
        use_cuda_fp16: bool = True,

        autotune_warmup_after_quantized: bool = False,
        calibration_enable_gpu_cache: bool = True,
    ):
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if self.quantize_config.format == FORMAT.MARLIN:
            _validate_marlin_compatibility(self.quantize_config, throwError=True)

        # TODO: lm_head quantization is yet ready but pending
        if self.quantize_config.lm_head:
            raise ValueError("lm_head quantization is currently inference only and not applicable for quantization. Please set `lm_head=False`.")

        if len(calibration_dataset) == 0:
            raise ValueError("Calibration dataset must not be empty.")

        min_calibration_dataset_size = 256
        min_calibration_dataset_input_ids_avg_length = 256

        if len(calibration_dataset) < min_calibration_dataset_size:
            logger.warning(f"Calibration dataset size should be greater than {min_calibration_dataset_size}. "
                             f"Current size: {len(calibration_dataset)}.")

        # Calculate the average length of the average input_ids
        total_input_ids_length = 0
        for e in calibration_dataset:
            input_ids_length = len(e["input_ids"])
            total_input_ids_length += input_ids_length
        avg = total_input_ids_length / len(calibration_dataset)

        if avg < min_calibration_dataset_input_ids_avg_length:
            logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
                             f"{min_calibration_dataset_input_ids_avg_length}! Current AVG is {avg}.")


        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logger.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        calibration_dataset = self._prepare_dataset_for_quantization(calibration_dataset, batch_size)

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(calibration_dataset)
        layers = get_module_by_name_prefix(self.model, self.layers_node)

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to(inp, data_device))
            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs["attention_mask"] is not None:
                attention_masks.append(kwargs["attention_mask"].to(data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == CPU:
            layers[0] = layers[0].to(CUDA_0)
            force_layer_back_to_cpu = True

        ori_outside_layer_module_devices = {}
        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to(module, cur_layer_device)

        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for example in calibration_dataset:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        handle.remove()

        move_to(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        layer_modules = self.layer_modules

        if not self.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        # dynamic expert layer index for model defs
        if self.dynamic_expert_index is not None:
            num_experts = getattr(self.model.config, self.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                      num_experts=num_experts)

        quantizers = {}

        # stores all per-layer quant stats such as avg loss and processing time
        quant_log = []

        layer_count = len(layers)
        layer_pb = tqdm(range(layer_count))
        for i in layer_pb:
            layer_pb.set_description(f"Quantizing layer {i + 1} of {layer_count}")
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to(layer, CUDA_0)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = find_layers(layer)
            for names in layer_modules:
                subset = {n: full[n] for n in names if n in full}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        self.quantize_config.bits,
                        perchannel=True,
                        sym=self.quantize_config.sym,
                        mse=False,
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        # gptq is mutable.
                        gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = []
                    for k, layer_inp in enumerate(layer_inputs[j]):
                        layer_input.append(move_to(layer_inp, cur_layer_device))

                    mask = attention_masks[j]
                    layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

                    additional_layer_inputs = {"attention_mask": layer_attention_mask}
                    layer_position_ids = (
                        None if not position_ids else move_to(position_ids[j], cur_layer_device)
                    )
                    if layer_position_ids is not None:
                        additional_layer_inputs["position_ids"] = layer_position_ids
                    for k, v in layer_input_kwargs[j].items():
                        additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)
                    layer(*layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                for name in subset:
                    layer_pb.set_description(f"Quantizing {name} in layer {i + 1} of {layer_count}")

                    try:
                        scale, zero, g_idx, duration, avg_loss = gptq[name].fasterquant(
                            percdamp=self.quantize_config.damp_percent,
                            group_size=self.quantize_config.group_size,
                            actorder=self.quantize_config.desc_act,
                            static_groups=self.quantize_config.static_groups,
                        )

                        stat = {"layer": i + 1, "module": name, "avg_loss": f"{avg_loss:.4f}",
                                "time": f"{duration:.4f}"}

                        quant_log.append(stat)
                        logger.info(stat)

                    except torch._C._LinAlgError as e:
                        if "not positive-definite" in str(e).lower():
                            logger.warning(
                                "Please increase damp or nsamples for calibration data to avoid the following quant error. "
                            )
                        raise e

                    quantizers[f"{self.layers_node}.{i}.{name}"] = (
                        gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device),
                    )
                    gptq[name].free()

            for j in range(num_batches):
                layer_input = []
                for k, layer_inp in enumerate(layer_inputs[j]):
                    layer_input.append(move_to(layer_inp, cur_layer_device))

                mask = attention_masks[j]
                layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = None if not position_ids else move_to(position_ids[j], cur_layer_device)
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)
                layer_output = move_to(
                    layer(*layer_input, **additional_layer_inputs)[0],
                    cur_layer_device if calibration_enable_gpu_cache else CPU,
                )
                layer_outputs.append([layer_output])

            layers[i] = move_to(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = (
                layer_outputs,
                [],
            )  # TODO: is it really OK to cache only the first positional argument?
            torch.cuda.empty_cache()

        logger.info(f"Quantization summary:\n{quant_log}")
        for module_log in quant_log:
            logger.info(module_log)

        self.qlinear_kernel = pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=self.quantize_config.desc_act,
            warmup_triton=autotune_warmup_after_quantized,
            force_layer_back_to_cpu=force_layer_back_to_cpu,
            use_marlin=self.quantize_config.format == FORMAT.MARLIN,
            use_bitblas=self.quantize_config.format == FORMAT.BITBLAS,
        )
        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = simple_dispatch_model(self.model, device_map)
        self.model.config.use_cache = forward_pass_use_cache

        self._quantized = True

        torch.cuda.empty_cache()

        return quant_log

    @property
    def device(self):
        if not self.hf_device_map:
            return self.model.device
        else:
            device = [d for d in self.hf_device_map.values() if d not in {"disk"}][0]
            return torch.device(device)

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_quantized(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        format: Optional[FORMAT] = None,
        use_safetensors: bool = True,
        max_shard_size: Optional[str] = None,
        model_base_name: Optional[str] = None
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        # write autogptq tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=META_QUANTIZER_GPTQMODEL,
            version=__version__,
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)
        quantize_config = copy.deepcopy(self.quantize_config)
        model = self.model

        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear as BitBLASQuantLinear
            # BitBLASQuantLinear does not have a pack method and needs to be converted to BitBLAS format when saving.
            logger.info("Converting model to BitBlas Format...")
            model = convert_to_bitblas(model, self.qlinear_kernel, quantize_config, quantize_config.sym,
                                       quantize_config.desc_act, repack=True)
            self.qlinear_kernel = BitBLASQuantLinear
        if model_base_name is None:
            model_base_name = (
                    self.quantize_config.model_file_base_name or
                    f"gptq_model-{self.quantize_config.bits}bit-{self.quantize_config.group_size}g"
            )

        if format == FORMAT.GPTQ_V2 or (format is None and quantize_config.format == FORMAT.GPTQ_V2):
            logger.warning(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPTQModel version >= {MIN_VERSION_WITH_V2}."
            )

        if format is not None and quantize_config.format != format:
            # Model qzeros may be edited in place.
            # TODO: avoid inplace modification of the weights
            model = copy.deepcopy(self.model)

            if format == FORMAT.GPTQ_V2:
                if quantize_config.format != FORMAT.GPTQ:
                    raise NotImplementedError(
                        f"Asked to serialize a model with `format={format}` but the model format is {quantize_config.format}. This is not supported."
                    )

                model = convert_gptq_v1_to_v2_format(
                    model,
                    quantize_config=quantize_config,
                    qlinear_kernel=self.qlinear_kernel,
                )

                quantize_config.format = FORMAT.GPTQ_V2
            elif format == FORMAT.GPTQ:
                if quantize_config.format != FORMAT.GPTQ_V2:
                    raise NotImplementedError(
                        f"Asked to serialize a model with `format={format}` but the model format is {quantize_config.format}. This is not supported."
                    )

                model = convert_gptq_v2_to_v1_format(
                    model, quantize_config=quantize_config, qlinear_kernel=self.qlinear_kernel
                )

                quantize_config.format = FORMAT.GPTQ

        # internal is always gptq v2 but allow users to pass gptq (v1) via config
        if format is None and quantize_config.format == FORMAT.GPTQ:
            # Model qzeros may be edited in place.
            # TODO: avoid inplace modification of the weights
            # fix ModelCloud/GPTQModel/issues/47
            # fix gptqmodel_cuda cannot be serialized
            # no need to set it back, no calculation below
            if quantize_config.bits != 4:
                cuda_name_modules = {}
                from gptqmodel.nn_modules.qlinear.qlinear_cuda import BaseCudaQuantLinear
                for name, module in model.named_modules():
                    if isinstance(module, BaseCudaQuantLinear):
                        cuda_name_modules[name] = module.gptqmodel_cuda
                        module.gptqmodel_cuda = None
                model = copy.deepcopy(self.model)

                for name, modules in model.named_modules():
                    if isinstance(module, BaseCudaQuantLinear) and name in cuda_name_modules:
                        module.gptqmodel_cuda = cuda_name_modules[name]

                del cuda_name_modules
            else:
                model = copy.deepcopy(self.model)
            model = convert_gptq_v2_to_v1_format(
                model, quantize_config=quantize_config, qlinear_kernel=self.qlinear_kernel
            )

        model.to(CPU)

        state_dict = model.state_dict()

        if quantize_config.model_file_base_name is None:
            if use_safetensors:
                model_base_name = "model"
            else:
                model_base_name = "pytorch_model"
        else:
            model_base_name = quantize_config.model_file_base_name

        if use_safetensors:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            model_save_name = model_base_name + ".safetensors"
        else:
            model_save_name = model_base_name + ".bin"

        if not self.qlinear_kernel.SUPPORTED_SHARDS and max_shard_size is not None:
            logger.warning("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        if max_shard_size is None:
            if use_safetensors:
                if safetensors_metadata is None:
                    safetensors_metadata = {}
                elif not isinstance(safetensors_metadata, dict):
                    raise TypeError("safetensors_metadata must be a dictionary.")
                else:
                    logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                    new_safetensors_metadata = {}
                    converted_keys = False
                    for key, value in safetensors_metadata.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            converted_keys = True
                            try:
                                new_key = str(key)
                                new_value = str(value)
                            except Exception as e:
                                raise TypeError(
                                    f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                                )
                            if new_key in new_safetensors_metadata:
                                logger.warning(
                                    f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                                )
                            new_safetensors_metadata[new_key] = new_value
                    safetensors_metadata = new_safetensors_metadata
                    if converted_keys:
                        logger.debug(
                            f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                        )

                # Format is required to enable Accelerate to load the metadata
                # otherwise it raises an OSError
                safetensors_metadata["format"] = "pt"
                safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
            else:
                logger.warning(
                    "We highly suggest saving quantized model using safetensors format for security reasons. Please set `use_safetensors=True` whenever possible.")
                torch.save(model.state_dict(), join(save_dir, model_save_name))
        else:
            # Shard checkpoint
            shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=model_save_name)

            # Clean the folder from a previous save
            for filename in os.listdir(save_dir):
                full_filename = join(save_dir, filename)

                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                if (
                        filename.startswith(model_base_name)
                        and isfile(full_filename)
                        and filename not in shards.keys()
                        and reg.fullmatch(filename_no_suffix) is not None
                ):
                    os.remove(full_filename)

            # Save the model
            for shard_file, shard in shards.items():
                if use_safetensors:
                    if safetensors_metadata is None:
                        safetensors_metadata = {}
                    elif not isinstance(safetensors_metadata, dict):
                        raise TypeError("safetensors_metadata must be a dictionary.")
                    else:
                        logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                        new_safetensors_metadata = {}
                        converted_keys = False
                        for key, value in safetensors_metadata.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                converted_keys = True
                                try:
                                    new_key = str(key)
                                    new_value = str(value)
                                except Exception as e:
                                    raise TypeError(
                                        f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}")
                                if new_key in new_safetensors_metadata:
                                    logger.warning(
                                        f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting.")
                                new_safetensors_metadata[new_key] = new_value
                        safetensors_metadata = new_safetensors_metadata
                        if converted_keys:
                            logger.debug(
                                f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}")

                    # Format is required to enable Accelerate to load the metadata
                    # otherwise it raises an OSError
                    safetensors_metadata["format"] = "pt"

                    safe_save(shard, join(save_dir, shard_file), safetensors_metadata)
                else:
                    torch.save(shard, join(save_dir, shard_file))

            if index is not None:
                index_save_name = model_save_name + ".index.json"
                index_save_path = join(save_dir, index_save_name)
                # Save the index as well
                with open(index_save_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
        config.quantization_config = quantize_config.to_dict()
        config.save_pretrained(save_dir)

        quantize_config.model_name_or_path = save_dir
        quantize_config.model_file_base_name = model_base_name
        quantize_config.save_pretrained(save_dir)

    def save_pretrained(
        self,
        save_dir: str,
        **kwargs,
    ):
        logger.warning("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: QuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        torch_dtype: [str | torch.dtype] = "auto",
        **model_init_kwargs,
    ):
        """load un-quantized pretrained model to cpu"""

        if not torch.cuda.is_available():
            raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_name_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        # allow models to define optional notes that output messages to users that want to use this model
        notes = cls.info.get("notes")
        if notes:
            logger.info(notes)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        model_init_kwargs["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        if torch_dtype == "auto":
            torch_dtype = auto_dtype_from_config(config)
        elif not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"torch_dtype value of `{torch_dtype}` is not a torch.dtype instance.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.tie_weights()

            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
                low_zero=False,
            )
            model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
            )
            del model
        else:
            model_init_kwargs["device_map"] = None

        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return cls(model, quantized=False, quantize_config=quantize_config)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,

        # TODO: refract this bewildering amount of ugly args @ZX-ModelCloud
        # combine into Backend.ENUM class of Backend.AUTO, Backend.TRITON, Backend.MARLIN
        # single arp of backend: Backend = Backend.AUTO (default to auto)
        use_triton: bool = True,
        use_marlin: bool = True,
        use_bitblas: bool = False,
        disable_exllama: bool = False,
        disable_exllamav2: bool = False,

        torch_dtype: [str | torch.dtype] = "auto",
        use_cuda_fp16: bool = True,
        quantize_config: Optional[QuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        format: Optional[FORMAT] = None,
        allow_unsafe_loading: bool = False,
        verify_hash: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        """load quantized model from local disk"""
        # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
        if disable_exllama is None:
            if disable_exllamav2:
                disable_exllama = False
            else:
                disable_exllama = True

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{model_name_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        if not disable_exllamav2 and not disable_exllama:
            logger.warning(
                "You have activated both exllama and exllamav2 kernel. Setting disable_exllama to True and keeping disable_exllamav2 to False"
            )
            disable_exllama = True

        # == step1: prepare configs and file names == #
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )

        if torch_dtype == "auto":
            torch_dtype = auto_dtype_from_config(config, quant_inference=True)
        elif not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"torch_dtype value of `{torch_dtype}` is not a torch.dtype instance.")

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = QuantizeConfig.from_pretrained(
                model_name_or_path, format=format, **cached_file_kwargs, **kwargs
            )
        else:
            if not isinstance(quantize_config, QuantizeConfig):
                quantize_config = QuantizeConfig.from_quant_config(quantize_config, format)

        if quantize_config.format == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            use_marlin = True

        marlin_compatible = _validate_marlin_device_support()

        if not use_marlin:
            unsupported = _validate_marlin_compatibility(quantize_config)
            if unsupported is None and marlin_compatible:
                logger.info(
                    "You passed a model that is compatible with the Marlin int4*fp16 GPTQ kernel but use_marlin is False. We recommend using `use_marlin=True` to use the optimized Marlin kernels for inference. Example: `model = GPTQModel.from_quantized(..., use_marlin=True)`."
                )

        if quantize_config.format == FORMAT.BITBLAS:
            # format bitblas requires bitblas kernel
            use_bitblas = True

        if model_basename is None:
            if quantize_config.model_file_base_name:
                possible_model_basenames = [quantize_config.model_file_base_name]
            else:
                possible_model_basenames = [
                    f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
                    "model",
                ]
        else:
            possible_model_basenames = [model_basename]

        quantize_config.model_name_or_path = model_name_or_path

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)

        # Retrieve (and if necessary download) the quantized checkpoint(s).
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
            model_name_or_path=model_name_or_path,
            extensions=extensions,
            possible_model_basenames=possible_model_basenames,
            **cached_file_kwargs,
        )

        # bin files have security issues: disable loading by default
        if ".bin" in resolved_archive_file:
            if allow_unsafe_loading:
                logger.warning(
                    "There are security risks when loading tensors from .bin files. Make sure you are loading model only from a trusted source."
                )
            else:
                raise ValueError(
                    "Loading of unsafe .bin files are not allowed by default. Pass allow_unsafe_loading=True to bypass."
                )

        quantize_config.model_file_base_name = true_model_basename

        model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.
        if verify_hash:
            if is_sharded:
                verfieid = verify_sharded_model_hashes(model_save_name, verify_hash)
            else:
                verfieid = verify_model_hash(model_save_name, verify_hash)
            if not verfieid:
                raise ValueError(f"Hash verification failed for {model_save_name}")
            logger.info(f"Hash verification succeeded for {model_save_name}")
        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        if torch_dtype != torch.float16:
            logger.warning("Overriding use_cuda_fp16 to False since torch_dtype is not torch.float16.")
            use_cuda_fp16 = False

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]

        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )

            if cls.dynamic_expert_index is not None:
                num_experts = getattr(config, cls.dynamic_expert_index)
                cls.layer_modules = get_moe_layer_modules(layer_modules=cls.layer_modules,
                                                          num_experts=num_experts)

            layers = find_layers(model)
            ignore_layers = [cls.lm_head] + cls.base_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == cls.lm_head:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                        not name.endswith(ignore_layer) for sublist in cls.layer_modules for ignore_layer in sublist
                ):
                    # log non-lm-head quantizerd layers only
                    if name is not cls.lm_head:
                        logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act,
                use_marlin=quantize_config.format == FORMAT.MARLIN,
                use_bitblas=quantize_config.format == FORMAT.BITBLAS,
            )
            model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == "cuda" else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = accelerate.utils.get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type],
                    low_zero=(device_map == "balanced_low_0"),
                )
        if not isinstance(device_map, dict):
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
            )

        if use_marlin:
            if is_sharded:
                raise ValueError(
                    "The loading of sharded checkpoints with Marlin is currently not supported."
                )
            if not _validate_marlin_device_support():
                raise ValueError(
                    f'Marlin kernel does not support this gpu with compute capability of `{torch.cuda.get_device_capability()}`. Please do not use `use_marlin=True`.'
                )

            # Validate the model can run in Marlin.
            if torch_dtype != torch.float16:
                raise ValueError("Marlin kernel requires torch_dtype=torch.float16.")

            _validate_marlin_compatibility(quantize_config, throwError=True)

            # Load the quant linear type we need.
            # TODO: load marlin directly with the right quantlinear class.
            quant_linear_class = select_quant_linear(
                bits=quantize_config.bits,
                group_size=quantize_config.group_size,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
                use_triton=use_triton,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_marlin=False,
                use_bitblas=False,
            )

            # Prepare model for marlin load.
            # If is marlin serialized load then load directly. Otherwise, convert to marlin.
            model = prepare_model_for_marlin_load(
                model=model,
                quantize_config=quantize_config,
                quant_linear_class=quant_linear_class,
                torch_dtype=torch_dtype,
                current_model_save_name=model_save_name,
                device_map=device_map,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
            )

        if use_bitblas:
            if is_sharded:
                raise ValueError(
                    "The loading of sharded checkpoints with BitBLAS is currently not supported. Please raise an issue in GPTQModel repository.")

            # Load the quant linear type we need.
            # TODO: load directy bitblas with the right quantlinear class.
            quant_linear_class = select_quant_linear(
                bits=quantize_config.bits,
                group_size=quantize_config.group_size,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
                use_triton=use_triton,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_marlin=False,
                use_bitblas=False,
            )

            # Prepare model for bitblas load.
            # If is bitblas serialized load then load directly. Otherwise, convert to bitblas.
            model = prepare_model_for_bitblas_load(
                model=model,
                quantize_config=quantize_config,
                quant_linear_class=quant_linear_class,
                torch_dtype=torch_dtype,
                model_save_name=model_save_name,
                device_map=device_map,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
            )

        # If we use marlin or bitblas to load the quantized model, the model is already a converted model,
        # and we no longer need to call load_checkpoint_in_model()
        if not (use_marlin or use_bitblas):
            accelerate.utils.modeling.load_checkpoint_in_model(
                model,
                dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )

        # TODO: Why are we using this custom function and not dispatch_model?
        model = simple_dispatch_model(model, device_map)

        qlinear_kernel = select_quant_linear(
            bits=quantize_config.bits,
            group_size=quantize_config.group_size,
            desc_act=quantize_config.desc_act,
            sym=quantize_config.sym,
            use_triton=use_triton,
            disable_exllama=disable_exllama,
            disable_exllamav2=disable_exllamav2,
            use_marlin=use_marlin,
            use_bitblas=use_bitblas,
        )

        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if quantize_config.format == FORMAT.GPTQ:
            # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
            if not quantize_config.sym and not quantize_config.is_quantized_or_packed_by_v2():
                raise ValueError(
                    f"Loading of a sym=False model with format={FORMAT.GPTQ} is only supported if produced by gptqmodel version >= {MIN_VERSION_WITH_V2}"
                )

            logger.info(f"Compatibility: converting `{FORMAT_FIELD_JSON}` from `{FORMAT.GPTQ}` to `{FORMAT.GPTQ_V2}`.")

            model = convert_gptq_v1_to_v2_format(
                model,
                quantize_config=quantize_config,
                qlinear_kernel=qlinear_kernel,
            )

            quantize_config.format = FORMAT.GPTQ_V2

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # Any post-initialization that require device information, for example buffers initialization on device.
        model = gptqmodel_post_init(model, use_act_order=quantize_config.desc_act)

        model.eval()

        # == step6: (optional) warmup triton == #
        if use_triton and warmup_triton:
            from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear

            QuantLinear.warmup(model, seqlen=model.seqlen)

        return cls(
            model,
            quantized=True,
            quantize_config=quantize_config,
            is_triton_backend=use_triton,
            qlinear_kernel=qlinear_kernel,
        )

    def warmup_triton(self, enabled: bool = True):
        if not enabled:
            return

        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear

        QuantLinear.warmup(self.model, seqlen=self.model.seqlen)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)


__all__ = ["BaseGPTQModel"]
