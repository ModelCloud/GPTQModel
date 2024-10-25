from __future__ import annotations

import copy
import json
import logging
import os
import re
from os.path import isfile, join
from typing import Dict, List, Optional, Union

import accelerate
import lm_eval
import torch
import torch.nn as nn
import transformers
from accelerate.hooks import remove_hook_from_module
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable
from packaging import version
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_utils import no_init_weights, shard_checkpoint
from transformers.models.mllama.modeling_mllama import MllamaCrossAttentionDecoderLayer
from transformers.utils.generic import ContextManagers

from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_qbits import QBitsQuantLinear, qbits_dtype
from ..quantization import GPTQ, QuantizeConfig
from ..quantization.config import (FORMAT, FORMAT_FIELD_JSON, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT,
                                   META_FIELD_QUANTIZER, META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI,
                                   MIN_VERSION_WITH_V2, QUANTIZE_BLACK_LIST, AutoRoundQuantizeConfig)
from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.importer import select_quant_linear
from ..utils.marlin import (_validate_marlin_compatibility,
                            _validate_marlin_device_support, prepare_model_for_marlin_load)
from ..utils.model import (auto_dtype_from_config, check_to_quantized, convert_gptq_v1_to_v2_format,
                           convert_gptq_v2_to_v1_format, copy_py_files, find_layers, get_checkpoints, get_device,
                           get_model_files_size, get_module_by_name_prefix, get_module_by_name_suffix,
                           get_moe_layer_modules, gptqmodel_post_init, make_quant, move_to, nested_move_to, pack_model,
                           simple_dispatch_model, verify_model_hash, verify_sharded_model_hashes)
from ..version import __version__
from ._const import CPU, CUDA_0, DEVICE, SUPPORTED_MODELS
from .loader import ModelLoader

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
    layer_type: Union[List[str], str] = None
    # for each repeating layer there are multiple modules within each layer
    layer_modules: List[List[str]] = None

    # some models require trust_remove_code = True (dbrx_converted)
    require_trust_remote_code = None

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    model_loader = AutoModelForCausalLM

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_name_or_path: str = None,
    ):
        super().__init__()

        self.model = model
        self._quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.quantize_config = quantize_config
        self.config = self.model.config

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_name_or_path = model_name_or_path

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
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
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
            if tokenizer:
                vocab = tokenizer.get_vocab()

                # auto select the best pad token to use
                for token in ["<|finetune_right_pad_id|>", "<|pad|>", "<pad>", "<|unk|>", "<unk>"]:
                    token_id = vocab.get(token)
                    if token_id is not None:
                        pad_token_id = token_id
                        break
            else:
                logger.warning("Model config does not have pad token mapped. Please pass in tokenizer to `quantize()` so GPTQModel can auto-select the best pad token.")

            if not pad_token_id and isinstance(self.config.eos_token_id, list): # Llama-3.1-8B-Instruct's eos_token_id is a list
                pad_token_id = self.config.eos_token_id[0]
            elif not pad_token_id:
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

    def quantize(
        self,
        calibration_dataset: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if self.quantize_config.format == FORMAT.MARLIN:
            _validate_marlin_compatibility(self.quantize_config, throw_error=True)

        if self.quantize_config.lm_head and not isinstance(self.quantize_config, AutoRoundQuantizeConfig):
            raise ValueError("`lm_head=True` quantization is only available with AutoRound quantizer. Please use `AutoRoundQuantizeConfig` instead of `QuantizeConfig` and set `lm_head=True` or set `lm_head=False`.")

        if len(calibration_dataset) == 0:
            raise ValueError("Calibration dataset must not be empty.")

        # Validate quant linear before quantization starts
        _ = select_quant_linear(
            bits=self.quantize_config.bits,
            dynamic=self.quantize_config.dynamic,
            group_size=self.quantize_config.group_size,
            desc_act=self.quantize_config.desc_act,
            sym=self.quantize_config.sym,
            backend=BACKEND.AUTO,
            format=self.quantize_config.format,
        )

        min_calibration_dataset_size = 256
        min_calibration_dataset_input_ids_avg_length = 256

        if len(calibration_dataset) < min_calibration_dataset_size:
            logger.warning(f"Calibration dataset size should be greater than {min_calibration_dataset_size}. "
                             f"Current size: {len(calibration_dataset)}.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.qlinear_bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # Calculate the average length of the average input_ids
        total_input_ids_length = 0
        max_input_id_length = 0
        for row in calibration_dataset:
            input_ids = row["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 1:
                    input_ids_length = input_ids.shape[0]
                else:
                    raise ValueError("Expected a 1-dimensional tensor for 'input_ids', but got a tensor with {0} dimensions.".format(input_ids.dim()))
            else:
                input_ids_length = len(input_ids)

            if input_ids_length > max_input_id_length:
                max_input_id_length = input_ids_length
            total_input_ids_length += input_ids_length
        avg = total_input_ids_length / len(calibration_dataset)

        if avg < min_calibration_dataset_input_ids_avg_length:
            logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
                             f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logger.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        calibration_dataset = self._prepare_dataset_for_quantization(calibration_dataset, batch_size, tokenizer,)

        if isinstance(self.quantize_config, AutoRoundQuantizeConfig):
            from auto_round import AutoRound
            from auto_round import __version__ as auto_round_version

            if version.parse(auto_round_version) < version.parse("0.3.0"):
                raise ValueError(f"AutoRound version must be >= 0.3.0: actual = {auto_round_version}")

            if self.quantize_config.lm_head:
                self.quantize_config.layer_config['lm_head'] = {"data_type": "int"}

            import torch.nn.functional as F
            from torch.utils.data import DataLoader

            # set the nsamples/seqlen according to the actual size of the calibration_dataset.
            nsamples = len(calibration_dataset)
            seqlen = max_input_id_length

            @torch.no_grad()
            def collate_batch(batch):
                input_ids_new = []
                attention_mask_new = []
                for text in batch:
                    input_ids, attention_mask = text["input_ids"][0], text["attention_mask"][0]

                    input_ids = input_ids[:seqlen]
                    input_ids_new.append(input_ids)

                    attention_mask = attention_mask[:seqlen]
                    attention_mask_new.append(attention_mask)

                if len(input_ids_new) == 0:
                    return None

                input_ids_new = [F.pad(t, (0, seqlen - t.size(0))) for t in input_ids_new]
                attention_mask_new = [F.pad(t, (0, seqlen - t.size(0))) for t in attention_mask_new]

                input_ids_new = torch.vstack(input_ids_new)
                attention_mask_new = torch.vstack(attention_mask_new)
                res = {"input_ids": input_ids_new, "attention_mask": attention_mask_new}
                return res

            dataloader = DataLoader(calibration_dataset, collate_fn=collate_batch, shuffle=False, batch_size=nsamples)

            self.autoround = AutoRound(self.model,
                                  tokenizer=None,
                                  bits=self.quantize_config.bits,
                                  group_size=self.quantize_config.group_size,
                                  sym=self.quantize_config.sym, batch_size=batch_size, n_samples=nsamples,
                                  dataset=dataloader, seqlen=seqlen, nblocks=self.quantize_config.nblocks,
                                  iters=self.quantize_config.iters, lr=self.quantize_config.lr,
                                  minmax_lr=self.quantize_config.minmax_lr,
                                  enable_quanted_input=self.quantize_config.enable_quanted_input,
                                  device=self.hf_device_map,
                                  amp=self.quantize_config.amp,
                                  low_gpu_mem_usage=self.quantize_config.low_gpu_mem_usage,
                                  seed=self.quantize_config.seed,
                                  gradient_accumulate_steps=self.quantize_config.gradient_accumulate_steps,
                                  scale_dtype=self.quantize_config.scale_dtype, layer_config=self.quantize_config.layer_config,
                                  enable_minmax_tuning=self.quantize_config.enable_minmax_tuning)

            model, _ = self.autoround.quantize()

            quantizers = {}
            for key in self.autoround.layer_config:
                info = self.autoround.layer_config[key]
                if not check_to_quantized(info):
                    continue
                quantizers[key] = (None, info["scale"], info["zp"].to(torch.float32), None)

            self.qlinear_kernel = pack_model(
                model=self.model,
                quantizers=quantizers,
                bits=self.quantize_config.bits,
                dynamic=self.quantize_config.dynamic,
                group_size=self.quantize_config.group_size,
                backend=BACKEND.AUTO,
                desc_act=self.quantize_config.desc_act,
                force_layer_back_to_cpu=True,
                format=self.quantize_config.format,
                parallel_packing=self.quantize_config.parallel_packing,
            )

            self.model = model
            self._quantized = True
            return

        forward_pass_use_cache = self.model.config.use_cache if hasattr(self.model.config, "use_cache") else False
        self.model.config.use_cache = False

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

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
            layer_pb.set_description(f"Quantizing layer {i} of {layer_count - 1}")
            layer = layers[i]
            if isinstance(layer, MllamaCrossAttentionDecoderLayer):
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

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
                    bits = self.quantize_config.bits
                    sym = self.quantize_config.sym
                    if self.quantize_config.dynamic is not None:
                        layer_name = f"{self.layers_node}.{i}.{name}"
                        bits = self.quantize_config.dynamic_get(layer_name, "bits", bits)
                        sym = self.quantize_config.dynamic_get(layer_name, "sym", sym)
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        bits,
                        perchannel=True,
                        sym=sym,
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
                    with torch.no_grad():
                        layer(*layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                for name in subset:
                    layer_pb.set_description(f"Quantizing {name} in layer {i} of {layer_count - 1}")

                    group_size = self.quantize_config.group_size
                    actorder = self.quantize_config.desc_act
                    if self.quantize_config.dynamic is not None:
                        layer_name = f"{self.layers_node}.{i}.{name}"
                        group_size = self.quantize_config.dynamic_get(layer_name, "group_size", group_size)
                        actorder = self.quantize_config.dynamic_get(layer_name, "actorder", actorder)

                    scale, zero, g_idx, duration, avg_loss, damp_percent = gptq[name].fasterquant(
                        percdamp=self.quantize_config.damp_percent,
                        group_size=group_size,
                        actorder=actorder,
                        static_groups=self.quantize_config.static_groups,
                    )
                    stat = {"layer": i, "module": name, "avg_loss": f"{avg_loss:.5f}",
                            "damp_percent": f"{damp_percent:.5f}", "time": f"{duration:.3f}"}
                    if self.quantize_config.dynamic is not None:
                        stat["dynamic"] = self.quantize_config.dynamic_get(layer_name=layer_name)

                    quant_log.append(stat)
                    logger.info(stat)

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
                with torch.no_grad():
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
            backend=BACKEND.AUTO,
            desc_act=self.quantize_config.desc_act,
            force_layer_back_to_cpu=force_layer_back_to_cpu,
            format=self.quantize_config.format,
            dynamic=self.quantize_config.dynamic,
            parallel_packing=self.quantize_config.parallel_packing,
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
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_quantized(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        use_safetensors: bool = True,
        max_shard_size: Optional[str] = None,
        model_base_name: Optional[str] = None
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        pre_quantized_size_mb = get_model_files_size(self.model_name_or_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        # write gptqmodel tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=META_QUANTIZER_GPTQMODEL,
            version=__version__,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=self.quantize_config.damp_percent
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=self.quantize_config.damp_auto_increment
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)
        quantize_config = copy.deepcopy(self.quantize_config)

        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if model_base_name is None:
            model_base_name = (
                    f"gptq_model-{self.quantize_config.bits}bit-{self.quantize_config.group_size}g"
            )

        if quantize_config.format == FORMAT.GPTQ_V2:
            logger.warning(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPTQModel version >= {MIN_VERSION_WITH_V2}."
            )

        if not self.load_quantized_model:
            model = self.model
            # # internal is always gptq v2 but allow users to pass gptq (v1) via config
            if quantize_config.format == FORMAT.GPTQ:
                # Model qzeros may be edited in place.
                model = convert_gptq_v2_to_v1_format(
                    model, quantize_config=quantize_config, qlinear_kernel=self.qlinear_kernel
                )
        else:
            model = self.get_model_with_quantize(quantize_config, self.model_name_or_path)
        model.to(CPU)
        state_dict = model.state_dict()

        model_base_name = "model"

        if use_safetensors:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            model_save_name = model_base_name + ".safetensors"
        else:
            model_save_name = model_base_name + ".pt"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
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
            total_size_mb = os.path.getsize(join(save_dir, model_save_name)) / (1024 * 1024)
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

            total_size_mb = 0
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
                shard_size_mb = os.path.getsize(join(save_dir, shard_file)) / (1024 * 1024)
                total_size_mb += shard_size_mb

            if index is not None:
                index_save_name = model_save_name + ".index.json"
                index_save_path = join(save_dir, index_save_name)
                # Save the index as well
                with open(index_save_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)

        total_size_gb = total_size_mb / 1024
        size_diff_mb = pre_quantized_size_mb - total_size_mb
        size_diff_gb = size_diff_mb / 1024
        percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
        logger.info(f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB")
        logger.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
        logger.info(f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%")

        config.quantization_config = quantize_config.to_dict()
        config.save_pretrained(save_dir)

        quantize_config.save_pretrained(save_dir)

        # need to copy .py files for model/tokenizers not yet merged to HF transformers
        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_name_or_path)

    def get_model_with_quantize(self, quantize_config, model_name_or_path):
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        transformers.modeling_utils._init_weights = False
        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = self.model_loader.from_config(
                config, torch_dtype=torch.float16
            )

            if self.dynamic_expert_index is not None:
                num_experts = getattr(config, self.dynamic_expert_index)
                self.layer_modules = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                           num_experts=num_experts)

            layers = find_layers(model)
            ignore_layers = [self.lm_head] + self.base_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == self.lm_head:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                        not name.endswith(ignore_layer) for sublist in self.layer_modules for ignore_layer in sublist
                ):
                    # log non-lm-head quantizerd layers only
                    if name is not self.lm_head:
                        logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                backend=BACKEND.AUTO,
                format=quantize_config.format,
                desc_act=quantize_config.desc_act,
                pack=True,
            )
            model.tie_weights()

        accelerate.load_checkpoint_in_model(
            model,
            dtype=torch.float16,
            # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=self.checkpoint_file_name,
            # device_map=device_map,
            # offload_state_dict=True,
            # offload_buffers=True,
        )
        torch.cuda.empty_cache()
        return model

    def save_pretrained(
        self,
        save_dir: str,
        **kwargs,
    ):
        logger.warning("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    def lm_eval(
        self,
        tasks: Optional[List[Union[str, dict, object]]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = 32,
        max_batch_size: Optional[int] = 64,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = True,
        evaluation_tracker: Optional[EvaluationTracker] = None,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_kwargs: Optional[str] = None,
        task_manager: Optional[TaskManager] = None,
        verbosity: str = "INFO",
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234,
        output_path: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        show_config: bool = False,
    ):
        LM = HFLM(
            pretrained=self,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        # evaluation_tracker need model_args cannot be None
        model_args = ""
        if evaluation_tracker is None and output_path is not None:
            evaluation_tracker = EvaluationTracker(output_path=output_path)

        results = lm_eval.simple_evaluate(
            model=LM,
            model_args=model_args,
            tasks=tasks,
            device=self.device,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            use_cache=use_cache,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            delete_requests_cache=delete_requests_cache,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            check_integrity=check_integrity,
            write_out=write_out,
            log_samples=log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            task_manager=task_manager,
            verbosity=verbosity,
            predict_only=predict_only,
            random_seed=random_seed,
            numpy_random_seed=numpy_random_seed,
            torch_random_seed=torch_random_seed,
            fewshot_random_seed=fewshot_random_seed,
        )

        if results is not None:
            if log_samples:
                samples = results.pop("samples")

            dumped = json.dumps(
                results, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
            if show_config:
                print(dumped)

            # Add W&B logging
            if wandb_project is not None:
                wandb_logger = WandbLogger(
                    project=wandb_project, job_type="eval", name=wandb_name
                )
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if log_samples:
                    wandb_logger.log_eval_samples(samples=samples)

            evaluation_tracker.save_results_aggregated(
                results=results, samples=samples if log_samples else None
            )

            if log_samples:
                for task_name, config in results["configs"].items():
                    evaluation_tracker.save_results_samples(
                        task_name=task_name, samples=samples[task_name]
                    )

            if (evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub):
                evaluation_tracker.recreate_metadata_card()

            return results
        else:
            raise ValueError('lm_eval run fail, check your code!!!')

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: QuantizeConfig,
        trust_remote_code: bool = False,
        use_liger_kernel: bool = False,
        torch_dtype: [str | torch.dtype] = "auto",
        **model_init_kwargs,
    ):
        model = ModelLoader.from_pretrained(pretrained_model_name_or_path, trust_remote_code, use_liger_kernel, torch_dtype, cls.require_trust_remote_code, cls.info, **model_init_kwargs)
        return cls(
            model,
            quantized=False,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            model_name_or_path=pretrained_model_name_or_path
        )

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        backend: BACKEND = BACKEND.AUTO,
        torch_dtype: [str | torch.dtype] = "auto",
        quantize_config: Optional[QuantizeConfig] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        format: Optional[FORMAT] = None,
        verify_hash: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        model, quantize_config, qlinear_kernel, load_quantized_model = ModelLoader.from_quantized(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
            quantize_config=quantize_config,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            format=format,
            verify_hash=verify_hash,
            require_trust_remote_code=cls.require_trust_remote_code,
            dynamic_expert_index=cls.dynamic_expert_index,
            base_modules=cls.base_modules,
            layer_modules=cls.layer_modules,
            lm_head=cls.lm_head,
            layer_type=cls.layer_type,
            **kwargs
        )

        return cls(
            model,
            quantized=True,
            quantize_config=quantize_config,
            qlinear_kernel=qlinear_kernel,
            load_quantized_model=load_quantized_model,
            trust_remote_code=trust_remote_code,
            model_name_or_path=model_name_or_path,
        )


    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)


__all__ = ["BaseGPTQModel"]
