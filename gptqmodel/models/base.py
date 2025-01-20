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

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, modeling_utils

from ..nn_modules.hooked_linear import replace_linear_with_hooked_linear
from ..quantization import GPTQ, QuantizeConfig
from ..quantization.config import FORMAT, QUANTIZE_BLACK_LIST, AutoRoundQuantizeConfig
from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from ..utils.importer import select_quant_linear
from ..utils.logger import setup_logger
from ..utils.model import (MODALITY, check_to_quantized, find_layers, get_device,
                           get_module, get_module_by_name_prefix, get_moe_layer_modules,
                           move_to, nested_move_to, normalize_tokenizer, pack_model)
from ..utils.progress import ProgressBar
from ..utils.torch import torch_empty_cache
from ._const import CPU, DEVICE, SUPPORTS_MODULE_TYPES
from .loader import ModelLoader
from .writer import (QUANT_LOG_DAMP, QUANT_LOG_FWD_TIME, QUANT_LOG_LAYER,
                     QUANT_LOG_LOSS, QUANT_LOG_MODULE, QUANT_LOG_TIME, ModelWriter)


def check_support_param_buffer_assignment(*args, **kwargs):
    return False


# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

logger = setup_logger()

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
    # some models require transformer version(internalm require '<=4.42.2')
    require_pkgs_version: Optional[List[str]] = None
    # some models require a specific dtype, such as float16
    require_dtype: Optional[str|torch.dtype] = None

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    loader = AutoModelForCausalLM

    # monkey patch api for trust_remote_code=True models that have broken transformer compat
    require_monkeypatch = False

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
    ):
        super().__init__()

        self.model = model
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.tokenizer = tokenizer
        self.quantize_config = quantize_config
        self.config = self.model.config if hasattr(self.model, "config") else None

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[List[int]]],
        batch_size: int = 1,
    ):
        if isinstance(calibration_dataset[0], (str, list)) or (isinstance(calibration_dataset[0], list) and all(isinstance(x, int) for x in calibration_dataset[0])):
            if self.tokenizer is None:
                raise ValueError(f"tokenizer must be provided when calibration_dataset is List[str] or List[int], type: {type(calibration_dataset[0])}")

            # Convert strings/ints to tokenized format
            new_calibration_dataset = []
            for data in calibration_dataset:
                # convert to tensor directly if already in token ids format (ints)
                if isinstance(data, list) and all(isinstance(x, int) for x in data):
                    input_ids = torch.tensor([data], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    new_calibration_dataset.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    })
                # call tokenizer if dataset still string format (str)
                else:
                    tokenized = self.tokenizer(data, return_tensors="pt")
                    new_calibration_dataset.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"]
                    })
            calibration_dataset = new_calibration_dataset

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

            new_calibration_dataset.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            if self.tokenizer:
                vocab = self.tokenizer.get_vocab()

                # auto select the best pad token to use
                for token in ["<|finetune_right_pad_id|>", "<|pad|>", "<pad>", "<|unk|>", "<unk>"]:
                    token_id = vocab.get(token)
                    if token_id is not None:
                        pad_token_id = token_id
                        break
            else:
                logger.warning("Model config does not have pad token mapped. Please pass in tokenizer to `quantize()` so GPTQModel can auto-select the best pad token.")

            if not pad_token_id and isinstance(self.config.eos_token_id, list):  # Llama-3.1-8B-Instruct's eos_token_id is a list
                pad_token_id = self.config.eos_token_id[0]
            elif not pad_token_id:
                pad_token_id = self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError("Calibration data requires model's `pad_token_id` or `eos_token_id` to be set: actual = `None`.")

        new_calibration_dataset_batched = [
            collate_data(new_calibration_dataset[start: start + batch_size], pad_token_id)
            for start in range(0, len(new_calibration_dataset), batch_size)
        ]


        return new_calibration_dataset_batched

    def quantize(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        logger_board: Optional[str] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
    ) -> List[Dict[str, str]]:
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if backend == BACKEND.IPEX:
            self.quantize_config.format = FORMAT.IPEX

        if self.quantize_config.format == FORMAT.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please switch to FORMAT.GPTQ. GPTQMOdel will auto-use Marlin kernel for accelerated inference for FORMAT.GPTQ."
            )

        if len(calibration_dataset) == 0:
            raise ValueError("Calibration dataset must not be empty.")

        if logger_board == "clearml":
            try:
                from clearml import Task
                from random_word import RandomWords

                from ..utils.plotly import create_plotly
            except ImportError as _:
                raise ImportError(
                    "The logger_board is set to 'clearml', but required dependencies are missing. "
                    "Please install them by running: pip install gptqmodel[logger]"
                )
            task = Task.init(project_name='GPTQModel', task_name=f'Experiment-{RandomWords().get_random_word()}', task_type=Task.TaskTypes.optimizer)
        else:
            task = None

        # Validate quant linear before quantization starts
        _ = select_quant_linear(
            bits=self.quantize_config.bits,
            dynamic=self.quantize_config.dynamic,
            group_size=self.quantize_config.group_size,
            desc_act=self.quantize_config.desc_act,
            sym=self.quantize_config.sym,
            backend=backend,
            device=DEVICE(self.quantize_config.device),
            pack=True,
            format=self.quantize_config.format,
        )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            self.tokenizer = tokenizer
            # after tokenizer is reset, need to normalize it again
            self.tokenizer = normalize_tokenizer(self.config, self.tokenizer)

        min_calibration_dataset_size = 256
        min_calibration_dataset_input_ids_avg_length = 256

        if len(calibration_dataset) < min_calibration_dataset_size:
            logger.warning(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                           f"Current: {len(calibration_dataset)}.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        calibration_dataset = self.prepare_dataset(calibration_dataset, batch_size,)

        # Calculate the average length of the average input_ids
        total_input_ids_length = 0
        max_input_id_length = 0
        for row in calibration_dataset:
            input_ids = row["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() <= 2:
                    input_ids_length = input_ids.shape[-1]
                else:
                    raise ValueError(
                        "Expected a 1-dimensional tensor or 2-dimensional tensor for 'input_ids', but got a tensor with {0} dimensions.".format(
                            input_ids.dim()))
            else:
                input_ids_length = len(input_ids)

            if input_ids_length > max_input_id_length:
                max_input_id_length = input_ids_length
            total_input_ids_length += input_ids_length
        avg = total_input_ids_length / len(calibration_dataset)

        if avg < min_calibration_dataset_input_ids_avg_length:
            logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
                           f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

        if isinstance(self.quantize_config, AutoRoundQuantizeConfig):
            from auto_round import AutoRound
            from auto_round import __version__ as auto_round_version

            if version.parse(auto_round_version) < version.parse("0.3.0"):
                raise ValueError(f"AutoRound version must be >= 0.3.0: actual = {auto_round_version}")

            if self.quantize_config.lm_head:
                self.quantize_config.layer_config[self.lm_head] = {"data_type": "int"}

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
                                       device=self.device,
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
                backend=backend,
                desc_act=self.quantize_config.desc_act,
                format=self.quantize_config.format,
                lm_head_name=self.lm_head,
                parallel_packing=self.quantize_config.parallel_packing,
            )

            self.model = model
            self.quantized = True
            return

        if self.quantize_config.lm_head:
            if self.model.config.tie_word_embeddings and hasattr(self.model.model, "_tied_weights_keys"):
                tied_keys = self.model._tied_weights_keys
                for item in tied_keys:
                    if self.lm_head in item:
                        raise NotImplementedError("quantizing lm_head with tied weights has not been supported "
                                                  "currently")

            lm_head_module = get_module(self.model, key=self.lm_head)
            if get_module(self.model, key=self.lm_head) is None:
                raise ValueError(f"could not find layer {self.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.quantize_config.dynamic is None:
                self.quantize_config.dynamic = {self.lm_head: lm_head_quant_config}
            elif self.quantize_config.dynamic_get(self.lm_head, default_value=None) is None:
                self.quantize_config.dynamic[self.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.model.config.use_cache if hasattr(self.model.config, "use_cache") else False
        self.model.config.use_cache = False

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
            self.model.to(self.quantize_config.device)

        num_batches = len(calibration_dataset)
        layers = get_module_by_name_prefix(self.model, self.layers_node)

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to(inp, data_device))
            if len(layer_input) == 0:
                # Some models put hidden_states in kwargs instead of args.
                # For example, gptj ...
                if kwargs.get("hidden_states") is not None:
                    layer_input.append(move_to(kwargs["hidden_states"], data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs.get("attention_mask") is not None:
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

            if not self.quantize_config.lm_head or self.quantize_config.lm_head_low_gpu_mem_usage:
                raise ValueError

        lm_head_inputs = []
        if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
            def store_lm_head_input_hook(_, args, kwargs):
                # Positional arguments.
                lm_head_layer_input = []
                for inp in args:
                    lm_head_layer_input.append(move_to(inp, data_device))
                if len(lm_head_layer_input) == 0:
                    # Some models put hidden_states in kwargs instead of args.
                    # For example, gptj ...
                    if kwargs.get("hidden_states") is not None:
                        lm_head_layer_input.append(move_to(kwargs["hidden_states"], data_device))

                lm_head_inputs.append(lm_head_layer_input)
                raise ValueError

        # move layer to target device
        layers[0] = layers[0].to(self.quantize_config.device)

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
        if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
            lm_head_handle = layers[0].register_forward_pre_hook(store_lm_head_input_hook, with_kwargs=True)
        is_ovis = self.__class__.__name__ == "OvisGPTQ"
        for example in calibration_dataset:
            for k, v in example.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        if len(v[i].shape) == 1:
                            v[i] = v[i].unsqueeze(0)
                        v[i] = move_to(v[i].to(torch.bfloat16) if is_ovis else v[i], cur_layer_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, cur_layer_device)
            try:
                if is_ovis:
                    self.generate(inputs=example.pop("input_ids"), max_new_tokens=1024, **example)
                else:
                    self.model(**example)
            except ValueError:
                pass
        handle.remove()
        if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
            lm_head_handle.remove()

        if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
            self.model.to(CPU)
        else:
            move_to(layers[0], CPU)

        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to(module, ori_outside_layer_module_devices[module_name])

        torch_empty_cache()

        layer_modules = self.layer_modules

        if not self.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        # dynamic expert layer index for model defs
        if self.dynamic_expert_index is not None:
            num_experts = getattr(self.model.config, self.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                  num_experts=num_experts)

        quantizers = {}

        layer_count = len(layers)
        layer_pb = ProgressBar(range(layer_count + 1 if self.quantize_config.lm_head else layer_count))
        gpu_memorys = []
        cpu_memorys = []
        durations = []
        avg_losses = []
        module_names = []
        shared_kv_cache_dict = {}

        # replace linear with hooked linear
        replace_linear_with_hooked_linear(self.model)

        for i in layer_pb:
            is_lm_head = i >= layer_count
            if is_lm_head:
                layer_pb.set_description("Quantizing lm_head")
                layer = get_module(self.model, key=self.lm_head)
                if self.quantize_config.lm_head and not self.quantize_config.lm_head_low_gpu_mem_usage:
                    layer_inputs = lm_head_inputs
            else:
                layer_pb.set_description(f"Quantizing layer {i} of {layer_count - 1}")
                layer = layers[i]

            if layer.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue
            if task is not None:
                gpu_memory = get_gpu_usage_memory()
                cpu_memory = get_cpu_usage_memory()
                task.get_logger().report_scalar(
                    title='GPU Memory',
                    series='GPU Memory',
                    value=gpu_memory,
                    iteration=i,
                )

                task.get_logger().report_scalar(
                    title='CPU Memory',
                    series='CPU Memory',
                    value=cpu_memory,
                    iteration=i,
                )
                gpu_memorys.append(gpu_memory)
                cpu_memorys.append(cpu_memory)

            if get_device(layer) == CPU and self.quantize_config.device != CPU:
                move_to(layer, self.quantize_config.device)

            cur_layer_device = get_device(layer)
            full = find_layers(layer, name=self.lm_head if is_lm_head else "")
            modules = [[self.lm_head]] if is_lm_head else layer_modules
            for index, names in enumerate(modules):
                subset = {n: full[n] for n in names if n in full}
                skipped_modules = []
                gptq = {}
                for name in subset:
                    bits = self.quantize_config.bits
                    sym = self.quantize_config.sym
                    mse = self.quantize_config.mse
                    if self.quantize_config.dynamic is not None:
                        layer_name = self.lm_head if is_lm_head else f"{self.layers_node}.{i}.{name}"

                        if self.quantize_config.dynamic_get(layer_name=layer_name) == False: # noqa: E712
                            logger.info(f"skip module: {layer_name}")

                            skipped_modules.append(name)
                            continue

                        bits = self.quantize_config.dynamic_get(layer_name, "bits", bits)
                        sym = self.quantize_config.dynamic_get(layer_name, "sym", sym)
                        mse = self.quantize_config.dynamic_get(layer_name, "mse", mse)
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        bits,
                        perchannel=True,
                        sym=sym,
                        mse=mse,
                    )

                for name in skipped_modules:
                    subset.pop(name)

                if len(gptq) == 0:
                    continue

                def add_batch(name):
                    def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                        # gptq is mutable.
                        gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handle = []
                for name in subset:
                    if hasattr(subset[name], 'forward_hook'):
                        subset[name].forward_hook = add_batch(name)
                    else:
                        handle.append(subset[name].register_forward_hook(add_batch(name)))

                fwd_start = time.time()
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
                        # reuse_kv is a flag to reuse the kv cache, only for the hamba model
                        if hasattr(layer, "reuse_kv"):
                            if layer.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(i - 1)

                            layer_output = layer(*layer_input) if is_lm_head else layer(*layer_input, **additional_layer_inputs)
                            if shared_kv_cache_dict.get(i) is None:
                                shared_kv_cache_dict[i] = layer_output[-1]
                        else:
                            layer(*layer_input) if is_lm_head else layer(*layer_input, **additional_layer_inputs)

                    del layer_input
                    del additional_layer_inputs

                fwd_end = time.time()
                fwd_time = fwd_end - fwd_start

                for h in handle:
                    h.remove()

                for name in subset:
                    if hasattr(subset[name], 'forward_hook'):
                        subset[name].forward_hook = None

                if index == len(layer_modules) - 1:
                    torch_empty_cache()

                for name_index, name in enumerate(subset):
                    layer_name = self.lm_head if is_lm_head else f"{self.layers_node}.{i}.{name}"
                    layer_pb.set_description(f"Quantizing {name} in layer {i} of {layer_count - 1}")

                    group_size = self.quantize_config.group_size
                    desc_act = self.quantize_config.desc_act
                    damp_percent = self.quantize_config.damp_percent
                    static_groups = self.quantize_config.static_groups
                    if self.quantize_config.dynamic is not None:
                        group_size = self.quantize_config.dynamic_get(layer_name, "group_size", group_size)
                        desc_act = self.quantize_config.dynamic_get(layer_name, "desc_act", desc_act)
                        damp_percent = self.quantize_config.dynamic_get(layer_name, "damp_percent", damp_percent)
                        static_groups = self.quantize_config.dynamic_get(layer_name, "static_groups", static_groups)

                    scale, zero, g_idx, duration, avg_loss, damp_percent = gptq[name].quantize(
                        percdamp=damp_percent,
                        group_size=group_size,
                        actorder=desc_act,
                        static_groups=static_groups,
                    )
                    if task is not None:
                        task.get_logger().report_scalar(
                            title='Quantization Loss',
                            series=f'layer_{i}_loss',
                            value=avg_loss,
                            iteration=name_index,
                        )

                        task.get_logger().report_scalar(
                            title='Quantization Time',
                            series=f'layer_{i}_time',
                            value=duration,
                            iteration=name_index,
                        )
                    durations.append(duration)
                    avg_losses.append(avg_loss)
                    module_names.append(f"layer-{i}-{name}")

                    stat = {QUANT_LOG_LAYER: i, QUANT_LOG_MODULE: name, QUANT_LOG_LOSS: f"{avg_loss:.5f}",
                            QUANT_LOG_DAMP: f"{damp_percent:.5f}", QUANT_LOG_TIME: f"{duration:.3f}", QUANT_LOG_FWD_TIME: f"{fwd_time:.3f}"}
                    if self.quantize_config.dynamic is not None:
                        stat["dynamic"] = self.quantize_config.dynamic_get(layer_name=layer_name)

                    self.quant_log.append(stat)
                    logger.info(stat)

                    quantizers[layer_name] = (
                        gptq[name].quantizer.to(CPU),
                        move_to(scale, CPU),
                        move_to(zero, CPU),
                        move_to(g_idx, CPU),
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

                if hasattr(layer, "reuse_kv"):
                    if layer.reuse_kv:
                        additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(i - 1)

                with torch.no_grad():
                    layer_output = move_to(
                        layer(*layer_input)[0] if is_lm_head else layer(*layer_input, **additional_layer_inputs)[0],
                        cur_layer_device if calibration_enable_gpu_cache else CPU,
                    )
                    layer_outputs.append([layer_output])

                del layer_input
                del additional_layer_inputs
                if num_batches > 1 and j == num_batches - 1:
                    torch_empty_cache()

            if not is_lm_head:
                layers[i] = move_to(layer, CPU)
            else:
                move_to(layer, CPU)

            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = (
                layer_outputs,
                [],
            )  # TODO: is it really OK to cache only the first positional argument?

            torch_empty_cache()

        logger.info(f"Quantization summary:\n{self.quant_log}")
        for module_log in self.quant_log:
            logger.info(module_log)
        if task is not None:
            x = list(range(layer_count))
            gpu_fig = create_plotly(x=x, y=gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
            cpu_fig = create_plotly(x=x, y=cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
            loss_fig = create_plotly(x=module_names, y=avg_losses, xaxis_title="layer", yaxis_title="loss")
            time_fig = create_plotly(x=module_names, y=durations, xaxis_title="layer", yaxis_title="time")
            task.get_logger().report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
            task.get_logger().report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
            task.get_logger().report_plotly('avg_loss', 'avg_loss', loss_fig)
            task.get_logger().report_plotly('quant_time', 'quant_time', time_fig)

        self.qlinear_kernel = pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            backend=backend,
            desc_act=self.quantize_config.desc_act,
            format=self.quantize_config.format,
            lm_head_name=self.lm_head,
            dynamic=self.quantize_config.dynamic,
            parallel_packing=self.quantize_config.parallel_packing,
        )

        self.model.config.use_cache = forward_pass_use_cache

        self.quantized = True
        torch_empty_cache()

        return self.quant_log

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise f"{self.model.__class__.__name__} does not support the to() method"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[str] = None,
            meta_quantizer: Optional[str] = None,
            **kwargs,
    ):
        extra_json_file_names = ["preprocessor_config.json", "chat_template.json"]
        for name in extra_json_file_names:
            json_path = os.path.join(self.model_local_path, name)
            if os.path.exists(json_path):
                os.makedirs(save_dir, exist_ok=True)

                shutil.copyfile(json_path, os.path.join(save_dir, name))

        if self.quantized:
            self.save_quantized(save_dir, safetensors_metadata, max_shard_size, meta_quantizer)

            # overwrite quant_override_files
            for name, value in self.quant_override_files.items():
                json_path = os.path.join(save_dir, name)
                with open(json_path, "w", encoding="utf-8") as f:
                    if isinstance(value, str):
                        f.write(value)
                    else:
                        f.write(json.dumps(value))
        else:
            self.save_pretrained(save_dir, **kwargs)

    def serve(self,
               host: str = "0.0.0.0",
               port: int = 80,
               async_mode: bool = False):
        from ..utils.openai_server import OpenAiServer
        self.server = OpenAiServer(model=self)
        self.server.start(host=host, port=port, async_mode=async_mode)

    def serve_shutdown(self):
        if self.server is not None:
            self.server.shutdown()

    def serve_wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        if self.server is not None:
            self.server.wait_until_ready(timeout=timeout, check_interval=check_interval)



    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)

__all__ = ["BaseGPTQModel"]

BaseGPTQModel = ModelLoader(ModelWriter(BaseGPTQModel))
