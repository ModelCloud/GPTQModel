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

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch._dynamo
import torch.nn as nn
from packaging import version
from packaging.version import Version
from tokenicer import Tokenicer
from transformers import (AutoModelForCausalLM, AutoProcessor, PreTrainedModel,
                          PreTrainedTokenizerBase, ProcessorMixin, modeling_utils)

from ..adapter.adapter import Adapter
from ..nn_modules.hooked_linear import replace_linear_with_hooked_linear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import GPTQ, QuantizeConfig
from ..quantization.config import FORMAT, QUANT_METHOD, QUANTIZE_BLACK_LIST, AutoRoundQuantizeConfig
from ..quantization.rotation.rotation import fuse_layer_norms, rotate_model
from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import setup_logger
from ..utils.model import (MODALITY, check_to_quantized, find_modules, get_device, get_module,
                           get_module_by_name_prefix, get_moe_layer_modules, move_to, nested_move_to, pack_model)
from ..utils.torch import torch_compile, torch_empty_cache
from ._const import CALIBRATION_DATASET_CONCAT_CHAR, CPU, DEFAULT_MAX_SHARD_SIZE, DEVICE, SUPPORTS_MODULE_TYPES
from .loader import ModelLoader
from .writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                     PROCESS_LOG_TIME, QUANT_LOG_DAMP, QUANT_LOG_LOSS, ModelWriter)

# pytorch 2.6.0 fixes many compilation errors
TORCH_MIN_VERSION_STR = '2.6.0'
PYTORCH_MIN_VERSION_WITH_COMPILE = Version(TORCH_MIN_VERSION_STR)

def check_support_param_buffer_assignment(*args, **kwargs):
    return False


# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

log = setup_logger()

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

    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = True

    pre_lm_head_norm_module: str = None

    # some models require trust_remove_code = True (dbrx_converted)
    require_trust_remote_code = None
    # some models require transformer version(internalm require '<=4.42.2')
    require_pkgs_version: Optional[List[str]] = None
    # some models require a specific dtype, such as float16
    require_dtype: Optional[str|torch.dtype] = None
    require_fast_init: bool = True

    # some models require Processor? For example, Qwen2VLImageProcessor.
    require_load_processor = False

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

        self.compiled = False # set to True while compile() is triggered successfully
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")
            self.model.tokenizer = self.tokenizer.tokenizer # helpful for CI tests
        else:
            self.tokenizer = tokenizer # TODO none?
            self.model.tokenizer = tokenizer # helpful for CI tests # TODO none?

        # auto-fix model config erors
        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=model_local_path)

        self.quantize_config = quantize_config

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        self.processor: ProcessorMixin = None
        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path)

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

        # hack: circular import
        from ..adapter.adapter import Lora

        # check adapter load and print info so users knows lora(s) are applied
        if isinstance(self.quantize_config.adapter, Lora):
            loaded_loras = 0
            qmodules = find_modules(self.model, layers=[BaseQuantLinear])
            for name, m in qmodules.items():
                if all(hasattr(m.adapter, name) for name in Lora.parameter_keys()):
                    loaded_loras += 1

            log.info(f"Adapter: `{loaded_loras}` EoRA/Lora adapters loaded for `{len(qmodules)}` modules.")

        # print kernel info:
        log.info(f"Kernel: loaded -> `[{', '.join(cls.__name__ for cls in self.kernels())}]`")

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[List[int]]],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_dataset_concat_size: Optional[int] = None,
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

        if calibration_dataset_concat_size:
            concatenated_data = []
            input_ids_buff = []
            attention_mask_buff = []
            current_length = 0

            new_line = self.tokenizer(CALIBRATION_DATASET_CONCAT_CHAR, return_tensors="pt")
            new_line_input_ids = _convert_tensor_to_list(new_line["input_ids"])[0]
            new_line_attention_mask = _convert_tensor_to_list(new_line["attention_mask"])[0]
            new_line_input_ids_len = len(new_line_input_ids)

            for example in new_calibration_dataset:
                input_ids = example["input_ids"][0]
                attention_mask = example["attention_mask"][0]

                if current_length + len(input_ids) + new_line_input_ids_len >= calibration_dataset_concat_size:
                    if len(input_ids_buff) > 0:
                        remaining_space = calibration_dataset_concat_size - current_length
                        # if there is remaining space, add the remaining input to the current block
                        if remaining_space > 0:
                            input_ids_buff.extend(new_line_input_ids)
                            input_ids_buff.extend(input_ids[:remaining_space - new_line_input_ids_len])
                            attention_mask_buff.extend(new_line_attention_mask)
                            attention_mask_buff.extend(attention_mask[:remaining_space - new_line_input_ids_len])

                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })
                        else:
                            # if there is no remaining space, add the current block to the concatenated data
                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })

                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                    else:
                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                else:
                    if len(input_ids_buff) > 0:
                        input_ids_buff.extend(new_line_input_ids)
                        attention_mask_buff.extend(new_line_attention_mask)
                        current_length += new_line_input_ids_len

                    input_ids_buff.extend(input_ids)
                    attention_mask_buff.extend(attention_mask)
                    current_length += len(input_ids)


            if input_ids_buff:
                padding_length = calibration_dataset_concat_size - len(input_ids_buff)
                if padding_length > 0:
                    input_ids_buff.extend([self.tokenizer.pad_token_id] * padding_length)
                    attention_mask_buff.extend([0] * padding_length)
                concatenated_data.append({
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff]
                })

            new_calibration_dataset = concatenated_data

        new_calibration_dataset_batched = [
            collate_data(new_calibration_dataset[start: start + batch_size], self.tokenizer.pad_token_id)
            for start in range(0, len(new_calibration_dataset), batch_size)
        ]


        return new_calibration_dataset_batched

    def quantize(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_dataset_concat_size: Optional[int] = None,
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        logger_board: Optional[str] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        # Experimental: enables the buffering of fwd inputs to cpu, slower than non-buffered, may reduce vram usage
        buffered_fwd: bool = False,
        # torch/cuda GC is auto enabled to reduce vram usage: disable to for small models or you know there is no possibility of oom due to vram to accelerate quantization
        auto_gc: bool = True,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
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
            pack_dtype=self.quantize_config.pack_dtype,
        )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # overwrite quantize_config.adapter
        if adapter is not None:
            self.quantize_config.adapter = adapter

        from ..adapter.adapter import Lora
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper

        # has lora process
        needs_lora = isinstance(self.quantize_config.adapter, Lora)

        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration_dataset": calibration_dataset,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_dataset_concat_size": calibration_dataset_concat_size,
            "batch_size": batch_size,
            "logger_board": logger_board,
            "retain_w": needs_lora,  # lora needs original w
        }

        # init processor with default GPTQ processor
        if self.quantize_config.quant_method == QUANT_METHOD.QQQ:
            from ..looper.qqq_processor import QQQProcessor
            quantize_processor = QQQProcessor(**args)

            # rotate model
            if self.quantize_config.rotation:
                from gptqmodel.models.definitions.llama import LlamaGPTQ
                from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ
                if not isinstance(self, (LlamaGPTQ, Qwen2GPTQ)):
                    raise ValueError(f"rotation only supports: llama/qwen2 model, "
                                     f"current model is {self.__class__.__name__}")
                module_name_args = {
                    "layers_node": self.layers_node,
                    "lm_head_name": self.lm_head
                }
                self.model = fuse_layer_norms(model=self.model,
                                              pre_lm_head_norm_module_name=self.pre_lm_head_norm_module,
                                              **module_name_args)

                self.model, _ = rotate_model(model=self.model, rotate_mode=self.quantize_config.rotation,
                                             device=self.quantize_config.device, **module_name_args)
                if auto_gc:
                    torch_empty_cache()

        else:
            from ..looper.gptq_processor import GPTQProcessor
            quantize_processor = GPTQProcessor(**args)


        processors = [quantize_processor]

        # Append EoRA processor for lora adapter
        if needs_lora:
            processors.append(
                EoraProcessor(
                    tokenizer=self.tokenizer,
                    qcfg=self.quantize_config,
                    calibration_dataset=adapter_calibration_dataset if adapter_calibration_dataset is not None else calibration_dataset,
                    prepare_dataset_func=self.prepare_dataset,
                    calibration_dataset_concat_size=calibration_dataset_concat_size,
                    batch_size=batch_size,
                    logger_board=logger_board,
                )
            )

        # prepare processor worker (looper)
        module_looper = ModuleLooper(self, processors=processors)

        return module_looper.loop(
            calibration_enable_gpu_cache=calibration_enable_gpu_cache,
            buffered_fwd=buffered_fwd,
            auto_gc=auto_gc,
            backend=backend,
        )

    def _eora_generate(
        self,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter,
        quantized_modules: Dict[str, TorchQuantLinear],
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        logger_board: Optional[str] = None,
        # Experimental: enables the buffering of fwd inputs to cpu, slower than non-buffered, may reduce vram usage
        buffered_fwd: bool = False,
        # torch/cuda GC is auto enabled to reduce vram usage: disable to for small models or you know there is no possibility of oom due to vram to accelerate quantization
        auto_gc: bool = True,
    ):
        if self.quantized:
            raise EnvironmentError("eora_generate() is called a model that is already quantized")

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..adapter.adapter import Lora
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # init processor with EoRA processor
        processors = [
            DequantizeProcessor(
                quantized_modules=quantized_modules,
            ),
            EoraProcessor(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                calibration_dataset=calibration_dataset,
                prepare_dataset_func=self.prepare_dataset,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
                batch_size=batch_size,
                logger_board=logger_board,
            ),
        ]

        # prepare processor worker (looper)
        module_looper = ModuleLooper(model=self, processors=processors)

        module_looper.loop(
            calibration_enable_gpu_cache=calibration_enable_gpu_cache,
            buffered_fwd=buffered_fwd,
            auto_gc=auto_gc,
        )

        self.eora_save(save_dir=adapter.path, model_save_dir=self.model_local_path)
        return

    @torch.no_grad()
    def quantize_old(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_dataset_concat_size: Optional[int] = None,
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        logger_board: Optional[str] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        # Experimental: enables the buffering of fwd inputs to cpu, slower than non-buffered, may reduce vram usage
        buffered_fwd: bool = False,
        # torch/cuda GC is auto enabled to reduce vram usage: disable to for small models or you know there is no possibility of oom due to vram to accelerate quantization
        auto_gc: bool = True,
    ) -> Tuple[List[Dict[str, str]], Dict[str, torch.Tensor]]:
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
            pack_dtype=self.quantize_config.pack_dtype,
        )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        min_calibration_dataset_size = 256
        min_calibration_dataset_input_ids_avg_length = 256

        if len(calibration_dataset) < min_calibration_dataset_size:
            log.warn(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                           f"Current: {len(calibration_dataset)}.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        calibration_dataset = self.prepare_dataset(calibration_dataset=calibration_dataset,
                                                   calibration_dataset_concat_size=calibration_dataset_concat_size,
                                                   batch_size=batch_size)

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
            log.warn(f"The average length of input_ids of calibration_dataset should be greater than "
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
                quant_result=quantizers,
                bits=self.quantize_config.bits,
                dynamic=self.quantize_config.dynamic,
                group_size=self.quantize_config.group_size,
                backend=backend,
                desc_act=self.quantize_config.desc_act,
                format=self.quantize_config.format,
                quant_method=self.quantize_config.quant_method,
                lm_head_name=self.lm_head,
                parallel_packing=self.quantize_config.parallel_packing,
            )

            self.model = model
            self.quantized = True
            return

        if self.quantize_config.lm_head:
            if self.model.config.tie_word_embeddings and hasattr(self.model, "_tied_weights_keys"):
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
            elif self.quantize_config.dynamic_get(self.lm_head, default=None) is None:
                self.quantize_config.dynamic[self.lm_head] = lm_head_quant_config

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

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to(inp, device=data_device))
            if len(layer_input) == 0:
                # Some models put hidden_states in kwargs instead of args.
                # For example, gptj ...
                if kwargs.get("hidden_states") is not None:
                    layer_input.append(move_to(kwargs["hidden_states"], device=data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise ValueError

        # move layer to target device
        layers[0] = layers[0].to(device=self.quantize_config.device)

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
        is_ovis = self.__class__.__name__ == "OvisGPTQ"
        self.pre_quantize_generate_hook_start()
        for example in calibration_dataset:
            for k, v in example.items():
                data_device = self.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for module_index in range(len(v)):
                        if len(v[module_index].shape) == 1:
                            v[module_index] = v[module_index].unsqueeze(0)
                        v[module_index] = move_to(v[module_index].to(self.model.visual_tokenizer.dtype) if is_ovis else v[module_index], data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, data_device)
            try:
                if is_ovis:
                    self.generate(inputs=example.pop("input_ids"), max_new_tokens=1024, **example)
                else:
                    self.model(**example)
            except ValueError:
                pass
        self.pre_quantize_generate_hook_end()
        handle.remove()

        move_to(layers[0], CPU)

        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to(module, ori_outside_layer_module_devices[module_name])

        if auto_gc:
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
        quant_modules_pb = log.pb(range(layer_count + 1 if self.quantize_config.lm_head else layer_count)).manual()
        gpu_memorys = []
        cpu_memorys = []
        durations = []
        avg_losses = []
        module_names = []
        shared_kv_cache_dict = {}

        # replace linear with hooked linear
        replace_linear_with_hooked_linear(self.model)

        quantized_weights = {}
        for module_index in quant_modules_pb:
            is_lm_head_module = module_index >= layer_count
            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.model, key=self.lm_head)
                layer_inputs = self.lm_head_pre_quantize_generate_hook(layer_inputs)
            else:
                quant_modules_pb.title(f"Quantizing layer {module_index} of {layer_count - 1}").draw()
                module = layers[module_index]

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue
            if task is not None:
                gpu_memory = get_gpu_usage_memory()
                cpu_memory = get_cpu_usage_memory()
                task.get_logger().report_scalar(
                    title='GPU Memory',
                    series='GPU Memory',
                    value=gpu_memory,
                    iteration=module_index,
                )

                task.get_logger().report_scalar(
                    title='CPU Memory',
                    series='CPU Memory',
                    value=cpu_memory,
                    iteration=module_index,
                )
                gpu_memorys.append(gpu_memory)
                cpu_memorys.append(cpu_memory)

            self.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.lm_head if is_lm_head_module else "")
            modules = [[self.lm_head]] if is_lm_head_module else layer_modules
            for index, names in enumerate(modules):
                subset = {n: full[n] for n in names if n in full}
                skipped_modules = []
                gptq = {}
                for name in subset:
                    qcfg_clone = copy.deepcopy(self.quantize_config)

                    # dynamic overrides
                    if self.quantize_config.dynamic is not None:
                        layer_name = self.lm_head if is_lm_head_module else f"{self.layers_node}.{module_index}.{name}"

                        if self.quantize_config.dynamic_get(layer_name=layer_name) == False: # noqa: E712
                            log.info(f"skip module: {layer_name}")

                            skipped_modules.append(name)
                            continue

                        qcfg_clone.bits = self.quantize_config.dynamic_get(layer_name, "bits", qcfg_clone.bits)
                        qcfg_clone.sym = self.quantize_config.dynamic_get(layer_name, "sym", qcfg_clone.sym)
                        qcfg_clone.mse = self.quantize_config.dynamic_get(layer_name, "mse", qcfg_clone.mse)
                        qcfg_clone.group_size = self.quantize_config.dynamic_get(layer_name, "group_size", qcfg_clone.group_size)
                        qcfg_clone.desc_act = self.quantize_config.dynamic_get(layer_name, "desc_act", qcfg_clone.desc_act)
                        qcfg_clone.damp_percent = self.quantize_config.dynamic_get(layer_name, "damp_percent", qcfg_clone.damp_percent)
                        qcfg_clone.static_groups = self.quantize_config.dynamic_get(layer_name, "static_groups", qcfg_clone.static_groups)

                    tmp = GPTQ(module=subset[name], qcfg=qcfg_clone)
                    gptq[name] = tmp

                    # models like DeepSeek v3/r1 has > 256 $ of sub-modules per layer
                    # use buffered mode go vram don't explode: gptq needs to store fwd inputs per each layer fwd
                    # all sub-modules within a single layer needs to store all the inputs.
                    # deepseek has massive # of sub-modules per layer, causing vram pressure
                    # buffered mode is slower due to gpu<->cpu movement
                    if buffered_fwd: # TODO tweak this number for masive MoE
                        log.info(f"Experimental: enabling fwd buffered mode for: `{name}`")
                        tmp.fwd_inputs_buffered = True

                    tmp.quantizer.configure(
                        perchannel=True,
                    )

                for name in skipped_modules:
                    subset.pop(name)

                if len(gptq) == 0:
                    continue

                def add_batch(name):
                    def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                        # gptq is mutable.
                        g = gptq[name] # noqa: F821
                        g.add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handle = []
                for name in subset:
                    if hasattr(subset[name], 'forward_hook'):
                        subset[name].forward_hook = add_batch(name)
                    else:
                        handle.append(subset[name].register_forward_hook(add_batch(name)))

                # logger.info(f"layer-{i}: Begin Forward() Pass")
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

                    # reuse_kv is a flag to reuse the kv cache, only for the hamba model
                    if hasattr(module, "reuse_kv"):
                        if module.reuse_kv:
                            additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(module_index - 1)

                        layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input, **additional_layer_inputs)
                        if shared_kv_cache_dict.get(module_index) is None:
                            shared_kv_cache_dict[module_index] = layer_output[-1]
                    else:
                        module(*layer_input) if is_lm_head_module else module(*layer_input, **additional_layer_inputs)

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
                    if auto_gc:
                        torch_empty_cache()

                for name_index, name in enumerate(subset):
                    layer_name = self.lm_head if is_lm_head_module else f"{self.layers_node}.{module_index}.{name}"
                    quant_modules_pb._subtitle(f"Quantizing {name} in layer {module_index} of {layer_count - 1}")

                    # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
                    ## Need to return the quantized_weight for offloading
                    quantized_weight, scale, zero, g_idx, duration, avg_loss, damp_percent = gptq[name].quantize()

                    ## Assign the quantized weight to the weight
                    gptq[name].module.weight.data = quantized_weight.to(device=gptq[name].device)
                    ## Offload the quantized weight to CPU for EoRA
                    quantized_weights['model.layers.%d.%s' % (module_index, name)] = quantized_weight.cpu()


                    if task is not None:
                        task.get_logger().report_scalar(
                            title='Quantization Loss',
                            series=f'layer_{module_index}_loss',
                            value=avg_loss,
                            iteration=name_index,
                        )

                        task.get_logger().report_scalar(
                            title='Quantization Time',
                            series=f'layer_{module_index}_time',
                            value=duration,
                            iteration=name_index,
                        )
                    durations.append(duration)
                    avg_losses.append(avg_loss)
                    module_names.append(f"layer-{module_index}-{name}")

                    stat = {PROCESS_LOG_LAYER: module_index, PROCESS_LOG_MODULE: name, QUANT_LOG_LOSS: f"{avg_loss:.5f}",
                            QUANT_LOG_DAMP: f"{damp_percent:.5f}", PROCESS_LOG_TIME: f"{duration:.3f}", PROCESS_LOG_FWD_TIME: f"{fwd_time:.3f}"}
                    if self.quantize_config.dynamic is not None:
                        stat["dynamic"] = self.quantize_config.dynamic_get(layer_name=layer_name)

                    self.quant_log.append(stat)
                    log.info(stat)

                    quantizers[layer_name] = (
                        gptq[name].quantizer.to(CPU),
                        move_to(scale, CPU),
                        move_to(zero, CPU),
                        move_to(g_idx, CPU),
                    )
                    gptq[name].free()
                    # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")

            # logger.info(f"layer-{i}: Begin Forward() Pass 2 Post-Quant")
            is_last_quant = module_index == len(quant_modules_pb) - 1
            if not is_last_quant:
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

                    if hasattr(module, "reuse_kv"):
                        if module.reuse_kv:
                            additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(module_index - 1)

                    layer_output = move_to(
                        module(*layer_input)[0] if is_lm_head_module else module(*layer_input, **additional_layer_inputs)[0],
                        cur_layer_device if calibration_enable_gpu_cache else CPU,
                    )
                    layer_outputs.append([layer_output])

                    del layer_input
                    del additional_layer_inputs
                    if num_batches > 1 and j == num_batches - 1:
                        if auto_gc:
                            torch_empty_cache()

            if not is_lm_head_module:
                layers[module_index] = self.post_quantize(module)
            else:
                self.post_quantize(module)

            del module
            del gptq
            del layer_inputs

            if not is_last_quant:
                layer_inputs, layer_outputs = (
                    layer_outputs,
                    [],
                )  # TODO: is it really OK to cache only the first positional argument?

            if auto_gc:
                torch_empty_cache()

        log.info(f"Quantization summary:\n{self.quant_log}")
        for module_log in self.quant_log:
            log.info(module_log)
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
            quant_result=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            backend=backend,
            desc_act=self.quantize_config.desc_act,
            format=self.quantize_config.format,
            quant_method=self.quantize_config.quant_method,
            lm_head_name=self.lm_head,
            dynamic=self.quantize_config.dynamic,
            parallel_packing=self.quantize_config.parallel_packing,
            pack_dtype=self.quantize_config.pack_dtype,
        )

        self.model.config.use_cache = forward_pass_use_cache

        self.quantized = True
        if auto_gc:
            torch_empty_cache()

        ## need to return quantized_weight for EoRA
        return self.quant_log, quantized_weights

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
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # placeholder, noop, and alert users to correct static api
    def push_to_hub(self,
                    repo_id: str,
                    quantized_path: str,  # saved local directory path
                    private: bool = False,
                    exists_ok: bool = False,  # set to true if repo already exists
                    token: Optional[str] = None):

        log.error("`push_to_hub()` api cannot be used on the model instance. Please use `GPTQModel.push_to_hub()` static api instead.")

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            **kwargs,
    ):
        if self.quantized:
            # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
            #untie_weights(self.model)

            self.save_quantized(
                save_dir=save_dir,
                safetensors_metadata=safetensors_metadata,
                max_shard_size=max_shard_size,
                meta_quantizer=meta_quantizer,
                eora_path=eora_path)

            # overwrite quant_override_files
            for name, value in self.quant_override_files.items():
                json_path = os.path.join(save_dir, name)
                with open(json_path, "w", encoding="utf-8") as f:
                    if isinstance(value, str):
                        f.write(value)
                    else:
                        f.write(json.dumps(value))
        else:
            self.save_pretrained(save_dir=save_dir, **kwargs)


    # returns all the loaded qlinear types, returns empty [] if non-found
    def kernels(self) -> List[Type[BaseQuantLinear]]:
        if not isinstance(self.model, nn.Module):
            return []
        loaded_kernels = set()
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for k, v in modules.items():
            loaded_kernels.add(v.__class__)

        return list(loaded_kernels)

    def compile(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        log.warn("Deprecation: `model.compile()` is deprecated. Please use `model.optimize()` instead.")
        return self.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if not self.quantized:
            log.warn("model is not quantized, skip compiling...")
            return self

        if Version(torch.__version__) < PYTORCH_MIN_VERSION_WITH_COMPILE:
            self.compiled = False
            log.warn(f"To use compile(), you need to have torch version >= {TORCH_MIN_VERSION_STR}, please "
                           f"upgrade it by `pip install -U torch torchaudio torchvision`")
            return self

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for name in modules.keys():
            modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # supress errors until PyTorch fixed: https://github.com/pytorch/pytorch/issues/132635
        # torch._dynamo.config.suppress_errors = True
        log.info(f"Compiling model with backend: `{backend}`, mode: `{mode}`")

        self.model = torch_compile(self.model, fullgraph=fullgraph, backend=backend, mode=mode)

        #trigger kernel compilation hooks
        # if self.compiled:
        #     modules = find_modules(self.model, layers=[BaseQuantLinear])
        #     for name in modules.keys():
        #         modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # logger.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        # modules = find_modules(self.model, layers=[BaseQuantLinear])
        # for name in modules.keys():
        #     modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        return self

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

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        pass

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.pre_lm_head_norm_module:
            norm = get_module_by_name_prefix(self.model, self.pre_lm_head_norm_module)
            self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(self, module: nn.Module) -> nn.Module:
        if get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self.quantize_config.device)
        return module

    def post_quantize(self, module: nn.Module) -> nn.Module:
        return move_to(module, device=CPU)

    ## overrides nn.module.train()
    # def train(self, mode=True):
    #     old_mode = self.training
    #     # Call the parent class's train() method to set the training mode
    #     super().train(mode)
    #
    #     if old_mode == mode:
    #         return
    #
    #     # Custom behavior when switching to training mode
    #     if mode:
    #         if not self.SUPPORTS_TRAINING:
    #             err = f"{self.__class__.__name__}: MODEL switching to training mode."
    #             log.error(err)
    #             raise NotImplementedError(err)
    #         else:
    #             log.info(f"{self.__class__.__name__}: MODEL switching to training mode.")
    #     else:
    #         log.info(f"{self.__class__.__name__}: `MODEL switching to eval mode.")

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)

__all__ = ["BaseGPTQModel"]

BaseGPTQModel = ModelLoader(ModelWriter(BaseGPTQModel))
