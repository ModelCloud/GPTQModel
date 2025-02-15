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

from typing import Callable, List, Tuple, Optional, Union, Dict

import torch
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models._const import CALIBRATION_DATASET_CONCAT_CHAR
from gptqmodel.quantization.config import QuantizeConfig
from torch import Tensor
from torch.nn import Module

from gptqmodel.utils.data import collate_data
from gptqmodel.utils.device import get_gpu_usage_memory, get_cpu_usage_memory
from gptqmodel.utils.logger import setup_logger

logger = setup_logger()


# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True):
        self.tokenizer = tokenizer
        self.qcfg = qcfg


        # if processor require fwd generate and hooks, set this to true
        # looper should bypass generate + hooks if this is false
        self.require_fwd = require_fwd

        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = {}

        self.pb = None
        self.logger_task = None
        self.fwd_time = None
        self.layer_count = None

        # logging
        self.log = []
        self.logger_board = logger_board
        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.module_names = []

        if self.logger_board == "clearml":
            try:
                from clearml import Task
                from random_word import RandomWords

                from ..utils.plotly import create_plotly
            except ImportError as _:
                raise ImportError(
                    "The logger_board is set to 'clearml', but required dependencies are missing. "
                    "Please install them by running: pip install gptqmodel[logger]"
                )
            self.logger_task = Task.init(project_name='GPTQModel',
                                         task_name=f'{self.__class__.__name__}-{RandomWords().get_random_word()}',
                                         task_type=Task.TaskTypes.optimizer)
        else:
            self.logger_task = None


        # prepare dataset
        if calibration_dataset is not None:
            if len(calibration_dataset) == 0:
                raise ValueError("Calibration dataset must not be empty.")

            min_calibration_dataset_size = 256
            min_calibration_dataset_input_ids_avg_length = 256
            if len(calibration_dataset) < min_calibration_dataset_size:
                logger.warning(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                               f"Current: {len(calibration_dataset)}.")

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
                logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
                               f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

            self.num_batches = len(calibration_dataset)

        self.calibration_dataset = calibration_dataset

    def prepare_dataset(
            self,
            calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[List[int]]],
            # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
            calibration_dataset_concat_size: Optional[int] = None,
            batch_size: int = 1,
    ):
        if isinstance(calibration_dataset[0], (str, list)) or (
                isinstance(calibration_dataset[0], list) and all(isinstance(x, int) for x in calibration_dataset[0])):
            if self.tokenizer is None:
                raise ValueError(
                    f"tokenizer must be provided when calibration_dataset is List[str] or List[int], type: {type(calibration_dataset[0])}")

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

    def collect_memory_info(self, layer_index: int):
        if self.logger_task is not None:
            gpu_memory = get_gpu_usage_memory()
            cpu_memory = get_cpu_usage_memory()
            self.logger_task.get_logger().report_scalar(
                title='GPU Memory',
                series='GPU Memory',
                value=gpu_memory,
                iteration=layer_index,
            )

            self.logger_task.get_logger().report_scalar(
                title='CPU Memory',
                series='CPU Memory',
                value=cpu_memory,
                iteration=layer_index,
            )
            self.gpu_memorys.append(gpu_memory)
            self.cpu_memorys.append(cpu_memory)

    def log_plotly(self):
        pass

    def set_calibration_dataset(self, calibration_dataset):
        pass

    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        pass

    # after preproces, this process may be skipped due to dynamic override (lora adapter = None)
    def is_skipped(self, module: NamedModule) -> bool:
        pass

    def receive_input_cache(self, input_cache: InputCache):
        self.inputs_cache = input_cache

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_inputs(self, layer_inputs: List[List[Tensor]]):
        self.inputs_cache.layer_inputs = layer_inputs

    def clear_cache_data(self):
        del self.tasks
        self.tasks = {}
        del self.inputs_cache.layer_inputs
        self.inputs_cache.layer_inputs = []

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        pass

    # do work and return processor.self state which will updated/merged
    def process(self, module: NamedModule):
        pass

    # step after `process` and before post_process generate()
    def post_process(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    def submodule_finalize(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    def finalize(self, model: BaseGPTQModel, **kwargs):
        del self.inputs_cache

    def release_calibration_dataset(self):
        del self.calibration_dataset

    def number_batches(self) -> int:
        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        pass

    @classmethod
    def name(cls) -> str:
        pass
