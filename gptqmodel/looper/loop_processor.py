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

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ..looper.input_cache import InputCache
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..quantization.config import QuantizeConfig
from ..utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from ..utils.logger import setup_logger

log = setup_logger()


# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True):

        # result is total collection of all module results mapped by module.full_name
        self._results: Dict[str, Any] = {}

        # toggle to enable stream from gpu to cpu
        self.stream = False

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
                log.warn(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                               f"Current: {len(calibration_dataset)}.")

            calibration_dataset = prepare_dataset_func(calibration_dataset=calibration_dataset,
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

            self.num_batches = len(calibration_dataset)

        self.calibration_dataset = calibration_dataset

    def result_save(self, key: str, value: Any):
        assert self.result_get(key) is None, f"key: {key} already exists in `self.result`"
        self._results[key] = value

    def result_get(self, key: str, default: Any = None) -> Any:
        return self._results.get(key, default)

    def results(self):
        return self._results

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

    def set_fwd_time(self, fwd_time: float):
        self.fwd_time = fwd_time

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
        self.tasks = {}
        self.inputs_cache.layer_inputs = []

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        pass

    # do work and return processor.self state which will updated/merged
    def process(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    # submodule_finalize is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    # finalize is called in reverse after all next sequential processes are called
    def finalize(self, model: BaseGPTQModel, **kwargs):
        del self.inputs_cache
        del self._results

    def release_calibration_dataset(self):
        del self.calibration_dataset

    def number_batches(self) -> int:
        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        pass

    @classmethod
    def name(cls) -> str:
        pass
