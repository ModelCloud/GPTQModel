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

import copy
import time
from typing import Callable, Tuple

import torch
from gptqmodel import QuantizeConfig
from gptqmodel.adapter.adapter import Lora
from gptqmodel.eora.eora import eora_compute_lora, eora_process_input, process_input
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                                     PROCESS_LOG_NAME, PROCESS_LOG_TIME, QUANT_LOG_DAMP, QUANT_LOG_LOSS)
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from gptqmodel.utils.logger import setup_logger
from torch.nn import Module

logger = setup_logger()


class EoraProcessor(LoopProcessor):
    def __init__(self, calibration_dataset, qcfg: QuantizeConfig):
        super().__init__(calibration_dataset=calibration_dataset, qcfg=qcfg)

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
            self.logger_task = Task.init(project_name='GPTQModel', task_name=f'EoraProcessor-{RandomWords().get_random_word()}', task_type=Task.TaskTypes.optimizer)
        else:
            self.logger_task = None

        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.avg_losses = []
        self.module_names = []

        # dict: key is module name, value is the accumulated eigen_scaling_diag_matrix
        self.eigen_scaling_diag_matrix = {}


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
        task = self.logger_task
        if task is not None:
            from gptqmodel.utils.plotly import create_plotly
            x = list(range(self.layer_count))
            gpu_fig = create_plotly(x=x, y=self.gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
            cpu_fig = create_plotly(x=x, y=self.cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
            loss_fig = create_plotly(x=self.module_names, y=self.avg_losses, xaxis_title="layer", yaxis_title="loss")
            time_fig = create_plotly(x=self.module_names, y=self.durations, xaxis_title="layer", yaxis_title="time")
            task.get_logger().report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
            task.get_logger().report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
            task.get_logger().report_plotly('avg_loss', 'avg_loss', loss_fig)
            task.get_logger().report_plotly('quant_time', 'quant_time', time_fig)

    def preprocess(self, module: NamedModule, buffered_fwd: bool):
        adapter_cfg = copy.deepcopy(self.qcfg.adapter)

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            adapter_cfg.adapter = self.qcfg.dynamic_get(module.full_name, "adapter", adapter_cfg)

        # hack store property inside module
        module.adapter_cfg = adapter_cfg
        return

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            eora_process_input(
                input=input,
                name=name,
                eigen_scaling_diag_matrix=self.eigen_scaling_diag_matrix,
                sample_size=len(self.calibration_dataset)
            )
        return tmp

    def process(self, module: NamedModule):
        assert (isinstance(module.adapter_cfg, Lora))

        self.pb.set_description(f"EoRA gen: {module.name} in layer {module.layer_index} of {self.layer_count - 1}")

        start = time.time()

        eigen_scaling_diag_matrix = self.eigen_scaling_diag_matrix[module.name]

        wq = module.state.get("wq"),

        A, B, computed_wq = eora_compute_lora(
            w=module.state.get("w"),
            wq=wq,
            module=module,
            eigen_scaling_diag_matrix=eigen_scaling_diag_matrix,
            rank=module.adapter_cfg.rank
        )

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq.to(module.weight.data.dtype)

        # lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(dtype=torch.float16)
        # lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(dtype=torch.float16)

        duration = time.time() - start
        self.durations.append(duration)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}"
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.log.append(stat)
        logger.info(stat)

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        module.state.update({
            "lora_A": A.to(dtype=torch.float16, device=CPU),
            "lora_B": B.to(dtype=torch.float16, device=CPU),
        })

    def post_process(self, module: NamedModule):
        pass

    def submodule_finalize(self, module: NamedModule):
        pass

    def finalize(self, model: BaseGPTQModel, **kwargs):
        del self.eigen_scaling_diag_matrix
        super().finalize(model=model, **kwargs)

    @classmethod
    def name(cls) -> str:
        return "eora_test"
