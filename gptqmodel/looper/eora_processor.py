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
from typing import Callable, Tuple, Optional

import torch
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.adapter.adapter import Lora
from gptqmodel.eora.eora import eora_compute_lora, eora_process_input
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                                     PROCESS_LOG_NAME, PROCESS_LOG_TIME)
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import move_to
from gptqmodel.utils.torch import torch_sync, torch_new_stream_ctx

from torch.nn import Module

logger = setup_logger()


class EoraProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True):
        super().__init__(tokenizer, qcfg, calibration_dataset, calibration_dataset_concat_size, batch_size,
                         logger_board, require_fwd)

        # dict: key is module name, value is the accumulated eigen_scaling_diag_matrix
        self.eigen_scaling_diag_matrix = {}

    def log_plotly(self):
        task = self.logger_task
        if task is not None:
            from gptqmodel.utils.plotly import create_plotly
            x = list(range(self.layer_count))
            gpu_fig = create_plotly(x=x, y=self.gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
            cpu_fig = create_plotly(x=x, y=self.cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
            time_fig = create_plotly(x=self.module_names, y=self.durations, xaxis_title="layer", yaxis_title="time")
            task.get_logger().report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
            task.get_logger().report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
            task.get_logger().report_plotly('quant_time', 'quant_time', time_fig)

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.num_batches = len(calibration_dataset)

    def preprocess(self, module: NamedModule, **kwargs):
        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:
            module.adapter_cfg = None # hack
            return

        adapter_cfg = copy.deepcopy(self.qcfg.adapter)

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            adapter_cfg.adapter = self.qcfg.dynamic_get(module.full_name, "adapter", adapter_cfg)

        # hack store property inside module
        module.adapter_cfg = adapter_cfg

        self.eigen_scaling_diag_matrix[module.name] = 0

        return

    def is_skipped(self, module: NamedModule) -> bool:
        # dynamic override removed eora processing for this module
        return module.adapter_cfg in [None, {}]

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            eora_process_input(
                input=input,
                name=name,
                eigen_scaling_diag_matrix=self.eigen_scaling_diag_matrix,
                sample_size=self.num_batches
            )
        return tmp

    def process(self, module: NamedModule):
        assert (isinstance(module.adapter_cfg, Lora))

        self.pb.set_description(f"EoRA gen: {module.name} in layer {module.layer_index} of {self.layer_count - 1}")

        start = time.time()

        eigen_scaling_diag_matrix = self.eigen_scaling_diag_matrix[module.name]

        w = module.state.pop("w")
        wq: torch.Tensor = module.state["wq"]

        A, B, computed_wq = eora_compute_lora(
            w=w,
            wq=wq,
            module=module,
            eigen_scaling_diag_matrix=eigen_scaling_diag_matrix,
            rank=module.adapter_cfg.rank
        )

        del w

        # wq is currently on GPU, stream to CPU if possible
        streamCtx = torch_new_stream_ctx()
        if streamCtx:
            wq_copy = torch.zeros_like(wq, device=CPU, pin_memory=True)
            with streamCtx:
                wq_copy.copy_(wq, non_blocking=True)

            module.state.update({
                "wq": wq_copy,
                "streaming": True,
            })

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq.to(dtype=module.weight.data.dtype)

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
        self.result_save(module.full_name, {
            "lora_A": move_to(A, device=CPU, stream=True), # A.to(dtype=torch.float16, device=CPU),
            "lora_B": move_to(B, device=CPU, stream=True), # B.to(dtype=torch.float16, device=CPU),
            "streaming": True,
        })

    def post_process(self, module: NamedModule):
        pass

    def submodule_finalize(self, module: NamedModule):
        if module.state.pop("streaming", False):
            torch_sync()

    def finalize(self, model: BaseGPTQModel, **kwargs):
        # block for streams
        torch_sync()

        del self.eigen_scaling_diag_matrix
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            if processor_index == 0:
                raise ValueError("EoraProcessor's calibration_dataset must be provided.")
            else:
                return False
        return True


    @classmethod
    def name(cls) -> str:
        return "eora_test"
