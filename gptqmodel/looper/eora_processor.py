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
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..adapter.adapter import Lora
from ..eora.eora import eora_compute_lora, eora_process_input
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER,
                             PROCESS_LOG_MODULE, PROCESS_LOG_NAME, PROCESS_LOG_TIME)
from ..quantization.config import QuantizeConfig
from ..quantization.gptq import CPU
from ..utils.logger import setup_logger
from ..utils.model import move_to
from ..utils.torch import torch_compile, torch_sync

log = setup_logger()


class EoraProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True,
                 ):
        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration_dataset=calibration_dataset,
                         calibration_dataset_concat_size=calibration_dataset_concat_size,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         logger_board=logger_board, require_fwd=require_fwd)

        # dict: key is module name, value is the accumulated eigen_scaling_diag_matrix
        self.eigen_scaling_diag_matrix: Dict[str, torch.float32] = {}


        # Increase the dynamo cache size limit, default of 8 is too low
        if torch._dynamo.config.cache_size_limit < 64:
            torch._dynamo.config.cache_size_limit = 64

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        #self.eora_compute_lora = torch_compile(eora_compute_lora)
        #self.eora_process_input = torch_compile(eora_process_input)

        self.eora_compute_lora = eora_compute_lora
        self.eora_process_input = eora_process_input

    def log_plotly(self):
        task = self.logger_task
        if task is not None:
            from ..utils.plotly import create_plotly
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

        # dynamic override of adapter.rank
        adapter_cfg.rank = self.qcfg.dynamic_get(
                module.full_name,
                key="adapter",
                sub_key="rank",
                default=adapter_cfg.rank)

        # hack store property inside module
        module.adapter_cfg = adapter_cfg

        self.eigen_scaling_diag_matrix[module.name] = 0 # torch.tensor(0.0, dtype=torch.float32)

        return

    def is_skipped(self, module: NamedModule) -> bool:
        # dynamic override removed eora processing for this module
        return module.adapter_cfg in [None, {}]

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            self.eora_process_input(
                input=input,
                name=name,
                eigen_scaling_diag_matrix=self.eigen_scaling_diag_matrix,
                sample_size=self.num_batches
            )
        return tmp

    def process(self, module: NamedModule):
        assert isinstance(module.adapter_cfg, Lora)

        self.pb.title(f"EoRA gen: {module.name} in layer").draw()

        start = time.time()

        eigen_scaling_diag_matrix = self.eigen_scaling_diag_matrix[module.name]

        w: torch.Tensor = module.state.pop("w")
        w_device = w.device  # TODO clear up device situation between w and wq
        wq: torch.Tensor = module.state["wq"]

        # print(f"types: w = `{w.dtype}`, device = `{w.device}`, wq = `{wq.dtype}`,  device = `{wq.device}`")
        if w.dtype != torch.float16:
            w_wq_delta = w.to(dtype=torch.float32) - wq # wq is float16
        else:
            w_wq_delta = w - wq

        assert w_wq_delta.dtype == torch.float32

        # print(f"types: w_q_delta = `{w_wq_delta.dtype}`,  device = `{w_wq_delta.device}`")
        del w

        A, B = self.eora_compute_lora(
            device=w_device,
            w_wq_delta=w_wq_delta,
            module=module,
            eigen_scaling_diag_matrix=eigen_scaling_diag_matrix,
            rank=module.adapter_cfg.rank
        )

        # wq with A/B applied
        computed_wq = wq + (B @ A)

        module.state.update({
            "wq": move_to(wq, device=CPU, stream=self.stream),
        })

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq.to(dtype=module.weight.data.dtype)

        # for assert weight
        # module.state.update({
        #     "wq_ab": move_to(computed_wq.to(dtype=module.weight.data.dtype), device=CPU, stream=self.stream),
        # })

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
        log.info(stat)

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        self.result_save(module.full_name, {
            "rank": module.adapter_cfg.rank,
            "lora_A.weight": move_to(A.to(dtype=torch.float16), device=CPU, stream=self.stream),
            "lora_B.weight": move_to(B.to(dtype=torch.float16), device=CPU, stream=self.stream),
        })

        # eora = Lora(rank=module.adapter_cfg.rank, lora_A=A, lora_B=B)
        #
        # module.state.update({
        #     "adapter": eora,
        # })

    def submodule_finalize(self, module: NamedModule):
        pass
        # adapter: Lora = module.state.pop("adapter")
        #
        # # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        # self.result_save(module.full_name, {
        #     "lora_A.weight": move_to(adapter.lora_A.to(dtype=torch.float16), device=CPU, stream=self.stream),
        #     # A.to(dtype=torch.float16, device=CPU),
        #     "lora_B.weight": move_to(adapter.lora_B.to(dtype=torch.float16), device=CPU, stream=self.stream),
        #     # B.to(dtype=torch.float16, device=CPU),
        # })

    def finalize(self, model: BaseGPTQModel, **kwargs):
        # block for streams
        if self.stream:
            torch_sync()

        del self.eigen_scaling_diag_matrix

        # hack: store loras into model until `save()` is called
        model.lora_results = self.results()

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
        return "eora"
