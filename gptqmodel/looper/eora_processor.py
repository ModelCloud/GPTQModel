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
from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch

from gptqmodel import QuantizeConfig
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models.writer import (QUANT_LOG_DAMP, QUANT_LOG_FWD_TIME, QUANT_LOG_LAYER,
                                     QUANT_LOG_LOSS, QUANT_LOG_MODULE, QUANT_LOG_TIME)
from gptqmodel.quantization import GPTQ
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.device import get_gpu_usage_memory, get_cpu_usage_memory
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import move_to, pack_model
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
        qcfg_clone = copy.deepcopy(self.qcfg)

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            qcfg_clone.adapter = self.qcfg.dynamic_get(module.full_name, "adapter", qcfg_clone.adapter)

        tmp = GPTQ(module=module, qcfg=qcfg_clone)

        self.tasks[module.name] = tmp
        return tmp

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            inp = input[0].detach().to(dtype=torch.float32) # TODO FIX ME: Do we really need to detach?
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)

            tmp = inp.shape[0]
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)

            nsamples = len(self.calibration_dataset)

            self.subset_eigen_scaling_diag_matrix[name] *= nsamples / (nsamples + tmp)
            self.subset_eigen_scaling_diag_matrix[name] += adds_sum / nsamples

            del inp, adds, adds_sum, output
        return tmp

    def process(self, module: NamedModule):
        self.pb.set_description(f"EoRA gen: {module.name} in layer {module.layer_index} of {self.layer_count - 1}")

        original_weight = module.state.get("w")
        quantized_weight = module.state.get("wq")

        dev = original_weight.device
        delta = original_weight - quantized_weight

        ## save this later for SVD
        raw_scaling_diag_matrix = self.subset_eigen_scaling_diag_matrix.pop(module.name).to(torch.float64).to(device=dev)

        L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
        if (L < 0).any().item():
            print(f"found negative eigenvalues in {module.name}")
            minimum = torch.min(L[L > 0])
            L[L < 0] = minimum

        sqrtEigenvalues = torch.sqrt(L)
        scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception:
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        ##
        delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

        r = self.qcfg.adapter.rank

        U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
        lowrank_r = r
        truc_s = S[:lowrank_r]
        truc_u = U[:, :lowrank_r]
        truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
        truc_sigma = torch.diag(truc_s)

        sqrtS = torch.sqrt(truc_sigma)
        B = torch.matmul(truc_u, sqrtS).to(quantized_weight.dtype)
        A = torch.matmul(sqrtS, truc_v).to(quantized_weight.dtype)

        # override module weight with computed weight with B@A delta
        comp_weight = quantized_weight + B @ A
        module.weight.data = comp_weight.to(module.weight.data.dtype)

        # lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(dtype=torch.float16)
        # lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(dtype=torch.float16)

        self.durations.append(duration)
        self.avg_losses.append(avg_loss)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = {QUANT_LOG_LAYER: module.layer_index, QUANT_LOG_MODULE: module.name, QUANT_LOG_LOSS: f"{avg_loss:.5f}",
                QUANT_LOG_DAMP: f"{damp_percent:.5f}", QUANT_LOG_TIME: f"{duration:.3f}",
                QUANT_LOG_FWD_TIME: f"{self.fwd_time:.3f}"}
        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.log.append(stat)
        logger.info(stat)

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        module.state.update({
            "lora_A": A.to(dtype=torch.float16, device=CPU),
            "lora_B": B.to(dtype=torch.float16, device=CPU),
        })

        del B, A, quantized_weight, U, S, V, L, Q

    def post_process(self, module: NamedModule):
        pass

    def submodule_finalize(self, module: NamedModule):
        pass

    def finalize(self, model: BaseGPTQModel, **kwargs):
        del self.eigen_scaling_diag_matrix
        super().finalize(model=model, **kwargs)

    def name(self) -> str:
        return "eora"
