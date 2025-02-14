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

from gptqmodel.utils.plotly import create_plotly

logger = setup_logger()

class GPTQProcessor(LoopProcessor):
    def __init__(self, calibration_dataset, qcfg: QuantizeConfig, logger_board: str = ""):
        super().__init__(calibration_dataset=calibration_dataset, qcfg=qcfg)
        self.quant_log = []
        self.quant_result = {}

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
            self.logger_task = Task.init(project_name='GPTQModel', task_name=f'GPTQProcessor-{RandomWords().get_random_word()}', task_type=Task.TaskTypes.optimizer)
        else:
            self.logger_task = None

        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.avg_losses = []
        self.module_names = []

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
            qcfg_clone.bits = self.qcfg.dynamic_get(module.full_name, "bits", qcfg_clone.bits)
            qcfg_clone.sym = self.qcfg.dynamic_get(module.full_name, "sym", qcfg_clone.sym)
            qcfg_clone.mse = self.qcfg.dynamic_get(module.full_name, "mse", qcfg_clone.mse)

            qcfg_clone.group_size = self.qcfg.dynamic_get(module.full_name, "group_size", qcfg_clone.group_size)
            qcfg_clone.desc_act = self.qcfg.dynamic_get(module.full_name, "desc_act", qcfg_clone.desc_act)
            qcfg_clone.damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", qcfg_clone.damp_percent)
            qcfg_clone.static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", qcfg_clone.static_groups)

        tmp = GPTQ(module=module, qcfg=qcfg_clone)

        # models like DeepSeek v3/r1 has > 256 $ of sub-modules per layer
        # use buffered mode go vram don't explode: gptq needs to store fwd inputs per each layer fwd
        # all sub-modules within a single layer needs to store all the inputs.
        # deepseek has massive # of sub-modules per layer, causing vram pressure
        # buffered mode is slower due to gpu<->cpu movement
        if buffered_fwd:  # TODO tweak this number for masive MoE
            logger.info(f"Experimental: enabling fwd buffered mode for: `{module.name}`")
            tmp.fwd_inputs_buffered = True

        tmp.quantizer.configure(
            perchannel=True,
        )
        self.tasks[module.name] = tmp
        return tmp

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            # gptq is mutable.
            g = self.tasks[name]  # noqa: F821
            g.add_batch(inp[0].data, out.data)  # noqa: F821
        return tmp

    def process(self, module: NamedModule):
        self.pb.set_description(f"Quantizing {module.name} in layer {module.layer_index} of {self.layer_count - 1}")
        gptq = self.tasks


        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        g = gptq[module.name]
        # TODO FIX ME, quantize does NOT need to pass any args! Check HF compat!
        wq, scale, zero, g_idx, duration, avg_loss, damp_percent = g.quantize()
        ## Assign the quantized weight to the weight
        #gptq[name].layer.weight.data = q_full_weight.to(device=gptq[name].device)

        ## Offload the quantized weight to CPU for EoRA
        #quantized_weights['model.layers.%d.%s' % (module_index, name)] = q_full_weights.cpu()

        # if task is not None:
        #     task.get_logger().report_scalar(
        #         title='Quantization Loss',
        #         series=f'layer_{module_index}_loss',
        #         value=avg_loss,
        #         iteration=name_index,
        #     )
        #
        #     task.get_logger().report_scalar(
        #         title='Quantization Time',
        #         series=f'layer_{module_index}_time',
        #         value=duration,
        #         iteration=name_index,
        #     )
        self.durations.append(duration)
        self.avg_losses.append(avg_loss)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = {QUANT_LOG_LAYER: module.layer_index, QUANT_LOG_MODULE: module.name, QUANT_LOG_LOSS: f"{avg_loss:.5f}",
                QUANT_LOG_DAMP: f"{damp_percent:.5f}", QUANT_LOG_TIME: f"{duration:.3f}",
                QUANT_LOG_FWD_TIME: f"{self.fwd_time:.3f}"}
        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.quant_log.append(stat)
        logger.info(stat)

        self.quant_result[module.full_name] = (
            move_to(scale, CPU),
            move_to(zero, CPU),
            move_to(g_idx, CPU),
        )
        w = module.weight.data
        # TODO FIXME data can't set to None
        # module.weight.data = None # Processor should fix this

        gptq[module.name].free()
        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        module.state.update({
            "w": w, # fp16, non-quantized weight
            "wq": wq, # fp16, quantized weight but not int4 (packed qweight)
        })

    def post_process(self, module: NamedModule):
        # prepare for module.foward post generate
        module.weight.data = module.state["wq"] # module.layer.weight or module.weight?

    def submodule_finalize(self, module: NamedModule):
        # generate complete, safe to move to cpu
        # TODO FIX: remove this? eora process need to override fwd in post_process so it can do wq + (A @ B)
        module.weight.data = module.state.pop("wq").cpu()
        module.state.pop("w") # no need for original weights now

    def model_finalize(self, model: BaseGPTQModel, **kwargs):
        backend = kwargs.pop("backend")
        model.qlinear_kernel = pack_model(
            model=model.model,
            quant_result=self.quant_result,
            bits=self.qcfg.bits,
            group_size=self.qcfg.group_size,
            backend=backend,
            desc_act=self.qcfg.desc_act,
            format=self.qcfg.format,
            lm_head_name=model.lm_head,
            dynamic=self.qcfg.dynamic,
            parallel_packing=self.qcfg.parallel_packing,
            pack_dtype=self.qcfg.pack_dtype,
        )
        model.quantized = True

        del self.quant_result

