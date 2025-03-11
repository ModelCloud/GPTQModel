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
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME, QUANT_LOG_DAMP, QUANT_LOG_LOSS)
from ..quantization import GPTQ
from ..quantization.config import QUANT_METHOD, QuantizeConfig
from ..quantization.gptq import CPU
from ..utils.logger import setup_logger
from ..utils.model import move_to, pack_model
from ..utils.torch import torch_sync

log = setup_logger()

class GPTQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True, retain_w: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration_dataset=calibration_dataset,
                         calibration_dataset_concat_size=calibration_dataset_concat_size,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         logger_board=logger_board, require_fwd=require_fwd)

        self.retain_w = retain_w
        self.avg_losses = []

    def log_plotly(self):
        task = self.logger_task
        if task is not None:
            from ..utils.plotly import create_plotly
            x = list(range(self.layer_count))
            gpu_fig = create_plotly(x=x, y=self.gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
            cpu_fig = create_plotly(x=x, y=self.cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
            loss_fig = create_plotly(x=self.module_names, y=self.avg_losses, xaxis_title="layer", yaxis_title="loss")
            time_fig = create_plotly(x=self.module_names, y=self.durations, xaxis_title="layer", yaxis_title="time")
            task.get_logger().report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
            task.get_logger().report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
            task.get_logger().report_plotly('avg_loss', 'avg_loss', loss_fig)
            task.get_logger().report_plotly('quant_time', 'quant_time', time_fig)

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("GPTQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, buffered_fwd: bool):
        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:
            return

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
            log.info(f"Experimental: enabling fwd buffered mode for: `{module.name}`")
            tmp.fwd_inputs_buffered = True

        tmp.quantizer.configure(
            perchannel=True,
        )
        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        # gptq has no dynamic method of full override (removal)
        t = self.tasks.get(module.name, False)
        if t == False:
            return True
        else:
            return False

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            # gptq is mutable.
            g = self.tasks[name]  # noqa: F821
            g.add_batch(inp[0].data, out.data)  # noqa: F821
        return tmp

    def process(self, module: NamedModule):
        self.pb.title(f"Quantizing {module.name} in layer ").draw()

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        g = self.tasks[module.name]
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

        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            QUANT_LOG_LOSS: f"{avg_loss:.5f}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.log.append(stat)
        log.info(stat)

        self.result_save(module.full_name, {
            "scale": move_to(scale, device=CPU, stream=self.stream),
            "zero": move_to(zero, device=CPU, stream=self.stream),
            "g_idx": move_to(g_idx, device=CPU, stream=self.stream),
        })

        if self.retain_w:
            # original weights
            w = module.weight.data
            module.state.update({
                "w": w,  # bf16/fp16, non-quantized native weight
            })

        self.tasks[module.name].free()

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        module.state.update({
            "wq": wq, # fp16, quantized weight but not int4 (packed qweight)
        })

        # prepare for module.forward post generate
        module.weight.data = wq

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule):
        # generate complete, safe to move to cpu
        module.weight.data = move_to(module.state.pop("wq"), device=CPU, stream=self.stream) # large weights is slow to init on cpu
        module.state.pop("w", None) # no need for original weights now

    def finalize(self, model: BaseGPTQModel, **kwargs):
        # block for streams
        if self.stream:
            torch_sync()

        backend = kwargs.pop("backend")
        model.qlinear_kernel = pack_model(
            model=model.model,
            quant_result=self.results(),
            bits=self.qcfg.bits,
            group_size=self.qcfg.group_size,
            backend=backend,
            desc_act=self.qcfg.desc_act,
            format=self.qcfg.format,
            quant_method=self.qcfg.quant_method,
            lm_head_name=model.lm_head,
            dynamic=self.qcfg.dynamic,
            parallel_packing=self.qcfg.parallel_packing,
            pack_dtype=self.qcfg.pack_dtype,
        )

        # set quantized state
        model.quantized = True

        model.quantize_config.quant_method = QUANT_METHOD.GPTQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    @classmethod
    def name(cls) -> str:
        return "gptq"
