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
import threading
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_MAX_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..quantization import GPTQ, GPTQv2
from ..quantization.config import QUANT_METHOD, QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.model import move_to, pack_model
from ..utils.torch import CPU, DEVICE_0, DEVICE_0_STREAM, DEVICE_1, torch_empty_cache, torch_streamCtx, torch_sync

log = setup_logger()
lock = threading.Lock()

class GPTQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True, calculate_w_wq_diff: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration_dataset=calibration_dataset,
                         calibration_dataset_concat_size=calibration_dataset_concat_size,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         logger_board=logger_board, require_fwd=require_fwd)

        self.calculate_w_wq_diff = calculate_w_wq_diff
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

    def preprocess(self, module: NamedModule, buffered_fwd: bool, fail_safe: bool):
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
            qcfg_clone.v2 = self.qcfg.dynamic_get(module.full_name, "v2", qcfg_clone.v2)
            qcfg_clone.v2_alpha = self.qcfg.dynamic_get(module.full_name, "v2_alpha", qcfg_clone.v2_alpha)

        # store last used qcfg_dynamic
        self.qcfg_dynamic = qcfg_clone

        if qcfg_clone.v2 is True:
            tmp = GPTQv2(module=module, qcfg=qcfg_clone)
        else:
            tmp = GPTQ(module=module, qcfg=qcfg_clone)
            tmp.fail_safe = fail_safe

        # models like DeepSeek v3/r1 has > 256 $ of sub-modules per layer
        # use buffered mode go vram don't explode: gptq needs to store fwd inputs per each layer fwd
        # all sub-modules within a single layer needs to store all the inputs.
        # deepseek has massive # of sub-modules per layer, causing vram pressure
        # buffered mode is slower due to gpu<->cpu movement
        if buffered_fwd:
            log.info.once(f"Quantize: Enabling fwd buffered mode for: `{module.name}`")
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

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            g = self.tasks[name]  # noqa: F821
            g.add_batch(inp[0].data, out.data)  # noqa: F821
            del inp, out
        return tmp

    def pre_process_streaming(self, module: NamedModule):
        g = self.tasks[module.name]
        with torch_streamCtx(module.target_device_stream):
            # log.debug(f"streaming module `{g.name}` to device = `{module.target_device}`")
            if g.H is not None:
                g.H = g.H.to(device=module.target_device, non_blocking=True)
            g.module.weight.data = g.module.weight.data.to(device=module.target_device, non_blocking=True)

    def process(self, module: NamedModule, auto_gc: bool = True):
        # Reset peak memory stats
        #torch.cuda.reset_peak_memory_stats()
        self.pb.title(f"Quantizing {module.name} in layer ").draw()

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        with self.lock:
            g = self.tasks[module.name]

        wq, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        with self.lock:
            self.result_save(module.full_name, {
                "scale": scale,
                "zero": zero,
                "g_idx": g_idx,
            })

            self.durations.append(duration)
            self.avg_losses.append(avg_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
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

        stats_0 = torch.cuda.memory_stats(DEVICE_0)
        active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2
        peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

        if torch.cuda.device_count() > 1:
            stats_1 = torch.cuda.memory_stats(DEVICE_1)
            active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2
            peak_active_1 = stats_1.get("active_bytes.all.peak", 0) / 1024 ** 2

            max_memory = f"{active_0:.2f}MB, {active_1:.2f}MB"
        else:
            max_memory = f"{active_0:.2f}MB"

        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            QUANT_LOG_LOSS: f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
            PROCESS_MAX_MEMORY: max_memory,
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        # Log the new row
        self.log_new_row(stat)

        if self.calculate_w_wq_diff:
            if module.weight.data.dtype == torch.float16:
                # diff in float16
                w_wq_diff = module.weight.data - wq
            else:
                # diff in float32
                w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

            module.state.update({
                "w_wq_diff": w_wq_diff,
            })

        with self.lock:
            self.tasks[module.name].free()

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")

        module.state.update({
            "wq": wq,  # fp16, quantized weight but not int4 (packed qweight)
        })

        module.weight.data = wq

        # if auto_gc:
        #     torch_empty_cache()

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

    def name(self) -> str:
        # TODO fix me..this hacks inherited base class logic, why not override name in gptqv2?
        qcfg = self.qcfg_dynamic if self.qcfg_dynamic is not None else self.qcfg
        return "gptq v2" if qcfg.v2 else "gptq"
