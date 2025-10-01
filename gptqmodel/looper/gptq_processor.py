# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import copy
import threading
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import LoopProcessor, get_max_memory
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_MAX_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..quantization import GPTQ, GPTQv2
from ..quantization.config import METHOD, QuantizeConfig
from ..utils.importer import select_quant_linear
from ..utils.logger import setup_logger
from ..utils.model import create_quant_module, find_modules, move_to, pack_model, pack_module
from ..utils.torch import HAS_CUDA, tf32_disable_guard, torch_streamCtx, torch_sync

log = setup_logger()
lock = threading.Lock()

class GPTQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True, calculate_w_wq_diff: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
                         calibration_concat_size=calibration_concat_size,
                         calibration_sort=calibration_sort,
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

    def preprocess(self, module: NamedModule, fail_safe: bool):
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
            with tf32_disable_guard():
                g.add_batch(inp[0].data, out.data)  # noqa: F821
            del inp, out
        return tmp

    def process(self, module: NamedModule):
        # Reset peak memory stats
        #torch.cuda.reset_peak_memory_stats()
        self.pb.title(f"Quantizing {module.name} in layer ").draw()

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        with self.lock:
            g = self.tasks[module.name]

        with tf32_disable_guard():
            wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        q_scales = q_scales.to(CPU)
        q_zeros = q_zeros.to(CPU)
        q_g_idx = q_g_idx.to(CPU)

        with self.lock:
            module.state.update({"q_scales": q_scales})
            module.state.update({"q_zeros": q_zeros})
            module.state.update({"q_g_idx": q_g_idx})

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



        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            QUANT_LOG_LOSS: f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
            PROCESS_MAX_MEMORY: get_max_memory(),
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        # Log the new row
        self.log_new_row(stat)

        if self.calculate_w_wq_diff:
            # diff in float32
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

            with self.lock:
                module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

        with self.lock:
            self.tasks[module.name].free()

            # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
            module.state.update({
                "wq": wq,  # fp16, quantized weight but not int4 (packed qweight)
            })

        # single largest deallocation of vram happens here
        module.weight.data = wq

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        # generate complete, safe to move to cpu
        # module.weight.data = move_to(module.state.pop("wq"), device=CPU, stream=self.stream) # large weights is slow to init on cpu

        # cleanup all memory or states vars persistently added by this processor
        with self.lock:
            module.state.pop("w", None) #
            module.state.pop("w_wq_diff", None)

            q_zeros = module.state.pop("q_zeros")
            q_scales = module.state.pop("q_scales")
            q_g_idx = module.state.pop("q_g_idx")

        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        assert q_g_idx.device == CPU

        layers = find_modules(model.model)

        # replace module with quantized module
        create_quant_module(
            name=module.full_name,
            linear_cls=model.qlinear_kernel,
            bits=self.qcfg.bits,
            desc_act=self.qcfg.desc_act,
            dynamic=self.qcfg.dynamic,
            group_size=self.qcfg.group_size,
            module=model.model,
            submodule=module,
            sym=self.qcfg.sym,
            device=self.qcfg.device,
            lm_head_name=model.lm_head,
            pack_dtype=self.qcfg.pack_dtype,
            register_buffers=False,
        )

        # pack module
        qModules = {name: submodule for name, submodule in find_modules(model.model, [model.qlinear_kernel]).items() if name == module.full_name}
        pack_module(
            name=module.full_name,
            qModules=qModules,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            layers=layers,
            quant_linear_cls=model.qlinear_kernel,
            lock=self.lock,
        )

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        # block for streams
        if self.stream:
            torch_sync()

        # print("finalize")
        # print_module_tree(model.model)

        # set quantized state
        model.quantized = True
        model.quantize_config.quant_method = METHOD.GPTQ

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
