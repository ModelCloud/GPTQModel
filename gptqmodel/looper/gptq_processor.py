# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import copy
import threading
import time
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..quantization import GPTQ, GPTQv2
from ..quantization.config import METHOD, QuantizeConfig
from ..utils.importer import select_quant_linear
from ..utils.logger import setup_logger, log_time_block
from ..utils.device import get_device
from ..utils.model import create_quant_module, find_modules, move_to, pack_model, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import tf32_disable_guard

log = setup_logger()
lock = threading.Lock()

class GPTQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int,
                 require_fwd: bool = True, calculate_w_wq_diff: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
                         calibration_concat_size=calibration_concat_size,
                         calibration_sort=calibration_sort,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         require_fwd=require_fwd)

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []

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
            desc_act_override = self.qcfg.dynamic_get(module.full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override
            act_group_aware_override = self.qcfg.dynamic_get(module.full_name, "act_group_aware", None)
            if act_group_aware_override is not None:
                qcfg_clone.act_group_aware = act_group_aware_override
            qcfg_clone.damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", qcfg_clone.damp_percent)
            qcfg_clone.static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", qcfg_clone.static_groups)
            qcfg_clone.v2 = self.qcfg.dynamic_get(module.full_name, "v2", qcfg_clone.v2)
            qcfg_clone.v2_alpha = self.qcfg.dynamic_get(module.full_name, "v2_alpha", qcfg_clone.v2_alpha)

            qcfg_clone._resolve_activation_ordering(desc_act_override, act_group_aware_override)

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
            batch_idx = self.current_batch_index()
            g.add_batch(inp[0].data, out.data, batch_index=batch_idx)  # noqa: F821
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

        expected_device = getattr(module, "target_device", None)
        if expected_device is None:
            expected_device = getattr(module.module, "target_device", None)
        if expected_device is None:
            expected_device = get_device(module.module)

        if expected_device is not None:
            expected_device = torch.device(expected_device)

            module_weight = getattr(module.module, "weight", None)
            if module_weight is not None:
                assert module_weight.device == expected_device, (
                    f"Module '{module.full_name}' weight device {module_weight.device} does not match "
                    f"assigned target device {expected_device}."
                )
                assert module_weight.data.device == expected_device, (
                    f"Module '{module.full_name}' weight.data device {module_weight.data.device} does not match "
                    f"assigned target device {expected_device}."
                )

            g_module = getattr(g, "module", None)
            g_weight = getattr(g_module, "weight", None) if g_module is not None else None
            if g_weight is not None:
                assert g_weight.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' expected device {expected_device}, "
                    f"but found weight on {g_weight.device}."
                )
                assert g_weight.data.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' weight.data on {g_weight.data.device} "
                    f"does not match target device {expected_device}."
                )

            g_h = getattr(g, "H", None)
            if g_h is not None:
                assert torch.device(g_h.device) == expected_device, (
                    f"GPTQ Hessian tensor for '{module.full_name}' lives on {g_h.device}, expected {expected_device}."
                )

            if expected_device.type == "cuda" and torch.cuda.is_available():
                current_cuda_device = torch.device("cuda", torch.cuda.current_device())
                assert current_cuda_device == expected_device, (
                    f"CUDA thread context {current_cuda_device} does not match expected device {expected_device} "
                    f"while processing '{module.full_name}'."
                )

        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        module.stream_state_payload_to_cpu(
            {
                "q_scales": q_scales,
                "q_zeros": q_zeros,
                "q_g_idx": q_g_idx,
            },
        )
        del q_scales, q_zeros, q_g_idx

        with self.lock:
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
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
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
            # assert module.weight.data.dtype in (torch.float16, torch.bfloat16)

            with self.lock:
                module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

        with self.lock:
            self.tasks[module.name].free()

            # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
            if self.calculate_w_wq_diff:
                module.state.update({
                    "wq": wq,  # fp16, quantized weight but not int4 (packed qweight)
                })

        # single largest deallocation of vram happens here
        module.weight.data = wq

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        # generate complete, safe to move to cpu
        # module.weight.data = move_to(module.state.pop("wq"), device=CPU) # large weights is slow to init on cpu

        # cleanup all memory or states vars persistently added by this processor
        module.stream_sync()
        with (self.lock):
            # if calculate_w_wq_diff is enabled (eora), we need to revert our original wq
            if self.calculate_w_wq_diff:
                module.weight.data = module.state.pop("wq").to(CPU)

            module.state.pop("w", None) #
            module.state.pop("w_wq_diff", None)

            # need to clone to due to steamed pinned memory and access on diff thread
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            q_g_idx = module.state.pop("q_g_idx").clone()

        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        assert q_g_idx.device == CPU

        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))

        # replace module with quantized module
        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
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
        if timer is not None and create_start is not None:
            timer.record(
                "submodule_finalize_create",
                time.perf_counter() - create_start,
                source=module_label,
            )

        # pack module
        qModules = {
            name: submodule
            for name, submodule in find_modules(model.model, [model.qlinear_kernel]).items()
            if name == module.full_name
        }
        pack_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "pack",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                packer_label = pack_module(
                    name=module.full_name,
                    qModules=qModules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=q_g_idx,
                    layers=layers,
                    quant_linear_cls=model.qlinear_kernel,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
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
