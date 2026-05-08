# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import DEVICE
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from ..quantization.config import METHOD, QuantizeConfig, resolve_quant_format
from ..utils.fallback import normalize_fallback
from ..quantization.qqq import QQQ
from ..utils.backend import BACKEND
from ..utils.logger import setup_logger, log_time_block
from ..utils.model import create_quant_module, find_modules, move_to, pack_module
from ..utils.torch import CPU

log = setup_logger()

class QQQProcessor(LoopProcessor):
    """Captures activations and quantizes modules with the QQQ workflow."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
    ):
        """Initializes QQQ processing and optional weight-delta tracking."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(require_fwd=require_fwd),
        )

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []

    def _quant_linear_kernel(self):
        device = self.qcfg.device
        if isinstance(device, DEVICE):
            return (QQQTorchLinear, BACKEND.QQQ_TORCH) if device == DEVICE.NPU else (QQQLinear, BACKEND.QQQ)
        if isinstance(device, torch.device):
            return (QQQTorchLinear, BACKEND.QQQ_TORCH) if device.type == "npu" else (QQQLinear, BACKEND.QQQ)
        if isinstance(device, str) and device.split(":")[0].lower() == "npu":
            return QQQTorchLinear, BACKEND.QQQ_TORCH
        return QQQLinear, BACKEND.QQQ

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because QQQ capture is fixed at construction."""

        raise NotImplementedError("QQQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Builds the per-module QQQ task after applying dynamic overrides."""

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

            qcfg_clone._resolve_activation_ordering(desc_act_override, act_group_aware_override)

        tmp = QQQ(module=module, qcfg=qcfg_clone)

        tmp.fallback = normalize_fallback(fallback, qcfg_clone.fallback)
        tmp.expected_nsamples = getattr(self, "total_calibration_tokens", None)

        if self.qcfg.mse > 0.0:
            mse = True
            norm = self.qcfg.mse
        else:
            mse = False
            norm = 100
        tmp.quantizer.configure(
            self.qcfg.bits,
            perchannel=True,
            sym=self.qcfg.sym,
            mse=mse,
            norm=norm,
            groupsize=self.qcfg.group_size,
        )

        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from QQQ work."""

        # gptq has no dynamic method of full override (removal)
        t = self.tasks.get(module.name, False)
        if t == False:
            return True
        else:
            return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the QQQ task."""

        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for QQQ statistics accumulation."""

            # gptq is mutable.
            q = self.tasks[name]  # noqa: F821
            q.add_batch(inp[0].data, out.data)  # noqa: F821
        return tmp

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Runs QQQ quantization for one module and stores pack-ready tensors."""

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)
        qqq = self.tasks

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        q = qqq[module.name]
        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, q_scales_extra, nsamples = q.quantize()

        q_scales = q_scales.to(CPU)
        q_zeros = q_zeros.to(CPU)
        q_g_idx = q_g_idx.to(CPU)

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
        if isinstance(avg_loss, str):
            loss_display = avg_loss
        else:
            loss_display = f"{avg_loss:.10f}" if isinstance(avg_loss, (int, float)) else "unknown"

        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.durations.append(duration)
            if isinstance(avg_loss, (int, float)):
                self.avg_losses.append(avg_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
            self.log.append(stat)

        self.log_new_row(stat)

        with self.lock:
            module.state.update({"q_scales": q_scales})
            module.state.update({"q_zeros": q_zeros})
            module.state.update({"q_g_idx": q_g_idx})
            module.state.update({"q_scales_extra": q_scales_extra})

        if self.calculate_w_wq_diff:
            if module.weight.data.dtype == torch.float16:
                # diff in float16
                w_wq_diff = module.weight.data - wq
            else:
                # diff in float32
                w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

            with self.lock:
                module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

        # with torch_streamCtx(DEVICE_0_STREAM):
        #     wq = wq.to(device=DEVICE_0, non_blocking=True) # move to d0 for post quant inference
        # wq = wq.to(device=DEVICE_0, non_blocking=False)

        # prepare for module.forward post generate
        module.weight.data = wq

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Creates the quantized module and packs the saved QQQ tensors into it."""

        # generate complete, safe to move to cpu
        module.weight.data = move_to(module.weight.data, device=CPU) # large weights is slow to init on cpu
        module.state.pop("w", None) # no need for original weights now

        # cleanup all memory or states vars persistently added by this processor
        with self.lock:
            module.state.pop("w", None)  #
            module.state.pop("w_wq_diff", None)

            q_zeros = module.state.pop("q_zeros")
            q_scales = module.state.pop("q_scales")
            q_g_idx = module.state.pop("q_g_idx")
            q_scales_extra = module.state.pop("q_scales_extra")

        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        quant_linear_cls, backend = self._quant_linear_kernel()

        # replace module with quantized module
        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            create_quant_module(
                name=module.full_name,
                linear_cls=quant_linear_cls,
                bits=self.qcfg.runtime_bits,
                desc_act=self.qcfg.desc_act,
                dynamic=self.qcfg.dynamic,
                group_size=self.qcfg.group_size,
                module=model.model,
                submodule=module,
                sym=self.qcfg.sym,
                device=self.qcfg.device,
                lm_head_name=model.lm_head,
                pack_dtype=self.qcfg.pack_dtype,
                format=resolve_quant_format(self.qcfg.format, self.qcfg.method),
                backend=backend,
                register_buffers=False,
            )

        # pack module
        qModules = {
            name: submodule
            for name, submodule in find_modules(model.model, [quant_linear_cls]).items()
            if name == module.full_name
        }
        with log_time_block(
            "pack",
            logger=log,
            module_name=module_label,
        ):
            pack_module(
                name=module.full_name,
                qModules=qModules,
                q_scales=q_scales,
                q_zeros=q_zeros,
                q_g_idx=q_g_idx,
                layers=layers,
                quant_linear_cls=quant_linear_cls,
                lock=self.lock,
                q_scales_extra=q_scales_extra,
                quantize_config=self.qcfg,
            )

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as QQQ-quantized and runs shared finalization logic."""

        # set quantized state
        model.quantized = True

        model.quantize_config.method = METHOD.QQQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures QQQ received calibration data before the quantization loop starts."""

        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    @classmethod
    def name(cls) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "qqq"

    def has_captured_input_ids(self, name: str) -> bool:
        """Reports whether the module saw at least one captured forward batch."""

        return self.tasks[name].fwd_counter > 0
