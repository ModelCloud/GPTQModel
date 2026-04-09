# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from typing import Optional

import torch

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..quantization.config import (
    BitsAndBytesConfig,
    FP8Config,
    GGUFConfig,
    METHOD,
    RTNConfig,
    clone_weight_only_config_for_module,
    resolve_quant_format,
)
from ..quantization.rtn import RTN
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, pack_module
from ..utils.module_locks import parent_module_lock


log = setup_logger()


class WeightOnlyProcessor(LoopProcessor):
    """Process weight-only modules without entering activation-based quantization flows."""

    def __init__(
        self,
        tokenizer,
        qcfg: RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig,
    ):
        """Initializes a weight-only processor for RTN, GGUF, FP8, or BitsAndBytes."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            calibration_concat_separator=None,
            batch_size=1,
            execution_config=ExecutionConfig(
                require_fwd=False,
                fwd_replay_after_process=False,
            ),
        )
        self.lock = threading.Lock()

    @staticmethod
    def _uses_direct_pack(qcfg: RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig) -> bool:
        """Returns whether the method packs directly from the original dense weights."""

        return qcfg.method in {METHOD.GGUF, METHOD.FP8, METHOD.BITSANDBYTES}

    def _update_logged_loss(self, module: NamedModule, avg_loss: str) -> None:
        """Backfills the logged loss field after late dequant-error measurement."""

        with self.lock:
            for entry in reversed(self.log):
                if entry.get(PROCESS_LOG_LAYER) == module.layer_index and entry.get(PROCESS_LOG_MODULE) == module.name:
                    entry[QUANT_LOG_LOSS] = avg_loss
                    return

    def quantize_module(
        self,
        module: NamedModule,
    ) -> Optional[RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig]:
        """Clones per-module config, quantizes weights, and logs the result."""

        qcfg_clone = clone_weight_only_config_for_module(self.qcfg, module.full_name)
        if qcfg_clone is None:
            return None

        if self._uses_direct_pack(qcfg_clone):
            start_time = time.time()
            duration = time.time() - start_time
            avg_loss = f"{qcfg_clone.method.value}: pending"
            damp_percent = 0.0
            nsamples = 0
        else:
            task = RTN(module=module, qcfg=qcfg_clone)
            wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = task.quantize()

            module.stream_state_payload_to_cpu(
                {
                    "q_scales": q_scales,
                    "q_zeros": q_zeros,
                    "q_g_idx": q_g_idx,
                },
            )
            del q_scales, q_zeros, q_g_idx

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: avg_loss if isinstance(avg_loss, str) else f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
            "lifecycle": "weight_only",
        }

        with self.lock:
            self.log.append(stat)
        self.log_new_row(stat)

        if not self._uses_direct_pack(qcfg_clone):
            module.weight.data = wq
        return qcfg_clone

    def submodule_finalize(
        self,
        module: NamedModule,
        model: BaseQModel,
        *,
        qcfg: Optional[RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig] = None,
        **kwargs,
    ):
        """Creates and packs the final quantized module into the model graph."""

        active_qcfg = qcfg or self.qcfg
        if not self._uses_direct_pack(active_qcfg):
            module.stream_sync()
            with self.lock:
                q_zeros = module.state.pop("q_zeros").clone()
                q_scales = module.state.pop("q_scales").clone()
                q_g_idx = module.state.pop("q_g_idx").clone()

            assert q_zeros.device == CPU
            assert q_scales.device == CPU
            assert q_g_idx.device == CPU

        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))
        original_layer = layers.get(module.full_name)
        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block("create_quant_module", logger=log, module_name=module_label):
            with parent_module_lock(parent_key):
                create_quant_module(
                    name=module.full_name,
                    linear_cls=model.qlinear_kernel,
                    bits=active_qcfg.runtime_bits,
                    desc_act=active_qcfg.desc_act,
                    dynamic=active_qcfg.dynamic,
                    group_size=active_qcfg.group_size,
                    module=model.model,
                    submodule=module,
                    sym=active_qcfg.sym,
                    device=active_qcfg.device,
                    lm_head_name=model.lm_head,
                    pack_dtype=active_qcfg.pack_dtype,
                    format=resolve_quant_format(active_qcfg.format, active_qcfg.method),
                    register_buffers=False,
                    init_kwargs=active_qcfg.quant_linear_init_kwargs(),
                )
        if timer is not None and create_start is not None:
            timer.record("submodule_finalize_create", time.perf_counter() - create_start, source=module_label)

        qmodules = {
            name: submodule
            for name, submodule in find_modules(model.model, [model.qlinear_kernel]).items()
            if name == module.full_name
        }

        if self._uses_direct_pack(active_qcfg):
            pack_start = time.perf_counter() if timer is not None else None
            with log_time_block("module.pack_original", logger=log, module_name=module_label):
                with parent_module_lock(parent_key):
                    qmodule = qmodules[module.full_name]
                    qmodule.pack_original(
                        linear=original_layer,
                        scales=None,
                        zeros=None,
                        g_idx=None,
                        smooth=active_qcfg.smooth,
                    )
            if timer is not None and pack_start is not None:
                timer.record(
                    "submodule_finalize_pack",
                    time.perf_counter() - pack_start,
                    source=f"{module_label} [module.pack_original]",
                )

            reference_weight = qmodule._weight_to_matrix(original_layer).detach().cpu().to(torch.float32)
            dequant_weight = qmodule.dequantize_weight().T.detach().cpu().to(torch.float32)
            mean_abs_err = (dequant_weight - reference_weight).abs().mean().item()
            self._update_logged_loss(module, f"{active_qcfg.method.value}: {mean_abs_err:.7f}")
            module.state.pop("tp_pad_info", None)
            module.state.pop("quant_source_module", None)
            module.unregister_parameter("weight")
            return

        pack_start = time.perf_counter() if timer is not None else None
        with log_time_block("pack", logger=log, module_name=module_label):
            with parent_module_lock(parent_key):
                packer_label = pack_module(
                    name=module.full_name,
                    qModules=qmodules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=q_g_idx,
                    layers=layers,
                    quant_linear_cls=model.qlinear_kernel,
                    lock=self.lock,
                    quantize_config=active_qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        del q_scales, q_zeros, q_g_idx
        module.state.pop("tp_pad_info", None)
        module.state.pop("quant_source_module", None)
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model quantized and runs shared processor finalization."""

        model.quantized = True
        super().finalize(model=model, **kwargs)

    def name(self) -> str:
        """Returns the method-specific processor label used in logs."""

        if self.qcfg.method == METHOD.GGUF:
            return "weight_only_gguf"
        if self.qcfg.method == METHOD.FP8:
            return "weight_only_fp8"
        if self.qcfg.method == METHOD.BITSANDBYTES:
            return "weight_only_bitsandbytes"
        return "weight_only_rtn"

__all__ = ["WeightOnlyProcessor"]
