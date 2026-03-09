# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Optional

import torch

from ..looper.gptq_processor import clone_gptq_config_for_module
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
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
from ..quantization import GPTQ
from ..quantization.config import CalibrationlessMethod, FailSafe, FailSafeStrategy, METHOD, QuantizeConfig
from ..quantization.gptq import get_number_of_rows_and_cols
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, pack_module
from ..utils.module_locks import parent_module_lock


log = setup_logger()


class CalibrationlessGPTQProcessor(LoopProcessor):
    _TP_TARGETS = (2, 4, 8)

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
    ):
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            calibration_concat_separator=None,
            batch_size=1,
            require_fwd=False,
            fwd_after_process=False,
        )
        self.lock = threading.Lock()
        self.qcfg_dynamic: Optional[QuantizeConfig] = None

    def _zero_calibration_failsafe(self) -> FailSafe:
        calibrationless = self.qcfg.calibrationless
        if calibrationless is None:
            raise ValueError("Calibration-less GPTQ processor requires `qcfg.calibrationless` to be configured.")
        if calibrationless.method != CalibrationlessMethod.RTN:
            raise NotImplementedError(
                f"Calibration-less GPTQ processor only supports `method={CalibrationlessMethod.RTN.value}` today."
            )
        return FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=calibrationless.smooth,
        )

    def _annotate_tp_padding(self, module: NamedModule, qcfg: QuantizeConfig) -> None:
        target_multiple = math.lcm(*self._TP_TARGETS)
        if qcfg.group_size > 0:
            target_multiple = math.lcm(target_multiple, qcfg.group_size)

        _, columns = get_number_of_rows_and_cols(module)
        pad_cols = (target_multiple - (columns % target_multiple)) % target_multiple
        if pad_cols == 0:
            module.state.pop("tp_pad_info", None)
            return

        module.state["tp_pad_info"] = {
            "pad_cols": pad_cols,
            "target_multiple": target_multiple,
            "original_columns": columns,
        }

    def quantize_module(self, module: NamedModule) -> None:
        qcfg_clone = clone_gptq_config_for_module(
            self.qcfg,
            module.full_name,
            failsafe=self._zero_calibration_failsafe(),
        )
        if qcfg_clone is None:
            return

        self.qcfg_dynamic = qcfg_clone
        self._annotate_tp_padding(module, qcfg_clone)

        task = GPTQ(module=module, qcfg=qcfg_clone)
        task.quantizer.configure(perchannel=True)
        task.failsafe = qcfg_clone.failsafe
        task.expected_nsamples = 0

        if getattr(self, "_pause_controller", None) is not None and self.pb is not None:
            base_title = f"Quantizing {module.name} in layer"
            self._pause_controller.register_and_draw_progress_bar(self.pb, title=base_title, subtitle="")

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
            "lifecycle": "calibrationless",
        }

        with self.lock:
            self.log.append(stat)
        self.log_new_row(stat)

        module.weight.data = wq
        task.free()

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
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
        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block("create_quant_module", logger=log, module_name=module_label):
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
            timer.record("submodule_finalize_create", time.perf_counter() - create_start, source=module_label)

        qmodules = {
            name: submodule
            for name, submodule in find_modules(model.model, [model.qlinear_kernel]).items()
            if name == module.full_name
        }
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
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        model.quantized = True
        model.quantize_config.quant_method = METHOD.GPTQ
        super().finalize(model=model, **kwargs)

    def name(self) -> str:
        return "calibrationless_gptq"


__all__ = ["CalibrationlessGPTQProcessor"]
