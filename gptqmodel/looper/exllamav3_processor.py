# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import transformers
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd

from ..exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
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
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..quantization import QuantizeConfig
from ..quantization.config import EXL3Config, FORMAT, GPTQConfig, METHOD
from ..quantization.gptq import GPTQ
from ..utils.device import get_device
from ..utils.exllamav3 import create_exllamav3_module
from ..utils.logger import setup_logger
from ..utils.module_locks import parent_module_lock


setup_logger()

_EXL3_SIGMA_REG = 0.025
_OUT_SCALES_TO_ARG = {
    "always": True,
    "never": False,
    "auto": None,
    None: None,
}


def clone_exllamav3_config_for_module(
    qcfg: EXL3Config,
    module_full_name: str,
) -> Optional[EXL3Config]:
    """Clones and applies per-module EXL3 dynamic overrides, or skips the module."""

    if qcfg.dynamic_get(layer_name=module_full_name) == False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        qcfg_clone.bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
        qcfg_clone.head_bits = qcfg.dynamic_get(module_full_name, "head_bits", qcfg_clone.head_bits)

        out_scales_override = qcfg.dynamic_get(module_full_name, "out_scales", None)
        if out_scales_override is not None:
            qcfg_clone.out_scales = out_scales_override

        codebook_override = qcfg.dynamic_get(module_full_name, "codebook", None)
        if codebook_override is not None:
            qcfg_clone.codebook = codebook_override

        calibration_override = qcfg.dynamic_get(module_full_name, "calibration", None)
        if calibration_override is not None:
            qcfg_clone.calibration = calibration_override

    qcfg_clone.__post_init__()
    return qcfg_clone


class EXL3Processor(LoopProcessor):
    """Captures activations and repacks modules into ExLlamaV3 format."""

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
        calibration_concat_separator: Optional[str] = None,
        lm_head_name: str = "lm_head",
    ):
        """Initializes EXL3 processing and tracks the lm_head naming convention."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(
                require_fwd=require_fwd,
                fwd_replay_after_process=True,
                subset_forward_early_stop=True,
            ),
        )

        self.avg_losses = []
        self.lm_head_name = lm_head_name
        self._stats_lock = threading.Lock()

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because EXL3 capture is fixed at construction."""

        raise NotImplementedError("EXL3Processor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Builds the capture task and effective EXL3 config for one module."""

        del fallback, kwargs

        module_qcfg = clone_exllamav3_config_for_module(self.qcfg, module.full_name)
        if module_qcfg is None:
            return

        capture_qcfg = GPTQConfig(
            bits=max(1, module_qcfg.runtime_bits),
            group_size=-1,
            desc_act=False,
            sym=True,
            device=module_qcfg.device,
            pack_dtype=module_qcfg.pack_dtype,
        )

        task = GPTQ(module=module, qcfg=capture_qcfg)
        task.expected_nsamples = getattr(self, "total_calibration_tokens", None)
        task.quantizer.configure(perchannel=True)

        self.tasks[module.name] = {
            "capture": task,
            "qcfg": module_qcfg,
        }

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from EXL3 work."""

        return self.tasks.get(module.name, False) is False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the EXL3 task."""

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for the EXL3 capture task."""

            capture = self.tasks[name]["capture"]
            batch_idx = self.current_batch_index()
            capture.add_batch(inp[0].data, out.data, batch_index=batch_idx)
            del inp, out

        return tmp

    def _is_lm_head(self, module: NamedModule) -> bool:
        """Returns whether the named module corresponds to the model lm_head."""

        if module.full_name == self.lm_head_name:
            return True
        return module.full_name.endswith(f".{self.lm_head_name}")

    def _target_bits(self, module: NamedModule, module_qcfg: EXL3Config) -> int:
        """Chooses lm_head-specific bitwidth overrides when configured."""

        if self._is_lm_head(module) and module_qcfg.head_bits is not None:
            return max(1, int(module_qcfg.head_bits))
        return max(1, module_qcfg.runtime_bits)

    def _build_quant_args(
        self,
        module: NamedModule,
        module_qcfg: EXL3Config,
        device: torch.device,
    ) -> Dict[str, object]:
        """Builds the argument bundle passed into the EXL3 quantizer."""

        quant_args: Dict[str, object] = {
            "K": self._target_bits(module, module_qcfg),
            "devices": [device],
            "apply_out_scales": _OUT_SCALES_TO_ARG.get(module_qcfg.out_scales, None),
            "sigma_reg": _EXL3_SIGMA_REG,
            "seed": 787,
        }

        if module_qcfg.codebook == "mcg":
            quant_args["mcg"] = True
        elif module_qcfg.codebook == "mul1":
            quant_args["mul1"] = True

        return quant_args

    def _quant_input_weight(self, capture: GPTQ, device: torch.device) -> torch.Tensor:
        """Exports the captured dense weight matrix in EXL3 quantizer layout."""

        normalized = capture.clone_module(copy=True, device=device)
        return normalized.t().contiguous().to(torch.float32)

    def _restore_module_weight(self, module: NamedModule, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Reshapes the EXL3 output weight back into the wrapped module layout."""

        target = module.module if isinstance(module, NamedModule) else module

        if isinstance(target, transformers.Conv1D):
            return quantized_weight.contiguous().view_as(target.weight.data)

        if isinstance(target, (torch.nn.Linear, _ConvNd)):
            return quantized_weight.t().contiguous().view_as(target.weight.data)

        raise NotImplementedError(f"Unsupported EXL3 module type: {target.__class__.__name__}")

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Runs EXL3 quantization for one module and stages its packed tensors."""

        del subset, previous_subset, subset_index, subset_total

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        task_entry = self.tasks[module.name]
        capture: GPTQ = task_entry["capture"]
        module_qcfg: EXL3Config = task_entry["qcfg"]

        target_device = device or get_device(module.module)
        target_device = torch.device(target_device)
        if target_device.type != "cuda":
            raise ValueError("EXL3 quantization requires CUDA/HIP execution.")

        start_time = time.perf_counter()
        capture.finalize_hessian(target_device=target_device)
        hessian = capture.H
        if hessian is None:
            raise RuntimeError(f"EXL3 failed to capture Hessian for module `{module.full_name}`.")
        if capture.nsamples <= 0:
            raise RuntimeError(f"EXL3 captured no calibration activations for module `{module.full_name}`.")

        h_data = {
            "H": hessian,
            "count": capture.nsamples,
            "finalized": False,
        }

        quant_args = self._build_quant_args(module, module_qcfg, target_device)
        input_weight = self._quant_input_weight(capture, target_device)
        weight_q, proxy_err, out_tensors = quantize_exl3(
            weight=input_weight,
            H_data=h_data,
            quant_args=quant_args,
            return_weight_q=True,
        )
        duration = time.perf_counter() - start_time

        stream_payload = dict(out_tensors)
        if module.bias is not None:
            stream_payload["bias"] = module.bias.detach()
        module.stream_state_payload_to_cpu(stream_payload)

        restored_weight = self._restore_module_weight(module, weight_q)
        module.weight.data = restored_weight.to(dtype=module.weight.dtype)

        workspace_summary = getattr(capture, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(capture, "_borrow_workspace_totals", None)

        if isinstance(proxy_err, str):
            loss_display = proxy_err
        else:
            loss_display = f"{proxy_err:.10f}" if isinstance(proxy_err, (int, float)) else "unknown"

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{capture.nsamples}",
            QUANT_LOG_DAMP: f"{_EXL3_SIGMA_REG:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
        }

        if workspace_summary:
            requests = int(workspace_summary.get("requests", 0) or 0)
            if requests:
                hit_rate = float(workspace_summary.get("hit_rate", 0.0) or 0.0)
                chunk_rows = workspace_summary.get("chunk_rows")
                stat["workspace_cache_requests"] = str(requests)
                stat["workspace_cache_hit_rate"] = f"{hit_rate:.1%}"
                stat["workspace_stage_dtype"] = workspace_summary.get("staging_dtype", "")
                if chunk_rows is not None:
                    stat["workspace_chunk_rows"] = str(chunk_rows)
        if workspace_totals:
            total_requests = int(workspace_totals.get("requests", 0) or 0)
            if total_requests:
                cumulative_hit_rate = (
                    float(workspace_totals.get("materialized_hits", 0) or 0.0) / total_requests
                )
                stat["workspace_total_requests"] = str(total_requests)
                stat["workspace_total_hit_rate"] = f"{cumulative_hit_rate:.1%}"

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self._stats_lock:
            self.durations.append(duration)
            if isinstance(proxy_err, (int, float)):
                self.avg_losses.append(proxy_err)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
            self.log.append(stat)

        self.log_new_row(stat)

        capture.free()
        del input_weight, restored_weight, weight_q, out_tensors, stream_payload

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Builds and installs the ExLlamaV3 module from the staged tensors."""

        del kwargs

        module.stream_sync()

        tensors: Dict[str, torch.Tensor] = {}
        with self._stats_lock:
            module.state.pop("w", None)
            for tensor_name in ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"):
                tensor = module.state.pop(tensor_name, None)
                if tensor is not None:
                    tensors[tensor_name] = tensor.clone()

        parent_key = getattr(module, "full_name", getattr(module, "name", None))
        with parent_module_lock(parent_key):
            create_exllamav3_module(
                module_root=model.model,
                name=module.full_name,
                submodule=module,
                tensors=tensors,
            )

        module.unregister_parameter("weight")
        if getattr(module, "bias", None) is not None:
            module.unregister_parameter("bias")

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as EXL3-quantized and runs shared finalization logic."""

        model.quantized = True
        model.quantize_config.method = METHOD.EXL3
        model.quantize_config.format = FORMAT.EXL3
        model.qlinear_kernel = ExllamaV3Linear
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures EXL3 received calibration data before the quantization loop starts."""

        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("EXL3Processor's calibration_dataset must be provided.")
        return True

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "exl3"


__all__ = ["EXL3Processor", "clone_exllamav3_config_for_module"]
