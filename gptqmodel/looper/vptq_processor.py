# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.qlinear.vptq import VPTQLinear
from ..quantization.config import FORMAT, METHOD, QuantizeConfig, resolve_quant_format
from ..quantization.vptq import VPTQ
from ..utils.backend import BACKEND
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, move_to
from ..utils.torch import CPU


log = setup_logger()


class VPTQProcessor(LoopProcessor):
    """Captures activations and quantizes modules with the VPTQ workflow.

    The calibration forward passes feed the same Hessian accumulator used by
    GPTQ, but VPTQ turns the Hessian into codebook-driven vector quantization
    instead of scalar weight rounding. No pre-computed `hessian_path` is
    required: Hessians are always built from the calibration data.
    """

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
        """Initializes VPTQ processing and optional weight-delta tracking."""

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

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because VPTQ capture is fixed at construction."""

        raise NotImplementedError("VPTQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Builds the per-module VPTQ task after applying dynamic overrides."""

        del kwargs

        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:  # noqa: E712
            return

        qcfg_clone = copy.deepcopy(self.qcfg)

        tmp = VPTQ(module=module, qcfg=qcfg_clone)

        tmp.fallback = normalize_fallback(fallback, qcfg_clone.fallback)
        tmp.expected_nsamples = getattr(self, "total_calibration_tokens", None)

        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from VPTQ work."""

        t = self.tasks.get(module.name, False)
        return t == False  # noqa: E712

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the VPTQ task."""

        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for the module Hessian."""

            q = self.tasks[name]
            q.add_batch(inp[0].data, out.data)
        return tmp

    @staticmethod
    def _result_to_cpu(result):
        """Moves every tensor of a `VPTQQuantResult` to host memory."""

        def to_cpu(value):
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if isinstance(value, torch.Tensor):
                return move_to(value, device=CPU)
            return value

        result.centroids = to_cpu(result.centroids)
        result.indices = to_cpu(result.indices)
        result.res_centroids = to_cpu(result.res_centroids)
        result.res_indices = to_cpu(result.res_indices)
        result.weight_scale = to_cpu(result.weight_scale)
        result.weight_bias = to_cpu(result.weight_bias)
        result.perm = to_cpu(result.perm)
        result.quantized_weight = to_cpu(result.quantized_weight)
        return result

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Runs VPTQ quantization for one module and stores the pack-ready result."""

        del device, subset, previous_subset, subset_index, subset_total

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        q = self.tasks[module.name]
        try:
            result, duration, avg_loss = q.quantize()
            nsamples = q.nsamples
        finally:
            q.free()

        wq = result.quantized_weight
        if self.calculate_w_wq_diff:
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)
        else:
            w_wq_diff = None

        # The packed codebooks/indices are small relative to the weight matrix,
        # so a direct host copy is cheaper than a pinned staging round trip.
        result = self._result_to_cpu(result)

        loss_display = f"{avg_loss:.10f}" if isinstance(avg_loss, (int, float)) else "unknown"

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{self.qcfg.percdamp:.5f}",
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

            module.state["vptq_result"] = result
            if self.calculate_w_wq_diff:
                module.state["w_wq_diff"] = w_wq_diff
                # EoRA rematerializes the dense module before consuming this.
                module.state["wq"] = wq.detach().to(device=CPU, copy=True)

        self.log_new_row(stat)

        # single largest deallocation of vram happens here
        module.weight.data = wq

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Creates the quantized module and loads the saved VPTQ codebooks into it."""

        del kwargs

        if self.calculate_w_wq_diff:
            module.weight.data = module.state.pop("wq").to(CPU)
        else:
            module.weight.data = move_to(module.weight.data, device=CPU)

        with self.lock:
            module.state.pop("w", None)
            module.state.pop("w_wq_diff", None)
            result = module.state.pop("vptq_result")

        layers = {module.full_name: model.model.get_submodule(module.full_name)}
        module_label = getattr(module, "full_name", getattr(module, "name", ""))

        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            create_quant_module(
                name=module.full_name,
                linear_cls=VPTQLinear,
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
                backend=BACKEND.VPTQ_TORCH,
                register_buffers=False,
                init_kwargs=self.qcfg.quant_linear_init_kwargs(),
                dtype=module.module_dtype,
            )

        qmodule = model.model.get_submodule(module.full_name)
        with log_time_block(
            "module.pack",
            logger=log,
            module_name=module_label,
        ):
            qmodule.pack(linear=layers[module.full_name], vptq_result=result)

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as VPTQ-quantized and runs shared finalization logic."""

        model.quantized = True
        model.quantize_config.method = METHOD.VPTQ
        model.quantize_config.format = FORMAT.VPTQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures VPTQ received calibration data before the quantization loop starts."""

        if self.calibration_dataset is None:
            raise ValueError("VPTQProcessor's calibration_dataset must be provided.")
        return True

    @classmethod
    def name(cls) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "vptq"

    def has_captured_input_ids(self, name: str) -> bool:
        """Reports whether the module saw at least one captured forward batch."""

        return self.tasks[name].fwd_counter > 0
