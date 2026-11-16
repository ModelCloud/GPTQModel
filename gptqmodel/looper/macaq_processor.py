# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-FileCopyrightText: 2026 the MACAQ authors
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# MACaQ calibration/quantization processor.
#
# The calibration lifecycle uses GPT-QModel's layer looper. Packed W4A4
# block replay, quantization math, and packed storage follow the Huawei CSL
# MACaQ reference recipe, ported in
# `gptqmodel/quantization/macaq.py` from
# https://github.com/huawei-csl/MACaQ (Apache-2.0; arXiv:2611.10649).

from __future__ import annotations

import copy
from types import MethodType
from typing import Callable, Dict, Optional, Tuple

import torch
import transformers
from torch.nn import Module

from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    CalibrationTypes,
    CheckpointSupport,
    ForwardConfig,
    LoopProcessor,
    ProcessorLifecycleConfig,
    bind_resolved_module_config,
    get_resolved_module_config,
)
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
from ..nn_modules.qlinear.macaq import MacaqLinear, MacaqTritonLinear
from ..quantization import macaq as macaq_core
from ..quantization.config import METHOD
from ..quantization.macaq import Macaq
from ..utils.backend import BACKEND
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, move_to
from ..utils.torch import CPU


log = setup_logger()


class MacaqProcessor(LoopProcessor):
    """Captures Hessians and quantizes Linear modules with the MACaQ recipe."""

    lifecycle = ProcessorLifecycleConfig(
        forward=ForwardConfig(
            required=True,
            replay_after_process=True,
            all_modules_in_single_pass=True,
            subset_early_stop=True,
        ),
        calibration=CalibrationTypes.EACH_MODULE_NEEDS_INPUT,
        checkpointing=CheckpointSupport.PRIMARY,
    )

    def __init__(
        self,
        tokenizer,
        qcfg,
        calibration,
        forward: Optional[ForwardConfig] = None,
        calculate_w_wq_diff: bool = False,
        backend: BACKEND = BACKEND.MACAQ_TORCH,
    ):
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            forward=forward,
        )
        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        if backend not in (BACKEND.MACAQ_TORCH, BACKEND.MACAQ_TRITON):
            raise ValueError(f"MACaQ requires a resolved Torch or Triton pack backend, got {backend!r}.")
        self.pack_backend = backend

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("MacaqProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        del kwargs
        resolved_cfg = self.qcfg.resolve_module_config(module.full_name)
        if resolved_cfg.excluded:
            return
        bind_resolved_module_config(module, resolved_cfg)

        qcfg_clone = copy.deepcopy(self.qcfg)
        overrides = resolved_cfg.overrides
        for key, value in overrides.items():
            setattr(qcfg_clone, key, value)
        if overrides:
            qcfg_clone.__post_init__()
        task = Macaq(module=module, qcfg=qcfg_clone)
        # A per-module `dynamic` fallback override outranks the processor-level
        # fallback, so it is merged as the value rather than as the default.
        task.fallback = normalize_fallback(
            resolved_cfg.get("fallback", None),
            normalize_fallback(fallback, self.qcfg.fallback),
        )
        task.expected_nsamples = getattr(self, "total_calibration_tokens", None)
        self.tasks[module.name] = task

    def is_skipped(self, module: NamedModule) -> bool:
        return self.tasks.get(module.name, False) == False  # noqa: E712

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            self.tasks[name].add_batch(inp[0].data, out.data)

        return hook

    @staticmethod
    def _effective_weight(result) -> torch.Tensor:
        """Undo the saved (D1 balance, block-Hadamard) frame for replay.

        The stored weight is ``Wr = (W / p) @ BH``; the effective dense weight
        in the original frame is ``(Wr @ BH.T) * p``.
        """

        dequant = result.dequantized_weight
        if dequant is None:
            raise ValueError("MacaqProcessor requires the quantizer's dequantized reference weight.")
        bh = result.BH.float()
        weight = macaq_core.rotate_blocks(dequant.float(), bh.T)
        if result.pre_scale is not None:
            weight = weight * result.pre_scale.float().unsqueeze(0)
        return weight

    @staticmethod
    def _result_to_cpu(result):
        result.W_packed = move_to(result.W_packed, device=CPU)
        result.W_scales = move_to(result.W_scales, device=CPU)
        result.BH = move_to(result.BH, device=CPU)
        if result.pre_scale is not None:
            result.pre_scale = move_to(result.pre_scale, device=CPU)
        result.dequantized_weight = None
        return result

    @staticmethod
    def _packed_replay_forward(linear: Module, x: torch.Tensor) -> torch.Tensor:
        return linear._macaq_replay(x)

    def _install_packed_replay(self, module: NamedModule, task: Macaq, result) -> None:
        """Replay with MXFP4 activations, as the upstream block loop does.

        The looper keeps the original Linear until finalization. A temporary
        packed child supplies its forward during subset and block replay, then
        is removed before the layer is offloaded and packed permanently.
        """

        linear = module.module
        if module.state.get("macaq_replay_active") or "_macaq_replay" in linear._modules:
            raise RuntimeError(f"MACaQ replay is already installed for {module.full_name}.")

        qlinear_cls, _ = self._linear_cls()
        qcfg = task.qcfg
        replay = qlinear_cls(
            bits=4,
            group_size=qcfg.group_size,
            in_features=task.columns,
            out_features=task.rows,
            bias=linear.bias is not None,
            pack_dtype=qcfg.pack_dtype,
            block_size=qcfg.block_size,
            had_block=qcfg.had_block,
            activation_precision=qcfg.activation_precision,
            pre_mode=qcfg.pre_mode,
            pre_alpha=qcfg.pre_alpha,
            rotation=qcfg.rotation,
            dtype=module.module_dtype,
        ).to(module.weight.device)
        replay.load_quant_result(result, bias=linear.bias)

        # Retain the original forward only for this calibration layer. In
        # particular, W4A4 activation rounding must affect the next block's
        # Hessian; a dense dequantized weight replay would be W4A16 instead.
        module.state["macaq_original_forward"] = linear.__dict__.get("forward")
        linear._macaq_replay = replay
        linear.forward = MethodType(self._packed_replay_forward, linear)
        module.state["macaq_replay_active"] = True

    @staticmethod
    def _remove_packed_replay(module: NamedModule) -> None:
        if not module.state.pop("macaq_replay_active", False):
            return
        linear = module.module
        original = module.state.pop("macaq_original_forward")
        if original is None:
            linear.__dict__.pop("forward", None)
        else:
            linear.forward = original
        del linear._macaq_replay

    def prepare_layer_post_quantize(
        self, *, model: BaseQModel, layer_module: Module, layer_index: int,
        processed_modules: Dict[str, NamedModule], is_lm_head_module: bool,
    ) -> None:
        del model, layer_module, layer_index, is_lm_head_module
        for module in processed_modules.values():
            if isinstance(module, NamedModule):
                self._remove_packed_replay(module)

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        del device, subset, previous_subset, subset_index, subset_total

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        task = self.tasks[module.name]
        try:
            result, duration, avg_loss = task.quantize()
            nsamples = task.nsamples
        finally:
            task.free()

        effective_weight = self._effective_weight(result)
        replay_weight = effective_weight.to(dtype=module.weight.dtype, device=module.weight.device)
        if isinstance(module.module, transformers.Conv1D):
            replay_weight = replay_weight.T.contiguous()
        if self.calculate_w_wq_diff:
            w_wq_diff = module.weight.data.to(torch.float32) - replay_weight.to(torch.float32)
        else:
            w_wq_diff = None

        self._install_packed_replay(module, task, result)
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
            QUANT_LOG_DAMP: f"{task.qcfg.ridge:.5f}",
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

            module.state["macaq_result"] = result
            if self.calculate_w_wq_diff:
                module.state["w_wq_diff"] = w_wq_diff
                module.state["wq"] = replay_weight.detach().to(device=CPU, copy=True)

        self.log_new_row(stat)
        module.weight.data = replay_weight

    def _linear_cls(self):
        if self.pack_backend == BACKEND.MACAQ_TRITON:
            return MacaqTritonLinear, BACKEND.MACAQ_TRITON
        return MacaqLinear, BACKEND.MACAQ_TORCH

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        del kwargs

        self._remove_packed_replay(module)

        if self.calculate_w_wq_diff:
            module.weight.data = module.state.pop("wq").to(CPU)
        else:
            module.weight.data = move_to(module.weight.data, device=CPU)

        with self.lock:
            module.state.pop("w", None)
            module.state.pop("w_wq_diff", None)
            result = module.state.pop("macaq_result")

        linear_cls, backend = self._linear_cls()
        layers = {module.full_name: model.model.get_submodule(module.full_name)}
        module_label = getattr(module, "full_name", getattr(module, "name", ""))

        with log_time_block("create_quant_module", logger=log, module_name=module_label):
            create_quant_module(
                name=module.full_name,
                linear_cls=linear_cls,
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
                format=self.qcfg.format,
                backend=backend,
                register_buffers=False,
                init_kwargs=self.qcfg.quant_linear_init_kwargs(),
                dtype=module.module_dtype,
                resolved_module_config=get_resolved_module_config(module),
            )

        qmodule = model.model.get_submodule(module.full_name)
        with log_time_block("module.pack", logger=log, module_name=module_label):
            qmodule.pack(linear=layers[module.full_name], macaq_result=result)

        with self.lock:
            self.result_pop(module.full_name)

        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        model.quantized = True
        model.quantize_config.method = METHOD.MACAQ
        model.quantize_config.format = model.quantize_config.checkpoint_format_for_settings()
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            raise ValueError("MacaqProcessor's calibration_dataset must be provided.")
        return True

    @classmethod
    def name(cls) -> str:
        return "macaq"

    def has_captured_input_ids(self, name: str) -> bool:
        return self.tasks[name].fwd_counter > 0
