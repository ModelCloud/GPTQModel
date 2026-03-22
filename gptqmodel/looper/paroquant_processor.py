# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from ..quantization.config import FORMAT, METHOD, QuantizeConfig, resolve_quant_format
from ..quantization.paroquant.optimization import optimize_paroquant_linear
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, move_to, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import CPU

log = setup_logger()


@dataclass
class _ParoQuantLayerState:
    modules: Dict[str, NamedModule] = field(default_factory=dict)
    subset_total: Optional[int] = None
    processed_subsets: Set[int] = field(default_factory=set)
    pending_modules: Set[str] = field(default_factory=set)
    quantized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ParoQuantProcessor(LoopProcessor):
    """Standalone ParoQuant lifecycle: capture, optimize, export, then pack."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        gptq_model,
        model,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
    ):
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
                fwd_after_process=True,
                subset_forward_early_stop=True,
                enable_activation_capture=True,
            ),
        )

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses: list[float] = []
        self.gptq_model = gptq_model
        self.model = model
        self.format = resolve_quant_format(qcfg.format, qcfg.method)
        self.qlinear_kernel = self._select_qlinear_kernel_for_format(self.format)
        self._layer_states: Dict[int, _ParoQuantLayerState] = {}
        self._layer_states_lock = threading.Lock()
        self.fallback = qcfg.fallback

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("ParoQuantProcessor's calibration_dataset cannot be modified")

    def _select_qlinear_kernel_for_format(self, format_value: FORMAT):
        fmt = FORMAT(format_value) if not isinstance(format_value, FORMAT) else format_value
        if fmt != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support this FORMAT: {format_value}")
        return ParoQuantQuantLinear

    def _resolve_qlinear_kernel(self, module_name: Optional[str] = None):
        format_override = self.qcfg.dynamic_get(module_name, "format", None) if module_name else None
        target_format = resolve_quant_format(format_override or self.qcfg.format, self.qcfg.method)
        if target_format != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support dynamic format override `{target_format}`.")
        return ParoQuantQuantLinear

    def _get_layer_state(self, layer_index: int) -> _ParoQuantLayerState:
        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _ParoQuantLayerState()
                self._layer_states[layer_index] = state
        return state

    def _record_input_feature(self, module_name: str, feature: torch.Tensor) -> None:
        if feature.dim() <= 2:
            feature = feature.unsqueeze(0)

        if feature.device.type != "cpu":
            feature = feature.detach().cpu()
        else:
            feature = feature.detach()

        with self.lock:
            entry = self.tasks.get(module_name)
            if entry is None:
                entry = {"inputs": []}
                self.tasks[module_name] = entry
            entry.setdefault("inputs", []).append(feature)

    def _layer_input_features(self, state: _ParoQuantLayerState) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        for name in list(state.modules):
            entry = self.tasks.get(name) or {}
            tensors: List[torch.Tensor] = entry.get("inputs", [])  # type: ignore[arg-type]
            if not tensors:
                features[name] = torch.empty(0)
                continue
            try:
                features[name] = torch.cat(tensors, dim=0)
                entry["inputs"] = [features[name]]
            except RuntimeError:
                features[name] = tensors[0]
        return features

    def _module_quant_params(self, module_name: str) -> tuple[int, int, bool]:
        bits = int(self.qcfg.dynamic_get(module_name, "bits", self.qcfg.runtime_bits))
        group_size = int(self.qcfg.dynamic_get(module_name, "group_size", self.qcfg.group_size))
        sym = bool(self.qcfg.dynamic_get(module_name, "sym", self.qcfg.sym))
        return bits, group_size, sym

    @staticmethod
    def _module_weight_matrix(module: NamedModule) -> torch.Tensor:
        weight = module.weight.data
        if weight.dim() != 2:
            raise ValueError(
                f"ParoQuant currently expects rank-2 module weights, got {tuple(weight.shape)} for `{module.full_name}`."
            )
        return weight

    def _quantize_one_module(
        self,
        module: NamedModule,
        inputs: torch.Tensor,
    ) -> tuple[float, float]:
        bits, group_size, sym = self._module_quant_params(module.full_name)
        weight = self._module_weight_matrix(module)
        bias = module.bias.data if getattr(module, "bias", None) is not None else None
        original_weight = weight.detach().clone()
        if inputs.numel() == 0:
            inputs = torch.empty((0, weight.shape[1]), dtype=weight.dtype, device=weight.device)

        with torch.inference_mode(False), torch.enable_grad():
            result = optimize_paroquant_linear(
                weight=weight,
                bias=bias,
                inputs=inputs,
                bits=bits,
                group_size=group_size,
                sym=sym,
                krot=self.qcfg.krot,
                pair_ratio=self.qcfg.opt_pair_ratio,
                train_rows=self.qcfg.opt_train_samples,
                val_rows=self.qcfg.opt_validation_samples,
                batch_size=self.qcfg.opt_batch_size,
                rotation_epochs=self.qcfg.opt_rotation_epochs,
                finetune_epochs=self.qcfg.opt_finetune_epochs,
                rotation_lr=self.qcfg.opt_rotation_lr,
                weight_lr=self.qcfg.opt_weight_lr,
                quantizer_lr=self.qcfg.opt_quantizer_lr,
                seed=self.qcfg.opt_seed + module.layer_index + hash(module.full_name),
            )

        pseudo_weight = result.pseudo_weight.to(device=weight.device, dtype=weight.dtype)
        pack_weight = result.pack_weight.to(dtype=weight.dtype).cpu()
        q_scales = result.q_scales.to(dtype=weight.dtype).cpu()
        q_zeros = result.q_zeros.cpu()
        pairs = result.pairs.to(dtype=torch.int16).cpu()
        theta = result.theta.to(dtype=weight.dtype).cpu()
        channel_scales = result.channel_scales.to(dtype=weight.dtype).cpu()

        with self.lock:
            module.state.update(
                {
                    "pack_weight": pack_weight,
                    "q_scales": q_scales,
                    "q_zeros": q_zeros,
                    "pairs": pairs,
                    "theta": theta,
                    "channel_scales": channel_scales,
                }
            )

        if self.calculate_w_wq_diff:
            if original_weight.dtype == torch.float16:
                w_wq_diff = original_weight - pseudo_weight
            else:
                w_wq_diff = original_weight.to(dtype=torch.float32) - pseudo_weight.to(dtype=torch.float32)
            with self.lock:
                module.state["w_wq_diff"] = w_wq_diff

        module.weight.data = pseudo_weight
        return result.train_loss, result.val_loss

    def _quantize_layer(self, layer_index: int, state: _ParoQuantLayerState) -> None:
        if state.quantized:
            return

        input_feat = self._layer_input_features(state)
        for module_name, tensor in input_feat.items():
            if tensor.numel() == 0 and not self.fallback:
                raise RuntimeError(
                    f"ParoQuantProcessor error: missing activation features for `{module_name}` with fallback disabled."
                )

        for module_name, named_module in list(state.modules.items()):
            feat = input_feat.get(module_name)
            if feat is None:
                feat = torch.empty(0)

            start = time.perf_counter()
            train_loss, val_loss = self._quantize_one_module(named_module, feat)
            duration = time.perf_counter() - start

            stat = {
                PROCESS_LOG_NAME: self.name(),
                PROCESS_LOG_LAYER: named_module.layer_index,
                PROCESS_LOG_MODULE: named_module.name,
                MODULE_FEATURE_COLUMN: self.module_feature_summary(named_module),
                DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named_module),
                QUANT_LOG_LOSS: f"{val_loss:.10f}",
                QUANT_LOG_NSAMPLES: f"{0 if feat.numel() == 0 else feat.reshape(-1, feat.shape[-1]).shape[0]}",
                PROCESS_LOG_TIME: f"{duration:.3f}",
                PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            }

            with self.lock:
                self.durations.append(duration)
                self.avg_losses.append(val_loss)
                self.module_names.append(f"layer-{named_module.layer_index}-{named_module.name}")
                self.log.append(stat)

            self.log_new_row(stat)

        state.quantized = True
        state.modules.clear()
        state.pending_modules.clear()
        state.processed_subsets.clear()
        state.subset_total = None

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        self.fallback = normalize_fallback(fallback, self.qcfg.fallback)
        layer_state = self._get_layer_state(module.layer_index)
        with layer_state.lock:
            layer_state.modules[module.name] = module
            layer_state.pending_modules.add(module.name)

        with self.lock:
            entry = self.tasks.get(module.name)
            if entry is None:
                self.tasks[module.name] = {"inputs": []}
            else:
                entry.setdefault("inputs", [])

    def is_skipped(self, module: NamedModule) -> bool:
        return self.tasks.get(module.name, False) is False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            del module, out
            if not inp:
                return
            feature = inp[0] if isinstance(inp, (tuple, list)) else inp
            self._record_input_feature(name, feature)

        return hook

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        del device, previous_subset
        layer_index = module.layer_index
        state = self._get_layer_state(layer_index)

        with state.lock:
            if subset is not None:
                state.modules.update(subset)
            if subset_total is not None:
                state.subset_total = subset_total
            if subset_index is not None:
                state.processed_subsets.add(subset_index)

            state.pending_modules.discard(module.name)

            should_quantize = (
                not state.quantized
                and bool(state.modules)
                and not state.pending_modules.intersection(state.modules.keys())
                and (state.subset_total is None or len(state.processed_subsets) >= state.subset_total)
            )
            if should_quantize:
                self._quantize_layer(layer_index, state)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        self.pack_module(module, model=model)

    def pack_module(self, module: NamedModule, model: BaseQModel):
        module.stream_sync()
        with self.lock:
            module.state.pop("w_wq_diff", None)
            pack_weight = module.state.pop("pack_weight").clone()
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            pairs = module.state.pop("pairs").clone()
            theta = module.state.pop("theta").clone()
            channel_scales = module.state.pop("channel_scales").clone()

        module.weight.data = move_to(pack_weight, device=CPU)
        quant_linear_cls = self._resolve_qlinear_kernel(module.full_name)
        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))

        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(module.full_name):
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
                    format=self.format,
                    register_buffers=False,
                    init_kwargs=self.qcfg.quant_linear_init_kwargs(),
                )

        qmodules = {
            name: submodule
            for name, submodule in find_modules(model.model, [quant_linear_cls]).items()
            if name == module.full_name
        }
        with log_time_block(
            "pack",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(module.full_name):
                pack_module(
                    name=module.full_name,
                    qModules=qmodules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=None,
                    layers=layers,
                    quant_linear_cls=quant_linear_cls,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                )

        qmodule = qmodules[module.full_name]
        if not isinstance(qmodule, ParoQuantQuantLinear):
            raise TypeError(
                f"Expected `{module.full_name}` to be packed as ParoQuantQuantLinear, got `{type(qmodule).__name__}`."
            )

        qmodule.pairs.copy_(pairs.to(device=qmodule.pairs.device, dtype=qmodule.pairs.dtype))
        qmodule.theta.copy_(theta.to(device=qmodule.theta.device, dtype=qmodule.theta.dtype))
        qmodule.channel_scales.copy_(
            channel_scales.to(device=qmodule.channel_scales.device, dtype=qmodule.channel_scales.dtype)
        )
        qmodule.post_init()

    def finalize(self, model: BaseQModel, **kwargs):
        model.quantized = True
        model.quantize_config.method = METHOD.PAROQUANT
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("ParoQuantProcessor's calibration_dataset must be provided.")
        return True

    @classmethod
    def name(cls) -> str:
        return "paroquant"

    def has_captured_input_ids(self, name: str) -> bool:
        entry = self.tasks.get(name) or {}
        tensors: List[torch.Tensor] = entry.get("inputs", [])
        return tensors is not None and len(tensors) > 0 and all(t.numel() > 0 for t in tensors)
