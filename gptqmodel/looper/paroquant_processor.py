# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant processor implementation adapted from the ParoQuant paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant looper integration.

This processor keeps ParoQuant separate from the AWQ lifecycle:
1. capture calibration activations for each module
2. run ParoQuant's transformed-domain optimization per layer
3. export packed runtime tensors plus learned rotation state
4. replace the float modules with ParoQuant runtime kernels
"""

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
from ..quantization.paroquant.optimization import optimize_paroquant_linear, optimize_paroquant_llama_mlp_block
from ..utils.env import env_flag
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, move_to, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import CPU

log = setup_logger()


@dataclass
class _ParoQuantLayerState:
    """Per-layer bookkeeping for activation capture and deferred quantization."""

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
        """Configure a looper that captures activations and quantizes after replay."""
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
        """Reject runtime dataset swaps because capture state is tied to the processor."""
        raise NotImplementedError("ParoQuantProcessor's calibration_dataset cannot be modified")

    def _select_qlinear_kernel_for_format(self, format_value: FORMAT):
        """Resolve the only supported runtime kernel class for ParoQuant."""
        fmt = FORMAT(format_value) if not isinstance(format_value, FORMAT) else format_value
        if fmt != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support this FORMAT: {format_value}")
        return ParoQuantQuantLinear

    def _resolve_qlinear_kernel(self, module_name: Optional[str] = None):
        """Resolve per-module dynamic overrides while enforcing ParoQuant format."""
        format_override = self.qcfg.dynamic_get(module_name, "format", None) if module_name else None
        target_format = resolve_quant_format(format_override or self.qcfg.format, self.qcfg.method)
        if target_format != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support dynamic format override `{target_format}`.")
        return ParoQuantQuantLinear

    def _get_layer_state(self, layer_index: int) -> _ParoQuantLayerState:
        """Fetch or create the shared state bucket for one transformer layer."""
        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _ParoQuantLayerState()
                self._layer_states[layer_index] = state
        return state

    def _record_input_feature(self, module_name: str, feature: torch.Tensor) -> None:
        """Store one batch of calibration activations for a named module."""
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

    def _ensure_task_bucket(self, module_name: str, layer_index: int) -> None:
        """Reset repeated relative module names when quantization advances to a new layer."""
        with self.lock:
            entry = self.tasks.get(module_name)
            if entry is None or entry.get("layer_index") != layer_index:
                self.tasks[module_name] = {
                    "inputs": [],
                    "layer_index": layer_index,
                }
                return
            entry.setdefault("inputs", [])

    def _layer_input_features(self, state: _ParoQuantLayerState) -> Dict[str, torch.Tensor]:
        """Materialize concatenated calibration features for all modules in a layer."""
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
        """Read effective bit-width, group size, and symmetry for one module."""
        bits = int(self.qcfg.dynamic_get(module_name, "bits", self.qcfg.runtime_bits))
        group_size = int(self.qcfg.dynamic_get(module_name, "group_size", self.qcfg.group_size))
        sym = bool(self.qcfg.dynamic_get(module_name, "sym", self.qcfg.sym))
        return bits, group_size, sym

    @staticmethod
    def _module_weight_matrix(module: NamedModule) -> torch.Tensor:
        """Return the 2D weight matrix expected by the ParoQuant optimizer."""
        weight = module.weight.data
        if weight.dim() != 2:
            raise ValueError(
                f"ParoQuant currently expects rank-2 module weights, got {tuple(weight.shape)} for `{module.full_name}`."
            )
        return weight

    def _apply_optimization_result(self, module: NamedModule, result, original_weight: torch.Tensor) -> None:
        """Store one optimization result into the wrapped module and its scratch state."""
        weight = self._module_weight_matrix(module)
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

    def _log_quant_result(self, module: NamedModule, feat: torch.Tensor, val_loss: float, duration: float) -> None:
        """Append one quantization log row using the same format as other processors."""
        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: f"{val_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{0 if feat.numel() == 0 else feat.reshape(-1, feat.shape[-1]).shape[0]}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
        }

        with self.lock:
            self.durations.append(duration)
            self.avg_losses.append(val_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
            self.log.append(stat)

        self.log_new_row(stat)

    @staticmethod
    def _llama_mlp_block_modules(state: _ParoQuantLayerState) -> Optional[Dict[str, NamedModule]]:
        """Return the three coupled Llama MLP projections when the layer matches that pattern."""
        required = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        if any(name not in state.modules for name in required):
            return None

        modules = {name: state.modules[name] for name in required}
        layer_module = next(
            (named_module.state.get("layer_module") for named_module in modules.values() if named_module.state.get("layer_module")),
            None,
        )
        if layer_module is None or type(layer_module).__name__ != "LlamaDecoderLayer":
            return None

        mlp_module = getattr(layer_module, "mlp", None)
        if mlp_module is None or type(mlp_module).__name__ != "LlamaMLP":
            return None
        if not callable(getattr(mlp_module, "act_fn", None)):
            return None

        expected_modules = {
            "mlp.gate_proj": getattr(mlp_module, "gate_proj", None),
            "mlp.up_proj": getattr(mlp_module, "up_proj", None),
            "mlp.down_proj": getattr(mlp_module, "down_proj", None),
        }
        if any(named_module.module is not expected_modules[name] for name, named_module in modules.items()):
            return None

        return modules

    @staticmethod
    def _llama_mlp_block_input(input_feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use the shared MLP input captured at gate/up as the block replay input."""
        gate_inputs = input_feat.get("mlp.gate_proj")
        if gate_inputs is not None and gate_inputs.numel() > 0:
            return gate_inputs
        up_inputs = input_feat.get("mlp.up_proj")
        if up_inputs is not None and up_inputs.numel() > 0:
            return up_inputs
        return torch.empty(0)

    def _quantize_llama_mlp_block(
        self,
        block_modules: Dict[str, NamedModule],
        input_feat: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """Optimize a Llama MLP trio jointly when all config and structure checks match."""
        configs = {
            name: self._module_quant_params(named_module.full_name)
            for name, named_module in block_modules.items()
        }
        base_config = configs["mlp.gate_proj"]
        if any(config != base_config for config in configs.values()):
            return None

        gate_module = block_modules["mlp.gate_proj"]
        up_module = block_modules["mlp.up_proj"]
        down_module = block_modules["mlp.down_proj"]
        bits, group_size, sym = base_config

        gate_weight = self._module_weight_matrix(gate_module)
        up_weight = self._module_weight_matrix(up_module)
        down_weight = self._module_weight_matrix(down_module)
        gate_bias = gate_module.bias.data if getattr(gate_module, "bias", None) is not None else None
        up_bias = up_module.bias.data if getattr(up_module, "bias", None) is not None else None
        down_bias = down_module.bias.data if getattr(down_module, "bias", None) is not None else None
        block_inputs = self._llama_mlp_block_input(input_feat)
        if block_inputs.numel() == 0:
            block_inputs = torch.empty((0, gate_weight.shape[1]), dtype=gate_weight.dtype, device=gate_weight.device)

        layer_module = (
            gate_module.state.get("layer_module")
            or up_module.state.get("layer_module")
            or down_module.state.get("layer_module")
        )
        original_weights = {
            name: self._module_weight_matrix(named_module).detach().clone()
            for name, named_module in block_modules.items()
        }

        with torch.inference_mode(False), torch.enable_grad():
            result = optimize_paroquant_llama_mlp_block(
                gate_weight=gate_weight,
                gate_bias=gate_bias,
                up_weight=up_weight,
                up_bias=up_bias,
                down_weight=down_weight,
                down_bias=down_bias,
                inputs=block_inputs,
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
                seed=self.qcfg.opt_seed + gate_module.layer_index + hash(f"{gate_module.full_name}:llama_mlp_block"),
                activation_fn=layer_module.mlp.act_fn,
                fused_rotation=self.qcfg.opt_fused_rotation,
            )

        for name, named_module in block_modules.items():
            self._apply_optimization_result(named_module, result.module_results[name], original_weights[name])
        return result.val_loss, block_inputs

    def _quantize_one_module(
        self,
        module: NamedModule,
        inputs: torch.Tensor,
    ) -> tuple[float, float]:
        """Optimize one module and stash its packed runtime tensors in `module.state`."""
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
                fused_rotation=self.qcfg.opt_fused_rotation,
            )

        self._apply_optimization_result(module, result, original_weight)
        return result.train_loss, result.val_loss

    def _quantize_layer(self, layer_index: int, state: _ParoQuantLayerState) -> None:
        """Quantize every captured module in a layer once all subsets are ready."""
        if state.quantized:
            return

        input_feat = self._layer_input_features(state)
        for module_name, tensor in input_feat.items():
            if tensor.numel() == 0 and not self.fallback:
                raise RuntimeError(
                    f"ParoQuantProcessor error: missing activation features for `{module_name}` with fallback disabled."
                )

        handled_modules: Set[str] = set()
        enable_llama_mlp_block = self._enable_llama_mlp_block()
        block_modules = self._llama_mlp_block_modules(state) if enable_llama_mlp_block else None
        if block_modules is not None:
            block_start = time.perf_counter()
            block_result = self._quantize_llama_mlp_block(block_modules, input_feat)
            if block_result is not None:
                block_val_loss, block_inputs = block_result
                block_duration = time.perf_counter() - block_start
                split_duration = block_duration / float(len(block_modules))
                for named_module in block_modules.values():
                    self._log_quant_result(
                        named_module,
                        block_inputs,
                        block_val_loss,
                        split_duration,
                    )
                handled_modules.update(block_modules.keys())

        for module_name, named_module in list(state.modules.items()):
            if module_name in handled_modules:
                continue
            feat = input_feat.get(module_name)
            if feat is None:
                feat = torch.empty(0)

            start = time.perf_counter()
            _train_loss, val_loss = self._quantize_one_module(named_module, feat)
            duration = time.perf_counter() - start
            self._log_quant_result(named_module, feat, val_loss, duration)

        state.quantized = True
        with self.lock:
            for module_name in list(state.modules):
                entry = self.tasks.get(module_name)
                if entry is not None and entry.get("layer_index") == layer_index:
                    entry["inputs"] = []
        state.modules.clear()
        state.pending_modules.clear()
        state.processed_subsets.clear()
        state.subset_total = None

    def _enable_llama_mlp_block(self) -> bool:
        """Resolve joint Llama MLP block optimization toggle from config with env override."""
        config_default = bool(getattr(self.qcfg, "opt_enable_llama_mlp_block", False))
        return env_flag("GPTQMODEL_PAROQUANT_ENABLE_LLAMA_MLP_BLOCK", default=config_default)

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Register a module for later activation capture and deferred quantization."""
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        self.fallback = normalize_fallback(fallback, self.qcfg.fallback)
        layer_state = self._get_layer_state(module.layer_index)
        with layer_state.lock:
            layer_state.modules[module.name] = module
            layer_state.pending_modules.add(module.name)

        self._ensure_task_bucket(module.name, module.layer_index)

    def is_skipped(self, module: NamedModule) -> bool:
        """Report whether a module has been excluded from ParoQuant processing."""
        return self.tasks.get(module.name, False) is False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Capture input activations during the calibration forward pass."""
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
        """Mark subset progress and quantize once the whole layer is capture-complete."""
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
        """Pack one optimized float module into its ParoQuant runtime form."""
        self.pack_module(module, model=model)

    def pack_module(self, module: NamedModule, model: BaseQModel):
        """Replace a float module with a packed ParoQuant quantized module."""
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
        """Mark the model as ParoQuant-quantized before shared finalization work."""
        model.quantized = True
        model.quantize_config.method = METHOD.PAROQUANT
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Require calibration data because ParoQuant always needs activation replay."""
        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("ParoQuantProcessor's calibration_dataset must be provided.")
        return True

    @classmethod
    def name(cls) -> str:
        """Return the processor registry name."""
        return "paroquant"

    def has_captured_input_ids(self, name: str) -> bool:
        """Report whether non-empty activation batches were captured for a module."""
        entry = self.tasks.get(name) or {}
        tensors: List[torch.Tensor] = entry.get("inputs", [])
        return tensors is not None and len(tensors) > 0 and all(t.numel() > 0 for t in tensors)
