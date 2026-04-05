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

import copy
import hashlib
import inspect
import math
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
    QUANT_LOG_DAMP,
)
from ..nn_modules.hooked_linear import HookedLinear
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..quantization.config import FORMAT, METHOD, QuantizeConfig, resolve_quant_format
from ..quantization.paroquant.optimization import (
    _ParoQuantOptimLinear,
    _activate_stage_params,
    _normalize_group_size,
    _normalize_opt_impl,
    _normalize_opt_optimizer,
    _normalize_quantizer_impl,
    _quantizer_sym_for_impl,
    _resolve_best_state_snapshot_dtype,
    _result_from_model,
    build_paroquant_optimizer,
    build_random_rotation_buffers,
    build_random_rotation_buffers_reference,
    optimize_paroquant_linear,
)
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import (
    create_quant_module,
    find_modules,
    get_module_by_name_prefix,
    move_to,
    nested_move_to,
    pack_module,
    recurse_getattr,
    recurse_setattr,
)
from ..utils.module_locks import parent_module_lock
from ..utils.paroquant import prewarm_paroquant_rotation_extension
from ..utils.torch import CPU, torch_empty_cache

log = setup_logger()


@dataclass
class _ParoQuantLayerState:
    """Per-layer bookkeeping for activation capture and deferred quantization."""

    modules: Dict[str, NamedModule] = field(default_factory=dict)
    layer_module: Optional[torch.nn.Module] = None
    pristine_layer_module: Optional[torch.nn.Module] = None
    prepared_group_source_module: Optional[torch.nn.Module] = None
    prepared_group_source_module_by_device: Optional[Dict[str, torch.nn.Module]] = None
    layer_inputs: Optional[List[List[torch.Tensor]]] = None
    layer_input_kwargs: Optional[List[Dict[str, torch.Tensor]]] = None
    layer_outputs: Optional[List[List[torch.Tensor]]] = None
    grouped_dataset: Optional[Any] = None
    grouped_dataset_by_device: Optional[Dict[str, Any]] = None
    replay_batches: Optional[Any] = None
    subset_total: Optional[int] = None
    processed_subsets: Set[int] = field(default_factory=set)
    pending_modules: Set[str] = field(default_factory=set)
    quantized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _ParoQuantReplayBatch:
    """One CPU-owned layer replay batch used by streamed grouped optimization."""

    inputs: List[torch.Tensor]
    input_kwargs: Dict[str, Any]
    target: torch.Tensor
    position_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    row_count: int


def _value_has_inference_tensor(value: Any) -> bool:
    """Detect nested inference-mode tensors so caches can rebuild autograd-safe values."""
    if isinstance(value, torch.Tensor):
        return value.is_inference()
    if isinstance(value, dict):
        return any(_value_has_inference_tensor(inner) for inner in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_has_inference_tensor(inner) for inner in value)
    return False


class _LayerShardLoader:
    """Stream replay batches from CPU to one device shard at a time."""

    def __init__(
        self,
        batches: list[_ParoQuantReplayBatch],
        *,
        target_device: torch.device,
        shard_batches: int,
        metadata_cache: Optional[dict[tuple[int, str], torch.Tensor]] = None,
    ) -> None:
        self.batches = batches
        self.target_device = torch.device(target_device)
        self.shard_batches = max(1, int(shard_batches))
        self.metadata_cache = metadata_cache

    @staticmethod
    def _tensor_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        non_blocking = value.device.type == CPU.type and value.is_pinned() and device.type == "cuda"
        if value.is_inference():
            # Replay tensors may be captured under worker inference-mode; moving
            # them with copy=True recreates normal tensors that autograd can use.
            with torch.inference_mode(False):
                return value.to(device=device, non_blocking=non_blocking, copy=True)
        if value.device == device:
            return value
        return value.to(device=device, non_blocking=non_blocking)

    def _metadata_tensor_to_device(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.device == self.target_device and not value.is_inference():
            return value
        cache = self.metadata_cache
        if cache is None:
            return self._tensor_to_device(value, self.target_device)
        cache_key = (id(value), str(self.target_device))
        cached = cache.get(cache_key)
        if cached is None:
            cached = self._tensor_to_device(value, self.target_device)
            cache[cache_key] = cached
        return cached

    @staticmethod
    def _move_value_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return _LayerShardLoader._tensor_to_device(value, device)
        if isinstance(value, dict):
            return {key: _LayerShardLoader._move_value_to_device(inner, device) for key, inner in value.items()}
        if isinstance(value, list):
            return [_LayerShardLoader._move_value_to_device(inner, device) for inner in value]
        if isinstance(value, tuple):
            return tuple(_LayerShardLoader._move_value_to_device(inner, device) for inner in value)
        return value

    def _materialize_batch(self, batch: _ParoQuantReplayBatch) -> _ParoQuantReplayBatch:
        return _ParoQuantReplayBatch(
            inputs=[self._tensor_to_device(tensor, self.target_device) for tensor in batch.inputs],
            input_kwargs={
                key: self._move_value_to_device(value, self.target_device)
                for key, value in batch.input_kwargs.items()
            },
            target=self._tensor_to_device(batch.target, self.target_device),
            position_ids=self._metadata_tensor_to_device(batch.position_ids),
            attention_mask=self._metadata_tensor_to_device(batch.attention_mask),
            row_count=batch.row_count,
        )

    def iter_shards(self) -> Iterator[list[_ParoQuantReplayBatch]]:
        for start in range(0, len(self.batches), self.shard_batches):
            end = min(len(self.batches), start + self.shard_batches)
            shard = [self._materialize_batch(batch) for batch in self.batches[start:end]]
            yield shard

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
        capture_group_layer_context = str(getattr(qcfg, "opt_scope", "module")).strip().lower() != "module"
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
                enable_activation_capture=True,
                capture_layer_forward_context=capture_group_layer_context,
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
        self._rotary_lock = threading.Lock()
        self._rotary_cache: Dict[str, nn.Module] = {}
        self._rotary_source_id: Optional[int] = None
        self._clean_group_layer_inputs: Optional[List[List[torch.Tensor]]] = None
        self._runtime_prewarmed = False
        self.fallback = qcfg.fallback

    def set_calibration_dataset(self, calibration_dataset):
        """Reject runtime dataset swaps because capture state is tied to the processor."""
        raise NotImplementedError("ParoQuantProcessor's calibration_dataset cannot be modified")

    def receive_input_cache(self, input_cache):
        """Seed grouped calibration with the clean first-layer input stream."""
        super().receive_input_cache(input_cache)
        self._clean_group_layer_inputs = input_cache.layer_inputs if self._train_on_noisy_inputs_enabled() else None

    def _select_qlinear_kernel_for_format(self, format_value: FORMAT):
        """Resolve the only supported runtime kernel class for ParoQuant."""
        fmt = FORMAT(format_value) if not isinstance(format_value, FORMAT) else format_value
        if fmt != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PARO does not support this FORMAT: {format_value}")
        return ParoLinear

    def prewarm_runtime(self) -> None:
        """Build optional fused ParoQuant runtime pieces before timed layer work starts."""
        if getattr(self, "_runtime_prewarmed", False):
            return

        self._runtime_prewarmed = True
        fused_rotation = bool(getattr(self.qcfg, "opt_fused_rotation", False))
        group_size = int(getattr(self.qcfg, "group_size", 128))
        krot = int(getattr(self.qcfg, "krot", 8))
        if fused_rotation and group_size in {128} and krot in {1, 8}:
            log.info("ParoQuant: prewarming fused rotation extension...")
        if not prewarm_paroquant_rotation_extension(
            fused_rotation=fused_rotation,
            group_size=group_size,
            krot=krot,
        ):
            return
        log.info("ParoQuant: prewarmed fused rotation extension.")

    def _resolve_qlinear_kernel(self, module_name: Optional[str] = None):
        """Resolve per-module dynamic overrides while enforcing ParoQuant format."""
        format_override = self.qcfg.dynamic_get(module_name, "format", None) if module_name else None
        target_format = resolve_quant_format(format_override or self.qcfg.format, self.qcfg.method)
        if target_format != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PARO does not support dynamic format override `{target_format}`.")
        return ParoLinear

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
        n_samples = 0 if feat.numel() == 0 else feat.reshape(-1, feat.shape[-1]).shape[0]

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: f"{val_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{n_samples}",
            QUANT_LOG_DAMP: "",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
        }

        with self.lock:
            self.durations.append(duration)
            self.avg_losses.append(val_loss)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")
        self.log.append(stat)

        self.log_new_row(stat)

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
        module_seed = self._module_seed(module.layer_index, module.full_name)

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
                seed=module_seed,
                optimizer_name=getattr(self.qcfg, "opt_optimizer", "adamw"),
                optimizer_weight_decay=float(getattr(self.qcfg, "opt_weight_decay", 0.01)),
                optimizer_betas=tuple(getattr(self.qcfg, "opt_betas", (0.9, 0.95))),
                optimizer_eps=float(getattr(self.qcfg, "opt_eps", 1e-10)),
                optimizer_amsgrad=bool(getattr(self.qcfg, "opt_amsgrad", False)),
                sgd_momentum=float(getattr(self.qcfg, "opt_sgd_momentum", 0.0)),
                sgd_dampening=float(getattr(self.qcfg, "opt_sgd_dampening", 0.0)),
                sgd_nesterov=bool(getattr(self.qcfg, "opt_sgd_nesterov", False)),
                fused_rotation=self.qcfg.opt_fused_rotation,
                gradient_checkpointing=bool(getattr(self.qcfg, "opt_gradient_checkpointing", False)),
                stage_cudagraph=self._module_scope_stage_cudagraph_enabled(),
                best_state_dtype=getattr(self.qcfg, "opt_best_state_dtype", "fp32"),
                stage_impl=self.qcfg.opt_stage_impl,
                pair_impl=self.qcfg.opt_pair_impl,
                quantizer_impl=self.qcfg.opt_quantizer_impl,
                scale_clamp_min=self.qcfg.opt_channel_scale_clamp_min,
                scale_clamp_max=self.qcfg.opt_channel_scale_clamp_max,
            )

        self._apply_optimization_result(module, result, original_weight)
        return result.train_loss, result.val_loss

    @staticmethod
    def _module_archetype(full_name: str) -> str:
        """Use the terminal module name as the shared seed key across layers."""
        return full_name.rsplit(".", 1)[-1]

    def _module_seed_key(self, full_name: str) -> str:
        """Keep module-scope seeds unique per full module while preserving grouped-scope behavior.

        Module scope optimizes every linear independently across the full model.
        Reusing only the terminal archetype name correlates rotations across layers
        (`...layers.0.self_attn.q_proj`, `...layers.1.self_attn.q_proj`, etc.),
        which materially hurts end-to-end recovery on full-model module runs.
        Grouped scopes keep the existing archetype behavior to avoid disturbing the
        separately tuned layer/compute-block path.
        """
        if self._opt_scope_mode() == "module":
            return full_name
        return self._module_archetype(full_name)

    def _module_seed(self, layer_index: int, full_name: str) -> int:
        """Derive a deterministic per-module seed from base seed, layer index, and module name."""
        module_name = self._module_seed_key(full_name)
        seed_material = f"{int(self.qcfg.opt_seed)}:{int(layer_index)}:{module_name}".encode("utf-8")
        digest = hashlib.blake2b(seed_material, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _opt_scope_mode(self) -> str:
        """Normalize the configured ParoQuant optimization scope."""
        return str(getattr(self.qcfg, "opt_scope", "module")).strip().lower()

    def _gradient_checkpointing_enabled(self) -> bool:
        """Resolve grouped-stage checkpointing from config, defaulting to layer scope only."""
        configured = getattr(self.qcfg, "opt_gradient_checkpointing", None)
        if configured is None:
            return self._opt_scope_mode() == "layer"
        return bool(configured)

    def uses_grouped_optimization(self) -> bool:
        """Return whether this layer should optimize compute_block/layer scopes instead of one linear at a time."""
        return self._opt_scope_mode() != "module"

    def needs_pristine_layer_clone(self) -> bool:
        """Whether stage-layer orchestration must build a separate pristine layer copy.

        Whole-layer scope replays clean targets before mutating the live layer, so
        it can use the untouched live layer directly. ComputeBlock scope still needs
        a dedicated pristine copy because later grouped clone construction may
        happen after subset-time hooks and wrapper replacement.
        """
        return self._opt_scope_mode() != "layer"

    def capture_layer_forward_context_during_subset(self) -> bool:
        """ParoQuant captures grouped pristine layer IO outside subset forwards only."""
        return False

    def _train_on_noisy_inputs_enabled(self) -> bool:
        """Enable official clean-target / noisy-input calibration only when explicitly requested."""
        qcfg = getattr(self, "qcfg", None)
        return bool(getattr(qcfg, "opt_train_on_noisy_inputs", False)) and self.uses_grouped_optimization()

    def _module_scope_stage_cudagraph_enabled(self) -> bool:
        """Disable per-linear stage CUDA graphs in the full quantization loop.

        Module scope calls the ParoQuant optimizer once per linear across the
        whole model. The captured train-step graph keeps CUDA graph private-pool
        allocations alive across calls, which makes active VRAM ratchet upward
        layer by layer. Grouped/layer scope does not use this path.
        """
        if self._opt_scope_mode() == "module":
            return False
        return bool(getattr(self.qcfg, "opt_stage_cudagraph", False))

    def _optimizer_param_group_kwargs(self) -> Dict[str, object]:
        """Return shared optimizer hyperparameters for ParoQuant stage param groups."""
        qcfg = getattr(self, "qcfg", None)
        return {
            "weight_decay": float(getattr(qcfg, "opt_weight_decay", 0.01)),
            "betas": tuple(getattr(qcfg, "opt_betas", (0.9, 0.95))),
            "eps": float(getattr(qcfg, "opt_eps", 1e-10)),
            "amsgrad": bool(getattr(qcfg, "opt_amsgrad", False)),
            "momentum": float(getattr(qcfg, "opt_sgd_momentum", 0.0)),
            "dampening": float(getattr(qcfg, "opt_sgd_dampening", 0.0)),
            "nesterov": bool(getattr(qcfg, "opt_sgd_nesterov", False)),
        }

    def clean_group_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Return the clean calibration stream used to build grouped layer targets."""
        del layer_index
        if not self._train_on_noisy_inputs_enabled():
            return layer_inputs
        return getattr(self, "_clean_group_layer_inputs", None) or layer_inputs

    def receive_clean_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
    ) -> None:
        """Advance the clean calibration stream for later-layer train-on-noisy-inputs replay."""
        del layer_index
        if self._train_on_noisy_inputs_enabled():
            self._clean_group_layer_inputs = layer_inputs

    def _build_group_optim_linear(self, module: NamedModule) -> _ParoQuantOptimLinear:
        """Materialize one ParoQuant optimizer wrapper from the current live module state."""
        bits, group_size, sym = self._module_quant_params(module.full_name)
        weight = self._module_weight_matrix(module)
        bias = module.bias.data if getattr(module, "bias", None) is not None else None
        normalized_group_size = _normalize_group_size(group_size, weight.shape[1])
        normalized_pair_impl = _normalize_opt_impl(self.qcfg.opt_pair_impl, field="pair_impl")
        quantizer_sym = _quantizer_sym_for_impl(sym, self.qcfg.opt_quantizer_impl)
        _normalize_quantizer_impl(self.qcfg.opt_quantizer_impl)
        module_seed = self._module_seed(module.layer_index, module.full_name)

        if normalized_pair_impl == "reference":
            pairs, theta_mask = build_random_rotation_buffers_reference(
                in_features=weight.shape[1],
                group_size=normalized_group_size,
                krot=self.qcfg.krot,
                pair_ratio=self.qcfg.opt_pair_ratio,
                seed=module_seed,
                device=weight.device,
            )
        else:
            pairs, theta_mask = build_random_rotation_buffers(
                in_features=weight.shape[1],
                group_size=normalized_group_size,
                krot=self.qcfg.krot,
                pair_ratio=self.qcfg.opt_pair_ratio,
                seed=module_seed,
                device=weight.device,
            )

        return _ParoQuantOptimLinear(
            weight.detach().to(device=weight.device, dtype=torch.float32),
            None if bias is None else bias.detach().to(device=weight.device, dtype=torch.float32),
            bits=bits,
            group_size=normalized_group_size,
            quantizer_sym=quantizer_sym,
            pairs=pairs,
            theta_mask=theta_mask,
            scale_clamp_min=self.qcfg.opt_channel_scale_clamp_min,
            scale_clamp_max=self.qcfg.opt_channel_scale_clamp_max,
            # Layer-scope live optimization must stay on the portable PyTorch
            # rotation path. The fused autograd kernel can fail to load on some
            # fleet GPUs and the error may surface asynchronously well after the
            # original rotation call.
            fused_rotation=False if self._opt_scope_mode() == "layer" else self.qcfg.opt_fused_rotation,
        ).to(device=weight.device, dtype=torch.float32)

    @staticmethod
    def _restore_linear_from_hooked(module: HookedLinear) -> torch.nn.Linear:
        """Drop HookedLinear's inference-only forward while preserving the shared parameter storage."""
        restored = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(restored)
        restored.in_features = module.in_features
        restored.out_features = module.out_features
        restored.weight = module.weight
        restored.bias = module.bias
        return restored

    def _strip_hooked_linear_wrappers(self, module: torch.nn.Module) -> int:
        """Layer-scope training must not run through HookedLinear, which always forwards in inference mode."""
        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, HookedLinear):
                setattr(module, child_name, self._restore_linear_from_hooked(child))
                replaced += 1
                continue
            replaced += self._strip_hooked_linear_wrappers(child)
        return replaced

    @staticmethod
    def _sync_named_modules_to_live_layer(
        layer: torch.nn.Module,
        named_modules: list[NamedModule],
    ) -> None:
        """Retarget NamedModule handles after live-layer wrapper surgery.

        Layer scope temporarily unwraps HookedLinear modules into plain
        ``nn.Linear`` so autograd can flow through the real decoder layer. The
        corresponding ``NamedModule`` wrappers must follow those in-place
        replacements; otherwise later result application and packing will update
        detached wrapper objects instead of the real live layer modules.
        """
        for named_module in named_modules:
            live_module = recurse_getattr(layer, named_module.name)
            named_module.module = live_module
            try:
                named_module.module_dtype = next(live_module.parameters()).dtype
            except (StopIteration, AttributeError):
                pass

    @staticmethod
    def _force_layer_eager_attention(layer: torch.nn.Module) -> list[tuple[object, str, object]]:
        """Temporarily force live-layer grouped optimization onto eager attention kernels."""
        overrides: list[tuple[object, str, object]] = []
        seen_configs: set[int] = set()
        candidate_configs = [
            getattr(layer, "config", None),
            getattr(getattr(layer, "self_attn", None), "config", None),
        ]
        for config in candidate_configs:
            if config is None:
                continue
            config_id = id(config)
            if config_id in seen_configs:
                continue
            seen_configs.add(config_id)
            for attr in ("_attn_implementation", "attn_implementation"):
                if not hasattr(config, attr):
                    continue
                original_value = getattr(config, attr)
                if original_value == "eager":
                    continue
                setattr(config, attr, "eager")
                overrides.append((config, attr, original_value))
        return overrides

    @staticmethod
    def _restore_layer_attention_impl(overrides: list[tuple[object, str, object]]) -> None:
        """Restore any attention implementation overrides after live-layer grouped optimization."""
        for config, attr, original_value in reversed(overrides):
            setattr(config, attr, original_value)

    @staticmethod
    def _materialize_live_layer_autograd_tensors(
        layer: torch.nn.Module,
    ) -> tuple[int, int]:
        """Replace inference-born params/buffers with fresh normal tensors for live autograd.

        The layer object itself stays in place, but worker-side model loading can
        create parameters and buffers under ``torch.inference_mode()``. Those
        tensors keep inference-only bookkeeping even after ``module.to(...)``,
        which breaks backward in the in-place layer optimizer. Rebuilding them
        in place gives the live layer normal autograd-safe storage without
        falling back to a full cloned layer path.
        """
        replaced_params = 0
        replaced_buffers = 0
        with torch.inference_mode(False):
            for module in layer.modules():
                for name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    rebuilt = nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad)
                    if rebuilt is not param:
                        module._parameters[name] = rebuilt
                        replaced_params += 1
                for name, buffer in list(module._buffers.items()):
                    if buffer is None:
                        continue
                    rebuilt = buffer.detach().clone()
                    if rebuilt is not buffer:
                        module._buffers[name] = rebuilt
                        replaced_buffers += 1
        return replaced_params, replaced_buffers

    def _build_group_optim_layer(
        self,
        state: _ParoQuantLayerState,
        group_modules: list[NamedModule],
    ) -> tuple[torch.nn.Module, dict[str, _ParoQuantOptimLinear]]:
        """Clone the layer and swap one selected group to ParoQuant optimizer wrappers."""
        if not group_modules:
            raise ValueError("ParoQuantProcessor grouped optimization requires at least one module.")

        prepared_source = getattr(state, "prepared_group_source_module", None)
        if prepared_source is None:
            source_layer = getattr(state, "pristine_layer_module", None) or state.layer_module
            if source_layer is None:
                raise RuntimeError("ParoQuantProcessor grouped optimization requires the source layer module.")

            prepared_source = copy.deepcopy(source_layer).to(device=CPU, dtype=torch.float32)
            # ComputeBlock/layer clone optimization cannot train through HookedLinear
            # because its forward is permanently wrapped in inference mode.
            self._strip_hooked_linear_wrappers(prepared_source)
            # Grouped optimization needs stable, differentiable attention semantics.
            # Keep the cloned calibration-time layer on eager attention even if the
            # live model prefers SDPA/flash kernels for inference throughput.
            layer_attn = getattr(prepared_source, "self_attn", None)
            layer_attn_config = getattr(layer_attn, "config", None)
            if layer_attn_config is not None and hasattr(layer_attn_config, "_attn_implementation"):
                layer_attn_config._attn_implementation = "eager"

            state.prepared_group_source_module = prepared_source

        target_device = group_modules[0].weight.device
        if self._opt_scope_mode() == "compute_block":
            device_cache = getattr(state, "prepared_group_source_module_by_device", None)
            if device_cache is None:
                device_cache = {}
                state.prepared_group_source_module_by_device = device_cache

            device_key = str(target_device)
            prepared_source_for_device = device_cache.get(device_key)
            if prepared_source_for_device is None:
                if device_key == str(CPU):
                    prepared_source_for_device = prepared_source
                else:
                    prepared_source_for_device = copy.deepcopy(prepared_source).to(device=target_device, dtype=torch.float32)
                device_cache[device_key] = prepared_source_for_device

            layer_clone = copy.deepcopy(prepared_source_for_device)
            if next(layer_clone.parameters()).device != target_device:
                layer_clone = layer_clone.to(device=target_device, dtype=torch.float32)
        else:
            layer_clone = copy.deepcopy(prepared_source)
            layer_clone = layer_clone.to(device=target_device, dtype=torch.float32)

        optim_modules: dict[str, _ParoQuantOptimLinear] = {}
        for named_module in group_modules:
            optim_module = self._build_group_optim_linear(named_module)
            recurse_setattr(layer_clone, named_module.name, optim_module)
            optim_modules[named_module.name] = optim_module

        return layer_clone, optim_modules

    def _get_root_rotary(self) -> Optional[nn.Module]:
        """Return the model rotary module used to refresh grouped layer replay kwargs."""
        model = getattr(self, "model", None)
        if self.gptq_model is not None and model is not None and getattr(self.gptq_model, "rotary_embedding", None):
            rotary, _ = get_module_by_name_prefix(model, [self.gptq_model.rotary_embedding])
            return rotary
        if model is None:
            return None
        return getattr(getattr(model, "model", model), "rotary_emb", None)

    @staticmethod
    def _get_rotary_device(rotary: Optional[nn.Module], fallback: Optional[torch.device] = None) -> Optional[torch.device]:
        """Resolve the active device for a rotary module, falling back safely."""
        if rotary is None:
            return fallback

        rotary_device = getattr(getattr(rotary, "inv_freq", None), "device", None)
        if rotary_device is not None:
            return rotary_device

        try:
            return next(rotary.parameters()).device
        except (StopIteration, AttributeError, RuntimeError):
            return fallback

    def _get_rotary_for_device(self, target_device: Optional[torch.device]) -> Optional[nn.Module]:
        """Return a rotary module materialized on the requested device when needed."""
        rotary = self._get_root_rotary()
        if rotary is None or target_device is None:
            return rotary

        target_device = torch.device(target_device)
        if self._get_rotary_device(rotary) == target_device:
            return rotary

        cache_key = str(target_device)
        with self._rotary_lock:
            rotary = self._get_root_rotary()
            if rotary is None:
                return None

            source_id = id(rotary)
            if self._rotary_source_id != source_id:
                self._rotary_cache.clear()
                self._rotary_source_id = source_id

            if self._get_rotary_device(rotary) == target_device:
                return rotary

            cached = self._rotary_cache.get(cache_key)
            if cached is None:
                try:
                    cached = copy.deepcopy(rotary)
                except Exception:
                    cached = rotary

                move_to(cached, device=target_device)
                if cached is not rotary:
                    self._rotary_cache[cache_key] = cached

            return cached

    @staticmethod
    def _can_cache_rotary_position_embeddings(rotary: Optional[nn.Module]) -> bool:
        """Allow memoized rotary embeddings only for known HF rotary classes that depend on ids, device, and dtype."""
        if rotary is None:
            return False
        rotary_type = type(rotary)
        return rotary_type.__module__.startswith("transformers.models.") and rotary_type.__name__.endswith("RotaryEmbedding")

    def _cached_group_position_ids(
        self,
        *,
        device: torch.device,
        batch_dim: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Reuse deterministic generated position ids for repeated grouped layer replay batches."""
        position_ids_cache = getattr(self, "_group_position_ids_cache", None)
        if position_ids_cache is None:
            position_ids_cache = {}
            self._group_position_ids_cache = position_ids_cache

        cache_key = (str(device), int(batch_dim), int(seq_len))
        cached_position_ids = position_ids_cache.get(cache_key)
        if cached_position_ids is None or cached_position_ids.is_inference():
            with torch.inference_mode(False):
                cached_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_dim, -1)
            position_ids_cache[cache_key] = cached_position_ids
        return cached_position_ids

    def _cached_group_rotary_position_embeddings(
        self,
        *,
        rotary: nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_device: Optional[torch.device],
    ) -> Any:
        """Reuse rotary outputs when ids, dtype, and device are unchanged across replay batches/epochs."""
        target_rotary_device = rotary_device or x.device
        rotary_cache = getattr(self, "_group_rotary_position_embeddings_cache", None)
        if rotary_cache is None:
            rotary_cache = {}
            self._group_rotary_position_embeddings_cache = rotary_cache

        cache_key = (
            id(rotary),
            id(position_ids),
            str(target_rotary_device),
            str(x.dtype),
        )
        cached_position_embeddings = rotary_cache.get(cache_key)
        if cached_position_embeddings is None or _value_has_inference_tensor(cached_position_embeddings):
            x_for_rotary = _LayerShardLoader._tensor_to_device(x, target_rotary_device)
            pos_for_rotary = _LayerShardLoader._tensor_to_device(position_ids, target_rotary_device)
            with torch.inference_mode(False):
                cached_position_embeddings = rotary(x_for_rotary, pos_for_rotary)
            rotary_cache[cache_key] = cached_position_embeddings
        return cached_position_embeddings

    def _prepare_group_forward_kwargs(
        self,
        layer: torch.nn.Module,
        *,
        x: torch.Tensor,
        input_kwargs: Dict[str, Any],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cache: bool = True,
    ) -> Dict[str, Any]:
        """Refresh grouped replay kwargs so real decoder layers can be replayed safely."""
        target_device = x.device

        prepared_cache_key = None
        prepared_cache = None
        if cache and not x.requires_grad:
            prepared_cache = getattr(self, "_group_forward_prepared_cache", None)
            if prepared_cache is None:
                prepared_cache = {}
                self._group_forward_prepared_cache = prepared_cache
            prepared_cache_key = (
                id(layer),
                id(x),
                id(input_kwargs),
                id(attention_mask) if attention_mask is not None else 0,
                id(position_ids) if position_ids is not None else 0,
                str(target_device),
            )
            cached_prepared = prepared_cache.get(prepared_cache_key)
            if cached_prepared is not None:
                return dict(cached_prepared)

        skip_kwargs = {"past_key_values", "past_key_value"}

        if cache:
            kwargs_cache = getattr(self, "_group_forward_kwargs_cache", None)
            if kwargs_cache is None:
                kwargs_cache = {}
                self._group_forward_kwargs_cache = kwargs_cache
            kwargs_cache_key = (id(input_kwargs), str(target_device))
            cached_module_kwargs = kwargs_cache.get(kwargs_cache_key)
            if cached_module_kwargs is None:
                cached_module_kwargs = {
                    key: nested_move_to(value, device=target_device)
                    for key, value in input_kwargs.items()
                    if key not in skip_kwargs
                }
                kwargs_cache[kwargs_cache_key] = cached_module_kwargs
            module_kwargs = dict(cached_module_kwargs)
        else:
            module_kwargs = {
                key: _LayerShardLoader._move_value_to_device(value, target_device)
                for key, value in input_kwargs.items()
                if key not in skip_kwargs
            }

        signature_cache = getattr(self, "_group_forward_signature_cache", None)
        if signature_cache is None:
            signature_cache = {}
            self._group_forward_signature_cache = signature_cache
        layer_type = type(layer)
        cached_signature = signature_cache.get(layer_type)
        if cached_signature is None:
            supports_position_ids = False
            supports_position_embeddings = False
            supports_attention_mask = False
            try:
                signature = inspect.signature(layer.forward).parameters
                supports_position_ids = "position_ids" in signature
                supports_position_embeddings = "position_embeddings" in signature
                supports_attention_mask = "attention_mask" in signature
            except (ValueError, TypeError):
                supports_attention_mask = True
            cached_signature = (supports_position_ids, supports_position_embeddings, supports_attention_mask)
            signature_cache[layer_type] = cached_signature
        supports_position_ids, supports_position_embeddings, supports_attention_mask = cached_signature

        if supports_attention_mask:
            module_kwargs["attention_mask"] = None if attention_mask is None else (
                _LayerShardLoader._tensor_to_device(attention_mask, target_device)
            )

        if x.dim() == 2 and (supports_position_ids or supports_position_embeddings):
            x = x.unsqueeze(0)

        seq_len: Optional[int]
        batch_dim: int
        if x.dim() >= 2:
            batch_dim = x.shape[0]
            seq_len = x.shape[1] if x.dim() >= 3 else x.shape[0]
        else:
            batch_dim = 1
            seq_len = x.shape[0]

        rotary = self._get_root_rotary()
        if seq_len is not None and rotary is not None and supports_position_embeddings:
            rotary = self._get_rotary_for_device(target_device or x.device)
            rotary_device = self._get_rotary_device(rotary, target_device or x.device)
            pos_for_rotary = position_ids if supports_position_ids else None
            if pos_for_rotary is None or pos_for_rotary.shape[-1] != seq_len:
                pos_values = self._cached_group_position_ids(
                    device=rotary_device or x.device,
                    batch_dim=batch_dim,
                    seq_len=seq_len,
                )
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_values
                pos_for_rotary = pos_values
            else:
                if rotary_device is not None and pos_for_rotary.device != rotary_device:
                    pos_for_rotary = _LayerShardLoader._tensor_to_device(pos_for_rotary, rotary_device)
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_for_rotary

            if self._can_cache_rotary_position_embeddings(rotary):
                module_kwargs["position_embeddings"] = self._cached_group_rotary_position_embeddings(
                    rotary=rotary,
                    x=x,
                    position_ids=pos_for_rotary,
                    rotary_device=rotary_device,
                )
            else:
                x_for_rotary = x if rotary_device is None or x.device == rotary_device else x.to(rotary_device)
                module_kwargs["position_embeddings"] = rotary(x_for_rotary, pos_for_rotary)
        elif supports_position_ids:
            if position_ids is None or position_ids.shape[-1] != seq_len:
                pos_values = self._cached_group_position_ids(
                    device=target_device or x.device,
                    batch_dim=batch_dim,
                    seq_len=seq_len,
                )
                module_kwargs["position_ids"] = pos_values
            else:
                module_kwargs["position_ids"] = _LayerShardLoader._tensor_to_device(position_ids, target_device)

        module_kwargs["use_cache"] = False
        module_kwargs = self._normalize_group_runtime_metadata(module_kwargs)
        if prepared_cache_key is not None and prepared_cache is not None:
            prepared_cache[prepared_cache_key] = dict(module_kwargs)
        return module_kwargs

    @staticmethod
    def _clone_group_runtime_metadata(value: Any) -> Any:
        """Rebuild replay/runtime tensor trees with fresh normal tensors.

        This is stricter than the inference-detection path. It protects the
        live layer optimizer from tensors that still alias inference-backed
        storage even when `is_inference()` does not fire on the exact view that
        reaches the layer entry point.
        """
        if isinstance(value, torch.Tensor):
            with torch.inference_mode(False):
                return value.clone()
        if isinstance(value, dict):
            return {key: ParoQuantProcessor._clone_group_runtime_metadata(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [ParoQuantProcessor._clone_group_runtime_metadata(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(ParoQuantProcessor._clone_group_runtime_metadata(inner) for inner in value)
        return value

    @staticmethod
    def _layer_batch_row_count(input_batch: List[torch.Tensor]) -> int:
        """Count flattened token rows for one cached layer-input batch."""
        if not input_batch:
            return 0
        primary = input_batch[0]
        if not isinstance(primary, torch.Tensor) or primary.numel() == 0:
            return 0
        if primary.dim() == 0:
            return 1
        return int(primary.numel() // max(1, primary.shape[-1]))

    def _prefix_batch_count_for_rows(
        self,
        input_batches: List[List[torch.Tensor]],
        row_budget: int,
    ) -> int:
        """Choose the smallest non-empty prefix whose cached rows meet the requested budget."""
        if not input_batches:
            return 0
        if row_budget <= 0:
            return 1
        total_rows = 0
        for index, batch in enumerate(input_batches, start=1):
            total_rows += self._layer_batch_row_count(batch)
            if total_rows >= row_budget:
                return index
        return len(input_batches)

    def _suffix_batch_count_for_rows(
        self,
        input_batches: List[List[torch.Tensor]],
        row_budget: int,
    ) -> int:
        """Choose the smallest non-empty suffix whose cached rows meet the requested budget."""
        if not input_batches:
            return 0
        if row_budget <= 0:
            return 1
        total_rows = 0
        count = 0
        for batch in reversed(input_batches):
            total_rows += self._layer_batch_row_count(batch)
            count += 1
            if total_rows >= row_budget:
                return count
        return len(input_batches)

    @staticmethod
    def _target_primary(target_batch: Any) -> torch.Tensor:
        """Normalize cached layer outputs to the single tensor used for the loss target."""
        if isinstance(target_batch, (list, tuple)):
            if not target_batch:
                raise ValueError("ParoQuant grouped optimization received an empty target batch.")
            target_batch = target_batch[0]
        if not isinstance(target_batch, torch.Tensor):
            raise TypeError(f"ParoQuant grouped optimization expected tensor targets, got `{type(target_batch).__name__}`.")
        return target_batch

    def _prepare_group_target(
        self,
        target_batch: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache: bool = True,
    ) -> torch.Tensor:
        """Cache grouped loss targets after primary-tensor normalization and dtype/device conversion."""
        target = self._target_primary(target_batch)
        if target.is_inference():
            with torch.inference_mode(False):
                target = target.to(device=device, dtype=dtype, copy=True)
            if not cache:
                return target
        if not cache:
            return target.to(device=device, dtype=dtype)
        target_cache = getattr(self, "_group_target_cache", None)
        if target_cache is None:
            target_cache = {}
            self._group_target_cache = target_cache
        cache_key = (id(target), str(device), str(dtype))
        cached_target = target_cache.get(cache_key)
        if cached_target is None:
            cached_target = target.to(device=device, dtype=dtype)
            target_cache[cache_key] = cached_target
        return cached_target

    @staticmethod
    def _move_group_value_to_cpu(value: Any) -> Any:
        """Recursively normalize replay metadata onto CPU-owned tensors."""
        if isinstance(value, torch.Tensor):
            if value.is_inference():
                with torch.inference_mode(False):
                    tensor = value.detach().to(device=CPU, copy=True)
            else:
                tensor = value.detach().cpu() if value.device.type != CPU.type else value.detach()
            if torch.cuda.is_available() and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor
        if isinstance(value, dict):
            return {key: ParoQuantProcessor._move_group_value_to_cpu(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [ParoQuantProcessor._move_group_value_to_cpu(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(ParoQuantProcessor._move_group_value_to_cpu(inner) for inner in value)
        return value

    @staticmethod
    def _normalize_group_runtime_metadata(value: Any) -> Any:
        """Rebuild inference-mode replay metadata once kwargs are fully assembled on the live device."""
        if isinstance(value, torch.Tensor):
            return _LayerShardLoader._tensor_to_device(value, value.device)
        if isinstance(value, dict):
            return {
                key: ParoQuantProcessor._normalize_group_runtime_metadata(inner)
                for key, inner in value.items()
            }
        if isinstance(value, list):
            return [ParoQuantProcessor._normalize_group_runtime_metadata(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(ParoQuantProcessor._normalize_group_runtime_metadata(inner) for inner in value)
        return value

    def _replay_batches_from_state(
        self,
        state: _ParoQuantLayerState,
    ) -> tuple[list[_ParoQuantReplayBatch], list[_ParoQuantReplayBatch]]:
        """Build CPU-owned train/validation replay batches for in-place layer optimization."""
        cached_batches = getattr(state, "replay_batches", None)
        if cached_batches is not None:
            return cached_batches

        input_batches = state.layer_inputs or []
        input_kwargs_batches = state.layer_input_kwargs or []
        output_batches = state.layer_outputs or []
        if not input_batches or not output_batches:
            raise RuntimeError("ParoQuant layer optimization requires captured layer inputs and outputs.")
        if len(input_batches) != len(output_batches):
            raise RuntimeError("ParoQuant layer optimization requires aligned input/output batch counts.")

        if not input_kwargs_batches:
            input_kwargs_batches = [{} for _ in range(len(input_batches))]
        elif len(input_kwargs_batches) != len(input_batches):
            raise RuntimeError("ParoQuant layer optimization requires aligned layer-input kwargs.")

        position_ids = list(self.inputs_cache.position_ids or [])
        attention_masks = list(self.inputs_cache.attention_masks or [])
        if len(position_ids) < len(input_batches):
            position_ids.extend([None] * (len(input_batches) - len(position_ids)))
        if len(attention_masks) < len(input_batches):
            attention_masks.extend([None] * (len(input_batches) - len(attention_masks)))

        replay_batches: list[_ParoQuantReplayBatch] = []
        for input_batch, input_kwargs, output_batch, pos_ids, attn_mask in zip(
            input_batches,
            input_kwargs_batches,
            output_batches,
            position_ids,
            attention_masks,
        ):
            cpu_inputs = [
                self._move_group_value_to_cpu(tensor)
                for tensor in input_batch
            ]
            replay_batches.append(
                _ParoQuantReplayBatch(
                    inputs=cpu_inputs,
                    input_kwargs={
                        key: self._move_group_value_to_cpu(value)
                        for key, value in input_kwargs.items()
                    },
                    target=self._move_group_value_to_cpu(self._target_primary(output_batch)),
                    position_ids=None if pos_ids is None else self._move_group_value_to_cpu(pos_ids),
                    attention_mask=None if attn_mask is None else self._move_group_value_to_cpu(attn_mask),
                    row_count=self._layer_batch_row_count(cpu_inputs),
                )
            )

        train_batch_count = self._prefix_batch_count_for_rows(
            [batch.inputs for batch in replay_batches],
            int(self.qcfg.opt_train_samples),
        )
        val_batch_count = self._suffix_batch_count_for_rows(
            [batch.inputs for batch in replay_batches],
            int(self.qcfg.opt_validation_samples),
        )
        val_start = max(0, len(replay_batches) - val_batch_count)
        replay_split = (replay_batches[:train_batch_count], replay_batches[val_start:])
        state.replay_batches = replay_split
        return replay_split

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> int:
        """Measure one tensor's storage footprint."""
        return int(value.numel() * value.element_size())

    def _nested_tensor_bytes(self, value: Any) -> int:
        """Measure nested replay metadata footprint."""
        if isinstance(value, torch.Tensor):
            return self._tensor_bytes(value)
        if isinstance(value, dict):
            return sum(self._nested_tensor_bytes(inner) for inner in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._nested_tensor_bytes(inner) for inner in value)
        return 0

    def _replay_batch_bytes(self, batch: _ParoQuantReplayBatch) -> int:
        """Estimate one replay batch's device footprint."""
        return (
            sum(self._tensor_bytes(tensor) for tensor in batch.inputs)
            + self._tensor_bytes(batch.target)
            + (0 if batch.position_ids is None else self._tensor_bytes(batch.position_ids))
            + (0 if batch.attention_mask is None else self._tensor_bytes(batch.attention_mask))
            + self._nested_tensor_bytes(batch.input_kwargs)
        )

    def _layer_train_shard_batches(
        self,
        layer: torch.nn.Module,
        *,
        param_groups: Sequence[dict[str, object]],
        replay_batches: list[_ParoQuantReplayBatch],
    ) -> int:
        """Choose a conservative train-shard size from current free GPU memory."""
        if not replay_batches:
            return 1

        try:
            target_device = next(layer.parameters()).device
        except (StopIteration, RuntimeError):
            target_device = CPU

        if target_device.type != "cuda":
            return len(replay_batches)

        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(target_device)
        except RuntimeError:
            return 1

        layer_bytes = sum(self._tensor_bytes(param.detach()) for param in layer.parameters())
        layer_bytes += sum(self._tensor_bytes(buffer.detach()) for buffer in layer.buffers())
        active_params = {
            id(param): param
            for group in param_groups
            for param in group.get("params", [])
            if isinstance(param, nn.Parameter)
        }
        optim_bytes = sum(int(param.numel()) * 16 for param in active_params.values())
        sample_count = min(4, len(replay_batches))
        avg_batch_bytes = max(
            1,
            sum(self._replay_batch_bytes(batch) for batch in replay_batches[:sample_count]) // sample_count,
        )
        activation_margin = 256 * 1024 * 1024
        headroom = max(avg_batch_bytes, int(free_bytes) - layer_bytes - optim_bytes - activation_margin)
        return max(1, min(len(replay_batches), headroom // avg_batch_bytes))

    def _group_dataset_from_state(
        self,
        state: _ParoQuantLayerState,
    ) -> tuple[
        list[List[torch.Tensor]],
        list[Dict[str, Any]],
        list[Any],
        list[Optional[torch.Tensor]],
        list[Optional[torch.Tensor]],
        list[List[torch.Tensor]],
        list[Dict[str, Any]],
        list[Any],
        list[Optional[torch.Tensor]],
        list[Optional[torch.Tensor]],
    ]:
        """Slice the preserved layer IO into train/validation batch lists for grouped optimization."""
        cached_dataset = getattr(state, "grouped_dataset", None)
        if cached_dataset is not None:
            return cached_dataset

        input_batches = state.layer_inputs or []
        input_kwargs_batches = state.layer_input_kwargs or []
        output_batches = state.layer_outputs or []
        if not input_batches or not output_batches:
            raise RuntimeError("ParoQuant grouped optimization requires captured layer inputs and outputs.")
        if len(input_batches) != len(output_batches):
            raise RuntimeError("ParoQuant grouped optimization requires aligned input/output batch counts.")

        if not input_kwargs_batches:
            input_kwargs_batches = [{} for _ in range(len(input_batches))]
        elif len(input_kwargs_batches) != len(input_batches):
            raise RuntimeError("ParoQuant grouped optimization requires aligned layer-input kwargs.")

        position_ids = list(self.inputs_cache.position_ids or [])
        attention_masks = list(self.inputs_cache.attention_masks or [])
        if len(position_ids) < len(input_batches):
            position_ids.extend([None] * (len(input_batches) - len(position_ids)))
        if len(attention_masks) < len(input_batches):
            attention_masks.extend([None] * (len(input_batches) - len(attention_masks)))

        train_batch_count = self._prefix_batch_count_for_rows(input_batches, int(self.qcfg.opt_train_samples))
        val_batch_count = self._suffix_batch_count_for_rows(input_batches, int(self.qcfg.opt_validation_samples))
        val_start = max(0, len(input_batches) - val_batch_count)

        grouped_dataset = (
            input_batches[:train_batch_count],
            input_kwargs_batches[:train_batch_count],
            output_batches[:train_batch_count],
            position_ids[:train_batch_count],
            attention_masks[:train_batch_count],
            input_batches[val_start:],
            input_kwargs_batches[val_start:],
            output_batches[val_start:],
            position_ids[val_start:],
            attention_masks[val_start:],
        )
        state.grouped_dataset = grouped_dataset
        return grouped_dataset

    def _group_dataset_for_device(
        self,
        state: _ParoQuantLayerState,
        target_device: Optional[torch.device],
    ) -> tuple[
        list[List[torch.Tensor]],
        list[Dict[str, Any]],
        list[Any],
        list[Optional[torch.Tensor]],
        list[Optional[torch.Tensor]],
        list[List[torch.Tensor]],
        list[Dict[str, Any]],
        list[Any],
        list[Optional[torch.Tensor]],
        list[Optional[torch.Tensor]],
    ]:
        """Materialize the cached grouped dataset on the target device once per layer/device pair."""
        grouped_dataset = self._group_dataset_from_state(state)
        device = torch.device(target_device) if target_device is not None else CPU
        cache_key = str(device)
        grouped_dataset_by_device = getattr(state, "grouped_dataset_by_device", None)
        if grouped_dataset_by_device is None:
            grouped_dataset_by_device = {}
            state.grouped_dataset_by_device = grouped_dataset_by_device

        cached_dataset = grouped_dataset_by_device.get(cache_key)
        if cached_dataset is not None:
            return cached_dataset

        def _move_tensor_batches(batches: list[List[torch.Tensor]]) -> list[List[torch.Tensor]]:
            return [[move_to(tensor, device=device) for tensor in batch] for batch in batches]

        def _move_kwargs_batches(kwargs_batches: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
            return [nested_move_to(kwargs, device=device) for kwargs in kwargs_batches]

        def _move_target_batches(target_batches: list[Any]) -> list[Any]:
            return [nested_move_to(batch, device=device) for batch in target_batches]

        def _move_optional_tensors(optional_tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
            return [None if tensor is None else move_to(tensor, device=device) for tensor in optional_tensors]

        device_dataset = (
            _move_tensor_batches(grouped_dataset[0]),
            _move_kwargs_batches(grouped_dataset[1]),
            _move_target_batches(grouped_dataset[2]),
            _move_optional_tensors(grouped_dataset[3]),
            _move_optional_tensors(grouped_dataset[4]),
            _move_tensor_batches(grouped_dataset[5]),
            _move_kwargs_batches(grouped_dataset[6]),
            _move_target_batches(grouped_dataset[7]),
            _move_optional_tensors(grouped_dataset[8]),
            _move_optional_tensors(grouped_dataset[9]),
        )
        grouped_dataset_by_device[cache_key] = device_dataset
        return device_dataset

    def _forward_group_batch(
        self,
        layer: torch.nn.Module,
        *,
        batch_index: int,
        input_batch: List[torch.Tensor],
        input_kwargs: Dict[str, Any],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one cached batch through a grouped layer clone and return its primary output."""
        del batch_index
        try:
            layer_device = next(layer.parameters()).device
        except (StopIteration, RuntimeError):
            layer_device = CPU

        inputs = [_LayerShardLoader._tensor_to_device(inp, layer_device) for inp in input_batch]
        if not inputs:
            raise RuntimeError("ParoQuant grouped optimization forward requires at least one input tensor.")

        additional_inputs = self._prepare_group_forward_kwargs(
            layer,
            x=inputs[0],
            input_kwargs=input_kwargs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        module_output = layer(*inputs, **additional_inputs)
        if module_output is None:
            raise RuntimeError("ParoQuant grouped optimization forward returned no output.")
        if isinstance(module_output, tuple):
            module_output = module_output[0]
        if not isinstance(module_output, torch.Tensor):
            raise TypeError(
                "ParoQuant grouped optimization expected tensor layer outputs, "
                f"got `{type(module_output).__name__}`."
            )
        return module_output

    def _forward_replay_batch(
        self,
        layer: torch.nn.Module,
        *,
        replay_batch: _ParoQuantReplayBatch,
        cache_kwargs: bool,
    ) -> torch.Tensor:
        """Run one streamed replay batch through the live grouped layer."""
        layer_device = replay_batch.inputs[0].device if replay_batch.inputs else CPU

        inputs = [
            self._clone_group_runtime_metadata(_LayerShardLoader._tensor_to_device(inp, layer_device))
            for inp in replay_batch.inputs
        ]
        if not inputs:
            raise RuntimeError("ParoQuant layer replay requires at least one input tensor.")

        additional_inputs = self._prepare_group_forward_kwargs(
            layer,
            x=inputs[0],
            input_kwargs=replay_batch.input_kwargs,
            attention_mask=replay_batch.attention_mask,
            position_ids=replay_batch.position_ids,
            cache=cache_kwargs,
        )
        additional_inputs = self._clone_group_runtime_metadata(additional_inputs)
        if any(inp.is_inference() for inp in inputs) or _value_has_inference_tensor(additional_inputs):
            raise RuntimeError(
                "ParoQuant layer replay assembled inference-mode live inputs. "
                f"inputs_inference={[inp.is_inference() for inp in inputs]} "
                f"kwargs_inference={_value_has_inference_tensor(additional_inputs)} "
                f"kwargs_keys={sorted(additional_inputs.keys())}"
            )
        module_output = layer(*inputs, **additional_inputs)
        if module_output is None:
            raise RuntimeError("ParoQuant grouped optimization forward returned no output.")
        if isinstance(module_output, tuple):
            module_output = module_output[0]
        if not isinstance(module_output, torch.Tensor):
            raise TypeError(
                "ParoQuant grouped optimization expected tensor layer outputs, "
                f"got `{type(module_output).__name__}`."
            )
        return module_output

    @staticmethod
    def _reset_group_angles(optim_modules: dict[str, _ParoQuantOptimLinear]) -> None:
        """Clamp masked dummy rotations back to zero after each grouped optimizer step."""
        for optim_module in optim_modules.values():
            optim_module.reset_masked_angles()

    def _evaluate_group_layer(
        self,
        layer: torch.nn.Module,
        *,
        input_batches: list[List[torch.Tensor]],
        input_kwargs_batches: list[Dict[str, Any]],
        target_batches: list[Any],
        position_ids: list[Optional[torch.Tensor]],
        attention_masks: list[Optional[torch.Tensor]],
        use_amp: bool,
    ) -> float:
        """Measure full-layer reconstruction error for one grouped optimization stage."""
        if not input_batches:
            return 0.0

        total_loss = 0.0
        with torch.inference_mode():
            for batch_index, (input_batch, input_kwargs, target_batch, pos_ids, attn_mask) in enumerate(
                zip(input_batches, input_kwargs_batches, target_batches, position_ids, attention_masks)
            ):
                autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
                with autocast_ctx:
                    preds = self._forward_group_batch(
                        layer,
                        batch_index=batch_index,
                        input_batch=input_batch,
                        input_kwargs=input_kwargs,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                    )
                    target = self._prepare_group_target(target_batch, device=preds.device, dtype=preds.dtype)
                    total_loss += float(F.smooth_l1_loss(preds, target).item())
        return total_loss / max(1, len(input_batches))

    def _forward_group_batch_train(
        self,
        layer: torch.nn.Module,
        *,
        batch_index: int,
        input_batch: list[torch.Tensor],
        input_kwargs: dict[str, Any],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Optionally checkpoint grouped train forwards to reduce activation residency."""
        if not self._gradient_checkpointing_enabled() or not input_batch:
            return self._forward_group_batch(
                layer,
                batch_index=batch_index,
                input_batch=input_batch,
                input_kwargs=input_kwargs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        runtime_inputs = [
            self._clone_group_runtime_metadata(_LayerShardLoader._tensor_to_device(inp, inp.device))
            for inp in input_batch
        ]
        runtime_kwargs = self._clone_group_runtime_metadata(input_kwargs)
        runtime_attention_mask = self._clone_group_runtime_metadata(attention_mask)
        runtime_position_ids = self._clone_group_runtime_metadata(position_ids)

        def _forward(*runtime_inputs: torch.Tensor) -> torch.Tensor:
            return self._forward_group_batch(
                layer,
                batch_index=batch_index,
                input_batch=list(runtime_inputs),
                input_kwargs=runtime_kwargs,
                attention_mask=runtime_attention_mask,
                position_ids=runtime_position_ids,
            )

        return torch_checkpoint(_forward, *tuple(runtime_inputs), use_reentrant=False)

    def _forward_replay_batch_train(
        self,
        layer: torch.nn.Module,
        *,
        replay_batch: _ParoQuantReplayBatch,
        cache_kwargs: bool,
    ) -> torch.Tensor:
        """Optionally checkpoint streamed grouped train forwards to reduce activation residency."""
        if not self._gradient_checkpointing_enabled() or not replay_batch.inputs:
            return self._forward_replay_batch(
                layer,
                replay_batch=replay_batch,
                cache_kwargs=cache_kwargs,
            )

        def _forward(*runtime_inputs: torch.Tensor) -> torch.Tensor:
            return self._forward_replay_batch(
                layer,
                replay_batch=_ParoQuantReplayBatch(
                    inputs=list(runtime_inputs),
                    input_kwargs=replay_batch.input_kwargs,
                    target=replay_batch.target,
                    position_ids=replay_batch.position_ids,
                    attention_mask=replay_batch.attention_mask,
                    row_count=replay_batch.row_count,
                ),
                cache_kwargs=cache_kwargs,
            )

        return torch_checkpoint(_forward, *tuple(replay_batch.inputs), use_reentrant=False)

    @staticmethod
    def _normalize_group_optimizer_param_groups(
        param_groups: List[dict[str, object]],
    ) -> List[dict[str, object]]:
        """Merge equivalent optimizer groups so grouped optimization pays less optimizer overhead."""
        normalized_groups: List[dict[str, object]] = []
        group_index_by_key: Dict[tuple[float, float, tuple[float, float], float, bool, float, float, bool], int] = {}
        seen_param_ids: set[int] = set()

        for param_group in param_groups:
            raw_params = param_group.get("params", [])
            if isinstance(raw_params, nn.Parameter):
                raw_params = [raw_params]

            params: list[nn.Parameter] = []
            for param in raw_params:
                if not isinstance(param, nn.Parameter):
                    continue
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
                params.append(param)

            if not params:
                continue

            lr = float(param_group["lr"])
            weight_decay = float(param_group.get("weight_decay", 0.01))
            betas_obj = tuple(float(beta) for beta in param_group.get("betas", (0.9, 0.95)))
            betas = (betas_obj[0], betas_obj[1])
            eps = float(param_group.get("eps", 1e-10))
            amsgrad = bool(param_group.get("amsgrad", False))
            momentum = float(param_group.get("momentum", 0.0))
            dampening = float(param_group.get("dampening", 0.0))
            nesterov = bool(param_group.get("nesterov", False))
            key = (lr, weight_decay, betas, eps, amsgrad, momentum, dampening, nesterov)

            bucket_index = group_index_by_key.get(key)
            if bucket_index is None:
                group_index_by_key[key] = len(normalized_groups)
                normalized_groups.append(
                    {
                        "params": list(params),
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "betas": betas,
                        "eps": eps,
                        "amsgrad": amsgrad,
                        "momentum": momentum,
                        "dampening": dampening,
                        "nesterov": nesterov,
                    }
                )
            else:
                normalized_groups[bucket_index]["params"].extend(params)

        return normalized_groups

    @staticmethod
    def _build_group_optimizer(
        normalized_groups: List[dict[str, object]],
        *,
        device: torch.device,
        optimizer_name: str = "adamw",
    ) -> torch.optim.Optimizer:
        """Construct the grouped stage optimizer after redundant groups are merged away."""
        return build_paroquant_optimizer(
            normalized_groups,
            device=device,
            optimizer_name=optimizer_name,
            graph_capture=False,
        )

    @staticmethod
    def _build_group_adamw(
        normalized_groups: List[dict[str, object]],
        *,
        device: torch.device,
    ) -> torch.optim.Optimizer:
        """Backward-compatible wrapper for tests that still exercise the AdamW path directly."""
        return ParoQuantProcessor._build_group_optimizer(
            normalized_groups,
            device=device,
            optimizer_name="adamw",
        )

    @staticmethod
    def _group_state_key_matches_prefixes(state_key: str, active_prefixes: tuple[str, ...]) -> bool:
        """Match state keys that belong to the active grouped modules only."""
        return any(state_key == prefix or state_key.startswith(f"{prefix}.") for prefix in active_prefixes)

    def _snapshot_group_best_state(
        self,
        layer: torch.nn.Module,
        *,
        active_prefixes: tuple[str, ...],
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> dict[str, torch.Tensor]:
        """Capture only the mutable grouped module state instead of the whole layer clone."""
        return {
            key: (
                tensor.detach().clone()
                if target_device is None and (target_dtype is None or not tensor.is_floating_point() or tensor.dtype == target_dtype)
                else tensor.detach().to(
                    device=target_device if target_device is not None else tensor.device,
                    dtype=target_dtype if target_dtype is not None and tensor.is_floating_point() else tensor.dtype,
                ).clone()
            )
            for key, tensor in layer.state_dict().items()
            if self._group_state_key_matches_prefixes(key, active_prefixes)
        }

    @staticmethod
    def _restore_group_best_state(
        layer: torch.nn.Module,
        *,
        best_state: dict[str, torch.Tensor],
    ) -> None:
        """Restore the captured grouped module state in-place."""
        if not best_state:
            return
        live_state = layer.state_dict(keep_vars=True)
        with torch.no_grad():
            for key, tensor in best_state.items():
                live_state[key].copy_(tensor)

    def _run_group_stage(
        self,
        layer: torch.nn.Module,
        *,
        optim_modules: dict[str, _ParoQuantOptimLinear],
        input_batches_train: list[List[torch.Tensor]],
        input_kwargs_train: list[Dict[str, Any]],
        target_batches_train: list[Any],
        position_ids_train: list[Optional[torch.Tensor]],
        attention_masks_train: list[Optional[torch.Tensor]],
        input_batches_val: list[List[torch.Tensor]],
        input_kwargs_val: list[Dict[str, Any]],
        target_batches_val: list[Any],
        position_ids_val: list[Optional[torch.Tensor]],
        attention_masks_val: list[Optional[torch.Tensor]],
        param_groups: List[dict[str, object]],
        epochs: int,
    ) -> tuple[float, float]:
        """Run one grouped optimization stage against preserved full-layer outputs."""
        _normalize_opt_impl(self.qcfg.opt_stage_impl, field="stage_impl")
        optimizer_name = _normalize_opt_optimizer(getattr(self.qcfg, "opt_optimizer", "adamw"))
        normalized_groups = self._normalize_group_optimizer_param_groups(param_groups)

        opt_device = next(layer.parameters()).device
        use_amp = opt_device.type == "cuda"
        with _activate_stage_params(layer, normalized_groups):
            if epochs <= 0 or not normalized_groups:
                train_loss = self._evaluate_group_layer(
                    layer,
                    input_batches=input_batches_train,
                    input_kwargs_batches=input_kwargs_train,
                    target_batches=target_batches_train,
                    position_ids=position_ids_train,
                    attention_masks=attention_masks_train,
                    use_amp=use_amp,
                )
                val_loss = self._evaluate_group_layer(
                    layer,
                    input_batches=input_batches_val,
                    input_kwargs_batches=input_kwargs_val,
                    target_batches=target_batches_val,
                    position_ids=position_ids_val,
                    attention_masks=attention_masks_val,
                    use_amp=use_amp,
                )
                return train_loss, val_loss

            optimizer = self._build_group_optimizer(
                normalized_groups,
                device=opt_device,
                optimizer_name=optimizer_name,
            )
            total_steps = max(1, epochs * max(1, len(input_batches_train)))
            base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            scaler = torch.amp.GradScaler(enabled=use_amp)
            active_prefixes = tuple(optim_modules.keys())
            needs_angle_reset = any(optim_module.theta.requires_grad for optim_module in optim_modules.values())
            best_state_dtype = _resolve_best_state_snapshot_dtype(
                best_state_dtype=getattr(self.qcfg, "opt_best_state_dtype", "fp32"),
                device=opt_device,
            )
            best_state: Optional[dict[str, torch.Tensor]] = None
            best_val_loss = float("inf")
            last_train_loss = 0.0
            global_step = 0

            for _epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                optimizer.zero_grad(set_to_none=True)

                for batch_index, (input_batch, input_kwargs, target_batch, pos_ids, attn_mask) in enumerate(
                    zip(
                        input_batches_train,
                        input_kwargs_train,
                        target_batches_train,
                        position_ids_train,
                        attention_masks_train,
                    )
                ):
                    autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
                    with autocast_ctx:
                        preds = self._forward_group_batch_train(
                            layer,
                            batch_index=batch_index,
                            input_batch=input_batch,
                            input_kwargs=input_kwargs,
                            attention_mask=attn_mask,
                            position_ids=pos_ids,
                        )
                        target = self._prepare_group_target(target_batch, device=preds.device, dtype=preds.dtype)
                        loss = F.smooth_l1_loss(preds, target)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = (base_lr / 20.0) + ((base_lr - (base_lr / 20.0)) * cosine_ratio)

                    self._reset_group_angles(optim_modules)
                    epoch_loss += float(loss.item())
                    batch_count += 1

                last_train_loss = epoch_loss / max(1, batch_count)
                val_loss = self._evaluate_group_layer(
                    layer,
                    input_batches=input_batches_val,
                    input_kwargs_batches=input_kwargs_val,
                    target_batches=target_batches_val,
                    position_ids=position_ids_val,
                    attention_masks=attention_masks_val,
                    use_amp=use_amp,
                )
                if best_state is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self._snapshot_group_best_state(
                        layer,
                        active_prefixes=active_prefixes,
                        target_dtype=best_state_dtype,
                    )

            if best_state is not None:
                self._restore_group_best_state(layer, best_state=best_state)
            self._reset_group_angles(optim_modules)
            return last_train_loss, best_val_loss

    def _evaluate_group_layer_streamed(
        self,
        layer: torch.nn.Module,
        *,
        replay_batches: list[_ParoQuantReplayBatch],
        use_amp: bool,
        target_device: torch.device,
        metadata_cache: Optional[dict[tuple[int, str], torch.Tensor]] = None,
    ) -> float:
        """Measure full-layer reconstruction error while streaming one validation batch at a time."""
        if not replay_batches:
            return 0.0

        total_loss = 0.0
        loader = _LayerShardLoader(
            replay_batches,
            target_device=target_device,
            shard_batches=1,
            metadata_cache=metadata_cache,
        )
        autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with torch.inference_mode(), autocast_ctx:
            for shard in loader.iter_shards():
                replay_batch = shard[0]
                preds = self._forward_replay_batch(
                    layer,
                    replay_batch=replay_batch,
                    cache_kwargs=False,
                )
                target = self._prepare_group_target(
                    replay_batch.target,
                    device=preds.device,
                    dtype=preds.dtype,
                    cache=False,
                )
                total_loss += float(F.smooth_l1_loss(preds, target).item())
        return total_loss / max(1, len(replay_batches))

    def _run_group_stage_streamed(
        self,
        layer: torch.nn.Module,
        *,
        optim_modules: dict[str, _ParoQuantOptimLinear],
        replay_batches_train: list[_ParoQuantReplayBatch],
        replay_batches_val: list[_ParoQuantReplayBatch],
        param_groups: List[dict[str, object]],
        epochs: int,
        metadata_cache: Optional[dict[tuple[int, str], torch.Tensor]] = None,
    ) -> tuple[float, float]:
        """Run one grouped layer stage while streaming train shards and validation batches from CPU."""
        _normalize_opt_impl(self.qcfg.opt_stage_impl, field="stage_impl")
        optimizer_name = _normalize_opt_optimizer(getattr(self.qcfg, "opt_optimizer", "adamw"))
        normalized_groups = self._normalize_group_optimizer_param_groups(param_groups)

        opt_device = next(layer.parameters()).device
        use_amp = opt_device.type == "cuda"
        with _activate_stage_params(layer, normalized_groups):
            if epochs <= 0 or not normalized_groups:
                train_loss = self._evaluate_group_layer_streamed(
                    layer,
                    replay_batches=replay_batches_train,
                    use_amp=use_amp,
                    target_device=opt_device,
                    metadata_cache=metadata_cache,
                )
                val_loss = self._evaluate_group_layer_streamed(
                    layer,
                    replay_batches=replay_batches_val,
                    use_amp=use_amp,
                    target_device=opt_device,
                    metadata_cache=metadata_cache,
                )
                return train_loss, val_loss

            optimizer = self._build_group_optimizer(
                normalized_groups,
                device=opt_device,
                optimizer_name=optimizer_name,
            )
            total_steps = max(1, epochs * max(1, len(replay_batches_train)))
            base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            scaler = torch.amp.GradScaler(enabled=use_amp)
            active_prefixes = tuple(optim_modules.keys())
            needs_angle_reset = any(optim_module.theta.requires_grad for optim_module in optim_modules.values())
            best_state_dtype = _resolve_best_state_snapshot_dtype(
                best_state_dtype=getattr(self.qcfg, "opt_best_state_dtype", "fp32"),
                device=CPU,
            )
            best_state: Optional[dict[str, torch.Tensor]] = None
            best_val_loss = float("inf")
            last_train_loss = 0.0
            global_step = 0
            shard_batches = self._layer_train_shard_batches(
                layer,
                param_groups=normalized_groups,
                replay_batches=replay_batches_train,
            )

            for _epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                optimizer.zero_grad(set_to_none=True)
                train_loader = _LayerShardLoader(
                    replay_batches_train,
                    target_device=opt_device,
                    shard_batches=shard_batches,
                    metadata_cache=metadata_cache,
                )

                autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
                with autocast_ctx:
                    for shard in train_loader.iter_shards():
                        for replay_batch in shard:
                            preds = self._forward_replay_batch_train(
                                layer,
                                replay_batch=replay_batch,
                                cache_kwargs=False,
                            )
                            target = self._prepare_group_target(
                                replay_batch.target,
                                device=preds.device,
                                dtype=preds.dtype,
                                cache=False,
                            )
                            loss = F.smooth_l1_loss(preds, target)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                            global_step += 1
                            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
                            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                                group["lr"] = (base_lr / 20.0) + ((base_lr - (base_lr / 20.0)) * cosine_ratio)

                            if needs_angle_reset:
                                self._reset_group_angles(optim_modules)
                            epoch_loss += float(loss.item())
                            batch_count += 1

                last_train_loss = epoch_loss / max(1, batch_count)
                val_loss = self._evaluate_group_layer_streamed(
                    layer,
                    replay_batches=replay_batches_val,
                    use_amp=use_amp,
                    target_device=opt_device,
                    metadata_cache=metadata_cache,
                )
                if best_state is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self._snapshot_group_best_state(
                        layer,
                        active_prefixes=active_prefixes,
                        target_device=CPU,
                        target_dtype=best_state_dtype,
                    )

            if best_state is not None:
                self._restore_group_best_state(layer, best_state=best_state)
            if needs_angle_reset:
                self._reset_group_angles(optim_modules)
            return last_train_loss, best_val_loss

    def _optimize_live_layer(
        self,
        state: _ParoQuantLayerState,
        group_modules: list[NamedModule],
    ) -> tuple[dict[str, object], float]:
        """Optimize the live layer in place for full-layer ParoQuant scope."""
        if not group_modules:
            raise ValueError("ParoQuantProcessor grouped optimization requires at least one module.")
        if state.layer_module is None:
            raise RuntimeError("ParoQuantProcessor layer-scope optimization requires the live layer module.")

        layer = state.layer_module
        target_device = group_modules[0].weight.device
        original_layer_dtype = group_modules[0].weight.dtype
        layer = layer.to(device=target_device, dtype=torch.float32)
        self._strip_hooked_linear_wrappers(layer)
        self._sync_named_modules_to_live_layer(layer, group_modules)
        attn_impl_overrides = self._force_layer_eager_attention(layer)
        replay_batches_train, replay_batches_val = self._replay_batches_from_state(state)
        metadata_cache: dict[tuple[int, str], torch.Tensor] = {}
        optim_modules: dict[str, _ParoQuantOptimLinear] = {}
        original_modules: dict[str, torch.nn.Module] = {}
        optimizer_group_kwargs = self._optimizer_param_group_kwargs()
        try:
            for param in layer.parameters():
                param.requires_grad_(False)

            for named_module in group_modules:
                original_modules[named_module.name] = recurse_getattr(layer, named_module.name)
                optim_module = self._build_group_optim_linear(named_module)
                recurse_setattr(layer, named_module.name, optim_module)
                optim_modules[named_module.name] = optim_module

            self._materialize_live_layer_autograd_tensors(layer)

            self._run_group_stage_streamed(
                layer,
                optim_modules=optim_modules,
                replay_batches_train=replay_batches_train,
                replay_batches_val=replay_batches_val,
                param_groups=[
                    {"params": [optim_module.channel_scales_opt], "lr": self.qcfg.opt_rotation_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ] + [
                    {"params": [optim_module.theta], "lr": self.qcfg.opt_rotation_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ],
                epochs=int(self.qcfg.opt_rotation_epochs),
                metadata_cache=metadata_cache,
            )

            for optim_module in optim_modules.values():
                optim_module.init_quantizer()

            train_loss, val_loss = self._run_group_stage_streamed(
                layer,
                optim_modules=optim_modules,
                replay_batches_train=replay_batches_train,
                replay_batches_val=replay_batches_val,
                param_groups=[
                    {"params": [optim_module.weight], "lr": self.qcfg.opt_weight_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ] + [
                    {"params": optim_module.quantizer.optim_params(), "lr": self.qcfg.opt_quantizer_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                    if optim_module.quantizer is not None
                ],
                epochs=int(self.qcfg.opt_finetune_epochs),
                metadata_cache=metadata_cache,
            )

            metadata_cache.clear()

            results: dict[str, object] = {}
            for named_module in group_modules:
                optim_module = optim_modules[named_module.name]
                results[named_module.name] = _result_from_model(
                    optim_module,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    used_identity=False,
                )
            return results, val_loss
        finally:
            for named_module in reversed(group_modules):
                original_module = original_modules.get(named_module.name)
                if original_module is not None:
                    recurse_setattr(layer, named_module.name, original_module)
            # Live-layer training happens in fp32, but replay and inference should
            # return to the layer's native dtype so flash-attn and downstream
            # kernels see the original half/bfloat activations again.
            layer.to(device=target_device, dtype=original_layer_dtype)
            self._restore_layer_attention_impl(attn_impl_overrides)

    @staticmethod
    def _supports_live_layer_scope(group_modules: list[NamedModule]) -> bool:
        """Restrict in-place layer optimization to dense decoder blocks.

        The official apples-to-apples path is one full dense decoder layer at a
        time. Expert-heavy layers can contain hundreds or thousands of modules,
        so keep those on the cloned grouped path until a dedicated MoE/shared
        rotation implementation lands.
        """
        if not group_modules:
            return False
        expert_markers = (
            "expert",
            "experts",
            "shared_expert",
            "gate_up_proj",
        )
        expert_prefixes = ("experts.", "mlp.experts.", "moe.")
        expert_like_modules = 0
        dense_modules = 0
        for module in group_modules:
            module_name = getattr(module, "name", "")
            leaf = module_name.rsplit(".", 1)[-1]
            if any(marker in module_name for marker in expert_markers) or module_name.startswith(expert_prefixes):
                expert_like_modules += 1
                continue
            dense_modules += 1
            if leaf not in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                return False
        return dense_modules > 0 and expert_like_modules == 0

    def _optimize_group(
        self,
        state: _ParoQuantLayerState,
        group_modules: list[NamedModule],
    ) -> tuple[dict[str, object], float]:
        """Optimize one compute_block or whole-layer group against the preserved full-layer target."""
        if self._opt_scope_mode() == "layer" and self._supports_live_layer_scope(group_modules):
            with torch.inference_mode(False), torch.enable_grad():
                return self._optimize_live_layer(state, group_modules)

        with torch.inference_mode(False), torch.enable_grad():
            layer_clone, optim_modules = self._build_group_optim_layer(state, group_modules)
            for param in layer_clone.parameters():
                param.requires_grad_(False)

            (
                input_batches_train,
                input_kwargs_train,
                target_batches_train,
                position_ids_train,
                attention_masks_train,
                input_batches_val,
                input_kwargs_val,
                target_batches_val,
                position_ids_val,
                attention_masks_val,
            ) = self._group_dataset_for_device(state, next(layer_clone.parameters()).device)
            optimizer_group_kwargs = self._optimizer_param_group_kwargs()

            self._run_group_stage(
                layer_clone,
                optim_modules=optim_modules,
                input_batches_train=input_batches_train,
                input_kwargs_train=input_kwargs_train,
                target_batches_train=target_batches_train,
                position_ids_train=position_ids_train,
                attention_masks_train=attention_masks_train,
                input_batches_val=input_batches_val,
                input_kwargs_val=input_kwargs_val,
                target_batches_val=target_batches_val,
                position_ids_val=position_ids_val,
                attention_masks_val=attention_masks_val,
                param_groups=[
                    {"params": [optim_module.channel_scales_opt], "lr": self.qcfg.opt_rotation_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ] + [
                    {"params": [optim_module.theta], "lr": self.qcfg.opt_rotation_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ],
                epochs=int(self.qcfg.opt_rotation_epochs),
            )

            for optim_module in optim_modules.values():
                optim_module.init_quantizer()

            train_loss, val_loss = self._run_group_stage(
                layer_clone,
                optim_modules=optim_modules,
                input_batches_train=input_batches_train,
                input_kwargs_train=input_kwargs_train,
                target_batches_train=target_batches_train,
                position_ids_train=position_ids_train,
                attention_masks_train=attention_masks_train,
                input_batches_val=input_batches_val,
                input_kwargs_val=input_kwargs_val,
                target_batches_val=target_batches_val,
                position_ids_val=position_ids_val,
                attention_masks_val=attention_masks_val,
                param_groups=[
                    {"params": [optim_module.weight], "lr": self.qcfg.opt_weight_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                ] + [
                    {"params": optim_module.quantizer.optim_params(), "lr": self.qcfg.opt_quantizer_lr, **optimizer_group_kwargs}
                    for optim_module in optim_modules.values()
                    if optim_module.quantizer is not None
                ],
                epochs=int(self.qcfg.opt_finetune_epochs),
            )

            results: dict[str, object] = {}
            for named_module in group_modules:
                optim_module = optim_modules[named_module.name]
                results[named_module.name] = _result_from_model(
                    optim_module,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    used_identity=False,
                )
            return results, val_loss

    @staticmethod
    def _module_compute_block_label(module_name: str) -> str:
        """Map common projection archetypes to compute_block optimization buckets."""
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in {"q_proj", "k_proj", "v_proj"}:
            return "attn_qkv"
        if leaf == "o_proj":
            return "attn_o"
        if leaf in {"gate_proj", "up_proj"}:
            return "mlp_gate_up"
        if leaf == "down_proj":
            return "mlp_down"
        return f"single:{module_name}"

    @staticmethod
    def _module_compute_block_order(module_name: str) -> tuple[int, str]:
        """Keep compute_block members in canonical architectural order."""
        leaf = module_name.rsplit(".", 1)[-1]
        order = {
            "q_proj": 0,
            "k_proj": 1,
            "v_proj": 2,
            "o_proj": 3,
            "gate_proj": 4,
            "up_proj": 5,
            "down_proj": 6,
        }
        return (order.get(leaf, 100), module_name)

    def _optimization_groups_for_layer(
        self,
        state: _ParoQuantLayerState,
    ) -> list[tuple[str, list[NamedModule]]]:
        """Resolve the optimization scope for the current layer.

        `module` keeps today's per-linear behavior. `compute_block` and `layer`
        are scaffolded here so the lifecycle can switch scopes explicitly once
        their execution paths land.
        """
        mode = self._opt_scope_mode()
        named_modules = [state.modules[name] for name in sorted(state.modules)]
        if mode == "module":
            return [(module.name, [module]) for module in named_modules]
        if mode == "compute_block":
            grouped: Dict[str, list[NamedModule]] = {}
            for module in named_modules:
                grouped.setdefault(self._module_compute_block_label(module.name), []).append(module)
            for label in grouped:
                grouped[label].sort(key=lambda module: self._module_compute_block_order(module.name))
            return [(label, grouped[label]) for label in sorted(grouped)]
        if mode == "layer":
            return [("layer", named_modules)]
        raise ValueError(f"ParoQuantProcessor: unsupported optimize scope `{self.qcfg.opt_scope}`.")

    def _quantize_layer(self, layer_index: int, state: _ParoQuantLayerState) -> None:
        """Quantize every captured module in a layer once all subsets are ready."""
        if state.quantized:
            return

        optimization_groups = self._optimization_groups_for_layer(state)
        mode = self._opt_scope_mode()

        input_feat = self._layer_input_features(state)
        for module_name, tensor in input_feat.items():
            if tensor.numel() == 0 and not self.fallback:
                raise RuntimeError(
                    f"ParoQuantProcessor error: missing activation features for `{module_name}` with fallback disabled."
                )

        if mode == "module":
            for module_name, named_module in list(state.modules.items()):
                feat = input_feat.get(module_name)
                if feat is None:
                    feat = torch.empty(0)

                start = time.perf_counter()
                _train_loss, val_loss = self._quantize_one_module(named_module, feat)
                duration = time.perf_counter() - start
                self._log_quant_result(named_module, feat, val_loss, duration)
        else:
            if state.layer_inputs is None or state.layer_outputs is None:
                raise RuntimeError(
                    "ParoQuantProcessor grouped optimization requires preserved layer inputs and outputs. "
                    f"Resolved groups for layer {layer_index}: {[label for label, _modules in optimization_groups]}"
                )

            for _group_label, group_modules in optimization_groups:
                start = time.perf_counter()
                group_results, group_val_loss = self._optimize_group(state, group_modules)
                duration = time.perf_counter() - start
                duration_per_module = duration / max(1, len(group_modules))

                for named_module in group_modules:
                    original_weight = self._module_weight_matrix(named_module).detach().clone()
                    result = group_results[named_module.name]
                    self._apply_optimization_result(named_module, result, original_weight)
                    if mode == "layer":
                        move_to(named_module.module, device=CPU)
                    feat = input_feat.get(named_module.name)
                    if feat is None:
                        feat = torch.empty(0)
                    self._log_quant_result(named_module, feat, group_val_loss, duration_per_module)

                if mode == "compute_block" and getattr(self.qcfg, "offload_to_disk", False):
                    flush_device = self._module_weight_matrix(group_modules[0]).device if group_modules else None
                    torch_empty_cache(device=flush_device, gc=False, sync=True)

        state.quantized = True
        with self.lock:
            for module_name in list(state.modules):
                entry = self.tasks.get(module_name)
                if entry is not None and entry.get("layer_index") == layer_index:
                    entry["inputs"] = []
        state.modules.clear()
        state.pending_modules.clear()
        state.processed_subsets.clear()
        state.layer_inputs = None
        state.layer_input_kwargs = None
        state.layer_outputs = None
        state.pristine_layer_module = None
        state.prepared_group_source_module = None
        state.prepared_group_source_module_by_device = None
        state.grouped_dataset = None
        state.grouped_dataset_by_device = None
        state.replay_batches = None
        state.subset_total = None
        if hasattr(self, "_group_forward_kwargs_cache"):
            self._group_forward_kwargs_cache.clear()
        if hasattr(self, "_group_forward_prepared_cache"):
            self._group_forward_prepared_cache.clear()
        if hasattr(self, "_group_target_cache"):
            self._group_target_cache.clear()
        if hasattr(self, "_group_position_ids_cache"):
            self._group_position_ids_cache.clear()
        if hasattr(self, "_group_rotary_position_embeddings_cache"):
            self._group_rotary_position_embeddings_cache.clear()

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Register a module for later activation capture and deferred quantization."""
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        self.fallback = normalize_fallback(fallback, self.qcfg.fallback)
        layer_state = self._get_layer_state(module.layer_index)
        with layer_state.lock:
            layer_state.modules[module.name] = module
            layer_state.layer_module = module.state.get("layer_module", layer_state.layer_module)
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

    def receive_layer_forward_context(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        layer_outputs: List[List[torch.Tensor]],
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Preserve noisy grouped inputs plus clean float layer targets."""
        del subset_index
        state = self._get_layer_state(layer_index)
        with state.lock:
            if state.layer_inputs is None:
                state.layer_inputs = layer_inputs
            if state.layer_input_kwargs is None:
                state.layer_input_kwargs = layer_input_kwargs
            if state.layer_outputs is None:
                state.layer_outputs = layer_outputs
            if subset_total is not None and state.subset_total is None:
                state.subset_total = subset_total

    def receive_pristine_layer_module(
        self,
        *,
        layer_index: int,
        layer_module: torch.nn.Module,
    ) -> None:
        """Preserve an untouched float layer snapshot for grouped optimization clones."""
        if self._opt_scope_mode() == "layer":
            return
        state = self._get_layer_state(layer_index)
        with state.lock:
            if state.pristine_layer_module is None:
                state.pristine_layer_module = copy.deepcopy(layer_module).to(device=CPU)

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
        if not isinstance(qmodule, ParoLinear):
            raise TypeError(
                f"Expected `{module.full_name}` to be packed as ParoLinear, got `{type(qmodule).__name__}`."
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
        model.quantize_config.method = METHOD.PARO
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
