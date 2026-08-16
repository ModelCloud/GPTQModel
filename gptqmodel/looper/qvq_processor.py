# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QVQ calibration, quantization, replacement, and checkpoint lifecycle."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import zlib
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import transformers
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.base import (
    MODULE_TREE_FLAG_GATE,
    MODULE_TREE_FLAG_K,
    MODULE_TREE_FLAG_Q,
    MODULE_TREE_FLAG_UP,
    MODULE_TREE_FLAG_V,
    module_tree_flags_are_moe,
)
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
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.qvq import QVQLinear
from ..quantization.config import FORMAT, METHOD, GPTQConfig, HessianConfig, QVQConfig
from ..quantization.gptq import GPTQ
from ..quantization.qvq import QVQQuantizationTelemetry, quantize_qvq_linear
from ..quantization.qvq_yaqa import capture_yaqa_sketch_b
from ..utils.attn_mask import apply_keep_mask_bt
from ..utils.backend import BACKEND
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.looper_helpers import normalize_device_like
from ..utils.model import find_modules, get_layers_with_prefixes, recurse_setattr
from ..utils.module_locks import parent_module_lock
from .qvq_output_alignment import QVQOutputAlignmentAttachment


log = setup_logger()


def clone_qvq_config_for_module(qcfg: QVQConfig, module_full_name: str) -> Optional[QVQConfig]:
    """Clone QVQ config, apply the one supported dynamic override, or skip the module."""

    dynamic_overrides = qcfg.dynamic_get(layer_name=module_full_name)
    if dynamic_overrides is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)
    if dynamic_overrides:
        qcfg_clone.bits = dynamic_overrides.get("bits", qcfg_clone.bits)
    qcfg_clone.__post_init__()
    return qcfg_clone


class QVQProcessor(LoopProcessor):
    """Capture masked activation Hessians and install pack-ready QVQ linear modules."""

    # QVQ uses the input Hessian only; routing bypass may capture expert inputs
    # directly without executing an otherwise-discarded dense projection.
    moe_input_capture_without_forward = True
    def __init__(
        self,
        tokenizer,
        qcfg: QVQConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        yaqa_calibration=None,
        require_fwd: bool = True,
        calibration_concat_separator: Optional[str] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """Initialize QVQ lifecycle state and validate currently supported capture modes."""

        if execution_config is None:
            execution_config = ExecutionConfig(
                require_fwd=require_fwd,
                fwd_replay_after_process=True,
                subset_forward_early_stop=True,
            )
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=execution_config,
        )
        # YAQA's full-model Fisher pass may use a larger, independent stream.
        # The ordinary calibration dataset remains the source for activation
        # Hessians and module replay.
        self.yaqa_calibration = self.calibration_dataset if yaqa_calibration is None else yaqa_calibration
        # Keep [B, S] membership until the hook can remove padding. Flattening
        # before the hook loses the only authoritative keep mask.
        self.preserve_batch_keep_mask = True
        self.avg_losses = []
        self._stats_lock = threading.Lock()
        self._yaqa_input_hessians: Dict[str, torch.Tensor] = {}
        self._yaqa_output_hessians: Dict[str, torch.Tensor] = {}
        self._yaqa_stats: Dict[str, Any] = {}
        self._yaqa_factor_lock = threading.Lock()
        self._yaqa_prepared = qcfg.rounding != "yaqa"
        self._output_alignment = (
            None
            if qcfg.output_alignment is None
            else QVQOutputAlignmentAttachment(qcfg.output_alignment)
        )
        self._output_alignment_stats: Dict[int, list[Dict[str, float]]] = {}
        self._pristine_hessian_lock = threading.RLock()
        self._pristine_hessian_modules: Dict[int, Dict[str, NamedModule]] = {}
        self._active_pristine_hessian_captures: Dict[str, GPTQ] = {}
        self._propagation_gates: Dict[str, tuple[torch.Tensor, torch.Tensor, Any, Any]] = {}
        self._automatic_propagation_gate_samples: Dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._additional_calibration_sample_counts: Dict[str, set[int]] = {}
        self._propagation_gates_lock = threading.RLock()

    def _record_automatic_propagation_gate(
        self, module_full_name: str, inputs: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Retain a bounded held-out slice collected by the normal dense hook."""

        max_rows = 512
        with self._propagation_gates_lock:
            retained = sum(
                rows.shape[0] for rows, _ in self._automatic_propagation_gate_samples.get(module_full_name, ())
            )
            remaining = max_rows - retained
            if remaining <= 0:
                return
            self._automatic_propagation_gate_samples.setdefault(module_full_name, []).append(
                (inputs[:remaining].detach().cpu(), targets[:remaining].detach().cpu())
            )

    def _materialize_automatic_propagation_gate(self, module_full_name: str, bias: torch.Tensor | None = None) -> None:
        with self._propagation_gates_lock:
            if module_full_name in self._propagation_gates:
                self._automatic_propagation_gate_samples.pop(module_full_name, None)
                return
            samples = self._automatic_propagation_gate_samples.pop(module_full_name, [])
        if not samples:
            return
        inputs = torch.cat([item[0] for item in samples], dim=0)
        targets = torch.cat([item[1] for item in samples], dim=0)
        bias_cpu = None if bias is None else bias.detach().to(device="cpu", dtype=torch.float32)

        def heldout_acceptance(proposal: torch.Tensor, baseline: torch.Tensor) -> bool:
            gate_inputs = inputs.to(device=proposal.device, dtype=torch.float32)
            gate_targets = targets.to(device=proposal.device, dtype=torch.float32)
            proposal_output = gate_inputs @ proposal.to(torch.float32).transpose(0, 1)
            baseline_output = gate_inputs @ baseline.to(torch.float32).transpose(0, 1)
            if bias_cpu is not None:
                gate_bias = bias_cpu.to(device=proposal.device)
                proposal_output = proposal_output + gate_bias
                baseline_output = baseline_output + gate_bias
            proposal_loss = (gate_targets - proposal_output).square().sum()
            baseline_loss = (gate_targets - baseline_output).square().sum()
            return bool(torch.isfinite(proposal_loss) and proposal_loss < baseline_loss)

        self.set_propagation_gate(module_full_name, inputs, targets, heldout_acceptance)

    def _has_propagation_gate(self, module_full_name: str) -> bool:
        with self._propagation_gates_lock:
            return module_full_name in self._propagation_gates

    def set_propagation_gate(
        self,
        module_full_name: str,
        inputs: torch.Tensor,
        target_output: torch.Tensor,
        acceptance_gate=None,
        candidate_score=None,
    ) -> None:
        """Attach disjoint held-out module rows for opt-in bank selection."""
        if inputs.ndim != 2 or target_output.ndim != 2 or inputs.shape[0] != target_output.shape[0]:
            raise ValueError("QVQ propagation gate tensors must be rank-2 with matching rows.")
        if not torch.isfinite(inputs).all() or not torch.isfinite(target_output).all():
            raise ValueError("QVQ propagation gate tensors must be finite.")
        with self._propagation_gates_lock:
            self._propagation_gates[module_full_name] = (
                inputs.detach().clone(), target_output.detach().clone(), acceptance_gate, candidate_score
            )

    def _get_propagation_gate(self, module_full_name: str, device: torch.device):
        with self._propagation_gates_lock:
            gate = self._propagation_gates.get(module_full_name)
            if gate is None:
                return None
            inputs, targets, callback, candidate_score = gate
            return (
                inputs.to(device=device, dtype=torch.float32),
                targets.to(device=device, dtype=torch.float32),
                callback,
                candidate_score,
                gate,
            )

    def _require_propagation_gate(self, module_full_name: str, device: torch.device):
        with self._propagation_gates_lock:
            gate = self._propagation_gates.get(module_full_name)
            if gate is None:
                raise RuntimeError(
                    f"QVQ propagated bank selection requires a disjoint held-out gate for `{module_full_name}`."
                )
            inputs, targets, callback, candidate_score = gate
            return (
                inputs.to(device=device, dtype=torch.float32),
                targets.to(device=device, dtype=torch.float32),
                callback,
                candidate_score,
                gate,
            )

    def _pop_propagation_gate(self, module_full_name: str, owner=None) -> None:
        with self._propagation_gates_lock:
            if owner is None or self._propagation_gates.get(module_full_name) is owner:
                self._propagation_gates.pop(module_full_name, None)

    def refine_subset_module_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Match QTIP's projection-at-a-time installation when alignment is enabled."""

        if self._output_alignment is None:
            return groups
        gptq_model = self._output_alignment.gptq_model
        if gptq_model is None:
            ordered_groups = groups
        else:
            # QTIP's authoritative Llama recipe installs V, Q, K and then UP,
            # GATE. Preserve the model-tree group boundaries (so attention
            # output and MLP down remain in their declared positions), and use
            # only explicit semantic tags to reorder siblings within a group.
            role_priority = {
                MODULE_TREE_FLAG_V: 0,
                MODULE_TREE_FLAG_Q: 1,
                MODULE_TREE_FLAG_K: 2,
                MODULE_TREE_FLAG_UP: 0,
                MODULE_TREE_FLAG_GATE: 1,
            }

            def priority(module_name: str) -> int:
                flags = gptq_model.get_module_tree_flags(module_name)
                priorities = [role_priority[flag] for flag in flags if flag in role_priority]
                return min(priorities, default=len(role_priority))

            ordered_groups = [sorted(group, key=priority) for group in groups]
        return [[module_name] for group in ordered_groups for module_name in group]

    def _yaqa_target_modules(
        self,
        gptq_model: BaseQModel,
    ) -> tuple[dict[str, torch.nn.Linear], list[Module]]:
        """Resolve exact lifecycle targets from the model's declared module tree."""

        layers, layer_names = get_layers_with_prefixes(gptq_model.model, gptq_model.extract_layers_node())
        if not layers:
            raise ValueError("QVQ YAQA requires at least one decoder layer.")
        layer_module_groups = gptq_model.simple_layer_modules(
            model_config=gptq_model.model.config,
            quantize_config=self.qcfg,
            is_awq_quantize=False,
            include_capture_only=False,
        )
        targets: dict[str, torch.nn.Linear] = {}
        target_ids: set[int] = set()
        for layer_index, (layer, layer_name) in enumerate(zip(layers, layer_names)):
            if not gptq_model.should_quantize_layer(
                layer,
                layer_name,
                layer_index,
                self.qcfg,
            ):
                continue
            full = find_modules(layer, layers=[torch.nn.Linear, transformers.Conv1D])
            for group in layer_module_groups:
                for token in group:
                    name, flags = BaseQModel._parse_module_flags(token)
                    if "!" in flags or "?" in flags or name not in full:
                        continue
                    full_name = f"{layer_name}.{name}"
                    if clone_qvq_config_for_module(self.qcfg, full_name) is None:
                        continue
                    module = full[name]
                    if not gptq_model.should_quantize_module(
                        gptq_model.model,
                        full_name,
                        module,
                        self.qcfg,
                    ):
                        continue
                    if not isinstance(module, torch.nn.Linear):
                        raise NotImplementedError(
                            "QVQ YAQA full-model Fisher collection currently supports nn.Linear targets only; "
                            f"module `{full_name}` is {module.__class__.__name__}."
                        )
                    if full_name in targets or id(module) in target_ids:
                        raise ValueError(f"QVQ YAQA target module `{full_name}` is duplicated or shared.")
                    targets[full_name] = module
                    target_ids.add(id(module))
        if not targets:
            raise ValueError("QVQ YAQA module-tree selection produced no quantization targets.")
        return targets, layers

    def prepare_yaqa(self, gptq_model: BaseQModel) -> None:
        """Collect immutable full-model Sketch-B factors before any layer is quantized."""

        if self._output_alignment is not None:
            self._output_alignment.bind_model(gptq_model)

        if self.qcfg.rounding != "yaqa":
            return
        if self._yaqa_prepared:
            raise RuntimeError("QVQ YAQA full-model Fisher factors were already prepared.")
        if getattr(gptq_model, "turtle_model", None) is not None:
            raise RuntimeError(
                "QVQ YAQA requires a directly loaded dense source model; reload with `offload_to_disk=False` "
                "because an exact full-model backward cannot traverse a LazyTurtle meta shell."
            )
        if any(isinstance(module, BaseQuantLinear) for module in gptq_model.model.modules()):
            raise NotImplementedError(
                "QVQ YAQA requires an entirely dense source model for the exact full-model backward; "
                "incremental YAQA requantization after installing quantized layers is not supported."
            )
        get_moe_module_name = getattr(gptq_model, "get_moe_module_name", None)
        if callable(get_moe_module_name) and get_moe_module_name():
            raise NotImplementedError(
                "QVQ YAQA full-model Fisher collection is not yet defined for conditionally routed MoE experts; "
                "use a dense model for the W1/W1.5 YAQA gate."
            )
        model = gptq_model.model
        named_tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
        meta_names = [name for name, tensor in named_tensors if tensor.device.type == "meta"]
        if meta_names:
            raise RuntimeError(
                "QVQ YAQA requires every source tensor to be materialized; found meta tensors including "
                f"{meta_names[:3]}. Reload with `offload_to_disk=False`."
            )
        source_devices = {tensor.device for _, tensor in named_tensors}
        if len(source_devices) != 1:
            raise RuntimeError(
                "QVQ YAQA currently requires the dense source model on one device before its full-model backward; "
                f"found devices {sorted(map(str, source_devices))}."
            )
        source_device = next(iter(source_devices))
        target_device = normalize_device_like(self.qcfg.device) or source_device
        targets, decoder_layers = self._yaqa_target_modules(gptq_model)
        log.info(
            "QVQ YAQA: collecting full-model Sketch-B factors targets=%d batches=%d device=%s seed=%d "
            "minimum_sequences=%d regularization=%.6g activation_checkpointing=true checkpointed_modules=%d",
            len(targets),
            len(self.yaqa_calibration),
            target_device,
            self.qcfg.yaqa.seed,
            self.qcfg.yaqa.minimum_sequences,
            self.qcfg.yaqa.regularization,
            len(decoder_layers),
        )
        moved = source_device != target_device
        try:
            if moved:
                model.to(target_device)
            with torch.inference_mode(False), torch.enable_grad():
                input_hessians, output_hessians, stats = capture_yaqa_sketch_b(
                    model,
                    self.yaqa_calibration,
                    targets,
                    device=target_device,
                    seed=self.qcfg.yaqa.seed,
                    minimum_sequences=self.qcfg.yaqa.minimum_sequences,
                    first_decoder_layer=decoder_layers[0],
                    checkpoint_modules=decoder_layers,
                )
        finally:
            if moved:
                model.to(source_device)
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
        self._yaqa_input_hessians = input_hessians
        self._yaqa_output_hessians = output_hessians
        self._yaqa_stats = stats
        self._yaqa_prepared = True
        log.info(
            "QVQ YAQA: collected independent_sequences=%d valid_output_samples=%d factor_storage_bytes=%d",
            stats["independent_sequences"],
            stats["valid_output_samples"],
            stats["factor_storage_bytes"],
        )

    def set_calibration_dataset(self, calibration_dataset):
        """Reject dataset replacement because QVQ capture is fixed at construction."""

        raise NotImplementedError("QVQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Build a masked input-Hessian capture task and effective QVQ config."""

        del fallback, kwargs
        module_qcfg = clone_qvq_config_for_module(self.qcfg, module.full_name)
        if module_qcfg is None:
            return
        automatic_propagation = (
            module_qcfg.propagated_bank_selection is None
            and module_qcfg.format == FORMAT.QVQ_V4
            and module_qcfg.bank_count == 4
            and module_qcfg.rounding == "block_ldlq"
        )
        if automatic_propagation:
            # Banked V4 quantization can use an automatic module-boundary
            # recovery gate. The stronger downstream/full-model refiner remains
            # a separate caller-supplied gate.
            module_qcfg.propagated_bank_selection = True
            module_qcfg._qvq_automatic_propagation = True
        if self._output_alignment is not None and module_tree_flags_are_moe(
            module.state.get("module_tree_flags", frozenset())
        ):
            raise NotImplementedError(
                "QVQ output alignment does not yet support MoE decoder layers; "
                "module-tree tags identified expert/router leaves."
            )
        if module_qcfg.rounding == "yaqa":
            if not self._yaqa_prepared:
                raise RuntimeError("QVQ YAQA full-model Fisher factors must be prepared before layer processing.")
            with self._yaqa_factor_lock:
                yaqa_input_hessian = self._yaqa_input_hessians.get(module.full_name)
                yaqa_output_hessian = self._yaqa_output_hessians.get(module.full_name)
            if yaqa_input_hessian is None or yaqa_output_hessian is None:
                raise RuntimeError(f"QVQ YAQA has no full-model Fisher factors for `{module.full_name}`.")
        else:
            yaqa_input_hessian = None
            yaqa_output_hessian = None

        capture_qcfg = GPTQConfig(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            device=module_qcfg.device,
            pack_dtype=torch.int32,
            act_group_aware=False,
            hessian=HessianConfig(length_aware=False),
        )
        capture = GPTQ(module=module, qcfg=capture_qcfg)
        capture.expected_nsamples = getattr(self, "total_calibration_tokens", None)
        capture.quantizer.configure(perchannel=True)
        self.tasks[module.name] = {
            "capture": capture,
            "qcfg": module_qcfg,
            "yaqa_input_hessian": yaqa_input_hessian,
            "yaqa_output_hessian": yaqa_output_hessian,
        }
        if self._output_alignment is not None:
            self._output_alignment.register_module(module)
            if self._output_alignment.config.pristine_hessian and module_qcfg.rounding != "yaqa":
                with self._pristine_hessian_lock:
                    self._pristine_hessian_modules.setdefault(module.layer_index, {})[module.name] = module

    def _new_pristine_hessian_capture(self, module: NamedModule, source: GPTQ) -> GPTQ:
        """Build an empty capture with the exact production Hessian settings."""

        capture = GPTQ(module=module, qcfg=copy.deepcopy(source.qcfg))
        capture.expected_nsamples = source.expected_nsamples
        capture.quantizer.configure(perchannel=True)
        return capture

    @contextmanager
    def pristine_quant_input_capture(self, *, layer_index: int):
        """Capture one layer's QVQ Hessians from its untouched dense replay.

        Temporary captures make the operation transactional: a replay failure
        restores every pre-existing hook and discards all partial Hessians.
        Successful captures replace the still-empty normal captures, and the
        subsequent noisy subset forwards are ignored for those exact tasks.
        """

        enabled = bool(
            self._output_alignment is not None
            and self._output_alignment.config.pristine_hessian
            and self.qcfg.rounding != "yaqa"
        )
        if not enabled:
            yield
            return

        with self._pristine_hessian_lock:
            modules = dict(self._pristine_hessian_modules.get(layer_index, {}))
            if self._active_pristine_hessian_captures:
                raise RuntimeError("QVQ pristine Hessian capture cannot overlap decoder layers.")
            temporary = {
                name: self._new_pristine_hessian_capture(module, self.tasks[name]["capture"])
                for name, module in modules.items()
            }
            self._active_pristine_hessian_captures = temporary

        installed = []
        committed = False
        try:
            for name, module in modules.items():
                hook = self.pre_process_fwd_hook(name)
                if hasattr(module.module, "forward_hook"):
                    installed.append(
                        (
                            "attribute",
                            module,
                            module.forward_hook,
                            module.forward_hook_last,
                        )
                    )
                    module.forward_hook = hook
                    module.forward_hook_last = False
                else:
                    installed.append(("handle", module.register_forward_hook(hook)))
            yield
            with self._pristine_hessian_lock:
                previous_captures = {
                    name: self.tasks[name]["capture"]
                    for name in temporary
                }
                for name, capture in temporary.items():
                    self.tasks[name]["capture"] = capture
                    self.tasks[name]["pristine_hessian_complete"] = True
                committed = True
            for previous in previous_captures.values():
                try:
                    previous.free()
                except BaseException:
                    log.warning("QVQ could not release a superseded noisy Hessian capture.", exc_info=True)
        finally:
            for entry in reversed(installed):
                if entry[0] == "attribute":
                    _, module, previous_hook, previous_last = entry
                    module.forward_hook = previous_hook
                    module.forward_hook_last = previous_last
                else:
                    entry[1].remove()
            with self._pristine_hessian_lock:
                self._active_pristine_hessian_captures = {}
                self._pristine_hessian_modules.pop(layer_index, None)
            if not committed:
                # The contextmanager necessarily propagates the replay error
                # after this cleanup, so coverage.py cannot observe a normal
                # generator-exit edge from this loop.
                for capture in temporary.values():  # pragma: no branch
                    capture.free()

    def uses_grouped_optimization(self) -> bool:
        """Request one pristine layer replay only when output alignment is enabled."""

        return self._output_alignment is not None

    def needs_pristine_layer_clone(self) -> bool:
        """Alignment works transactionally on the live layer and does not clone it."""

        return False

    def receive_pristine_layer_module(self, *, layer_index: int, layer_module: Module) -> None:
        """Delegate the generic pristine-layer attachment point."""

        if self._output_alignment is not None:
            self._output_alignment.receive_pristine_layer_module(
                layer_index=layer_index,
                layer_module=layer_module,
            )

    def clean_group_layer_inputs(self, *, layer_index: int, layer_inputs):
        """Select QTIP's pristine per-layer input stream for output alignment."""

        if self._output_alignment is None:
            return layer_inputs
        return self._output_alignment.clean_group_layer_inputs(
            layer_index=layer_index,
            layer_inputs=layer_inputs,
        )

    def receive_clean_layer_inputs(self, *, layer_index: int, layer_inputs) -> None:
        """Advance the attachment-owned pristine stream to the next layer."""

        if self._output_alignment is not None:
            self._output_alignment.receive_clean_layer_inputs(
                layer_index=layer_index,
                layer_inputs=layer_inputs,
            )

    def receive_layer_forward_context(
        self,
        *,
        layer_index: int,
        layer_inputs,
        layer_input_kwargs,
        layer_outputs,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Capture bounded CPU-owned replay data through the generic layer hook."""

        del subset_index, subset_total
        if self._output_alignment is None:
            return
        self._output_alignment.receive_layer_forward_context(
            layer_index=layer_index,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            layer_outputs=layer_outputs,
            position_ids=list(self.inputs_cache.position_ids or []),
            attention_masks=list(self.inputs_cache.attention_masks or []),
        )

    def cleanup_subset(
        self,
        subset: Optional[Dict[str, NamedModule]] = None,
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Align once, after every worker in the final decoder-layer subset finishes."""

        if self._output_alignment is None or not subset:
            return
        if subset_index is None or subset_total is None:
            return
        layer_indices = {module.layer_index for module in subset.values()}
        if len(layer_indices) != 1:
            raise RuntimeError("QVQ output alignment cleanup received modules from multiple layers.")
        layer_index = layer_indices.pop()
        # cleanup_subset() is a finally hook. Never start alignment or mask the
        # original quantization error when a worker left an incomplete codec.
        if not self._output_alignment.modules_are_fully_staged(layer_index, set(subset)):
            self._output_alignment.discard_layer(layer_index)
            return
        final_subset = subset_index + 1 == subset_total
        alignment_device = get_device(next(iter(subset.values())).module)
        if alignment_device.type == "cuda":
            result = DEVICE_THREAD_POOL.do(
                alignment_device,
                self._output_alignment.align_layer,
                layer_index,
                finalize=final_subset,
            )
        else:
            result = self._output_alignment.align_layer(layer_index, finalize=final_subset)
        if result is not None:
            with self._stats_lock:
                self._output_alignment_stats.setdefault(layer_index, []).append(result)
                for stat in self.log:
                    if stat.get(PROCESS_LOG_LAYER) == layer_index and stat.get(PROCESS_LOG_MODULE) in subset:
                        stat.update(
                            {
                                f"output_alignment_{key}": value
                                for key, value in result.items()
                            }
                        )

    def is_skipped(self, module: NamedModule) -> bool:
        """Return whether dynamic rules omitted this module from QVQ work."""

        return self.tasks.get(module.name, False) is False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Return a hook that excludes padded positions before Hessian accumulation."""

        def capture_input(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            del module
            task_entry = self.tasks[name]
            with self._pristine_hessian_lock:
                task = self._active_pristine_hessian_captures.get(name)
                if task is None and task_entry.get("pristine_hessian_complete", False):
                    return
            if task is None:
                task = task_entry["capture"]
            source = inp[0]
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
            prepared_source = source
            prepared_output = out if torch.is_tensor(out) else None
            if (
                torch.is_tensor(keep_mask)
                and torch.is_tensor(source)
                and source.dim() >= 3
                and keep_mask.ndim == 2
                and keep_mask.shape[:2] == source.shape[:2]
            ):
                prepared_source = apply_keep_mask_bt(source, keep_mask)
                if torch.is_tensor(prepared_output) and prepared_output.shape[:2] == source.shape[:2]:
                    prepared_output = apply_keep_mask_bt(prepared_output, keep_mask)
            capture_source = prepared_source.data
            capture_output = None if prepared_output is None else prepared_output.data
            qcfg = task_entry["qcfg"]
            if (
                qcfg.propagated_bank_selection
                and qcfg.format != FORMAT.QVQ_V2B2_P32
                and task is task_entry["capture"]
                and not self._has_propagation_gate(name)
                and capture_output is not None
                and capture_source.ndim >= 2
                and capture_output.ndim >= 2
            ):
                source_2d = capture_source.reshape(-1, capture_source.shape[-1])
                output_2d = capture_output.reshape(-1, capture_output.shape[-1])
                if source_2d.shape[0] == output_2d.shape[0] and source_2d.shape[0] >= 2:
                    heldout_rows = max(1, source_2d.shape[0] // 8)
                    train_rows = source_2d.shape[0] - heldout_rows
                    with self._propagation_gates_lock:
                        self._additional_calibration_sample_counts.setdefault(name, set()).update(
                            {int(source_2d.shape[0]), int(train_rows)}
                        )
                    self._record_automatic_propagation_gate(
                        name, source_2d[train_rows:], output_2d[train_rows:]
                    )
                    capture_source = source_2d[:train_rows]
                    capture_output = output_2d[:train_rows]
            task.add_batch(capture_source, capture_output, batch_index=self.current_batch_index())

        return capture_input

    @staticmethod
    def _restore_module_weight(module: NamedModule, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Map canonical QVQ [out, in] reconstruction back to the wrapped module layout."""

        target = module.module
        if isinstance(target, transformers.Conv1D):
            return quantized_weight.t().contiguous().view_as(target.weight.data)
        if isinstance(target, torch.nn.Linear):
            return quantized_weight.contiguous().view_as(target.weight.data)
        raise NotImplementedError(f"Unsupported QVQ module type: {target.__class__.__name__}")

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Quantize one module and stage its exact serialized QVQ tensors on CPU."""

        del subset, previous_subset, subset_index, subset_total
        self.draw_progress(f"Quantizing {module.name} in layer")
        task_entry = self.tasks[module.name]
        capture: GPTQ = task_entry["capture"]
        module_qcfg: QVQConfig = task_entry["qcfg"]
        target_device = torch.device(device or get_device(module.module))

        started = time.perf_counter()
        canonical_weight = None
        propagation_gate = None
        try:
            output_hessian = None
            if module_qcfg.rounding == "yaqa":
                quantization_hessian = task_entry["yaqa_input_hessian"]
                output_hessian = task_entry["yaqa_output_hessian"]
                if quantization_hessian is None or output_hessian is None:
                    raise RuntimeError(f"QVQ YAQA factors disappeared for module `{module.full_name}`.")
            else:
                capture.finalize_hessian(target_device=target_device)
                if capture.H is None:
                    raise RuntimeError(f"QVQ failed to capture Hessian for module `{module.full_name}`.")
                quantization_hessian = capture.H
            if capture.nsamples <= 0:
                raise RuntimeError(f"QVQ captured no calibration activations for module `{module.full_name}`.")
            self._assert_calibration_sample_count(module.name, capture.nsamples)

            canonical_weight = capture.clone_module(copy=True, device=target_device)
            seed = zlib.crc32(module.full_name.encode("utf-8")) & 0x7FFFFFFF
            damp_percent = module_qcfg.yaqa.regularization if module_qcfg.rounding == "yaqa" else 0.01
            telemetry = (
                QVQQuantizationTelemetry()
                if os.environ.get("GPTQMODEL_QVQ_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
                else None
            )
            if module_qcfg.propagated_bank_selection:
                self._materialize_automatic_propagation_gate(
                    module.full_name,
                    None if module.bias is None else module.bias,
                )
                if getattr(module_qcfg, "_qvq_automatic_propagation", False) and not self._has_propagation_gate(
                    module.full_name
                ):
                    # Tiny/fully-masked calibration batches cannot provide a
                    # disjoint held-out row. Preserve the exact local path
                    # instead of failing late after Hessian capture.
                    module_qcfg.propagated_bank_selection = False
                else:
                    propagation_gate = self._require_propagation_gate(module.full_name, target_device)
            else:
                propagation_gate = None
            result = quantize_qvq_linear(
                canonical_weight,
                quantization_hessian,
                bits=module_qcfg.bits,
                output_hessian=output_hessian,
                bias=None if module.bias is None else module.bias.detach().to(target_device),
                seed=seed,
                damp_percent=damp_percent,
                codebook_version=module_qcfg.codebook,
                vector_size=module_qcfg.vector_size,
                trellis_window=module_qcfg.trellis_window,
                dual_v2=module_qcfg.format == FORMAT.QVQ_DUAL_V2,
                v2b4_p64=module_qcfg.format == FORMAT.QVQ_V2B4_P64,
                v2b2_p32=module_qcfg.format == FORMAT.QVQ_V2B2_P32,
                module_scale_search=module_qcfg.module_scale_search,
                output_channel_scale_optimization=module_qcfg.output_channel_scale_optimization,
                viterbi_objective=module_qcfg.viterbi_objective,
                tail_biting_candidates=module_qcfg.tail_biting_candidates,
                rounding=module_qcfg.rounding,
                yaqa_v2b2_family_mode=module_qcfg.yaqa.v2b2_family_mode,
                yaqa_spectral_refinement=module_qcfg.yaqa.spectral_refinement,
                yaqa_spectral_ranks=module_qcfg.yaqa.spectral_ranks,
                yaqa_spectral_lambdas=module_qcfg.yaqa.spectral_lambdas,
                yaqa_spectral_push=module_qcfg.yaqa.spectral_push,
                yaqa_spectral_push_alphas=module_qcfg.yaqa.spectral_push_alphas,
                yaqa_spectral_localized=module_qcfg.yaqa.spectral_localized,
                yaqa_spectral_localized_alphas=module_qcfg.yaqa.spectral_localized_alphas,
                yaqa_spectral_localized_max_segments=module_qcfg.yaqa.spectral_localized_max_segments,
                yaqa_spectral_localized_max_changes=module_qcfg.yaqa.spectral_localized_max_changes,
                yaqa_spectral_localized_replay_candidates=module_qcfg.yaqa.spectral_localized_replay_candidates,
                viterbi_minimum_proxy_improvement=module_qcfg.viterbi_minimum_proxy_improvement,
                telemetry=telemetry,
                bank_count=module_qcfg.bank_count,
                propagated_inputs=None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[0],
                propagated_target_output=None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[1],
                propagated_acceptance=None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[2],
                propagated_candidate_score=(
                    None
                    if propagation_gate is None or not module_qcfg.propagated_bank_selection
                    else propagation_gate[3]
                ),
            )
            duration = time.perf_counter() - started

            # Quantized replay temporarily overwrites the dense module. The
            # rollback image must be complete before the asynchronous payload
            # transfer starts: a failed stream event cannot itself be trusted
            # to have produced a valid host backup.
            original_weight = module.weight.detach().to(device="cpu", copy=True)
            with parent_module_lock(module.full_name):
                module.state["_qvq_original_weight"] = original_weight
            module.stream_state_payload_to_cpu(result.serialized_tensors())
            # LoopProcessor clears its per-subset task map before StageLayer drains
            # concurrent submodule finalizers. Keep the immutable decoder contract
            # with the module payload whose lifetime spans that hand-off.
            with parent_module_lock(module.full_name):
                module.state["_qvq_runtime_config"] = (
                    module_qcfg.bits,
                    module_qcfg.codebook,
                    module_qcfg.vector_size,
                    module_qcfg.bank_count,
                    module_qcfg.trellis_window,
                    module_qcfg.format == FORMAT.QVQ_DUAL_V2,
                    module_qcfg.format == FORMAT.QVQ_V2B4_P64,
                    module_qcfg.format == FORMAT.QVQ_V2B2_P32,
                )
            restored_weight = self._restore_module_weight(module, result.weight)
            module.weight.data = restored_weight.to(dtype=module.weight.dtype)
            loss = float(result.proxy_loss.item())
            selector_entropy = None
            selector_nonzero_fraction = None
            if result.bank_ids is not None and result.bank_ids.numel() > 0:
                selector_counts = torch.bincount(
                    result.bank_ids.to(device="cpu", dtype=torch.long),
                    minlength=module_qcfg.bank_count,
                ).to(torch.float64)
                selector_probabilities = selector_counts / selector_counts.sum()
                positive_probabilities = selector_probabilities[selector_probabilities > 0]
                selector_entropy = float(
                    -(positive_probabilities * torch.log2(positive_probabilities)).sum().item()
                )
                selector_nonzero_fraction = float((1.0 - selector_probabilities[0]).item())
            stat = {
                PROCESS_LOG_NAME: self.name(),
                PROCESS_LOG_LAYER: module.layer_index,
                PROCESS_LOG_MODULE: module.name,
                "full_name": module.full_name,
                MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
                DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
                QUANT_LOG_LOSS: f"{loss:.10f}",
                QUANT_LOG_NSAMPLES: str(capture.nsamples),
                QUANT_LOG_DAMP: f"{damp_percent:.5g}",
                PROCESS_LOG_TIME: f"{duration:.3f}",
                PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
                PROCESS_USED_MEMORY: self.device_memory_report(),
                "bits": module_qcfg.bits,
                "codebook": module_qcfg.codebook,
                "module_scale_search_selected": getattr(result, "module_scale_search_selected", False),
                "module_scale_multiplier": getattr(result, "module_scale_multiplier", 1.0),
                "module_scale_reencoded": getattr(result, "module_scale_reencoded", False),
                "output_scale_optimized_channels": result.output_scale_optimized_channels,
                "hessian_viterbi_selected": result.hessian_viterbi_selected,
                "rounding": result.rounding,
                "yaqa_independent_sequences": self._yaqa_stats.get("independent_sequences"),
                "yaqa_minimum_sequences": self._yaqa_stats.get("minimum_sequences"),
                "yaqa_sequence_loss_reduction": self._yaqa_stats.get("sequence_loss_reduction"),
                "yaqa_activation_checkpointing": self._yaqa_stats.get("activation_checkpointing"),
                "yaqa_checkpointed_modules": self._yaqa_stats.get("checkpointed_modules"),
                "yaqa_v2b2_family_mode": (
                    module_qcfg.yaqa.v2b2_family_mode
                    if module_qcfg.rounding == "yaqa" and module_qcfg.format == FORMAT.QVQ_V2B2_P32
                    else None
                ),
                "yaqa_bank_fallback_to_v2": result.yaqa_bank_fallback_to_v2,
                "yaqa_selector_churn": result.yaqa_selector_churn,
                "yaqa_family_changed": result.yaqa_family_changed,
                "yaqa_block_family_id": result.yaqa_block_family_id,
                "yaqa_spectral_selected": result.yaqa_spectral_selected,
                "yaqa_spectral_method": result.yaqa_spectral_method,
                "yaqa_spectral_rank": result.yaqa_spectral_rank,
                "yaqa_spectral_lambda": result.yaqa_spectral_lambda,
                "yaqa_spectral_alpha": result.yaqa_spectral_alpha,
                "yaqa_spectral_svd_device": result.yaqa_spectral_svd_device,
                "yaqa_spectral_concentration": result.yaqa_spectral_concentration,
                "yaqa_spectral_oracle_losses": result.yaqa_spectral_oracle_losses,
                "yaqa_spectral_candidates": result.yaqa_spectral_candidates,
                "yaqa_spectral_absorption_efficiency": result.yaqa_spectral_absorption_efficiency,
                "yaqa_spectral_selector_churn": result.yaqa_spectral_selector_churn,
                "yaqa_spectral_family_changed": result.yaqa_spectral_family_changed,
                "bank_selected_family_id": (
                    None if result.bank_alt_id is None else int(result.bank_alt_id.reshape(-1)[0].item())
                ),
                "bank_selector_entropy_bits": selector_entropy,
                "bank_selector_nonzero_fraction": selector_nonzero_fraction,
                "yaqa_kronecker_proxy_loss": (
                    None if result.kronecker_proxy_loss is None else float(result.kronecker_proxy_loss.item())
                ),
            }
            if result.telemetry is not None:
                stat["qvq_telemetry"] = result.telemetry
                log.info(
                    "QVQ telemetry module=%s bits=%s vector_size=%d %s",
                    module.full_name,
                    module_qcfg.bits,
                    module_qcfg.vector_size,
                    json.dumps(result.telemetry, sort_keys=True),
                )
            if self.qcfg.dynamic is not None:
                stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)
            with self._stats_lock:
                self.durations.append(duration)
                self.avg_losses.append(loss)
                self.module_names.append(f"layer-{module.layer_index}-{module.name}")
                self.log.append(stat)
            self.log_new_row(stat)
        except BaseException:
            # A failed asynchronous host transfer may still own result tensors.
            # Drain it before removing staged state, but never replace the
            # original quantization exception with a cleanup failure.
            try:
                module.stream_sync()
            except BaseException:
                pass
            with parent_module_lock(module.full_name):
                for key in (
                    "trellis",
                    "SU",
                    "SV",
                    "bias",
                    "bank_ids",
                    "bank_alt_id",
                    "_qvq_original_weight",
                    "_qvq_runtime_config",
                ):
                    module.state.pop(key, None)
            if canonical_weight is not None:
                try:
                    original_weight = self._restore_module_weight(module, canonical_weight)
                    module.weight.data = original_weight.to(device=module.weight.device, dtype=module.weight.dtype)
                except BaseException:
                    pass
            raise
        finally:
            # Gate tensors are module-scoped resources. Release both the
            # retained host snapshot and any device copy as soon as this
            # module finishes, including failure paths.
            if propagation_gate is not None:
                self._pop_propagation_gate(module.full_name, propagation_gate[4])
            capture.free()
            if module_qcfg.rounding == "yaqa":
                task_entry.pop("yaqa_input_hessian", None)
                task_entry.pop("yaqa_output_hessian", None)
                with self._yaqa_factor_lock:
                    self._yaqa_input_hessians.pop(module.full_name, None)
                    self._yaqa_output_hessians.pop(module.full_name, None)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Replace the replayed dense leaf with its serialized QVQ runtime module."""

        del kwargs
        original = module.module
        with parent_module_lock(module.full_name):
            original_weight = module.state["_qvq_original_weight"].clone()

        transfer_complete = False
        try:
            module.stream_sync()
            transfer_complete = True
            tensors = {}
            with parent_module_lock(module.full_name):
                runtime_config = module.state["_qvq_runtime_config"]
                bits, codebook = runtime_config[:2]
                vector_size = runtime_config[2] if len(runtime_config) > 2 else 2
                bank_count = runtime_config[3] if len(runtime_config) > 3 else 1
                trellis_window = runtime_config[4] if len(runtime_config) > 4 else 16
                dual_v2 = runtime_config[5] if len(runtime_config) > 5 else False
                v2b4_p64 = runtime_config[6] if len(runtime_config) > 6 else False
                v2b2_p32 = runtime_config[7] if len(runtime_config) > 7 else False
                for tensor_name in ("trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id"):
                    tensor = module.state.get(tensor_name)
                    if tensor is not None:
                        tensors[tensor_name] = tensor.clone()

            if isinstance(original, torch.nn.Linear):
                in_features, out_features = original.in_features, original.out_features
            elif isinstance(original, transformers.Conv1D):
                in_features, out_features = original.weight.shape
            else:
                raise NotImplementedError(f"Unsupported QVQ module type: {original.__class__.__name__}")

            qmodule = QVQLinear(
                bits=bits,
                in_features=in_features,
                out_features=out_features,
                bias="bias" in tensors,
                backend=BACKEND.QVQ,
                name=module.full_name,
                dtype=module.module_dtype,
                tensors=tensors,
                codebook_version=codebook,
                vector_size=vector_size,
                trellis_window=trellis_window,
                bank_count=bank_count,
                dual_v2=dual_v2,
                v2b4_p64=v2b4_p64,
                v2b2_p32=v2b2_p32,
            )
            # Materialized layer leaves may be freshly constructed with
            # ``training=True`` even while the authoritative model is in eval
            # mode. Inheriting that stale leaf flag makes the live QVQ module
            # use its dense training reference while reload uses native eval
            # inference, breaking exact live/reload parity.
            qmodule.train(model.model.training)
            with parent_module_lock(module.full_name):
                recurse_setattr(model.model, module.full_name, qmodule)
            # Match the loader lifecycle: initialize backend state only after
            # the fully populated module is installed at its final path.
            qmodule.post_init()
        except BaseException:
            with parent_module_lock(module.full_name):
                original.weight.data.copy_(
                    original_weight.to(device=original.weight.device, dtype=original.weight.dtype)
                )
                recurse_setattr(model.model, module.full_name, original)
                if not transfer_complete:
                    # A failed event makes the asynchronous payload unusable.
                    # Restore the dense module and discard it rather than allow
                    # a retry to install potentially incomplete tensors.
                    for key in (
                        "trellis",
                        "SU",
                        "SV",
                        "bias",
                        "bank_ids",
                        "bank_alt_id",
                        "_qvq_original_weight",
                        "_qvq_runtime_config",
                    ):
                        module.state.pop(key, None)
            raise
        with parent_module_lock(module.full_name):
            module.state.pop("w", None)
            for key in (
                "trellis",
                "SU",
                "SV",
                "bias",
                "bank_ids",
                "bank_alt_id",
                "_qvq_original_weight",
                "_qvq_runtime_config",
            ):
                module.state.pop(key, None)
        module.unregister_parameter("weight")
        if getattr(module, "bias", None) is not None:
            module.unregister_parameter("bias")
        return qmodule

    def finalize(self, model: BaseQModel, **kwargs):
        """Mark the model and checkpoint metadata as QVQ after replacement completes."""

        with self._yaqa_factor_lock:
            self._yaqa_input_hessians.clear()
            self._yaqa_output_hessians.clear()
        with self._propagation_gates_lock:
            self._propagation_gates.clear()
            self._automatic_propagation_gate_samples.clear()
            self._additional_calibration_sample_counts.clear()
        if self._output_alignment is not None:
            self._output_alignment.clear()
        model.quantized = True
        model.quantize_config.method = METHOD.QVQ
        # Preserve the selected QVQ codec format so V4 artifacts remain
        # distinguishable from the default V2 format after lifecycle finalization.
        if model.quantize_config.format == FORMAT.QVQ_V2B2_P32:
            model.quantize_config.format = FORMAT.QVQ_V2B2_P32
        elif model.quantize_config.format == FORMAT.QVQ_V2B4_P64:
            model.quantize_config.format = FORMAT.QVQ_V2B4_P64
        elif model.quantize_config.format == FORMAT.QVQ_DUAL_V2:
            model.quantize_config.format = FORMAT.QVQ_DUAL_V2
        elif model.quantize_config.trellis_window == 18:
            model.quantize_config.format = FORMAT.QVQ_V4_L18
        elif model.quantize_config.vector_size == 4:
            model.quantize_config.format = FORMAT.QVQ_V4
        else:
            model.quantize_config.format = FORMAT.QVQ
        model.qlinear_kernel = QVQLinear
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Require calibration data before entering the QVQ lifecycle."""

        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("QVQProcessor's calibration_dataset must be provided.")
        return True

    def name(self) -> str:
        """Return the processor label used by lifecycle telemetry."""

        return "qvq"


__all__ = ["QVQProcessor", "clone_qvq_config_for_module"]
