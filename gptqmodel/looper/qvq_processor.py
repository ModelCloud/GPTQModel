# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QVQ calibration, quantization, replacement, and checkpoint lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import zlib
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    ExecutionConfig,
    LoopProcessor,
)
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
from ..quantization.swiglu import (
    apply_swiglu_reparameterization,
    choose_swiglu_scales,
    select_swiglu_candidate_triplet,
)
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
    """Clone QVQ config, apply supported dynamic overrides, or skip the module."""

    dynamic_overrides = qcfg.dynamic_get(layer_name=module_full_name)
    if dynamic_overrides is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)
    if dynamic_overrides:
        qcfg_clone.bits = dynamic_overrides.get("bits", qcfg_clone.bits)
    if qcfg_clone.rounding == "yaqa":
        # Materialize the effective rate-specific damping on the per-module
        # clone consumed by ``process``.  Keeping this only in
        # ``regularization_by_rate`` made every solve use the global fallback
        # and caused nominal W2 damping sweeps to produce identical payloads.
        qcfg_clone.yaqa.regularization = qcfg_clone.yaqa.regularization_for_rate(
            qcfg_clone.bits
        )
        if dynamic_overrides and "yaqa_regularization" in dynamic_overrides:
            qcfg_clone.yaqa.regularization = dynamic_overrides["yaqa_regularization"]
            qcfg_clone.yaqa.__post_init__()
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
        module_replay_search_calibration=None,
        module_replay_confirmation_calibration=None,
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
        self._module_replay_search_calibration = module_replay_search_calibration
        self._module_replay_confirmation_calibration = module_replay_confirmation_calibration
        self._module_replay_model: Optional[BaseQModel] = None
        self._module_replay_rows: Dict[str, list[Dict[str, torch.Tensor]]] = {}
        self._module_replay_teacher_logits: Dict[str, list[torch.Tensor]] = {}
        self._module_replay_stats: Dict[str, Dict[str, Any]] = {}
        self._module_replay_lock = threading.RLock()
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
        self._smooth_swiglu_stats: Dict[str, Dict[str, Any]] = {}
        self._smooth_swiglu_prepared = False
        self._atomic_swiglu_inputs: Dict[str, torch.Tensor] = {}
        self._atomic_swiglu_candidates: Dict[str, Dict[str, Any]] = {}
        self._atomic_swiglu_lock = threading.RLock()

    @property
    def smooth_swiglu_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return JSON-safe statistics for the offline preconditioning pass."""

        return copy.deepcopy(self._smooth_swiglu_stats)

    def prepare_smooth_swiglu(self, gptq_model: BaseQModel) -> None:
        """Choose and fold Smooth-SwiGLU scales before Hessian/YAQA capture.

        The calibration stream is used only to estimate the nonlinear
        sensitivity.  All weights are transformed before any QVQ capture, so
        the down-projection Hessian sees the rescaled hidden state.  The
        transformation is exact in dense arithmetic and adds no checkpoint
        tensors or inference operations.
        """

        smooth_config = self.qcfg.smooth_swiglu
        replay_config = self.qcfg.module_granular_replay
        atomic_enabled = replay_config is not None and replay_config.strategy == "atomic_swiglu"
        if (smooth_config is None or not smooth_config.enabled) and not atomic_enabled:
            return
        if self._smooth_swiglu_prepared:
            raise RuntimeError("QVQ Smooth-SwiGLU preparation was already completed.")
        model = gptq_model.model
        if any(isinstance(module, BaseQuantLinear) for module in model.modules()):
            raise RuntimeError("QVQ Smooth-SwiGLU requires an entirely dense source model.")
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            raise RuntimeError("QVQ Smooth-SwiGLU requires model input embeddings.")
        mlps = {}
        for module_name, module in model.named_modules():
            if not module_name.endswith(".mlp"):
                continue
            gate = getattr(module, "gate_proj", None)
            up = getattr(module, "up_proj", None)
            down = getattr(module, "down_proj", None)
            if all(isinstance(item, torch.nn.Linear) for item in (gate, up, down)):
                mlps[module_name] = (module, gate, up, down)
        if not mlps:
            raise ValueError("QVQ Smooth-SwiGLU could not find Llama-style gate/up/down projection triples.")
        if not self.calibration_dataset:
            raise ValueError("QVQ Smooth-SwiGLU requires a nonempty calibration dataset.")

        max_tokens = 512 if smooth_config is None else smooth_config.max_calibration_tokens
        activations: Dict[str, list[torch.Tensor]] = {name: [] for name in mlps}
        retained_tokens = {name: 0 for name in mlps}
        mask_tls = threading.local()
        hooks = []

        def capture_mlp_input(module_name: str):
            def hook(module, inputs):
                del module
                if retained_tokens[module_name] >= max_tokens or not inputs:
                    return
                source = inputs[0]
                if not torch.is_tensor(source) or source.ndim < 2:
                    return
                source = source.detach()
                if source.ndim == 3:
                    keep_mask = getattr(mask_tls, "value", None)
                    if (
                        torch.is_tensor(keep_mask)
                        and keep_mask.ndim == 2
                        and keep_mask.shape == source.shape[:2]
                    ):
                        source = source[keep_mask.to(device=source.device, dtype=torch.bool)]
                    source = source.reshape(-1, source.shape[-1])
                else:
                    source = source.reshape(-1, source.shape[-1])
                remaining = max_tokens - retained_tokens[module_name]
                source = source[:remaining].to(device="cpu", dtype=torch.float32)
                activations[module_name].append(source)
                retained_tokens[module_name] += int(source.shape[0])

            return hook

        for module_name, (module, _, _, _) in mlps.items():
            hooks.append(module.register_forward_pre_hook(capture_mlp_input(module_name)))
        was_training = model.training
        model.eval()
        input_device = input_embeddings.weight.device
        try:
            with torch.no_grad():
                for row in self.calibration_dataset:
                    mask_tls.value = row.get("attention_mask")
                    model(
                        **self._module_replay_model_inputs(row, input_device),
                        use_cache=False,
                        return_dict=True,
                    )
                    if all(count >= max_tokens for count in retained_tokens.values()):
                        break
        finally:
            mask_tls.value = None
            for hook in hooks:
                hook.remove()
            model.train(was_training)

        for module_name, (_, gate, up, down) in mlps.items():
            if not activations[module_name]:
                raise RuntimeError(f"QVQ Smooth-SwiGLU captured no inputs for `{module_name}`.")
            inputs = torch.cat(activations[module_name], dim=0)
            if atomic_enabled:
                self._atomic_swiglu_inputs[module_name] = inputs.contiguous()
            if smooth_config is None or not smooth_config.enabled:
                continue
            if up.bias is not None:
                raise ValueError(
                    f"QVQ Smooth-SwiGLU requires a biasless `up_proj`; `{module_name}.up_proj` has a bias."
                )
            weight_device = gate.weight.device
            inputs = inputs.to(device=weight_device)
            scales, stats = choose_swiglu_scales(
                inputs,
                gate.weight,
                up.weight,
                down.weight,
                group_size=smooth_config.group_size,
                candidate_exponents=smooth_config.candidate_exponents,
                scale_min=smooth_config.scale_min,
                scale_max=smooth_config.scale_max,
            )
            with torch.no_grad():
                _, transformed_up, transformed_down = apply_swiglu_reparameterization(
                    gate.weight,
                    up.weight,
                    down.weight,
                    scales.to(device=gate.weight.device),
                )
                dense_gate = inputs.to(torch.float32) @ gate.weight.to(torch.float32).transpose(0, 1)
                dense_up = inputs.to(torch.float32) @ up.weight.to(torch.float32).transpose(0, 1)
                dense_output = (F.silu(dense_gate) * dense_up) @ down.weight.to(torch.float32).transpose(0, 1)
                transformed_up_output = inputs.to(torch.float32) @ transformed_up.to(torch.float32).transpose(0, 1)
                transformed_down_output = (
                    F.silu(dense_gate) * transformed_up_output
                ) @ transformed_down.to(torch.float32).transpose(0, 1)
                parity_error = (transformed_down_output - dense_output).to(torch.float32)
                parity_relative_l2 = parity_error.norm() / dense_output.norm().clamp_min(torch.finfo(torch.float32).eps)
                if not torch.isfinite(parity_error).all() or not torch.isfinite(parity_relative_l2):
                    raise RuntimeError(f"QVQ Smooth-SwiGLU produced non-finite dense parity for `{module_name}`.")
                stats["dense_parity_max_abs"] = float(parity_error.abs().max().item())
                stats["dense_parity_relative_l2"] = float(parity_relative_l2.item())
                parity_tolerance = smooth_config.dense_parity_relative_l2_tolerance
                if parity_relative_l2.item() > parity_tolerance:
                    raise RuntimeError(
                        f"QVQ Smooth-SwiGLU dense parity exceeded tolerance for `{module_name}`: "
                        f"relative_l2={parity_relative_l2.item():.8g}, tolerance={parity_tolerance:.8g}."
                    )
                up.weight.copy_(transformed_up)
                down.weight.copy_(transformed_down)
            stats["module"] = module_name
            stats["scale_search"] = "jacobian_proxy"
            self._smooth_swiglu_stats[module_name] = stats
        if smooth_config is not None and smooth_config.enabled:
            setattr(gptq_model, "qvq_smooth_swiglu_stats", self.smooth_swiglu_stats)
        self._smooth_swiglu_prepared = True
        log.info(
            "QVQ Smooth-SwiGLU: %s for %d MLPs using %d-token calibration slices",
            "folded scales" if smooth_config is not None and smooth_config.enabled else "captured atomic candidates",
            len(mlps),
            max(retained_tokens.values()),
        )

    @staticmethod
    def _module_replay_row_fingerprint(row: Dict[str, torch.Tensor]) -> str:
        input_ids = row.get("input_ids")
        if not torch.is_tensor(input_ids):
            raise ValueError("QVQ module replay rows must contain tensor `input_ids`.")
        payload = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _module_replay_model_inputs(row: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            name: value.to(device=device)
            for name, value in row.items()
            if torch.is_tensor(value) and name not in {"labels", "label"}
        }

    def _ensure_module_replay_residency(self, model: torch.nn.Module) -> torch.device:
        """Keep every tensor needed by final-logit replay on one device."""

        named_tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
        meta_names = [name for name, tensor in named_tensors if tensor.device.type == "meta"]
        if meta_names:
            raise RuntimeError(
                "QVQ module-granular replay requires a fully materialized model; "
                f"found meta tensors including {meta_names[:3]}."
            )
        source_devices = {tensor.device for _, tensor in named_tensors}
        if not source_devices:
            raise RuntimeError("QVQ module-granular replay found no model tensors to place.")
        target_device = normalize_device_like(self.qcfg.device) or next(iter(source_devices))
        if source_devices != {target_device}:
            # Stage finalization deliberately returns completed leaves to CPU.
            # Rehome once before the next module search so all candidates reuse
            # a fully resident model; never transfer layers inside each replay.
            model.to(target_device)
        return target_device

    def prepare_module_granular_replay(self, gptq_model: BaseQModel) -> None:
        """Cache exact dense FP32 logits for later propagated replay."""

        replay_config = self.qcfg.module_granular_replay
        if replay_config is None:
            return
        if self._module_replay_search_calibration is None or self._module_replay_confirmation_calibration is None:
            raise ValueError("QVQ module-granular replay requires explicit search and confirmation streams.")
        if self._module_replay_model is not None:
            raise RuntimeError("QVQ module-granular replay teacher targets were already prepared.")
        model = gptq_model.model
        if any(isinstance(module, BaseQuantLinear) for module in model.modules()):
            raise RuntimeError("QVQ module-granular replay teacher capture requires an entirely dense source model.")
        self._ensure_module_replay_residency(model)
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            raise RuntimeError("QVQ module-granular replay requires model input embeddings.")
        input_device = input_embeddings.weight.device
        streams = {
            "search": list(self._module_replay_search_calibration),
            "confirmation": list(self._module_replay_confirmation_calibration),
        }
        if len(streams["search"]) < replay_config.search_folds:
            raise ValueError("QVQ module replay search rows must cover every configured search fold.")
        fingerprints = {
            name: {self._module_replay_row_fingerprint(row) for row in rows}
            for name, rows in streams.items()
        }
        overlap = fingerprints["search"].intersection(fingerprints["confirmation"])
        if overlap:
            raise ValueError("QVQ module replay search and confirmation rows must be prompt-disjoint.")
        calibration_fingerprints = {
            self._module_replay_row_fingerprint(row) for row in self.calibration_dataset
        }
        yaqa_fingerprints = {
            self._module_replay_row_fingerprint(row) for row in self.yaqa_calibration
        }
        for split_name, split_fingerprints in fingerprints.items():
            if split_fingerprints.intersection(calibration_fingerprints):
                raise ValueError(
                    f"QVQ module replay {split_name} rows must be prompt-disjoint from ordinary calibration."
                )
            if split_fingerprints.intersection(yaqa_fingerprints):
                raise ValueError(f"QVQ module replay {split_name} rows must be prompt-disjoint from YAQA calibration.")

        was_training = model.training
        model.eval()
        teacher_logits: Dict[str, list[torch.Tensor]] = {"search": [], "confirmation": []}
        try:
            with torch.no_grad():
                for split_name, rows in streams.items():
                    for row in rows:
                        logits = model(
                            **self._module_replay_model_inputs(row, input_device),
                            use_cache=False,
                            return_dict=True,
                        ).logits[:, :-1]
                        teacher_logits[split_name].append(logits.detach().to(device="cpu", dtype=torch.float32))
        finally:
            model.train(was_training)
        self._module_replay_model = gptq_model
        self._module_replay_rows = {
            name: [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in row.items()}
                for row in rows
            ]
            for name, rows in streams.items()
        }
        self._module_replay_teacher_logits = teacher_logits

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

    def _module_replay_metrics(self, split_name: str, *, fold_index: int = 0, folds: int = 1) -> Dict[str, float]:
        """Replay one split and compare exact final logits against cached dense hidden states."""

        if self._module_replay_model is None:
            raise RuntimeError("QVQ module replay model is unavailable.")
        model = self._module_replay_model.model
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            raise RuntimeError("QVQ module replay requires input embeddings.")
        input_device = input_embeddings.weight.device
        rows = self._module_replay_rows[split_name][fold_index::folds]
        teacher_logits_rows = self._module_replay_teacher_logits[split_name][fold_index::folds]
        if not rows:
            raise ValueError("QVQ module replay produced an empty search fold.")
        totals = {"kl": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "tokens": 0}
        with torch.no_grad():
            for row, teacher_logits_cpu in zip(rows, teacher_logits_rows, strict=True):
                student_logits = model(**self._module_replay_model_inputs(row, input_device), use_cache=False).logits[
                    :, :-1
                ].to(torch.float32)
                teacher_logits = teacher_logits_cpu.to(device=student_logits.device)
                attention_mask = row.get("attention_mask")
                if attention_mask is None:
                    valid = torch.ones(teacher_logits.shape[:-1], dtype=torch.bool, device=student_logits.device)
                else:
                    attention_mask = attention_mask.to(device=student_logits.device, dtype=torch.bool)
                    # A next-token comparison is valid only when both the query
                    # position and its target position are real tokens. Requiring
                    # both sides excludes left/right padding boundaries.
                    valid = attention_mask[:, :-1] & attention_mask[:, 1:]
                teacher_logits = teacher_logits[valid]
                student_logits = student_logits[valid]
                if teacher_logits.numel() == 0:
                    continue
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                token_count = int(teacher_logits.shape[0])
                totals["kl"] += float(
                    F.kl_div(student_log_probs, teacher_log_probs, reduction="sum", log_target=True).item()
                )
                teacher_top10 = teacher_logits.topk(10, dim=-1).indices
                student_top10 = student_logits.topk(10, dim=-1).indices
                totals["top1"] += float((teacher_top10[..., 0] == student_top10[..., 0]).sum().item())
                for topn in (5, 10):
                    overlap = (
                        teacher_top10[..., :topn].unsqueeze(-1)
                        == student_top10[..., :topn].unsqueeze(-2)
                    ).any(dim=-1)
                    totals[f"top{topn}"] += float(overlap.sum().item()) / topn
                totals["tokens"] += token_count
        tokens = int(totals["tokens"])
        if tokens == 0:
            raise ValueError("QVQ module replay split contains no valid next-token positions.")
        return {
            "kl_forward": totals["kl"] / tokens,
            "top1_agreement": totals["top1"] / tokens,
            "top5_overlap": totals["top5"] / tokens,
            "top10_overlap": totals["top10"] / tokens,
            "tokens": tokens,
        }

    @staticmethod
    def _module_replay_score(candidate: list[Dict[str, float]], baseline: list[Dict[str, float]]) -> float:
        ratios = []
        for candidate_fold, baseline_fold in zip(candidate, baseline, strict=True):
            candidate_kl = candidate_fold["kl_forward"]
            baseline_kl = baseline_fold["kl_forward"]
            if not math.isfinite(candidate_kl) or not math.isfinite(baseline_kl) or baseline_kl <= 0:
                return math.inf
            ratios.append(candidate_kl / baseline_kl)
        return max(ratios)

    @staticmethod
    def _module_replay_confirmation_passes(
        candidate: Dict[str, float], baseline: Dict[str, float], config
    ) -> bool:
        required = baseline["kl_forward"] * config.minimum_relative_kl_improvement
        if not math.isfinite(candidate["kl_forward"]) or baseline["kl_forward"] - candidate["kl_forward"] < required:
            return False
        return all(
            candidate[metric] >= baseline[metric] - config.topn_regression_limit
            for metric in ("top1_agreement", "top5_overlap", "top10_overlap")
        )

    @staticmethod
    def _module_replay_qlinear_from_tensors(
        original: torch.nn.Linear,
        module_full_name: str,
        module_qcfg: QVQConfig,
        serialized_tensors: Dict[str, torch.Tensor],
    ) -> QVQLinear:
        tensors = {
            name: value.detach().to(device=original.weight.device)
            for name, value in serialized_tensors.items()
        }
        candidate = QVQLinear(
            bits=module_qcfg.bits,
            in_features=original.in_features,
            out_features=original.out_features,
            bias=original.bias is not None,
            backend=BACKEND.QVQ,
            name=module_full_name,
            dtype=original.weight.dtype,
            tensors=tensors,
            codebook_version=module_qcfg.codebook,
            vector_size=2,
            trellis_window=16,
            bank_count=2,
            v2b2_p32=True,
        ).eval()
        candidate.post_init()
        return candidate

    @staticmethod
    def _module_replay_qlinear(
        original: torch.nn.Linear,
        module_full_name: str,
        module_qcfg: QVQConfig,
        result,
    ) -> QVQLinear:
        return QVQProcessor._module_replay_qlinear_from_tensors(
            original,
            module_full_name,
            module_qcfg,
            result.serialized_tensors(),
        )

    def _select_module_granular_replay_candidate(
        self,
        module: NamedModule,
        module_qcfg: QVQConfig,
        canonical_weight: torch.Tensor,
        quantization_hessian: torch.Tensor,
        quantization_kwargs: Dict[str, Any],
    ):
        """Choose one complete serialized bank arm by live final-logit replay."""

        replay_config = module_qcfg.module_granular_replay
        if replay_config is None or not replay_config.includes_module(module.full_name):
            return quantize_qvq_linear(
                canonical_weight,
                quantization_hessian,
                bits=module_qcfg.bits,
                **quantization_kwargs,
            )
        if self._module_replay_model is None:
            raise RuntimeError("QVQ module-granular replay was enabled without prepared teacher targets.")
        quantization_kwargs = dict(quantization_kwargs)
        quantization_kwargs.pop("yaqa_v2b2_family_mode", None)
        quantization_kwargs.pop("yaqa_v2b2_fixed_family_id", None)
        # ``quantize_qvq_linear`` finalizes its telemetry collector before it
        # returns. Replay evaluates several complete candidates for one live
        # module, so sharing the caller's one-shot collector makes candidate
        # two fail while trying to record into an already-finalized object.
        # Give every candidate an independent collector and retain the one
        # attached to whichever complete result is ultimately selected.
        telemetry_enabled = quantization_kwargs.pop("telemetry", None) is not None
        model = self._module_replay_model.model
        self._ensure_module_replay_residency(model)
        original = model.get_submodule(module.full_name)
        if original is not module.module or not isinstance(original, torch.nn.Linear):
            raise RuntimeError(
                f"QVQ module replay target `{module.full_name}` is not the authoritative live dense linear."
            )
        candidate_results = {}
        candidate_records = []
        baseline_folds = None
        parent_name, _, child_name = module.full_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        with self._module_replay_lock:
            try:
                for alternative_bank_id in (0, *replay_config.alternative_bank_ids):
                    candidate_kwargs = dict(quantization_kwargs)
                    candidate_kwargs["telemetry"] = (
                        QVQQuantizationTelemetry() if telemetry_enabled else None
                    )
                    result = quantize_qvq_linear(
                        canonical_weight,
                        quantization_hessian,
                        bits=module_qcfg.bits,
                        yaqa_v2b2_family_mode="fixed_block_ldlq",
                        yaqa_v2b2_fixed_family_id=alternative_bank_id,
                        **candidate_kwargs,
                    )
                    candidate = self._module_replay_qlinear(
                        original,
                        module.full_name,
                        module_qcfg,
                        result,
                    )
                    setattr(parent, child_name, candidate)
                    folds = [
                        self._module_replay_metrics(
                            "search",
                            fold_index=fold_index,
                            folds=replay_config.search_folds,
                        )
                        for fold_index in range(replay_config.search_folds)
                    ]
                    if alternative_bank_id == 0:
                        baseline_folds = folds
                        score = 1.0
                    else:
                        assert baseline_folds is not None
                        score = self._module_replay_score(folds, baseline_folds)
                    candidate_results[alternative_bank_id] = result
                    candidate_records.append(
                        {
                            "alternative_bank_id": alternative_bank_id,
                            "score": score,
                            "fold_kl_forward": [fold["kl_forward"] for fold in folds],
                            "fold_metrics": folds,
                            "selected": False,
                        }
                    )
                    log.info(
                        "QVQ module replay: module=%s alternative_bank_id=%d score=%.8f fold_kl=%s",
                        module.full_name,
                        alternative_bank_id,
                        score,
                        [fold["kl_forward"] for fold in folds],
                    )
                    setattr(parent, child_name, original)

                eligible = [
                    record
                    for record in candidate_records
                    if record["alternative_bank_id"] != 0
                    and record["score"] < 1 - replay_config.minimum_relative_kl_improvement
                ]
                selected_id = (
                    int(min(eligible, key=lambda record: record["score"])["alternative_bank_id"])
                    if eligible
                    else 0
                )
                baseline_candidate = self._module_replay_qlinear(
                    original,
                    module.full_name,
                    module_qcfg,
                    candidate_results[0],
                )
                setattr(parent, child_name, baseline_candidate)
                baseline_confirmation = self._module_replay_metrics("confirmation")
                candidate_confirmation = baseline_confirmation
                if selected_id:
                    selected_candidate = self._module_replay_qlinear(
                        original,
                        module.full_name,
                        module_qcfg,
                        candidate_results[selected_id],
                    )
                    setattr(parent, child_name, selected_candidate)
                    candidate_confirmation = self._module_replay_metrics("confirmation")
                    if not self._module_replay_confirmation_passes(
                        candidate_confirmation,
                        baseline_confirmation,
                        replay_config,
                    ):
                        selected_id = 0
                selected_result = candidate_results[selected_id]
                for record in candidate_records:
                    record["selected"] = record["alternative_bank_id"] == selected_id
                self._module_replay_stats[module.full_name] = {
                    "selected_alternative_bank_id": selected_id,
                    "candidates": candidate_records,
                    "confirmation_baseline": baseline_confirmation,
                    "confirmation_candidate": candidate_confirmation,
                }
                log.info(
                    "QVQ module replay: module=%s selected_alternative_bank_id=%d "
                    "confirmation_baseline_kl=%.8g confirmation_candidate_kl=%.8g",
                    module.full_name,
                    selected_id,
                    baseline_confirmation["kl_forward"],
                    candidate_confirmation["kl_forward"],
                )
                return selected_result
            finally:
                setattr(parent, child_name, original)

    def _is_atomic_swiglu_module(
        self,
        module: NamedModule,
        subset: Optional[Dict[str, NamedModule]],
    ) -> bool:
        """Return whether ``module`` belongs to a complete atomic MLP subset."""

        replay_config = self.qcfg.module_granular_replay
        if replay_config is None or replay_config.strategy != "atomic_swiglu" or not subset:
            return False
        role = module.full_name.rsplit(".", 1)[-1]
        if role not in {"gate_proj", "up_proj", "down_proj"}:
            return False
        parent_name = module.full_name.rpartition(".")[0]
        roles = {
            name.rsplit(".", 1)[-1]
            for name in subset
            if name.rpartition(".")[0] == parent_name
        }
        required = {"gate_proj", "up_proj", "down_proj"}
        if roles != required:
            raise RuntimeError(
                "QVQ atomic_swiglu requires one complete gate/up/down subset; "
                f"`{parent_name}` contains {sorted(roles)}."
            )
        return True

    def _quantize_atomic_swiglu_candidates(
        self,
        module: NamedModule,
        module_qcfg: QVQConfig,
        canonical_weight: torch.Tensor,
        quantization_hessian: torch.Tensor,
        quantization_kwargs: Dict[str, Any],
    ):
        """Generate canonical reselect plus all fixed-family alternatives for one MLP leaf."""

        replay_config = module_qcfg.module_granular_replay
        assert replay_config is not None and replay_config.strategy == "atomic_swiglu"
        candidate_results = {}
        base_telemetry = quantization_kwargs.get("telemetry")
        candidate_kwargs_base = dict(quantization_kwargs)
        candidate_kwargs_base.pop("yaqa_v2b2_family_mode", None)
        candidate_kwargs_base.pop("yaqa_v2b2_fixed_family_id", None)
        for candidate_id, fixed_family_id in self._atomic_swiglu_candidate_specs(replay_config):
            candidate_kwargs = dict(candidate_kwargs_base)
            if fixed_family_id is None:
                # Candidate zero is the configured canonical YAQA result. In
                # the atomic experiment this is normal V2B2-P32 reselect, not
                # fixed family zero (which is only the independent V2
                # fallback artifact).
                candidate_kwargs["yaqa_v2b2_family_mode"] = module_qcfg.yaqa.v2b2_family_mode
            else:
                candidate_kwargs["yaqa_v2b2_family_mode"] = "fixed_block_ldlq"
                candidate_kwargs["yaqa_v2b2_fixed_family_id"] = fixed_family_id
            candidate_kwargs["telemetry"] = (
                base_telemetry
                if candidate_id == 0
                else QVQQuantizationTelemetry() if base_telemetry is not None else None
            )
            candidate_results[candidate_id] = quantize_qvq_linear(
                canonical_weight,
                quantization_hessian,
                bits=module_qcfg.bits,
                **candidate_kwargs,
            )
        return candidate_results[0], candidate_results

    @staticmethod
    def _atomic_swiglu_candidate_specs(replay_config) -> tuple[tuple[int, Optional[int]], ...]:
        """Map stable candidate IDs to canonical reselect or fixed family IDs.

        Candidate zero is reserved for the normal configured reselect result.
        Candidate one is fixed family zero, while candidates two through four
        correspond to configured fixed families one through three. Keeping
        candidate IDs separate from family IDs prevents the independent V2
        fallback from being confused with the canonical reselect artifact.
        """

        return ((0, None), (1, 0)) + tuple(
            (family_id + 1, family_id) for family_id in replay_config.alternative_bank_ids
        )

    def _remember_atomic_swiglu_candidates(
        self,
        module: NamedModule,
        module_qcfg: QVQConfig,
        canonical_weight: torch.Tensor,
        candidate_results: Dict[int, Any],
    ) -> None:
        """Keep only CPU snapshots needed by the deferred nonlinear selector."""

        candidates = {}
        for alternative_bank_id, result in candidate_results.items():
            candidates[int(alternative_bank_id)] = {
                "weight": result.weight.detach().to(device="cpu", copy=True).contiguous(),
                "serialized_tensors": {
                    name: tensor.detach().to(device="cpu", copy=True).contiguous()
                    for name, tensor in result.serialized_tensors().items()
                },
            }
        with self._atomic_swiglu_lock:
            self._atomic_swiglu_candidates[module.full_name] = {
                "dense_weight": canonical_weight.detach().to(device="cpu", copy=True).contiguous(),
                "candidates": candidates,
                "module_qcfg": copy.deepcopy(module_qcfg),
            }

    def _select_atomic_swiglu_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int],
        subset_total: Optional[int],
    ) -> None:
        """Select and stage one complete gate/up/down triplet by propagated replay."""

        del subset_index, subset_total
        replay_config = self.qcfg.module_granular_replay
        if replay_config is None or replay_config.strategy != "atomic_swiglu" or not subset:
            return
        role_names = {}
        for name in subset:
            role = name.rsplit(".", 1)[-1]
            if role in {"gate_proj", "up_proj", "down_proj"}:
                role_names[role] = name
        if set(role_names) != {"gate_proj", "up_proj", "down_proj"}:
            return
        parent_name = role_names["gate_proj"].rpartition(".")[0]
        if any(name.rpartition(".")[0] != parent_name for name in role_names.values()):
            raise RuntimeError(f"QVQ atomic_swiglu found mismatched MLP parents in {sorted(role_names.values())}.")
        mlp_inputs = self._atomic_swiglu_inputs.get(parent_name)
        if mlp_inputs is None:
            log.warning(
                "QVQ atomic_swiglu skipped incomplete subset `%s`: no captured MLP inputs.",
                parent_name,
            )
            return
        with self._atomic_swiglu_lock:
            records = {role: self._atomic_swiglu_candidates.get(name) for role, name in role_names.items()}
        if any(record is None for record in records.values()):
            log.warning(
                "QVQ atomic_swiglu skipped incomplete subset `%s`: candidate bank is incomplete.",
                parent_name,
            )
            return
        if self._module_replay_model is None:
            raise RuntimeError("QVQ atomic_swiglu requires prepared final-logit replay targets.")

        gate_record = records["gate_proj"]
        up_record = records["up_proj"]
        down_record = records["down_proj"]
        assert gate_record is not None and up_record is not None and down_record is not None
        candidate_specs = self._atomic_swiglu_candidate_specs(replay_config)
        candidate_ids = tuple(candidate_id for candidate_id, _ in candidate_specs)
        candidate_family_ids = dict(candidate_specs)
        preselector = select_swiglu_candidate_triplet(
            mlp_inputs,
            gate_record["dense_weight"],
            up_record["dense_weight"],
            down_record["dense_weight"],
            [gate_record["candidates"][candidate_id]["weight"] for candidate_id in candidate_ids],
            [up_record["candidates"][candidate_id]["weight"] for candidate_id in candidate_ids],
            [down_record["candidates"][candidate_id]["weight"] for candidate_id in candidate_ids],
            beam_size=min(4, len(candidate_ids)),
        )
        triplets = [
            (
                candidate_ids[item["gate_index"]],
                candidate_ids[item["up_index"]],
                candidate_ids[item["down_index"]],
            )
            for item in preselector["beam"]
        ]
        canonical_triplet = (0, 0, 0)
        if canonical_triplet not in triplets:
            triplets.append(canonical_triplet)

        model = self._module_replay_model.model
        self._ensure_module_replay_residency(model)
        originals = {role: model.get_submodule(name) for role, name in role_names.items()}
        parents = {role: name.rpartition(".")[0] for role, name in role_names.items()}
        if any(not isinstance(original, torch.nn.Linear) for original in originals.values()):
            raise RuntimeError(f"QVQ atomic_swiglu targets must be dense Linear modules: {role_names}.")
        candidate_records = []

        def install(triplet):
            for role, candidate_id in zip(("gate_proj", "up_proj", "down_proj"), triplet, strict=True):
                original = originals[role]
                candidate = self._module_replay_qlinear_from_tensors(
                    original,
                    role_names[role],
                    records[role]["module_qcfg"],
                    records[role]["candidates"][candidate_id]["serialized_tensors"],
                )
                setattr(model.get_submodule(parents[role]), role_names[role].rsplit(".", 1)[-1], candidate)

        def restore():
            for role, original in originals.items():
                setattr(model.get_submodule(parents[role]), role_names[role].rsplit(".", 1)[-1], original)

        with self._module_replay_lock:
            try:
                install(canonical_triplet)
                baseline_folds = [
                    self._module_replay_metrics(
                        "search", fold_index=fold_index, folds=replay_config.search_folds
                    )
                    for fold_index in range(replay_config.search_folds)
                ]
                for triplet in triplets:
                    if triplet == canonical_triplet:
                        folds = baseline_folds
                        score = 1.0
                    else:
                        install(triplet)
                        folds = [
                            self._module_replay_metrics(
                                "search", fold_index=fold_index, folds=replay_config.search_folds
                            )
                            for fold_index in range(replay_config.search_folds)
                        ]
                        score = self._module_replay_score(folds, baseline_folds)
                    candidate_records.append(
                        {
                            "gate_candidate_id": triplet[0],
                            "up_candidate_id": triplet[1],
                            "down_candidate_id": triplet[2],
                            "gate_fixed_family_id": candidate_family_ids[triplet[0]],
                            "up_fixed_family_id": candidate_family_ids[triplet[1]],
                            "down_fixed_family_id": candidate_family_ids[triplet[2]],
                            "score": score,
                            "fold_kl_forward": [fold["kl_forward"] for fold in folds],
                            "fold_metrics": folds,
                            "selected": False,
                        }
                    )
                    restore()

                eligible = [
                    record for record in candidate_records
                    if (record["gate_candidate_id"], record["up_candidate_id"], record["down_candidate_id"])
                    != canonical_triplet
                    and record["score"] < 1 - replay_config.minimum_relative_kl_improvement
                ]
                selected_triplet = canonical_triplet
                if eligible:
                    winner = min(eligible, key=lambda record: record["score"])
                    selected_triplet = (
                        winner["gate_candidate_id"],
                        winner["up_candidate_id"],
                        winner["down_candidate_id"],
                    )
                install(canonical_triplet)
                baseline_confirmation = self._module_replay_metrics("confirmation")
                candidate_confirmation = baseline_confirmation
                if selected_triplet != canonical_triplet:
                    install(selected_triplet)
                    candidate_confirmation = self._module_replay_metrics("confirmation")
                    if not self._module_replay_confirmation_passes(
                        candidate_confirmation, baseline_confirmation, replay_config
                    ):
                        selected_triplet = canonical_triplet
                for record in candidate_records:
                    record["selected"] = (
                        record["gate_candidate_id"],
                        record["up_candidate_id"],
                        record["down_candidate_id"],
                    ) == selected_triplet
                replay_stats = {
                    "strategy": "atomic_swiglu",
                    "selected_triplet": selected_triplet,
                    "preselector": preselector,
                    "candidates": candidate_records,
                    "confirmation_baseline": baseline_confirmation,
                    "confirmation_candidate": candidate_confirmation,
                }
                for name in role_names.values():
                    self._module_replay_stats[name] = replay_stats
            finally:
                restore()

        # The selection above temporarily installs QVQLinear candidates only
        # for replay. Restore the original dense module first, then update its
        # weight to the selected reconstruction. StageLayer replays the rest
        # of the layer before submodule_finalize() installs the serialized
        # runtime module, so leaving candidate zero here would propagate stale
        # data into later layers and output alignment.
        for role, name in role_names.items():
            module = subset[name]
            selected_id = selected_triplet[({"gate_proj": 0, "up_proj": 1, "down_proj": 2}[role])]
            selected_weight = records[role]["candidates"][selected_id]["weight"]
            restored_weight = self._restore_module_weight(module, selected_weight)
            module.module.weight.data.copy_(
                restored_weight.to(device=module.module.weight.device, dtype=module.module.weight.dtype)
            )
            payload = records[role]["candidates"][selected_id]["serialized_tensors"]
            module.stream_sync()
            with parent_module_lock(name):
                for key in ("trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id"):
                    module.state.pop(key, None)
            module.stream_state_payload_to_cpu(payload)
            for stat in self.log:
                if stat.get("full_name") == name:
                    stat["module_granular_replay"] = self._module_replay_stats[name]

        with self._atomic_swiglu_lock:
            for name in role_names.values():
                self._atomic_swiglu_candidates.pop(name, None)

    def refine_subset_module_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Match QTIP's projection-at-a-time installation when alignment is enabled."""

        replay_config = self.qcfg.module_granular_replay
        if self._output_alignment is None and replay_config is None:
            return groups
        if replay_config is not None:
            role_order = {role: index for index, role in enumerate(replay_config.module_order)}
            groups = [
                sorted(group, key=lambda name: role_order.get(name.rsplit(".", 1)[-1], len(role_order)))
                for group in groups
            ]
            if replay_config.strategy == "atomic_swiglu":
                # The model tree presents gate/up and down as separate sibling
                # groups. Rejoin them before StageLayer creates plans so all
                # three projections are processed and cleaned up atomically.
                atomic_roles = {"gate_proj", "up_proj", "down_proj"}
                atomic_names_by_parent: Dict[str, list[str]] = {}
                for group in groups:
                    for name in group:
                        role = name.rsplit(".", 1)[-1]
                        if role in atomic_roles:
                            parent = name.rpartition(".")[0]
                            atomic_names_by_parent.setdefault(parent, []).append(name)
                merged_groups: list[list[str]] = []
                emitted_parents: set[str] = set()
                for group in groups:
                    remaining = []
                    parents_in_group = []
                    for name in group:
                        role = name.rsplit(".", 1)[-1]
                        parent = name.rpartition(".")[0]
                        if role in atomic_roles:
                            if parent not in emitted_parents and parent not in parents_in_group:
                                parents_in_group.append(parent)
                        else:
                            remaining.append(name)
                    for parent in parents_in_group:
                        merged_groups.append(
                            sorted(
                                atomic_names_by_parent[parent],
                                key=lambda name: role_order.get(
                                    name.rsplit(".", 1)[-1], len(role_order)
                                ),
                            )
                        )
                        emitted_parents.add(parent)
                    if remaining:
                        merged_groups.append(remaining)
                groups = merged_groups
        ordered_groups = groups
        gptq_model = None if self._output_alignment is None else self._output_alignment.gptq_model
        if gptq_model is not None:
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

            ordered_groups = [sorted(group, key=priority) for group in ordered_groups]
        if replay_config is not None and replay_config.strategy == "atomic_swiglu":
            return ordered_groups
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

    @staticmethod
    def _yaqa_factor_bytes(module: torch.nn.Linear) -> int:
        return (module.in_features * module.in_features + module.out_features * module.out_features) * 4

    @staticmethod
    def _yaqa_packed_factor_bytes(module: torch.nn.Linear) -> int:
        return (
            module.in_features * (module.in_features + 1)
            + module.out_features * (module.out_features + 1)
        ) * 2

    @staticmethod
    def _yaqa_default_max_factor_bytes(target_device: torch.device, total_factor_bytes: int) -> int:
        """Bound the MPS Gram working set; larger one-pass captures regress end-to-end time."""

        if target_device.type != "mps":
            return total_factor_bytes
        # A full 512-row Llama 3.2 1B capture measured 24.00 minutes at
        # 4 GiB/two passes versus 41.55 minutes at 8 GiB/one pass. The larger
        # resident target set makes MPS Gram contraction 1.73x slower despite
        # eliminating one model traversal. Keep the measured throughput-safe
        # default; users can still override this explicitly in YaqaConfig.
        return 4 * 1024**3

    @classmethod
    def _yaqa_target_chunks(
        cls,
        targets: dict[str, torch.nn.Linear],
        decoder_layers: list[Module],
        max_factor_bytes: int,
        *,
        packed_symmetric: bool = False,
    ) -> list[dict[str, torch.nn.Linear]]:
        """Pack whole decoder layers into bounded Sketch-B passes.

        Keeping layer boundaries intact avoids changing the semantic target set
        within a layer while bounding persistent Gram tensors and their transient
        device-to-host updates. Every pass still traverses the complete model.
        """

        owner_by_module_id = {}
        for layer_index, layer in enumerate(decoder_layers):
            for child in layer.modules():
                owner_by_module_id[id(child)] = layer_index
        by_layer: dict[int, dict[str, torch.nn.Linear]] = {}
        for name, module in targets.items():
            layer_index = owner_by_module_id.get(id(module))
            if layer_index is None:
                raise ValueError(f"QVQ YAQA target `{name}` is not owned by a decoder layer.")
            by_layer.setdefault(layer_index, {})[name] = module

        chunks: list[dict[str, torch.nn.Linear]] = []
        current: dict[str, torch.nn.Linear] = {}
        current_bytes = 0
        for layer_index in sorted(by_layer):
            layer_targets = by_layer[layer_index]
            byte_counter = cls._yaqa_packed_factor_bytes if packed_symmetric else cls._yaqa_factor_bytes
            layer_bytes = sum(byte_counter(module) for module in layer_targets.values())
            if current and current_bytes + layer_bytes > max_factor_bytes:
                chunks.append(current)
                current = {}
                current_bytes = 0
            current.update(layer_targets)
            current_bytes += layer_bytes
        if current:
            chunks.append(current)
        return chunks

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
        max_factor_bytes = self.qcfg.yaqa.max_factor_bytes_per_pass
        if max_factor_bytes is None:
            total_factor_bytes = sum(self._yaqa_factor_bytes(module) for module in targets.values())
            max_factor_bytes = self._yaqa_default_max_factor_bytes(
                target_device,
                total_factor_bytes,
            )
        packed_symmetric_accumulators = target_device.type == "mps"
        target_chunks = self._yaqa_target_chunks(
            targets,
            decoder_layers,
            max_factor_bytes,
            packed_symmetric=packed_symmetric_accumulators,
        )
        log.info(
            "QVQ YAQA: collecting full-model Sketch-B factors targets=%d batches=%d device=%s seed=%d "
            "minimum_sequences=%d regularization=%.6g batch_size=%d activation_checkpointing=%s "
            "checkpointed_modules=%d factor_passes=%d max_factor_bytes_per_pass=%d packed_symmetric=%s",
            len(targets),
            len(self.yaqa_calibration),
            target_device,
            self.qcfg.yaqa.seed,
            self.qcfg.yaqa.minimum_sequences,
            self.qcfg.yaqa.regularization,
            self.qcfg.yaqa.batch_size,
            self.qcfg.yaqa.activation_checkpointing,
            len(decoder_layers) if self.qcfg.yaqa.activation_checkpointing else 0,
            len(target_chunks),
            max_factor_bytes,
            packed_symmetric_accumulators,
        )
        progress_stride = max(1, len(self.yaqa_calibration) // 16)
        moved = source_device != target_device
        try:
            if moved:
                model.to(target_device)
            input_hessians = {}
            output_hessians = {}
            pass_stats = []
            with torch.inference_mode(False), torch.enable_grad():
                for pass_index, pass_targets in enumerate(target_chunks, start=1):

                    def log_progress(progress, *, current_pass=pass_index):
                        completed = progress["completed_batches"]
                        total = progress["total_batches"]
                        if completed == 1 or completed == total or completed % progress_stride == 0:
                            log.info(
                                "QVQ YAQA Sketch-B: pass=%d/%d targets=%d batches=%d/%d "
                                "sequences=%d valid_tokens=%d",
                                current_pass,
                                len(target_chunks),
                                len(pass_targets),
                                completed,
                                total,
                                progress["completed_sequences"],
                                progress["valid_tokens"],
                            )

                    pass_inputs, pass_outputs, current_stats = capture_yaqa_sketch_b(
                        model,
                        self.yaqa_calibration,
                        pass_targets,
                        device=target_device,
                        seed=self.qcfg.yaqa.seed,
                        minimum_sequences=self.qcfg.yaqa.minimum_sequences,
                        first_decoder_layer=decoder_layers[0],
                        checkpoint_modules=decoder_layers if self.qcfg.yaqa.activation_checkpointing else (),
                        progress_callback=log_progress,
                        mps_cleanup_interval=self.qcfg.yaqa.mps_cleanup_interval,
                        chat_template_config=self.qcfg.yaqa.chat_template,
                    )
                    input_hessians.update(pass_inputs)
                    output_hessians.update(pass_outputs)
                    pass_stats.append(current_stats)
                    if target_device.type == "mps":
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
        finally:
            if moved:
                model.to(source_device)
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
        if not pass_stats:
            raise RuntimeError("QVQ YAQA produced no factor passes.")
        stats = dict(pass_stats[0])
        sum_fields = (
            "accumulator_bytes",
            "capture_wall_seconds",
            "final_host_transfer_seconds",
            "input_factor_elements",
            "output_factor_elements",
            "factor_storage_bytes",
            "mps_cleanup_count",
        )
        for field in sum_fields:
            stats[field] = sum(item[field] for item in pass_stats)
        stats["phase_wall_seconds"] = {
            phase: sum(item["phase_wall_seconds"][phase] for item in pass_stats)
            for phase in pass_stats[0]["phase_wall_seconds"]
        }
        stats["factor_passes"] = len(pass_stats)
        stats["max_factor_bytes_per_pass"] = max_factor_bytes
        stats["pass_target_counts"] = [len(chunk) for chunk in target_chunks]
        stats["pass_factor_bytes"] = [item["factor_storage_bytes"] for item in pass_stats]
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

        self._select_atomic_swiglu_subset(
            subset or {},
            subset_index=subset_index,
            subset_total=subset_total,
        )
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

        del previous_subset, subset_index, subset_total
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
            quantization_kwargs = {
                "output_hessian": output_hessian,
                "bias": None if module.bias is None else module.bias.detach().to(target_device),
                "seed": seed,
                "damp_percent": damp_percent,
                "codebook_version": module_qcfg.codebook,
                "vector_size": module_qcfg.vector_size,
                "trellis_window": module_qcfg.trellis_window,
                "dual_v2": module_qcfg.format == FORMAT.QVQ_DUAL_V2,
                "v2b4_p64": module_qcfg.format == FORMAT.QVQ_V2B4_P64,
                "v2b2_p32": module_qcfg.format == FORMAT.QVQ_V2B2_P32,
                "module_scale_search": module_qcfg.module_scale_search,
                "output_channel_scale_optimization": module_qcfg.output_channel_scale_optimization,
                "viterbi_objective": module_qcfg.viterbi_objective,
                "tail_biting_candidates": module_qcfg.tail_biting_candidates,
                "rounding": module_qcfg.rounding,
                "yaqa_v2b2_family_mode": module_qcfg.yaqa.v2b2_family_mode,
                "yaqa_sample_strategy": module_qcfg.yaqa.sample_strategy,
                "yaqa_spectral_refinement": module_qcfg.yaqa.spectral_refinement,
                "yaqa_spectral_ranks": module_qcfg.yaqa.spectral_ranks,
                "yaqa_spectral_lambdas": module_qcfg.yaqa.spectral_lambdas,
                "yaqa_spectral_push": module_qcfg.yaqa.spectral_push,
                "yaqa_spectral_push_alphas": module_qcfg.yaqa.spectral_push_alphas,
                "yaqa_spectral_localized": module_qcfg.yaqa.spectral_localized,
                "yaqa_spectral_localized_alphas": module_qcfg.yaqa.spectral_localized_alphas,
                "yaqa_spectral_localized_max_segments": module_qcfg.yaqa.spectral_localized_max_segments,
                "yaqa_spectral_localized_max_changes": module_qcfg.yaqa.spectral_localized_max_changes,
                "yaqa_spectral_localized_replay_candidates": module_qcfg.yaqa.spectral_localized_replay_candidates,
                "yaqa_spectral_localized_direct_replay_candidates": (
                    module_qcfg.yaqa.spectral_localized_direct_replay_candidates
                ),
                "viterbi_minimum_proxy_improvement": module_qcfg.viterbi_minimum_proxy_improvement,
                "viterbi_pruning": module_qcfg.viterbi_pruning,
                "telemetry": telemetry,
                "bank_count": module_qcfg.bank_count,
                "propagated_inputs": None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[0],
                "propagated_target_output": None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[1],
                "propagated_acceptance": None if propagation_gate is None or not module_qcfg.propagated_bank_selection else propagation_gate[2],
                "propagated_candidate_score": (
                    None
                    if propagation_gate is None or not module_qcfg.propagated_bank_selection
                    else propagation_gate[3]
                ),
            }
            atomic_swiglu = self._is_atomic_swiglu_module(module, subset)
            if atomic_swiglu:
                result, candidate_results = self._quantize_atomic_swiglu_candidates(
                    module,
                    module_qcfg,
                    canonical_weight,
                    quantization_hessian,
                    quantization_kwargs,
                )
            else:
                candidate_results = None
                result = self._select_module_granular_replay_candidate(
                    module,
                    module_qcfg,
                    canonical_weight,
                    quantization_hessian,
                    quantization_kwargs,
                )
            duration = time.perf_counter() - started

            # Quantized replay temporarily overwrites the dense module. The
            # rollback image must be complete before the asynchronous payload
            # transfer starts: a failed stream event cannot itself be trusted
            # to have produced a valid host backup.
            original_weight = module.weight.detach().to(device="cpu", copy=True)
            with parent_module_lock(module.full_name):
                module.state["_qvq_original_weight"] = original_weight
            if atomic_swiglu:
                assert candidate_results is not None
                self._remember_atomic_swiglu_candidates(
                    module,
                    module_qcfg,
                    canonical_weight,
                    candidate_results,
                )
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
                "yaqa_sketch_b_telemetry": self._yaqa_stats if module_qcfg.rounding == "yaqa" else None,
                "yaqa_v2b2_family_mode": (
                    module_qcfg.yaqa.v2b2_family_mode
                    if module_qcfg.rounding == "yaqa" and module_qcfg.format == FORMAT.QVQ_V2B2_P32
                    else None
                ),
                "yaqa_sample_strategy": (
                    module_qcfg.yaqa.sample_strategy
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
                "module_granular_replay": self._module_replay_stats.get(module.full_name),
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

        self._module_replay_rows.clear()
        self._module_replay_teacher_logits.clear()
        self._module_replay_model = None
        with self._atomic_swiglu_lock:
            self._atomic_swiglu_inputs.clear()
            self._atomic_swiglu_candidates.clear()
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
