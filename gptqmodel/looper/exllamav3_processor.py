# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import ctypes
import gc
import math
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import transformers
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd

from ..exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    quantize_exl3,
    reconstruct_exl3_tensors,
)
from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    ExecutionConfig,
    LoopProcessor,
)
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
    QUANT_LOG_LOSS_KIND,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, EXL3Config, GPTQConfig
from ..quantization.gptq import GPTQ
from ..utils.device import get_device
from ..utils.exl3_error_ledger import (
    JOURNAL_ENV,
    ROUTE_EVIDENCE_SCHEMA,
    ROUTE_EVIDENCE_SCHEMA_VERSION,
    ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
    ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    ZERO_ROUTE_RECOVERY_MODE_IDENTITY,
    ZERO_ROUTE_RECOVERY_MODE_MIXED,
    ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR,
    ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
    ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
    ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
    ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
    ZERO_ROUTE_RECOVERY_SCHEMA,
    ZERO_ROUTE_RECOVERY_SCHEMA_VERSION,
    ZERO_ROUTE_RECOVERY_TRIGGER,
    append_exl3_error_journal,
    build_projection_record,
    compact_projection_record,
    route_evidence_required,
    routed_expert_identity,
    validate_route_evidence,
    validate_zero_route_recovery,
    validate_zero_route_recovery_authorization,
    zero_route_recovery_enabled,
    zero_route_recovery_recipe,
)
from ..utils.exl3_capture_frontier import (
    CAPTURE_FRONTIER_ENV,
    EXL3CaptureDescriptor,
    EXL3CaptureFrontierStore,
    EXL3CaptureRecord,
)
from ..utils.exl3_capture_batch_spool import (
    CAPTURE_BATCH_CHECKPOINT_INTERVAL_ENV,
    CAPTURE_BATCH_SPOOL_ENV,
    EXL3CaptureBatchSpool,
)
from ..utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    canonical_json_bytes,
    checkpoint_root_from_provenance,
    sha256_bytes,
)
from ..utils.exl3_inline_mixed import (
    InlineMixedTierPlanStore,
    PROJECTION_ORDER,
    build_layer_tier_plan,
    inline_mixed_policy,
    projection_score,
)
from ..utils.exl3_remote import (
    CoordinatorSlot,
    EXL3_HESSIAN_CAPTURE_CONTRACT,
    ExecutionSlotLease,
    RemoteEndpoint,
    exl3_quantization_failure_message,
    remote_client_from_provenance,
    validate_exl3_hessian_metrics,
)
from ..utils.exl3_router_candidates import (
    ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT,
    has_learned_router_recovery,
    learned_router_ranked_choices,
)
from ..utils.exllamav3 import create_exllamav3_module
from ..utils.logger import setup_logger
from ..utils.model import recurse_setattr
from ..utils.module_locks import parent_module_lock
from ..utils.offload import offload_to_disk, offload_to_safetensors_reference

HOST_RSS_LIMIT_ENV = "GPTQMODEL_EXL3_HOST_RSS_LIMIT_BYTES"
CUDA_ALLOCATION_LIMIT_ENV = "GPTQMODEL_EXL3_CUDA_ALLOCATION_LIMIT_BYTES"
MEMORY_TELEMETRY_INTERVAL_ENV = (
    "GPTQMODEL_EXL3_MEMORY_TELEMETRY_INTERVAL_BATCHES"
)
HESSIAN_OWNER_POLICY_META = "ds4rt_hessian_owner_policy"
HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN = (
    "ds4rt.exl3-hessian-owner-weighted-round-robin-v1"
)

log = setup_logger()

_EXL3_SIGMA_REG = 0.025
_DEFERRED_RUNTIME_WEIGHT_STATE = "exl3_deferred_runtime_weight"
_DEFERRED_FINALIZE_WEIGHT_STATE = "exl3_deferred_finalize_weight"
_OUT_SCALES_TO_ARG = {
    "always": True,
    "never": False,
    "auto": None,
    None: None,
}


def _inline_candidate_role(policy) -> str:
    return f"candidate_k{policy.base_bits}"


def _inline_selected_role(policy) -> str:
    return f"selected_k{policy.upgrade_bits}"


def resolve_hessian_owner_device(
    *,
    expert_index: int,
    devices: list[torch.device],
    policy: dict[str, Any] | None,
) -> torch.device:
    """Resolve deterministic capture ownership, optionally with device weights."""

    if not devices:
        raise ValueError("routed EXL3 Hessian capture has no ownership device")
    if policy is None:
        return devices[expert_index % len(devices)]
    weights = policy.get("device_weights") if isinstance(policy, dict) else None
    if (
        not isinstance(policy, dict)
        or policy.get("contract") != HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN
        or not isinstance(weights, list)
        or len(weights) != len(devices)
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
            for weight in weights
        )
    ):
        raise ValueError("EXL3 Hessian owner policy is invalid")
    slot = expert_index % sum(weights)
    for device, weight in zip(devices, weights, strict=True):
        if slot < weight:
            return device
        slot -= weight
    raise AssertionError("validated Hessian owner weights did not select a device")


def prepare_exl3_hessian(
    capture: GPTQ,
    *,
    target_device: torch.device,
    module_full_name: str,
) -> torch.Tensor:
    """Restore GPTQ's normalized capture to EXL3's raw X^T X contract."""

    capture.finalize_hessian(target_device=target_device)
    hessian = capture.H
    if hessian is None:
        raise RuntimeError(
            f"EXL3 failed to capture Hessian for module `{module_full_name}`."
        )
    if capture.nsamples <= 0:
        raise RuntimeError(
            f"EXL3 captured no calibration activations for module `{module_full_name}`."
        )
    # GPTQ capture stores 2/N * X^T X. ExLlamaV3's H_data finalizer divides
    # by `count`, so restore the raw X^T X sum before handing it over.
    # Gate/up projections may share one normalized accumulator. EXL3 mutates
    # H_data while finalizing it, so each consumer gets a bounded working copy
    # while the canonical shared Hessian remains immutable on its owner GPU.
    if bool(getattr(capture, "hessian_is_shared", False)):
        hessian = hessian.detach().to(
            device=target_device,
            dtype=torch.float32,
            copy=True,
        )
    hessian.mul_(float(capture.nsamples) / 2.0)
    return hessian


def _router_recovery_candidates(
    router: Module,
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    candidate_rank_min: int,
    candidate_rank_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adjacent learned-router candidates, or none for hash routing."""

    if hasattr(router, "tid2eid"):
        rows = indices.shape[0]
        return (
            indices.new_empty((rows, 0), dtype=torch.int64),
            logits.new_empty((rows, 0), dtype=torch.float32),
        )
    if not has_learned_router_recovery(router):
        raise RuntimeError(
            "EXL3 recovery candidate capture requires a learned top-k or hash router"
        )
    router_top_k = int(indices.shape[-1])
    if candidate_rank_min != router_top_k + 1:
        raise RuntimeError(
            "EXL3 recovery candidates must start immediately after "
            f"router top-k: top_k={router_top_k} "
            f"candidate_rank_min={candidate_rank_min}"
        )
    ranked_scores, ranked_indices = learned_router_ranked_choices(
        router,
        logits,
        rank_max=candidate_rank_max,
        selected_indices=indices,
    )
    candidate_indices = ranked_indices[
        :, candidate_rank_min - 1 : candidate_rank_max
    ]
    candidate_score_gaps = (
        ranked_scores[:, router_top_k - 1 : router_top_k]
        - ranked_scores[:, candidate_rank_min - 1 : candidate_rank_max]
    ).float()
    return candidate_indices, candidate_score_gaps


class _EXL3NaturalRouteCapture:
    """Accumulate per-expert exposure during one native-router subset replay."""

    def __init__(
        self,
        processor: "EXL3Processor",
        *,
        layer_module: Module,
        subset: Dict[str, NamedModule],
    ) -> None:
        self.processor = processor
        self.spool = getattr(processor, "_active_capture_batch_spool", None)
        self.collect_route_evidence = True
        self._batch_payloads: dict[int, dict[str, Any]] = {}
        self.targets: dict[str, dict[str, Any]] = {}
        for task_name, named_module in subset.items():
            full_name = getattr(named_module, "full_name", None)
            identity = (
                processor._routed_expert_identity(full_name)
                if isinstance(full_name, str)
                else None
            )
            if identity is not None and task_name in processor.tasks:
                self.targets[task_name] = identity
        family_ids = {
            (identity["block_namespace"], identity["logical_layer"])
            for identity in self.targets.values()
        }
        if not self.targets or len(family_ids) != 1:
            raise ValueError(
                "EXL3 natural-route capture requires one routed block per subset"
            )
        self.family_id = next(iter(family_ids))
        provenance = processor._ledger_provenance()
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        self.recovery_recipe = zero_route_recovery_recipe(family_join)

        mlp = getattr(layer_module, "mlp", None)
        if mlp is None:
            mlp = getattr(layer_module, "ffn", None)
        self.router = getattr(mlp, "gate", None)
        if not isinstance(self.router, Module):
            raise ValueError("EXL3 natural-route capture could not resolve the router")

        self._lock = threading.Lock()
        self._handle = None
        self._router_calls = 0
        self._router_token_count = 0
        self._router_selected_route_count = 0
        self._router_top_k: int | None = None
        self._expert_counts: torch.Tensor | None = None
        self._gate_sums: torch.Tensor | None = None
        self._gate_squared_mass: torch.Tensor | None = None
        self._router_weight_dtypes: set[str] = set()
        self._mask_modes: set[str] = set()
        restored = getattr(processor, "_restored_route_accumulators", {}).pop(
            self.family_id, None
        )
        if restored is not None:
            self._expert_counts = restored["expert_counts"]
            self._gate_sums = restored["gate_sums"]
            self._gate_squared_mass = restored["gate_squared_mass"]
            self._router_calls = restored["router_calls"]
            self._router_token_count = restored["router_token_count"]
            self._router_selected_route_count = restored[
                "router_selected_route_count"
            ]
            self._router_top_k = restored["router_top_k"]
            self._router_weight_dtypes = set(
                restored["router_weight_dtypes"]
            )
            self._mask_modes = set(restored["mask_modes"])

    def __enter__(self) -> "_EXL3NaturalRouteCapture":
        if getattr(self.processor, "_active_natural_route_capture", None) is not None:
            raise RuntimeError("EXL3 batch capture context is already active")
        self.processor._active_natural_route_capture = self
        self._handle = self.router.register_forward_hook(self._capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.processor._active_natural_route_capture = None
        if exc_type is None and self._batch_payloads:
            raise RuntimeError("EXL3 capture ended with uncommitted batches")
        if exc_type is None and self.collect_route_evidence:
            self.processor._commit_natural_route_capture(self)
        return False

    def _capture(self, _module, inputs, output) -> None:
        paused_tls = getattr(self.processor, "_hooks_paused_tls", None)
        if paused_tls is not None and getattr(paused_tls, "value", False):
            return
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise RuntimeError("EXL3 natural-route capture requires router triplets")
        logits, weights, indices = output
        router_input = (
            inputs[0]
            if isinstance(inputs, (tuple, list)) and inputs
            else None
        )
        if (
            not isinstance(logits, torch.Tensor)
            or not isinstance(weights, torch.Tensor)
            or not isinstance(indices, torch.Tensor)
            or weights.shape != indices.shape
            or weights.ndim != 2
            or logits.ndim != 2
            or logits.shape[0] != weights.shape[0]
            or not isinstance(router_input, torch.Tensor)
        ):
            raise RuntimeError(
                "EXL3 natural-route capture received malformed router output"
            )

        router_input = router_input.reshape(-1, router_input.shape[-1])
        # Expert implementations fan out the router's complete token matrix,
        # including padding rows.  Keep that raw view solely for proving the
        # input observed by each expert hook; route evidence and recovery data
        # below must continue to retain only calibration-valid rows.
        verification_router_input = router_input
        verification_top_indices = indices
        keep_mask = getattr(getattr(self.processor, "_mask_tls", None), "value", None)
        if keep_mask is None:
            mask_mode = "absent"
        else:
            flat_keep = keep_mask.reshape(-1).to(
                device=indices.device, dtype=torch.bool
            )
            if flat_keep.numel() != indices.shape[0]:
                raise RuntimeError(
                    "EXL3 natural-route mask does not align with router rows"
                )
            mask_mode = "all-valid" if bool(flat_keep.all().item()) else "filtered"
            weights = weights[flat_keep]
            indices = indices[flat_keep]
            logits = logits[flat_keep]
            router_input = router_input[flat_keep]

        if self.spool is not None:
            batch_index = self.processor.current_batch_index()
            if batch_index is None:
                raise RuntimeError(
                    "EXL3 router capture has no calibration batch index"
                )
            candidate_rank_min = self.recovery_recipe["candidate_rank_min"]
            candidate_rank_max = self.recovery_recipe["candidate_rank_max"]
            candidate_indices, candidate_score_gaps = _router_recovery_candidates(
                self.router,
                logits,
                indices,
                candidate_rank_min=candidate_rank_min,
                candidate_rank_max=candidate_rank_max,
            )
            with self._lock:
                if batch_index in self._batch_payloads:
                    raise RuntimeError("EXL3 router ran twice for one capture batch")
                self._batch_payloads[batch_index] = {
                    "router_input": router_input.detach(),
                    "top_weights": weights.detach(),
                    "top_indices": indices.detach(),
                    "verification_router_input": (
                        verification_router_input.detach()
                    ),
                    "verification_top_indices": (
                        verification_top_indices.detach()
                    ),
                    "candidate_indices": candidate_indices.detach(),
                    "candidate_score_gaps": candidate_score_gaps.detach(),
                    "num_experts": int(logits.shape[-1]),
                    "expert_rows": {},
                    "verified_gate_experts": set(),
                    "mask_mode": mask_mode,
                }

        num_experts = int(logits.shape[-1])
        top_k = int(indices.shape[-1])
        # CUDA weighted bincount uses atomic additions whose final low bits can
        # vary across otherwise identical runs. Route evidence is part of the
        # projection checkpoint identity, so reduce this small vector on CPU in
        # float64 and keep resume identities bit-stable.
        flat_indices = indices.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.int64,
        )
        flat_weights = weights.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.float64,
        )
        if flat_indices.numel() == 0:
            return
        counts = torch.bincount(flat_indices, minlength=num_experts)
        gate_sums = torch.bincount(
            flat_indices,
            weights=flat_weights,
            minlength=num_experts,
        )
        gate_squared_mass = torch.bincount(
            flat_indices,
            weights=flat_weights.square(),
            minlength=num_experts,
        )
        if any(
            value.numel() != num_experts
            for value in (counts, gate_sums, gate_squared_mass)
        ):
            raise RuntimeError(
                "EXL3 natural-route capture observed an invalid expert id"
            )
        counts = counts.to(dtype=torch.int64)
        gate_sums = gate_sums.to(dtype=torch.float64)
        gate_squared_mass = gate_squared_mass.to(dtype=torch.float64)

        with self._lock:
            if self._router_top_k is None:
                self._router_top_k = top_k
                self._expert_counts = torch.zeros_like(counts)
                self._gate_sums = torch.zeros_like(gate_sums)
                self._gate_squared_mass = torch.zeros_like(gate_squared_mass)
            elif (
                self._router_top_k != top_k
                or self._expert_counts.numel() != num_experts
            ):
                raise RuntimeError("EXL3 natural-route geometry changed during replay")
            self._expert_counts.add_(counts)
            self._gate_sums.add_(gate_sums)
            self._gate_squared_mass.add_(gate_squared_mass)
            self._router_calls += 1
            self._router_token_count += int(indices.shape[0])
            self._router_selected_route_count += int(flat_indices.numel())
            self._router_weight_dtypes.add(str(weights.dtype))
            self._mask_modes.add(mask_mode)

    def capture_expert_input(self, task_name: str, value: torch.Tensor) -> None:
        """Prove gate inputs or retain exact down inputs for the active batch."""

        if self.spool is None:
            return
        identity = self.targets.get(task_name)
        if identity is None:
            return
        batch_index = self.processor.current_batch_index()
        if batch_index is None:
            raise RuntimeError("EXL3 expert capture has no calibration batch index")
        with self._lock:
            payload = self._batch_payloads.get(batch_index)
            if payload is None:
                raise RuntimeError("EXL3 expert capture preceded its router record")
            rows = value.detach().reshape(-1, value.shape[-1])
            expert = identity["expert"]
            if self.spool.phase == "gate-up":
                verification_indices = payload["verification_top_indices"]
                verification_input = payload["verification_router_input"]
                row_major = torch.nonzero(
                    verification_indices.eq(expert), as_tuple=False
                )[:, 0]
                # Hugging Face's fused-expert reference implementations use
                # one_hot(...).permute(expert, top_k, token) followed by
                # torch.where(), which yields top-k-major token order.  Other
                # implementations use the row-major order above.  Both are
                # exact fan-outs of the same selected router rows.
                topk_major = torch.where(
                    verification_indices.transpose(0, 1).eq(expert)
                )[1]
                matches_router = False
                for expected_indices in (row_major, topk_major):
                    expected = verification_input.index_select(
                        0,
                        expected_indices.to(verification_input.device),
                    )
                    if rows.shape == expected.shape and torch.equal(
                        rows,
                        expected.to(device=rows.device),
                    ):
                        matches_router = True
                        break
                if not matches_router:
                    raise RuntimeError(
                        "EXL3 pre-fanout recovery rows differ from the gate hook"
                    )
                if identity["projection"] == "w1":
                    payload["verified_gate_experts"].add(expert)
            elif identity["projection"] == "w2":
                previous = payload["expert_rows"].setdefault(expert, rows)
                if previous is not rows:
                    raise RuntimeError("EXL3 down expert ran twice in one batch")

    def commit_batch(self, batch_index: int) -> None:
        """Durably publish one exact additive recovery input record."""

        if self.spool is None:
            return
        with self._lock:
            payload = self._batch_payloads.pop(batch_index, None)
        if payload is None:
            if batch_index in self.spool.committed_indices:
                return
            raise RuntimeError("EXL3 completed a batch without capture payload")
        selected_experts = set(
            int(value)
            for value in payload["top_indices"].detach().unique().tolist()
        )
        tensors = {
            "router_input": payload["router_input"],
            "top_weights": payload["top_weights"],
            "top_indices": payload["top_indices"],
            "candidate_indices": payload["candidate_indices"],
            "candidate_score_gaps": payload["candidate_score_gaps"],
        }
        if self.spool.phase == "gate-up":
            if not selected_experts <= payload["verified_gate_experts"]:
                raise RuntimeError("EXL3 did not verify every selected gate input")
        else:
            if selected_experts != set(payload["expert_rows"]):
                raise RuntimeError("EXL3 did not capture every selected down input")
            tensors.update(
                {
                    f"expert_{expert:06d}": rows
                    for expert, rows in payload["expert_rows"].items()
                }
            )
        self.spool.commit(
            batch_index,
            tensors=tensors,
            metadata={
                "batch_index": batch_index,
                "mask_mode": payload["mask_mode"],
                "num_experts": payload["num_experts"],
                "selected_experts": sorted(selected_experts),
                "pre_fanout_gate_input_verified": self.spool.phase == "gate-up",
            },
        )

    def evidence_by_expert(self) -> dict[int, dict[str, Any]]:
        """Finalize the bounded route metrics needed by expert-family tiering."""

        if (
            self._router_calls <= 0
            or self._router_token_count <= 0
            or self._router_top_k is None
            or self._expert_counts is None
            or self._gate_sums is None
            or self._gate_squared_mass is None
        ):
            raise RuntimeError("EXL3 natural-route capture observed no router traffic")
        total_gate_sum = float(self._gate_sums.sum().item())
        total_gate_sq = float(self._gate_squared_mass.sum().item())
        if total_gate_sum <= 0 or total_gate_sq <= 0:
            raise RuntimeError(
                "EXL3 natural-route capture observed no positive gate mass"
            )
        namespace, logical_layer = next(
            iter(
                {
                    (identity["block_namespace"], identity["logical_layer"])
                    for identity in self.targets.values()
                }
            )
        )
        evidence: dict[int, dict[str, Any]] = {}
        for expert in sorted(
            {identity["expert"] for identity in self.targets.values()}
        ):
            route_count = int(self._expert_counts[expert].item())
            gate_sum = float(self._gate_sums[expert].item())
            gate_sq = float(self._gate_squared_mass[expert].item())
            evidence[expert] = {
                "schema": ROUTE_EVIDENCE_SCHEMA,
                "schema_version": ROUTE_EVIDENCE_SCHEMA_VERSION,
                "block_namespace": namespace,
                "logical_layer": logical_layer,
                "expert": expert,
                "router_calls": self._router_calls,
                "router_token_count": self._router_token_count,
                "router_selected_route_count": self._router_selected_route_count,
                "router_top_k": self._router_top_k,
                "expert_route_count": route_count,
                "expert_gate_weight_sum": gate_sum,
                "expert_gate_squared_mass": gate_sq,
                "total_gate_weight_sum": total_gate_sum,
                "total_gate_squared_mass": total_gate_sq,
                "expert_route_fraction": route_count
                / self._router_selected_route_count,
                "expert_gate_weight_mass_fraction": gate_sum / total_gate_sum,
                "expert_gate_squared_mass_fraction": gate_sq / total_gate_sq,
                "expert_gate_weight_mean": gate_sum / max(route_count, 1),
                "expert_gate_weight_rms": math.sqrt(gate_sq / max(route_count, 1)),
                "router_weight_dtypes": sorted(self._router_weight_dtypes),
                "mask_modes": sorted(self._mask_modes),
            }
        return evidence


def clone_exllamav3_config_for_module(
    qcfg: EXL3Config,
    module_full_name: str,
) -> Optional[EXL3Config]:
    """Clones and applies per-module EXL3 dynamic overrides, or skips the module."""

    if not qcfg.module_is_included(module_full_name):
        return None

    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        qcfg_clone.bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
        qcfg_clone.head_bits = qcfg.dynamic_get(
            module_full_name, "head_bits", qcfg_clone.head_bits
        )

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
        self._natural_route_evidence_cache: dict[
            tuple[str, int], dict[int, dict[str, Any]]
        ] = {}
        self._remote_client_initialized = False
        self._remote_client = None
        self._projection_checkpoint_store_initialized = False
        self._projection_checkpoint_store = None
        self._capture_frontier_store_initialized = False
        self._capture_frontier_store = None
        self._distributed_local_quant_locks: dict[str, threading.Lock] = {}
        self._hessian_family_owners: dict[
            tuple[str, int, int], tuple[str, GPTQ]
        ] = {}
        self._pending_hessian_family_aliases: dict[
            tuple[str, int, int], list[tuple[str, GPTQ]]
        ] = {}
        self._active_capture_batch_spool: EXL3CaptureBatchSpool | None = None
        self._active_capture_batch_layer: int | None = None
        self._active_natural_route_capture = None
        self._restored_route_accumulators: dict[
            tuple[str, int], dict[str, Any]
        ] = {}
        self.error_journal_path = os.getenv(JOURNAL_ENV) or str(
            Path(self.log_tmp_log_file_name).with_suffix(".exl3-error-ledger.jsonl")
        )
        inline_policy = self._inline_mixed_policy()
        candidate_bits = inline_policy.base_bits if inline_policy is not None else 2
        self.inline_mixed_candidate_journal_path = (
            f"{self.error_journal_path}.k{candidate_bits}-candidates"
        )

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because EXL3 capture is fixed at construction."""

        raise NotImplementedError(
            "EXL3Processor's calibration_dataset cannot be modified"
        )

    def _ledger_provenance(self) -> dict[str, Any] | None:
        """Return the run provenance shared by every EXL3 projection."""

        if not isinstance(self.qcfg.meta, dict):
            return None
        provenance = self.qcfg.meta.get("ds4rt_error_ledger")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(
                "EXL3 `meta.ds4rt_error_ledger` provenance must be a dictionary"
            )
        return provenance

    def _inline_mixed_policy(self, module_full_name=None, *, namespace=None):
        """Return the exact-rational layer-local mixed policy, if configured."""

        if module_full_name is not None:
            identity = self._routed_expert_identity(module_full_name)
            if identity is None:
                return None
            namespace = identity["block_namespace"]
        return inline_mixed_policy(getattr(self.qcfg, "meta", None), namespace=namespace)

    def _routed_expert_identity(self, module_full_name):
        provenance = self._ledger_provenance()
        return routed_expert_identity(
            module_full_name,
            family_join=provenance.get("family_join") if isinstance(provenance, dict) else None,
        )

    def _remote_client_for_run(self, provenance: dict[str, Any] | None):
        """Construct the immutable remote-worker client at most once per run."""

        lock = getattr(self, "_stats_lock", None)
        context = lock if lock is not None else nullcontext()
        with context:
            if not getattr(self, "_remote_client_initialized", False):
                self._remote_client = remote_client_from_provenance(provenance)
                self._remote_client_initialized = True
            return getattr(self, "_remote_client", None)

    def _projection_checkpoint_store_for_run(
        self, provenance: dict[str, Any] | None
    ) -> EXL3ProjectionCheckpointStore | None:
        """Construct one shared coordinator store and request index per run."""

        checkpoint_root = checkpoint_root_from_provenance(provenance)
        with self._stats_lock:
            if not getattr(
                self, "_projection_checkpoint_store_initialized", False
            ):
                self._projection_checkpoint_store = (
                    EXL3ProjectionCheckpointStore(checkpoint_root)
                    if checkpoint_root is not None
                    else None
                )
                self._projection_checkpoint_store_initialized = True
            store = getattr(self, "_projection_checkpoint_store", None)
        if store is not None and store.root != checkpoint_root:
            raise ValueError("EXL3 projection-checkpoint root changed during the run")
        return store

    def _capture_frontier_store_for_run(
        self, provenance: dict[str, Any] | None
    ) -> EXL3CaptureFrontierStore | None:
        """Construct the opt-in durable Hessian store at most once per run."""

        configured_root = os.getenv(CAPTURE_FRONTIER_ENV)
        root = Path(configured_root).expanduser().resolve() if configured_root else None
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        with self._stats_lock:
            if not getattr(self, "_capture_frontier_store_initialized", False):
                self._capture_frontier_store = (
                    EXL3CaptureFrontierStore(root, family_join=family_join)
                    if root is not None
                    else None
                )
                self._capture_frontier_store_initialized = True
            store = getattr(self, "_capture_frontier_store", None)
        if store is not None and store.root != root:
            raise ValueError("EXL3 capture-frontier root changed during the run")
        return store

    def _subset_capture_phase(
        self,
        subset: Dict[str, NamedModule],
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        identities = {
            task_name: identity
            for task_name, named_module in subset.items()
            if isinstance(getattr(named_module, "full_name", None), str)
            and (identity := self._routed_expert_identity(named_module.full_name))
            is not None
        }
        projections = {identity["projection"] for identity in identities.values()}
        if projections and projections <= {"w1", "w3"}:
            return "gate-up", identities
        if projections == {"w2"}:
            return "down", identities
        raise RuntimeError("EXL3 capture subset is not one routed projection phase")

    @staticmethod
    def _route_batch_statistics(
        *,
        weights: torch.Tensor,
        indices: torch.Tensor,
        num_experts: int,
        mask_mode: str,
    ) -> dict[str, Any]:
        flat_indices = indices.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
        flat_weights = weights.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        return {
            "expert_counts": torch.bincount(
                flat_indices, minlength=num_experts
            ).to(torch.int64),
            "gate_sums": torch.bincount(
                flat_indices, weights=flat_weights, minlength=num_experts
            ).to(torch.float64),
            "gate_squared_mass": torch.bincount(
                flat_indices,
                weights=flat_weights.square(),
                minlength=num_experts,
            ).to(torch.float64),
            "router_calls": 1,
            "router_token_count": int(indices.shape[0]),
            "router_selected_route_count": int(flat_indices.numel()),
            "router_top_k": int(indices.shape[-1]),
            "router_weight_dtypes": {str(weights.dtype)},
            "mask_modes": {str(mask_mode)},
        }

    @staticmethod
    def _merge_route_statistics(
        target: dict[str, Any] | None,
        addition: dict[str, Any],
    ) -> dict[str, Any]:
        if target is None:
            return addition
        for key in ("expert_counts", "gate_sums", "gate_squared_mass"):
            target[key].add_(addition[key])
        for key in (
            "router_calls",
            "router_token_count",
            "router_selected_route_count",
        ):
            target[key] += addition[key]
        if target["router_top_k"] != addition["router_top_k"]:
            raise RuntimeError("restored router top-k changed between batches")
        target["router_weight_dtypes"].update(addition["router_weight_dtypes"])
        target["mask_modes"].update(addition["mask_modes"])
        return target

    def restore_subset_capture_batches(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        expected_batches: int,
        subset: Dict[str, NamedModule],
    ) -> frozenset[int]:
        """Rebuild live Hessians from durable batches before missing forwards."""

        configured_root = os.getenv(CAPTURE_BATCH_SPOOL_ENV)
        if not configured_root:
            self._active_capture_batch_spool = None
            self._active_capture_batch_layer = None
            return frozenset()
        phase, identities = self._subset_capture_phase(subset)
        try:
            checkpoint_interval = int(
                os.getenv(CAPTURE_BATCH_CHECKPOINT_INTERVAL_ENV, "1")
            )
        except ValueError as error:
            raise RuntimeError(
                "EXL3 capture checkpoint interval is invalid"
            ) from error
        if checkpoint_interval <= 0:
            raise RuntimeError("EXL3 capture checkpoint interval must be positive")
        family_ids = {
            (identity["block_namespace"], identity["logical_layer"])
            for identity in identities.values()
        }
        if len(family_ids) != 1:
            raise RuntimeError("capture batch subset crosses routed blocks")
        ownership: dict[str, str] = {}
        for task_name, identity in identities.items():
            task = self.tasks.get(task_name)
            capture = task.get("capture") if isinstance(task, dict) else None
            owner = getattr(capture, "hessian_accumulator_device", None)
            if owner is None:
                raise RuntimeError("capture batch task has no Hessian owner")
            previous = ownership.setdefault(str(identity["expert"]), str(owner))
            if previous != str(owner):
                raise RuntimeError("expert Hessian ownership differs by projection")
        provenance = self._ledger_provenance()
        spool = EXL3CaptureBatchSpool(
            configured_root,
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            expected_batches=expected_batches,
            payload_contract=ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT,
            phase=phase,
            module_names=sorted(
                named.full_name for named in subset.values()
            ),
            provenance={
                "family_join": (
                    provenance.get("family_join")
                    if isinstance(provenance, dict)
                    else None
                )
            },
            ownership=ownership,
            checkpoint_interval=checkpoint_interval,
        )
        self._active_capture_batch_spool = spool
        self._active_capture_batch_layer = layer_index

        task_by_expert: dict[int, tuple[str, GPTQ]] = {}
        for task_name, identity in identities.items():
            if (
                phase == "gate-up" and identity["projection"] != "w1"
            ) or (phase == "down" and identity["projection"] != "w2"):
                continue
            task = self.tasks[task_name]
            task_by_expert[identity["expert"]] = (task_name, task["capture"])

        restored_stats = None
        for batch_index in sorted(spool.committed_indices):
            tensors, metadata = spool.load(batch_index)
            if phase == "gate-up":
                router_input = tensors["router_input"]
                indices = tensors["top_indices"].to(torch.int64)
                weights = tensors["top_weights"]
                num_experts = metadata.get("num_experts")
                if (
                    isinstance(num_experts, bool)
                    or not isinstance(num_experts, int)
                    or num_experts <= 0
                ):
                    raise RuntimeError(
                        "restored EXL3 capture batch lacks router geometry"
                    )
                restored_stats = self._merge_route_statistics(
                    restored_stats,
                    self._route_batch_statistics(
                        weights=weights,
                        indices=indices,
                        num_experts=num_experts,
                        mask_mode=metadata.get("mask_mode", "absent"),
                    ),
                )
                for expert, (_task_name, capture) in task_by_expert.items():
                    row_indices = torch.nonzero(
                        indices.eq(expert), as_tuple=False
                    )[:, 0]
                    if row_indices.numel():
                        capture.add_batch(
                            router_input.index_select(0, row_indices),
                            torch.empty(0),
                            batch_index=batch_index,
                        )
            else:
                for expert, (_task_name, capture) in task_by_expert.items():
                    key = f"expert_{expert:06d}"
                    rows = tensors.get(key)
                    if rows is not None:
                        capture.add_batch(
                            rows,
                            torch.empty(0),
                            batch_index=batch_index,
                        )
            del tensors
        if restored_stats is not None:
            self._restored_route_accumulators[next(iter(family_ids))] = (
                restored_stats
            )
        return spool.committed_indices

    def forward_committed_batch_indices(
        self, *, layer_index: int
    ) -> frozenset[int]:
        spool = getattr(self, "_active_capture_batch_spool", None)
        if spool is None or getattr(
            self, "_active_capture_batch_layer", None
        ) != layer_index:
            return frozenset()
        return spool.committed_indices

    def forward_batch_completed(self, *, layer_index: int, batch_index: int) -> None:
        capture = getattr(self, "_active_natural_route_capture", None)
        if capture is not None and getattr(
            self, "_active_capture_batch_layer", None
        ) == layer_index:
            capture.commit_batch(batch_index)
            interval = int(os.getenv(MEMORY_TELEMETRY_INTERVAL_ENV, "0"))
            spool = getattr(self, "_active_capture_batch_spool", None)
            complete = (
                spool is not None
                and len(spool.committed_indices)
                == int(spool.key["expected_batches"])
            )
            if interval > 0 and (batch_index % interval == 0 or complete):
                self._enforce_capture_memory_limits(
                    context=f"layer-{layer_index}-batch-{batch_index}"
                )

    def _enforce_capture_memory_limits(self, *, context: str) -> dict[str, Any]:
        """Stop after a durable batch if host or device capture memory is unsafe."""

        try:
            host_limit = int(os.getenv(HOST_RSS_LIMIT_ENV, "0"))
            cuda_limit = int(os.getenv(CUDA_ALLOCATION_LIMIT_ENV, "0"))
        except ValueError as error:
            raise RuntimeError("EXL3 memory-safety environment is invalid") from error
        summary = self.log_capture_memory_summary(context)
        rss = int(summary.get("process_rss_bytes", 0))
        if host_limit > 0 and rss > host_limit:
            raise RuntimeError(
                f"EXL3 host RSS safety limit exceeded after {context}: "
                f"actual={rss} limit={host_limit}"
            )
        if cuda_limit > 0:
            cuda_devices = summary.get("cuda_devices", {})
            for device, values in cuda_devices.items():
                allocated = int(values.get("allocated_bytes", 0))
                if allocated > cuda_limit:
                    raise RuntimeError(
                        f"EXL3 CUDA allocation safety limit exceeded after "
                        f"{context}: device={device} actual={allocated} "
                        f"limit={cuda_limit}"
                    )
            if any(
                int(values.get("reserved_bytes", 0)) > cuda_limit
                for values in cuda_devices.values()
            ):
                # Long attention forwards can leave large, fragmented cache
                # segments around the live Hessians. They are not recovery
                # state and can prevent the next workspace mapping.
                torch.cuda.empty_cache()
                summary = self.log_capture_memory_summary(
                    f"{context}-after-cache-trim"
                )
                for device, values in summary.get("cuda_devices", {}).items():
                    allocated = int(values.get("allocated_bytes", 0))
                    if allocated > cuda_limit:
                        raise RuntimeError(
                            f"EXL3 CUDA allocation safety limit exceeded after "
                            f"{context}: device={device} actual={allocated} "
                            f"limit={cuda_limit}"
                        )
        return summary

    @staticmethod
    def _subset_task_names_by_full_name(
        subset: Dict[str, NamedModule],
    ) -> dict[str, str]:
        names: dict[str, str] = {}
        for task_name, named_module in subset.items():
            full_name = getattr(named_module, "full_name", None)
            if not isinstance(full_name, str) or full_name in names:
                raise RuntimeError("EXL3 capture subset has invalid module identities")
            names[full_name] = task_name
        return names

    def restore_subset_capture_frontier(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: Dict[str, NamedModule],
    ) -> bool:
        """Index normalized Hessians and route evidence before subset replay."""

        provenance = self._ledger_provenance()
        store = self._capture_frontier_store_for_run(provenance)
        if store is None:
            return False
        task_names = self._subset_task_names_by_full_name(subset)
        records = store.restore_index(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            subset=subset,
        )
        if records is None:
            return False
        if set(records) != set(task_names):
            raise RuntimeError("EXL3 capture frontier restored incomplete tasks")

        route_cache = getattr(self, "_natural_route_evidence_cache", None)
        if route_cache is None:
            route_cache = {}
            self._natural_route_evidence_cache = route_cache
        requires_routes = route_evidence_required(provenance)
        restored_shared_states: dict[int, EXL3CaptureRecord] = {}
        for full_name, record in records.items():
            task_name = task_names[full_name]
            task = self.tasks.get(task_name)
            if not isinstance(task, dict):
                raise RuntimeError(f"EXL3 capture task disappeared: `{full_name}`")
            capture: GPTQ = task["capture"]
            state_id = id(getattr(capture, "_hessian_state", capture))
            shared_record = restored_shared_states.get(state_id)
            if shared_record is None:
                if (
                    getattr(capture, "nsamples", 0) != 0
                    or getattr(capture, "H", None) is not None
                    or getattr(capture, "_device_hessian_partials", {})
                    or getattr(capture, "_device_sample_counts", {})
                ):
                    raise RuntimeError(
                        f"EXL3 refused to mix live and restored capture for `{full_name}`"
                    )
                restored_shared_states[state_id] = record
            elif (
                shared_record.path != record.path
                or shared_record.sample_count != record.sample_count
            ):
                raise RuntimeError(
                    "EXL3 shared gate/up frontier records differ"
                )
            if requires_routes and record.route_evidence is None:
                raise RuntimeError(
                    f"EXL3 restored no required route evidence for `{full_name}`"
                )
            if shared_record is None:
                capture.nsamples = record.sample_count
                capture._device_hessian_partials.clear()
                capture._device_sample_counts.clear()
                capture._hessian_dirty = False
                capture._final_hessian_device_hint = None
            task["capture_frontier_record"] = record
            task["route_evidence"] = copy.deepcopy(record.route_evidence)
            if record.zero_route_recovery is not None:
                task["zero_route_recovery"] = copy.deepcopy(
                    record.zero_route_recovery
                )

            identity = self._routed_expert_identity(full_name)
            if identity is not None and record.route_evidence is not None:
                family_id = (
                    identity["block_namespace"],
                    identity["logical_layer"],
                )
                family = route_cache.setdefault(family_id, {})
                previous = family.setdefault(
                    identity["expert"], copy.deepcopy(record.route_evidence)
                )
                if previous != record.route_evidence:
                    raise RuntimeError(
                        f"EXL3 restored conflicting route evidence for `{full_name}`"
                    )
        log.info(
            "EXL3 capture frontier indexed lazily: layer=%s subset=%s/%s modules=%s",
            layer_index,
            subset_index + 1,
            subset_total,
            len(records),
        )
        return True

    def commit_subset_capture_frontier(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: Dict[str, NamedModule],
    ) -> None:
        """Durably commit captures before any projection quantization starts."""

        provenance = self._ledger_provenance()
        store = self._capture_frontier_store_for_run(provenance)
        if store is None:
            return
        requires_routes = route_evidence_required(provenance)
        descriptors = []
        captures_by_name: dict[str, GPTQ] = {}
        for task_name, named_module in subset.items():
            task = self.tasks.get(task_name)
            full_name = getattr(named_module, "full_name", None)
            if not isinstance(task, dict) or not isinstance(full_name, str):
                raise RuntimeError("EXL3 cannot persist an incomplete capture task")
            capture: GPTQ = task["capture"]
            route_evidence = task.get("route_evidence")
            if requires_routes and route_evidence is None:
                raise RuntimeError(
                    f"EXL3 captured no required route evidence for `{full_name}`"
                )
            captures_by_name[full_name] = capture
            descriptors.append(
                EXL3CaptureDescriptor(
                    module=full_name,
                    sample_count=int(capture.nsamples),
                    route_evidence=copy.deepcopy(route_evidence),
                    zero_route_recovery=copy.deepcopy(
                        task.get("zero_route_recovery")
                    ),
                )
            )
        before = self.capture_memory_summary()
        manifest = store.commit_streaming(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            subset=subset,
            descriptors=descriptors,
            hessian_loader=lambda full_name: captures_by_name[
                full_name
            ].snapshot_hessian(target_device=torch.device("cpu")),
        )
        policy = self._inline_mixed_policy(next(iter(subset.values())).full_name)
        retain_for_inline = policy is not None and any(
            (identity := self._routed_expert_identity(named_module.full_name)) is not None
            and identity["block_namespace"] == policy.namespace
            for named_module in subset.values()
        )
        if retain_for_inline:
            # The same durable frontier is also the bounded-memory source for
            # an inline upgraded tier. Keep only lightweight descriptors in the
            # tasks; selected jobs hydrate and release one Hessian apiece.
            records = store.restore_index(
                layer_index=layer_index,
                subset_index=subset_index,
                subset_total=subset_total,
                subset=subset,
            )
            if records is None:
                raise RuntimeError("EXL3 capture frontier disappeared after commit")
            for task_name, named_module in subset.items():
                task = self.tasks[task_name]
                task["capture_frontier_record"] = records[named_module.full_name]
            # Commit completed every snapshot before this point. Drop the
            # original live accumulators so base-tier candidate jobs use the same
            # one-Hessian-at-a-time path as an interrupted/resumed run. Gate
            # and up may share one accumulator, hence the shared-state guard.
            demoted_states: set[int] = set()
            for task_name, named_module in subset.items():
                task = self.tasks[task_name]
                capture: GPTQ = task["capture"]
                record = records[named_module.full_name]
                state_id = id(getattr(capture, "_hessian_state", capture))
                if state_id in demoted_states:
                    continue
                capture_lock = getattr(capture, "lock", None)
                lock_context = (
                    capture_lock if capture_lock is not None else nullcontext()
                )
                with lock_context:
                    capture.H = None
                    capture._device_hessian_partials.clear()
                    capture._device_sample_counts.clear()
                    capture.nsamples = record.sample_count
                    capture._hessian_dirty = False
                    capture._final_hessian_device_hint = None
                demoted_states.add(state_id)
        batch_spool = getattr(self, "_active_capture_batch_spool", None)
        if batch_spool is not None:
            batch_spool.discard()
            self._active_capture_batch_spool = None
            self._active_capture_batch_layer = None
        after = self.capture_memory_summary()
        log.info(
            "EXL3 capture frontier committed streaming: layer=%s subset=%s/%s "
            "modules=%s manifest=%s memory_before=%s memory_after=%s",
            layer_index,
            subset_index + 1,
            subset_total,
            len(manifest["captures"]),
            manifest["manifest_sha256"],
            before,
            after,
        )

    def _hydrate_capture_frontier(
        self,
        *,
        task_entry: dict[str, Any],
        capture: GPTQ,
        target_device: torch.device,
        retain_for_inline: bool | None = None,
    ) -> None:
        """Load only the restored Hessian needed by the current projection."""

        record = task_entry.get("capture_frontier_record")
        if record is None:
            return
        if retain_for_inline is None:
            retain_for_inline = self._inline_mixed_policy() is not None
        if not isinstance(record, EXL3CaptureRecord):
            raise RuntimeError("EXL3 restored capture record is malformed")
        store = self._capture_frontier_store_for_run(self._ledger_provenance())
        if store is None:
            raise RuntimeError("EXL3 restored capture store disappeared")
        capture_lock = getattr(capture, "lock", None)
        lock_context = capture_lock if capture_lock is not None else nullcontext()
        with lock_context:
            if getattr(capture, "_device_hessian_partials", {}):
                raise RuntimeError("EXL3 refused to mix lazy and live capture state")
            if capture.H is None:
                # Keep durable frontier payloads in owned host storage.
                host_hessian = store.load_record_hessian(record, device="cpu")
                hessian = host_hessian.to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=False,
                    copy=True,
                )
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                del host_hessian
                capture.H = hessian
                capture.nsamples = record.sample_count
                capture._hessian_dirty = False
                capture._final_hessian_device_hint = target_device
            elif capture.nsamples != record.sample_count:
                raise RuntimeError("EXL3 shared lazy capture sample count differs")
            if not retain_for_inline:
                task_entry.pop("capture_frontier_record", None)
            # Inline mixed mode does not consume the descriptor: its selected
            # upgraded pass reloads the same immutable Hessian exactly once.

    @staticmethod
    def _tensor_storage_summary(value: Any) -> dict[str, int]:
        """Count unique tensor storage reachable from a bounded object tree."""

        summary = {
            "tensor_count": 0,
            "storage_count": 0,
            "meta_tensor_count": 0,
            "host_bytes": 0,
            "device_bytes": 0,
        }
        seen_containers: set[int] = set()
        seen_storages: set[tuple[str, int, int]] = set()
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, torch.Tensor):
                summary["tensor_count"] += 1
                if current.device.type == "meta":
                    summary["meta_tensor_count"] += 1
                    continue
                try:
                    storage = current.untyped_storage()
                    storage_bytes = int(storage.nbytes())
                    data_ptr = int(storage.data_ptr())
                except (RuntimeError, TypeError):
                    storage_bytes = current.numel() * current.element_size()
                    data_ptr = id(current)
                storage_key = (str(current.device), data_ptr, storage_bytes)
                if storage_key in seen_storages:
                    continue
                seen_storages.add(storage_key)
                summary["storage_count"] += 1
                byte_key = (
                    "host_bytes"
                    if current.device.type == "cpu"
                    else "device_bytes"
                )
                summary[byte_key] += storage_bytes
                continue
            if isinstance(current, dict):
                identity = id(current)
                if identity not in seen_containers:
                    seen_containers.add(identity)
                    pending.extend(current.values())
                continue
            if isinstance(current, (list, tuple, set)):
                identity = id(current)
                if identity not in seen_containers:
                    seen_containers.add(identity)
                    pending.extend(current)
        return summary

    @staticmethod
    def _process_memory_summary() -> dict[str, int]:
        """Read resident-memory ownership classes from Linux procfs."""

        summary = {
            "process_rss_bytes": 0,
            "process_pss_bytes": 0,
            "process_private_dirty_bytes": 0,
            "process_anon_bytes": 0,
            "process_file_bytes": 0,
            "process_shmem_bytes": 0,
            "process_swap_bytes": 0,
        }
        proc_fields = {
            "Rss": "process_rss_bytes",
            "Pss": "process_pss_bytes",
            "Private_Dirty": "process_private_dirty_bytes",
            "Anonymous": "process_anon_bytes",
            "Pss_File": "process_file_bytes",
            "Pss_Shmem": "process_shmem_bytes",
            "Swap": "process_swap_bytes",
        }
        try:
            with open("/proc/self/smaps_rollup", "r", encoding="ascii") as source:
                for line in source:
                    name, separator, remainder = line.partition(":")
                    target = proc_fields.get(name)
                    if target is None or not separator:
                        continue
                    fields = remainder.split()
                    if fields:
                        summary[target] = int(fields[0]) * 1024
        except (OSError, ValueError):
            try:
                with open("/proc/self/statm", "r", encoding="ascii") as source:
                    resident_pages = int(source.read().split()[1])
                summary["process_rss_bytes"] = (
                    resident_pages * os.sysconf("SC_PAGE_SIZE")
                )
            except (OSError, IndexError, ValueError):
                pass
        return summary

    @staticmethod
    def _model_tensor_summary(model: Any | None) -> dict[str, int]:
        """Inventory materialized model storage without touching tensor data."""

        candidate = model
        if not isinstance(candidate, torch.nn.Module):
            candidate = getattr(model, "model", None)
        if not isinstance(candidate, torch.nn.Module):
            return EXL3Processor._tensor_storage_summary(None)
        tensors = list(candidate.parameters()) + list(candidate.buffers())
        return EXL3Processor._tensor_storage_summary(tensors)

    def capture_memory_summary(self, model: Any | None = None) -> dict[str, Any]:
        """Report process, cache, model, and capture tensor ownership."""

        summary = {
            "task_count": 0,
            "lazy_frontier_records": 0,
            "host_hessian_bytes": 0,
            "device_hessian_bytes": 0,
            "host_partial_bytes": 0,
            "device_partial_bytes": 0,
            "host_dense_quant_source_bytes": 0,
            "device_dense_quant_source_bytes": 0,
        }
        summary.update(self._process_memory_summary())

        cache = getattr(self, "inputs_cache", None)
        cache_payload = (
            getattr(cache, "layer_inputs", []),
            getattr(cache, "layer_input_kwargs", []),
            getattr(cache, "position_ids", []),
            getattr(cache, "attention_masks", []),
        )
        cache_summary = self._tensor_storage_summary(cache_payload)
        for key, value in cache_summary.items():
            summary[f"input_cache_{key}"] = value

        model_summary = self._model_tensor_summary(model)
        for key, value in model_summary.items():
            summary[f"model_{key}"] = value

        summary["cuda_allocated_bytes"] = 0
        summary["cuda_reserved_bytes"] = 0
        summary["cuda_devices"] = {}
        try:
            for device_index in range(torch.cuda.device_count()):
                allocated = int(torch.cuda.memory_allocated(device_index))
                reserved = int(torch.cuda.memory_reserved(device_index))
                summary["cuda_allocated_bytes"] += allocated
                summary["cuda_reserved_bytes"] += reserved
                summary["cuda_devices"][f"cuda:{device_index}"] = {
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                }
        except (RuntimeError, TypeError):
            pass

        seen_sources: set[int] = set()
        seen_capture_storages: set[tuple[str, int, int]] = set()
        for task in self.tasks.values():
            if not isinstance(task, dict):
                continue
            summary["task_count"] += 1
            if isinstance(task.get("capture_frontier_record"), EXL3CaptureRecord):
                summary["lazy_frontier_records"] += 1
            capture = task.get("capture")
            hessian = getattr(capture, "H", None)
            if isinstance(hessian, torch.Tensor):
                storage = hessian.untyped_storage()
                storage_key = (
                    str(hessian.device),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                )
                if storage_key not in seen_capture_storages:
                    seen_capture_storages.add(storage_key)
                    key = (
                        "host_hessian_bytes"
                        if hessian.device.type == "cpu"
                        else "device_hessian_bytes"
                    )
                    summary[key] += storage.nbytes()
            for partial in getattr(capture, "_device_hessian_partials", {}).values():
                storage = partial.untyped_storage()
                storage_key = (
                    str(partial.device),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                )
                if storage_key in seen_capture_storages:
                    continue
                seen_capture_storages.add(storage_key)
                key = (
                    "host_partial_bytes"
                    if partial.device.type == "cpu"
                    else "device_partial_bytes"
                )
                summary[key] += storage.nbytes()
            named_module = getattr(capture, "_named_module", None)
            state = getattr(named_module, "state", None)
            source = state.get("quant_source_module") if isinstance(state, dict) else None
            if isinstance(source, torch.nn.Module) and id(source) not in seen_sources:
                seen_sources.add(id(source))
                for tensor in list(source.parameters()) + list(source.buffers()):
                    key = (
                        "host_dense_quant_source_bytes"
                        if tensor.device.type == "cpu"
                        else "device_dense_quant_source_bytes"
                    )
                    summary[key] += tensor.numel() * tensor.element_size()
        return summary

    def log_capture_memory_summary(
        self, context: str, model: Any | None = None
    ) -> dict[str, Any]:
        summary = self.capture_memory_summary(model=model)
        log.info("EXL3 capture memory: context=%s summary=%s", context, summary)
        return summary

    @staticmethod
    def _malloc_trim() -> int | None:
        """Return unused glibc arenas when the platform exposes malloc_trim."""

        try:
            trim = getattr(ctypes.CDLL(None), "malloc_trim")
        except (AttributeError, OSError):
            return None
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return int(trim(0))

    def release_host_memory(
        self, context: str, model: Any | None = None
    ) -> dict[str, Any]:
        """Collect unreachable objects, trim glibc, and report the exact delta."""

        before = self.capture_memory_summary(model=model)
        collected = gc.collect()
        trim_result = self._malloc_trim()
        after = self.capture_memory_summary(model=model)
        result = {
            "gc_collected": int(collected),
            "malloc_trim_result": trim_result,
            "rss_released_bytes": max(
                0,
                int(before["process_rss_bytes"])
                - int(after["process_rss_bytes"]),
            ),
            "before": before,
            "after": after,
        }
        log.info("EXL3 host memory release: context=%s result=%s", context, result)
        return result

    def discard_capture_frontiers_through(self, layer_index: int) -> None:
        """Drop captures covered by a durable layer-output boundary."""

        store = self._capture_frontier_store_for_run(self._ledger_provenance())
        if store is not None:
            store.discard_through(layer_index, block_namespace="base")

    def _distributed_local_quant_lock(self, device: torch.device) -> threading.Lock:
        """Serialize trellis search independently on each coordinator GPU."""

        device_key = str(device)
        with self._stats_lock:
            locks = getattr(self, "_distributed_local_quant_locks", None)
            if locks is None:
                locks = {}
                self._distributed_local_quant_locks = locks
            return locks.setdefault(device_key, threading.Lock())

    @staticmethod
    def _projection_assignment_key(
        module: NamedModule, *, phase: str | None = None
    ) -> str:
        """Return the durable scheduler identity for one independent projection."""

        return module.full_name if phase is None else f"{module.full_name}::{phase}"

    def subset_forward_capture_context(
        self,
        *,
        layer_module: Module,
        subset: Dict[str, NamedModule],
    ):
        """Capture natural router exposure in the existing subset forward."""

        provenance = self._ledger_provenance()
        if not route_evidence_required(provenance):
            return nullcontext()
        targets = {
            task_name: identity
            for task_name, named_module in subset.items()
            if task_name in self.tasks
            and isinstance(getattr(named_module, "full_name", None), str)
            and (identity := self._routed_expert_identity(named_module.full_name)) is not None
        }
        if not targets:
            return nullcontext()
        family_ids = {
            (identity["block_namespace"], identity["logical_layer"])
            for identity in targets.values()
        }
        if len(family_ids) != 1:
            raise ValueError(
                "EXL3 natural-route capture requires one routed block per subset"
            )
        family_id = next(iter(family_ids))
        cache = getattr(self, "_natural_route_evidence_cache", None)
        if cache is None:
            cache = {}
            self._natural_route_evidence_cache = cache
        cached = cache.get(family_id)
        target_experts = {identity["expert"] for identity in targets.values()}
        if cached is not None and target_experts.issubset(cached):
            self._attach_natural_route_evidence(targets, cached)
            if getattr(self, "_active_capture_batch_spool", None) is None:
                return nullcontext()
            capture = _EXL3NaturalRouteCapture(
                self,
                layer_module=layer_module,
                subset=subset,
            )
            capture.collect_route_evidence = False
            return capture
        return _EXL3NaturalRouteCapture(
            self,
            layer_module=layer_module,
            subset=subset,
        )

    def _commit_natural_route_capture(
        self,
        capture: _EXL3NaturalRouteCapture,
    ) -> None:
        """Bind one subset's route evidence to its projection capture tasks."""

        evidence_by_expert = capture.evidence_by_expert()
        family_id = next(
            iter(
                {
                    (identity["block_namespace"], identity["logical_layer"])
                    for identity in capture.targets.values()
                }
            )
        )
        cache = getattr(self, "_natural_route_evidence_cache", None)
        if cache is None:
            cache = {}
            self._natural_route_evidence_cache = cache
        previous_family = cache.get(family_id)
        if previous_family is not None:
            for expert, evidence in evidence_by_expert.items():
                previous = previous_family.get(expert)
                if previous is not None and previous != evidence:
                    raise RuntimeError(
                        f"EXL3 natural-route evidence changed for {family_id} expert {expert}"
                    )
            previous_family.update(evidence_by_expert)
            evidence_by_expert = previous_family
        else:
            cache[family_id] = evidence_by_expert
        self._attach_natural_route_evidence(capture.targets, evidence_by_expert)

    def _attach_natural_route_evidence(
        self,
        targets: dict[str, dict[str, Any]],
        evidence_by_expert: dict[int, dict[str, Any]],
    ) -> None:
        """Attach one immutable block distribution to projection tasks."""

        for task_name, identity in targets.items():
            task = self.tasks.get(task_name)
            if task is None:
                raise RuntimeError(
                    f"EXL3 route capture lost task `{task_name}` before commit"
                )
            evidence = evidence_by_expert.get(identity["expert"])
            if evidence is None:
                raise RuntimeError(
                    f"EXL3 route capture lacks expert {identity['expert']}"
                )
            previous = task.get("route_evidence")
            if previous is not None and previous != evidence:
                raise RuntimeError(
                    f"EXL3 route evidence changed for task `{task_name}`"
                )
            task["route_evidence"] = evidence

    def plan_subset_zero_route_recovery(
        self,
        *,
        subset: Dict[str, NamedModule],
        layer_module: Module,
    ) -> tuple[str, ...]:
        """Plan deterministic top-ups for under-covered learned top-k routers."""

        provenance = self._ledger_provenance()
        if not route_evidence_required(provenance):
            return ()
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        recipe = zero_route_recovery_recipe(family_join)
        authorization = self._zero_route_recovery_authorization(provenance)
        mlp = getattr(layer_module, "mlp", None)
        if mlp is None:
            mlp = getattr(layer_module, "ffn", None)
        router = getattr(mlp, "gate", None)
        learned_topk_router = has_learned_router_recovery(router)
        recovery_tasks: list[str] = []
        counts_by_expert: dict[int, int] = {}
        family_ids: set[tuple[str, int]] = set()
        for task_name in sorted(subset):
            named_module = subset[task_name]
            identity = self._routed_expert_identity(
                getattr(named_module, "full_name", "")
            )
            task = self.tasks.get(task_name)
            if identity is None or not isinstance(task, dict):
                continue
            capture = task.get("capture")
            route_evidence = task.get("route_evidence")
            if not isinstance(route_evidence, dict):
                raise RuntimeError(
                    f"EXL3 coverage census lacks natural-route evidence for "
                    f"`{named_module.full_name}`"
                )
            natural_count = route_evidence.get("expert_route_count")
            captured_count = getattr(capture, "nsamples", None)
            if (
                isinstance(natural_count, bool)
                or not isinstance(natural_count, int)
                or natural_count < 0
                or isinstance(captured_count, bool)
                or not isinstance(captured_count, int)
                or captured_count != natural_count
            ):
                raise RuntimeError(
                    "EXL3 natural capture and router census disagree for "
                    f"`{named_module.full_name}`: capture={captured_count} "
                    f"routes={natural_count}"
                )
            family_ids.add(
                (identity["block_namespace"], identity["logical_layer"])
            )
            previous = counts_by_expert.setdefault(
                identity["expert"], natural_count
            )
            if previous != natural_count:
                raise RuntimeError(
                    "EXL3 projection siblings have inconsistent natural counts "
                    f"for expert {identity['expert']}"
                )
            if (
                authorization is not None
                and learned_topk_router
                and natural_count < recipe["target_sample_count"]
            ):
                recovery_tasks.append(task_name)
        if len(family_ids) > 1:
            raise RuntimeError("EXL3 coverage census crossed routed blocks")
        if counts_by_expert:
            values = list(counts_by_expert.values())
            log.info(
                "EXL3 natural-route census: family=%s experts=%s min=%s mean=%.3f "
                "max=%s zero=%s",
                next(iter(family_ids)),
                len(values),
                min(values),
                sum(values) / len(values),
                max(values),
                sum(value == 0 for value in values),
            )
        return tuple(recovery_tasks)

    def _zero_route_recovery_authorization(
        self,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve either a plan-native or content-bound continuation authority."""

        provenance = self._ledger_provenance() if provenance is None else provenance
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(family_join, dict):
            return None
        recipe = zero_route_recovery_recipe(family_join)
        if zero_route_recovery_enabled(provenance):
            family_digest = sha256_bytes(canonical_json_bytes(family_join))
            authorization = {
                "schema": ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
                "schema_version": ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
                "kind": "immutable-family-join",
                "recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
                "trigger": recipe["trigger"],
                "sample_source": recipe["sample_source"],
                "capture_method": recipe["capture_method"],
                "selection_policy": recipe["selection_policy"],
                "candidate_rank_min": recipe["candidate_rank_min"],
                "candidate_rank_max": recipe["candidate_rank_max"],
                "target_sample_count": recipe["target_sample_count"],
                "identity_calibration_policy": recipe[
                    "identity_calibration_policy"
                ],
                "family_join_sha256": family_digest,
                "authorization_sha256": family_digest,
            }
        else:
            meta = getattr(self.qcfg, "meta", None)
            authorization = (
                meta.get("ds4rt_zero_route_recovery")
                if isinstance(meta, dict)
                else None
            )
            if authorization is None:
                return None
        return validate_zero_route_recovery_authorization(
            authorization,
            family_join=family_join,
        )

    def finish_subset_zero_route_recovery(
        self,
        *,
        subset: Dict[str, NamedModule],
        task_names: tuple[str, ...],
    ) -> None:
        """Bind one completed direct-expert top-up to each recovered capture."""

        if not task_names:
            return
        provenance = self._ledger_provenance()
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        authorization = self._zero_route_recovery_authorization(provenance)
        if not isinstance(family_join, dict) or authorization is None:
            raise RuntimeError("EXL3 route-coverage top-up lost its authorization")
        recipe = zero_route_recovery_recipe(family_join)
        recovered_by_expert: dict[int, tuple[str, int, int]] = {}
        for task_name in task_names:
            named_module = subset.get(task_name)
            task = self.tasks.get(task_name)
            identity = self._routed_expert_identity(
                getattr(named_module, "full_name", "")
            )
            if identity is None or not isinstance(task, dict):
                raise RuntimeError("EXL3 route-coverage top-up lost a target task")
            route_evidence = task.get("route_evidence")
            capture = task.get("capture")
            recovery_capture = task.get("zero_route_recovery_capture")
            total_count = getattr(capture, "nsamples", None)
            natural_count = (
                route_evidence.get("expert_route_count")
                if isinstance(route_evidence, dict)
                else None
            )
            if (
                not isinstance(route_evidence, dict)
                or isinstance(natural_count, bool)
                or not isinstance(natural_count, int)
                or not 0
                <= natural_count
                < recipe["target_sample_count"]
                or not isinstance(recovery_capture, dict)
                or isinstance(total_count, bool)
                or not isinstance(total_count, int)
                or total_count != recipe["target_sample_count"]
            ):
                raise RuntimeError(
                    "EXL3 under-coverage recovery did not produce a valid "
                    f"capture for `{named_module.full_name}`"
                )
            recovery_mode = recovery_capture.get("recovery_mode")
            router_augmented_count = recovery_capture.get(
                "router_augmented_sample_count"
            )
            identity_count = recovery_capture.get("identity_calibration_count")
            if (
                recovery_mode
                not in {
                    ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR,
                    ZERO_ROUTE_RECOVERY_MODE_IDENTITY,
                    ZERO_ROUTE_RECOVERY_MODE_MIXED,
                }
                or isinstance(router_augmented_count, bool)
                or not isinstance(router_augmented_count, int)
                or router_augmented_count < 0
                or isinstance(identity_count, bool)
                or not isinstance(identity_count, int)
                or identity_count < 0
                or natural_count + router_augmented_count + identity_count
                != total_count
            ):
                raise RuntimeError(
                    "EXL3 under-coverage recovery metadata disagrees with the "
                    f"capture for `{named_module.full_name}`"
                )
            previous = recovered_by_expert.setdefault(
                identity["expert"],
                (recovery_mode, router_augmented_count, identity_count),
            )
            if previous != (
                recovery_mode,
                router_augmented_count,
                identity_count,
            ):
                raise RuntimeError(
                    "EXL3 route-coverage siblings captured different row counts"
                )
            task["zero_route_recovery"] = {
                "schema": ZERO_ROUTE_RECOVERY_SCHEMA,
                "schema_version": ZERO_ROUTE_RECOVERY_SCHEMA_VERSION,
                "trigger": recipe["trigger"],
                "sample_source": recipe["sample_source"],
                "capture_method": recipe["capture_method"],
                "selection_policy": recipe["selection_policy"],
                "candidate_rank_min": recipe["candidate_rank_min"],
                "candidate_rank_max": recipe["candidate_rank_max"],
                "selection_cap": recipe["selection_cap"],
                "target_sample_count": recipe["target_sample_count"],
                "identity_calibration_policy": recipe[
                    "identity_calibration_policy"
                ],
                "block_namespace": identity["block_namespace"],
                "logical_layer": identity["logical_layer"],
                "expert": identity["expert"],
                "natural_sample_count": natural_count,
                "router_augmented_sample_count": router_augmented_count,
                "identity_calibration_count": identity_count,
                "total_sample_count": total_count,
                "forced_pass_count": 1,
                "recovery_mode": recovery_mode,
                "candidate_rows_observed": recovery_capture[
                    "candidate_rows_observed"
                ],
                "candidate_rows_selected": recovery_capture[
                    "candidate_rows_selected"
                ],
                "candidate_rank_histogram": copy.deepcopy(
                    recovery_capture["candidate_rank_histogram"]
                ),
                "candidate_score_gap": copy.deepcopy(
                    recovery_capture["candidate_score_gap"]
                ),
                "authorization": copy.deepcopy(authorization),
            }
            task.pop("zero_route_recovery_capture", None)
        log.info(
            "EXL3 route-coverage top-up complete: experts=%s modules=%s rows=%s",
            len(recovered_by_expert),
            len(task_names),
            sorted(set(recovered_by_expert.values())),
        )

    def validate_subset_capture_readiness(
        self,
        *,
        subset: Dict[str, NamedModule],
        layer_module: Module,
    ) -> None:
        """Fail before fan-out unless every natural or recovered capture is exact."""

        provenance = self._ledger_provenance()
        if not route_evidence_required(provenance):
            return
        family_join = provenance.get("family_join")
        recipe = zero_route_recovery_recipe(family_join)
        authorization = self._zero_route_recovery_authorization(provenance)
        mlp = getattr(layer_module, "mlp", None)
        if mlp is None:
            mlp = getattr(layer_module, "ffn", None)
        router = getattr(mlp, "gate", None)
        learned_topk_router = has_learned_router_recovery(router)
        for task_name in sorted(subset):
            named_module = subset[task_name]
            identity = self._routed_expert_identity(
                getattr(named_module, "full_name", "")
            )
            task = self.tasks.get(task_name)
            if identity is None or not isinstance(task, dict):
                continue
            capture = task.get("capture")
            sample_count = getattr(capture, "nsamples", None)
            route_evidence = task.get("route_evidence")
            recovery = task.get("zero_route_recovery")
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or not isinstance(route_evidence, dict)
            ):
                raise RuntimeError(
                    "EXL3 capture readiness failed for "
                    f"`{named_module.full_name}`"
                )
            natural_count = route_evidence.get("expert_route_count")
            recovery_required = (
                authorization is not None
                and learned_topk_router
                and natural_count < recipe["target_sample_count"]
            )
            if recovery_required:
                if not isinstance(recovery, dict):
                    raise RuntimeError(
                        "EXL3 under-covered learned-router capture lacks recovery evidence for "
                        f"`{named_module.full_name}`"
                    )
                validate_zero_route_recovery(
                    recovery,
                    identity=identity,
                    sample_count=sample_count,
                    family_join=family_join,
                    expected_authorization=authorization,
                )
                validate_route_evidence(
                    route_evidence,
                    identity=identity,
                    sample_count=natural_count,
                    allow_zero=natural_count == 0,
                )
            else:
                if recovery is not None:
                    raise RuntimeError(
                        "EXL3 sufficient natural capture was relabeled as recovery for "
                        f"`{named_module.full_name}`"
                    )
                validate_route_evidence(
                    route_evidence,
                    identity=identity,
                    sample_count=sample_count,
                )

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
            # This short-lived config owns only the in-memory Hessian capture
            # for one projection. Enabling the generic completed-module
            # offloader creates one unused TemporaryDirectory per projection
            # (768 per DeepSeek V4 MoE block) and repeats that setup every
            # layer. The enclosing EXL3 config remains responsible for actual
            # module offload through its explicitly configured shared path.
            offload_to_disk=False,
            device=module_qcfg.device,
            pack_dtype=module_qcfg.pack_dtype,
            act_group_aware=False,
        )

        task = GPTQ(module=module, qcfg=capture_qcfg)
        task.expected_nsamples = getattr(self, "total_calibration_tokens", None)
        task.quantizer.configure(perchannel=True)

        identity = self._routed_expert_identity(module.full_name)
        capture_contract = None
        if identity is not None:
            configured_devices = list(
                getattr(self.qcfg, "moe_vram_strategy_devices", None) or []
            )
            devices = [torch.device(device) for device in configured_devices]
            if not devices:
                configured = getattr(self.qcfg, "device", None)
                if configured is not None:
                    devices = [torch.device(configured)]
            owner_policy = (
                self.qcfg.meta.get(HESSIAN_OWNER_POLICY_META)
                if isinstance(self.qcfg.meta, dict)
                else None
            )
            owner_device = resolve_hessian_owner_device(
                expert_index=identity["expert"],
                devices=devices,
                policy=owner_policy,
            )
            task.set_hessian_accumulator_device(owner_device)
            family = (
                identity["block_namespace"],
                identity["logical_layer"],
                identity["expert"],
            )
            owners = getattr(self, "_hessian_family_owners", None)
            if owners is None:
                owners = {}
                self._hessian_family_owners = owners
            pending_aliases = getattr(
                self, "_pending_hessian_family_aliases", None
            )
            if pending_aliases is None:
                pending_aliases = {}
                self._pending_hessian_family_aliases = pending_aliases
            if identity["projection"] == "w1":
                if family in owners:
                    raise RuntimeError("duplicate gate Hessian owner")
                owners[family] = (module.name, task)
                pending = pending_aliases.pop(family, [])
                for _alias_name, alias in pending:
                    alias.share_hessian_state_from(task)
                if pending:
                    owners.pop(family, None)
            elif identity["projection"] == "w3":
                owner = owners.pop(family, None)
                if owner is None:
                    pending_aliases.setdefault(family, []).append(
                        (module.name, task)
                    )
                else:
                    task.share_hessian_state_from(owner[1])
            capture_contract = {
                "schema": "ds4rt.exl3-shared-sharded-hessian",
                "schema_version": 1,
                "owner_device": str(owner_device),
                "owner_projection": (
                    "w1" if identity["projection"] in {"w1", "w3"}
                    else "w2"
                ),
                "capture_enabled": identity["projection"] != "w3",
            }
            if owner_policy is not None:
                capture_contract["owner_policy"] = copy.deepcopy(owner_policy)

        self.tasks[module.name] = {
            "capture": task,
            "qcfg": module_qcfg,
            "hessian_capture": capture_contract,
        }
        remote_client = self._remote_client_for_run(self._ledger_provenance())
        if remote_client is not None:
            if self._routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 distributed dispatch only accepts routed-expert projections"
                )

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from EXL3 work."""

        return self.tasks.get(module.name, False) is False

    def has_captured_input_ids(self, name: str) -> bool:
        """Report live or lazily restored calibration coverage."""

        task = self.tasks.get(name)
        if not isinstance(task, dict):
            return False
        capture = task.get("capture")
        return bool(getattr(capture, "nsamples", 0) > 0)

    def pre_process_fwd_hook(
        self, name: str
    ) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the EXL3 task."""

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for the EXL3 capture task."""

            capture = self.tasks[name]["capture"]
            batch_idx = self.current_batch_index()
            active_batch_capture = getattr(
                self, "_active_natural_route_capture", None
            )
            if active_batch_capture is not None:
                active_batch_capture.capture_expert_input(name, inp[0].data)
            capture.add_batch(inp[0].data, out.data, batch_index=batch_idx)
            del inp, out

        return tmp

    def _is_lm_head(self, module: NamedModule) -> bool:
        """Returns whether the named module corresponds to the model lm_head."""

        return module.full_name == self.lm_head_name or module.name == self.lm_head_name

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
        *,
        bits_override: int | None = None,
    ) -> Dict[str, object]:
        """Builds the argument bundle passed into the EXL3 quantizer."""

        quant_args: Dict[str, object] = {
            "K": (
                self._target_bits(module, module_qcfg)
                if bits_override is None
                else int(bits_override)
            ),
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

    def _restore_module_weight(
        self, module: NamedModule, quantized_weight: torch.Tensor
    ) -> torch.Tensor:
        """Reshapes the EXL3 output weight back into the wrapped module layout."""

        target = module.module if isinstance(module, NamedModule) else module

        if isinstance(target, transformers.Conv1D):
            return quantized_weight.contiguous().view_as(target.weight.data)

        if isinstance(target, (torch.nn.Linear, _ConvNd)):
            return quantized_weight.t().contiguous().view_as(target.weight.data)

        raise NotImplementedError(
            f"Unsupported EXL3 module type: {target.__class__.__name__}"
        )

    def _stage_runtime_weight(
        self,
        *,
        module: NamedModule,
        out_tensors: dict[str, torch.Tensor],
        target_device: torch.device,
    ) -> None:
        """Drop the dense source and defer reconstruction until layer replay.

        ``process()`` has already started streaming the packed result to the
        module's CPU state and, when configured, committed the same tensors to
        the projection checkpoint.  Reconstructing every result here makes one
        complete dense expert family accumulate either on CUDA or in host RAM
        while Hessian/capture state is still live.  A META placeholder retains
        only shape and dtype; the replay preparation hook reconstructs directly
        onto the final balanced forward device after quantization has released
        the Hessians.
        """

        del target_device
        required = {"trellis", "suh", "svh"}
        if not required.issubset(out_tensors):
            raise RuntimeError(
                f"EXL3 packed result is incomplete for `{module.full_name}`"
            )
        target = module.module
        weight = target.weight
        marker = {
            "dtype": str(weight.dtype),
            "shape": list(weight.shape),
            "requires_grad": bool(weight.requires_grad),
        }
        placeholder = torch.nn.Parameter(
            torch.empty(tuple(weight.shape), dtype=weight.dtype, device="meta"),
            requires_grad=weight.requires_grad,
        )
        with parent_module_lock(module.full_name):
            target.weight = placeholder
            module.state[_DEFERRED_RUNTIME_WEIGHT_STATE] = marker
        module.state.pop("quant_source_module", None)

    def prepare_runtime_weight_for_forward(
        self,
        *,
        module: NamedModule,
        target_device: torch.device | str,
    ) -> torch.nn.Module | None:
        """Materialize one deferred dense EXL3 replay weight on its final GPU."""

        marker = module.state.get(_DEFERRED_RUNTIME_WEIGHT_STATE)
        if marker is None:
            return None
        if not isinstance(marker, dict):
            raise RuntimeError(
                f"EXL3 deferred replay marker is malformed for `{module.full_name}`"
            )
        target = module.module
        if not getattr(target.weight, "is_meta", False):
            raise RuntimeError(
                f"EXL3 deferred replay weight is unexpectedly materialized for "
                f"`{module.full_name}`"
            )
        if marker.get("shape") != list(target.weight.shape) or marker.get(
            "dtype"
        ) != str(target.weight.dtype):
            raise RuntimeError(
                f"EXL3 deferred replay geometry changed for `{module.full_name}`"
            )

        module.stream_sync()
        packed_names = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
        packed = {
            name: module.state[name]
            for name in packed_names
            if isinstance(module.state.get(name), torch.Tensor)
        }
        if not {"trellis", "suh", "svh"}.issubset(packed):
            raise RuntimeError(
                f"EXL3 deferred replay tensors are incomplete for `{module.full_name}`"
            )

        target_device = torch.device(target_device)
        with self._distributed_local_quant_lock(target_device):
            runtime_weight = reconstruct_exl3_tensors(
                packed,
                device=target_device,
                dtype=target.weight.dtype,
            )
            restored_weight = self._restore_module_weight(module, runtime_weight)
            replay_weight = torch.nn.Parameter(
                restored_weight,
                requires_grad=bool(marker.get("requires_grad", False)),
            )
            with parent_module_lock(module.full_name):
                target.weight = replay_weight
                module.state.pop(_DEFERRED_RUNTIME_WEIGHT_STATE, None)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            del runtime_weight, restored_weight
        return target

    def prepare_layer_post_quantize(
        self,
        *,
        model: BaseQModel,
        layer_module: Module,
        layer_index: int,
        processed_modules: Dict[str, NamedModule],
        is_lm_head_module: bool,
    ) -> None:
        """Retire deferred META weights when a layer needs no final replay.

        EXL3 drops each dense source weight after packing and reconstructs it
        only for propagation replay. The terminal decoder layer has no next
        layer, so reconstructing its complete dense expert family would waste
        substantial memory and compute. Replace only those authenticated META
        placeholders with zero-sized CPU sentinels; ``submodule_finalize``
        immediately swaps each owner for its packed EXL3 module.
        """

        del model, layer_module, layer_index, is_lm_head_module
        for module in processed_modules.values():
            if not isinstance(module, NamedModule):
                continue
            marker = module.state.get(_DEFERRED_RUNTIME_WEIGHT_STATE)
            if marker is None:
                continue
            if not isinstance(marker, dict):
                raise RuntimeError(
                    f"EXL3 deferred replay marker is malformed for `{module.full_name}`"
                )
            target = module.module
            weight = getattr(target, "weight", None)
            if not isinstance(weight, torch.Tensor) or not weight.is_meta:
                raise RuntimeError(
                    "EXL3 terminal finalization expected a deferred META weight for "
                    f"`{module.full_name}`"
                )
            if marker.get("shape") != list(weight.shape) or marker.get(
                "dtype"
            ) != str(weight.dtype):
                raise RuntimeError(
                    f"EXL3 deferred replay geometry changed for `{module.full_name}`"
                )
            sentinel = torch.nn.Parameter(
                torch.empty(0, dtype=weight.dtype, device="cpu"),
                requires_grad=bool(marker.get("requires_grad", False)),
            )
            with parent_module_lock(module.full_name):
                target.weight = sentinel
                module.state[_DEFERRED_FINALIZE_WEIGHT_STATE] = marker
                module.state.pop(_DEFERRED_RUNTIME_WEIGHT_STATE, None)

    def _publish_inline_candidate(self, task_entry: dict[str, Any]) -> None:
        """Promote an unselected K2 candidate into the authoritative journal."""

        stat = task_entry.get("inline_candidate_stat")
        ledger_record = task_entry.get("inline_candidate_ledger_record")
        if not isinstance(stat, dict) or not isinstance(ledger_record, dict):
            raise RuntimeError("EXL3 inline-mixed candidate state is incomplete")
        with self._stats_lock:
            digest = append_exl3_error_journal(
                self.error_journal_path, ledger_record
            )
            stat["exl3_error_journal"] = self.error_journal_path
            stat["exl3_error_record_sha256"] = digest
            duration = float(stat[PROCESS_LOG_TIME])
            self.durations.append(duration)
            try:
                self.avg_losses.append(float(stat[QUANT_LOG_LOSS]))
            except (TypeError, ValueError):
                pass
            self.module_names.append(
                f"layer-{stat[PROCESS_LOG_LAYER]}-{stat[PROCESS_LOG_MODULE]}"
            )
            self.log.append(stat)

    def _requantize_inline_selected(
        self,
        *,
        model: BaseQModel,
        module: NamedModule,
        policy,
        tier_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Reload one candidate's source/Hessian at the selected higher tier."""

        task_entry = self.tasks[module.name]
        candidate_stat = task_entry.get("inline_candidate_stat")
        candidate_record = task_entry.get("inline_candidate_ledger_record")
        frontier = task_entry.get("capture_frontier_record")
        old_capture = task_entry.get("capture")
        if (
            not isinstance(candidate_stat, dict)
            or not isinstance(candidate_record, dict)
            or not isinstance(frontier, EXL3CaptureRecord)
            or not isinstance(old_capture, GPTQ)
        ):
            raise RuntimeError(
                "EXL3 selected tier lost its bounded replay state for "
                f"`{module.full_name}`"
            )
        candidate_request_sha256 = candidate_stat.get(
            "exl3_projection_checkpoint"
        )
        if not isinstance(candidate_request_sha256, str):
            raise RuntimeError("EXL3 selected tier lacks its candidate checkpoint")

        target_device = torch.device(
            task_entry.get("inline_source_device", self.qcfg.device)
        )
        prepared = model.shell_module_materialize(
            target_submodule=module.module,
            device=target_device,
            role="quant_source",
            named_module=module,
        )
        if prepared is not module.module:
            module.module = prepared

        capture = GPTQ(module=module, qcfg=old_capture.qcfg)
        capture.expected_nsamples = getattr(
            self, "total_calibration_tokens", None
        )
        capture.quantizer.configure(perchannel=True)
        task_entry["capture"] = capture

        inline_context = {
            "role": _inline_selected_role(policy),
            "policy_sha256": policy.policy_sha256,
            "base_bits": policy.base_bits,
            "upgrade_bits": policy.upgrade_bits,
            "candidate_request_sha256": candidate_request_sha256,
            "tier_plan_sha256": tier_plan["tier_plan_sha256"],
        }
        ledger_provenance = self._ledger_provenance()
        remote_client = self._remote_client_for_run(ledger_provenance)
        execution_lease = None
        execution_slot = None
        if remote_client is not None:
            execution_lease = remote_client.acquire_slot(
                self._projection_assignment_key(
                    module, phase=f"inline-k{policy.upgrade_bits}-selected"
                )
            )
            execution_slot = execution_lease.slot
        try:
            final_stat = self._process_on_slot(
                module=module,
                device=target_device,
                ledger_provenance=ledger_provenance,
                remote_client=remote_client,
                execution_slot=execution_slot,
                execution_lease=execution_lease,
                bits_override=policy.upgrade_bits,
                inline_context=inline_context,
                defer_stats=False,
            )
        finally:
            if execution_lease is not None:
                execution_lease.release()

        # The public per-module row describes the final selected-tier result,
        # while processor duration includes both searches this choice cost.
        candidate_duration = float(candidate_stat[PROCESS_LOG_TIME])
        with self._stats_lock:
            self.durations.append(candidate_duration)
            final_stat[PROCESS_LOG_TIME] = (
                f"{float(final_stat[PROCESS_LOG_TIME]) + candidate_duration:.3f}"
            )
        return final_stat

    def prepare_layer_replay(
        self,
        *,
        model: BaseQModel,
        layer_module: Module,
        layer_index: int,
        layer_count: int,
        processed_modules: Dict[str, NamedModule],
        is_lm_head_module: bool,
    ) -> None:
        """Choose and execute upgrades before the only propagation replay."""

        del layer_module
        routed_names = [
            module.full_name for module in processed_modules.values()
            if isinstance(module, NamedModule)
            and self._routed_expert_identity(module.full_name) is not None
        ]
        policy = self._inline_mixed_policy(routed_names[0]) if routed_names else None
        if policy is None or is_lm_head_module:
            return
        candidates: list[tuple[NamedModule, dict[str, Any], dict[str, Any]]] = []
        for module in processed_modules.values():
            if not isinstance(module, NamedModule):
                continue
            identity = self._routed_expert_identity(module.full_name)
            if identity is None or identity["block_namespace"] != policy.namespace:
                continue
            task_entry = self.tasks.get(module.name)
            if not isinstance(task_entry, dict):
                raise RuntimeError("EXL3 inline-mixed task disappeared")
            record = task_entry.get("inline_candidate_ledger_record")
            stat = task_entry.get("inline_candidate_stat")
            if not isinstance(record, dict) or not isinstance(stat, dict):
                raise RuntimeError(
                    "EXL3 inline base-tier candidate is absent for "
                    f"`{module.full_name}`"
                )
            candidates.append((module, task_entry, record))
        if not candidates:
            return

        logical_layers = {
            self._routed_expert_identity(module.full_name)["logical_layer"]
            for module, _task, _record in candidates
        }
        if len(logical_layers) != 1:
            raise RuntimeError("EXL3 inline-mixed candidates crossed logical layers")
        logical_layer = next(iter(logical_layers))
        if logical_layer != layer_index and policy.namespace == "base":
            raise RuntimeError("EXL3 inline-mixed base layer identity drifted")
        plan = build_layer_tier_plan(
            policy=policy,
            layer_index=logical_layer,
            layer_count=layer_count,
            candidate_records=[record for _module, _task, record in candidates],
        )
        plan = InlineMixedTierPlanStore(policy).commit(plan)
        selected = {entry["module"] for entry in plan["selected"]}
        candidate_by_name = {
            module.full_name: (module, task_entry)
            for module, task_entry, _record in candidates
        }
        if not selected.issubset(candidate_by_name):
            raise RuntimeError("EXL3 inline-mixed tier plan selected an unknown module")

        for module_name, (_module, task_entry) in candidate_by_name.items():
            if module_name not in selected:
                self._publish_inline_candidate(task_entry)

        # Reuse the normal per-device worker pool so coordinator GPUs and all
        # remote slots stay dynamically fed. Only one immutable Hessian is
        # hydrated per active worker; no layer-wide Hessian array is retained.
        from .. import DEVICE_THREAD_POOL

        futures = []
        for module_name in sorted(selected):
            module, task_entry = candidate_by_name[module_name]
            target_device = torch.device(
                task_entry.get("inline_source_device", self.qcfg.device)
            )
            futures.append(
                DEVICE_THREAD_POOL.submit(
                    target_device,
                    self._requantize_inline_selected,
                    model=model,
                    module=module,
                    policy=policy,
                    tier_plan=plan,
                )
            )
        for future in futures:
            future.result()

        log.info(
            "EXL3 inline-mixed layer finalized: namespace=%s layer=%s "
            "candidates=%s selected_k%s=%s quotas=%s plan=%s",
            policy.namespace,
            logical_layer,
            len(candidates),
            policy.upgrade_bits,
            len(selected),
            plan["quotas"],
            plan["tier_plan_sha256"],
        )

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Dynamically lease one slot, then quantize and stage one projection."""

        del subset, previous_subset, subset_index, subset_total

        ledger_provenance = self._ledger_provenance()
        policy = self._inline_mixed_policy(module.full_name)
        identity = self._routed_expert_identity(module.full_name)
        inline_context = None
        bits_override = None
        assignment_phase = None
        if (
            policy is not None
            and identity is not None
            and identity["block_namespace"] == policy.namespace
        ):
            bits_override = policy.base_bits
            assignment_phase = f"inline-k{policy.base_bits}-candidate"
            inline_context = {
                "role": _inline_candidate_role(policy),
                "policy_sha256": policy.policy_sha256,
                "base_bits": policy.base_bits,
                "upgrade_bits": policy.upgrade_bits,
            }
            if (
                self._projection_checkpoint_store_for_run(ledger_provenance)
                is None
                or self._capture_frontier_store_for_run(ledger_provenance)
                is None
            ):
                raise ValueError(
                    "EXL3 inline mixed quantization requires durable projection "
                    "checkpoints and capture frontiers"
                )
        remote_client = self._remote_client_for_run(ledger_provenance)
        execution_lease = None
        execution_slot = None
        if remote_client is not None:
            if self._routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 distributed dispatch only accepts routed-expert projections"
                )
            execution_lease = remote_client.acquire_slot(
                self._projection_assignment_key(
                    module, phase=assignment_phase
                )
            )
            execution_slot = execution_lease.slot
        try:
            return self._process_on_slot(
                module=module,
                device=device,
                ledger_provenance=ledger_provenance,
                remote_client=remote_client,
                execution_slot=execution_slot,
                execution_lease=execution_lease,
                bits_override=bits_override,
                inline_context=inline_context,
                defer_stats=inline_context is not None,
            )
        finally:
            if execution_lease is not None:
                execution_lease.release()

    def _process_on_slot(
        self,
        *,
        module: NamedModule,
        device: torch.device | None,
        ledger_provenance: dict[str, Any] | None,
        remote_client,
        execution_slot,
        execution_lease: ExecutionSlotLease | None,
        bits_override: int | None = None,
        inline_context: dict[str, Any] | None = None,
        defer_stats: bool = False,
    ):
        """Run one projection while its physical execution slot is exclusively held."""

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        task_entry = self.tasks[module.name]
        capture: GPTQ = task_entry["capture"]
        module_qcfg: EXL3Config = task_entry["qcfg"]

        target_device = device or get_device(module.module)
        if isinstance(execution_slot, CoordinatorSlot):
            target_device = execution_slot.device
        target_device = torch.device(target_device)
        if target_device.type != "cuda":
            raise ValueError("EXL3 quantization requires CUDA/HIP execution.")
        policy = self._inline_mixed_policy(module.full_name)
        candidate_pass = (
            isinstance(inline_context, dict)
            and policy is not None
            and inline_context.get("role") == _inline_candidate_role(policy)
        )
        if candidate_pass:
            task_entry["inline_source_device"] = str(target_device)

        restored_frontier = task_entry.get("capture_frontier_record") is not None
        accumulator_device = getattr(capture, "hessian_accumulator_device", None)
        staging_device = (
            torch.device("cpu")
            if restored_frontier
            else (accumulator_device or target_device)
        )
        self._hydrate_capture_frontier(
            task_entry=task_entry,
            capture=capture,
            target_device=staging_device,
            retain_for_inline=candidate_pass,
        )
        hessian = prepare_exl3_hessian(
            capture,
            target_device=staging_device,
            module_full_name=module.full_name,
        )

        remote_endpoint = None
        execution_contract = None
        projection_provenance = ledger_provenance
        if remote_client is not None:
            if self._routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 remote dispatch only accepts routed-expert projections"
                )
            if not isinstance(execution_slot, (CoordinatorSlot, RemoteEndpoint)):
                raise ValueError("EXL3 distributed execution slot is invalid")
            remote_endpoint = (
                execution_slot if isinstance(execution_slot, RemoteEndpoint) else None
            )
            execution_contract = remote_client.execution_contract(execution_slot)
            projection_provenance = copy.deepcopy(ledger_provenance)
            projection_provenance["execution"] = copy.deepcopy(execution_contract)
        quant_args = self._build_quant_args(
            module,
            module_qcfg,
            target_device,
            bits_override=bits_override,
        )
        input_weight = self._quant_input_weight(capture, staging_device)
        checkpoint_store = self._projection_checkpoint_store_for_run(
            ledger_provenance
        )
        if remote_client is not None and checkpoint_store is None:
            raise ValueError(
                "EXL3 remote dispatch requires a coordinator projection checkpoint"
            )
        checkpoint_request = None
        checkpoint_result = None
        checkpoint_hit = False
        if checkpoint_store is not None:
            family_join = (
                ledger_provenance.get("family_join")
                if isinstance(ledger_provenance, dict)
                else None
            )
            quantizer_contract = {
                "bits": int(quant_args["K"]),
                "codebook": module_qcfg.codebook,
                "apply_out_scales": quant_args["apply_out_scales"],
                "sigma_reg": float(quant_args["sigma_reg"]),
                "seed": int(quant_args["seed"]),
                "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
                "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
                "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
            }
            if inline_context is not None:
                quantizer_contract["inline_mixed"] = copy.deepcopy(
                    inline_context
                )
            hessian_ownership = task_entry.get("hessian_capture")
            if isinstance(hessian_ownership, dict):
                quantizer_contract["hessian_ownership"] = copy.deepcopy(
                    hessian_ownership
                )
            if execution_contract is not None:
                quantizer_contract["execution"] = copy.deepcopy(execution_contract)
            checkpoint_request = build_projection_request(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                input_weight=input_weight,
                hessian=hessian,
                sample_count=capture.nsamples,
                quantizer_contract=quantizer_contract,
                family_join=family_join,
                route_evidence=task_entry.get("route_evidence"),
                zero_route_recovery=task_entry.get("zero_route_recovery"),
            )
            checkpoint_store.reserve_module_request(checkpoint_request)
            loaded_checkpoint = checkpoint_store.load(checkpoint_request)
        else:
            loaded_checkpoint = None

        if loaded_checkpoint is None:
            if remote_endpoint is not None:
                remote_started = time.perf_counter()
                out_tensors, remote_result, transport = remote_client.quantize(
                    endpoint=remote_endpoint,
                    request_manifest=checkpoint_request,
                    input_weight=input_weight,
                    hessian=hessian,
                )
                remote_elapsed = time.perf_counter() - remote_started
                duration = remote_result.get("duration_seconds")
                proxy_err = remote_result.get("proxy_error")
                device_names = remote_result.get("device_names")
                quantizer_metrics = remote_result.get("quantizer_metrics")
                worker = remote_result.get("worker")
                worker_queue_wait_seconds = remote_result.get(
                    "worker_queue_wait_seconds"
                )
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or isinstance(proxy_err, bool)
                    or not isinstance(proxy_err, (int, float))
                    or not isinstance(device_names, list)
                    or not all(isinstance(value, str) for value in device_names)
                    or not isinstance(quantizer_metrics, dict)
                    or not isinstance(worker, dict)
                    or isinstance(worker_queue_wait_seconds, bool)
                    or not isinstance(worker_queue_wait_seconds, (int, float))
                    or not math.isfinite(worker_queue_wait_seconds)
                    or worker_queue_wait_seconds < 0
                ):
                    raise RuntimeError(
                        f"EXL3 remote worker returned malformed results for `{module.full_name}`."
                    )
                execution_result = {
                    "kind": "remote_worker",
                    "worker": copy.deepcopy(worker),
                    "transport": copy.deepcopy(transport),
                    "coordinator_elapsed_seconds": remote_elapsed,
                    "scheduler_wait_seconds": execution_lease.wait_seconds,
                    "scheduler_new_assignment": execution_lease.new_assignment,
                    "scheduler_assignment_key": execution_lease.assignment_key,
                    "worker_queue_wait_seconds": worker_queue_wait_seconds,
                }
            else:
                wait_started = time.perf_counter()
                # EXL3 caches trellis scratch tensors and a quantization stream
                # per physical device.  The device worker pool may contain
                # several host workers for staging/transport overlap, but two
                # quantize_exl3 calls on the same GPU would alias that scratch
                # state.  Serialize the device-local kernel region for both
                # local-only and distributed runs; different GPUs retain their
                # independent locks and still execute concurrently.
                quant_lock = self._distributed_local_quant_lock(target_device)
                with quant_lock:
                    if hessian.device != target_device:
                        hessian = hessian.to(
                            device=target_device,
                            dtype=torch.float32,
                            non_blocking=False,
                            copy=True,
                        )
                    if input_weight.device != target_device:
                        input_weight = input_weight.to(
                            device=target_device,
                            dtype=torch.float32,
                            non_blocking=False,
                            copy=True,
                        )
                    torch.cuda.synchronize(target_device)
                    h_data = {
                        "H": hessian,
                        "single_use": True,
                        "count": capture.nsamples,
                        "finalized": False,
                    }
                    quant_started = time.perf_counter()
                    try:
                        _weight_q, proxy_err, out_tensors = quantize_exl3(
                            weight=input_weight,
                            H_data=h_data,
                            quant_args=quant_args,
                            return_weight_q=False,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            exl3_quantization_failure_message(
                                error=error,
                                module_full_name=module.full_name,
                                request_sha256=(
                                    checkpoint_request.get("request_sha256")
                                    if checkpoint_request is not None
                                    else None
                                ),
                                hessian=hessian,
                                sample_count=capture.nsamples,
                                sigma_reg=float(quant_args["sigma_reg"]),
                            )
                        ) from error
                    del _weight_q
                    duration = time.perf_counter() - quant_started
                device_names = [str(device) for device in quant_args["devices"]]
                quantizer_metrics = quant_args.get("error_metrics")
                execution_result = {
                    "kind": "coordinator",
                    "coordinator_quant_lock_wait_seconds": quant_started - wait_started,
                }
                if execution_lease is not None:
                    execution_result.update(
                        {
                            "scheduler_wait_seconds": execution_lease.wait_seconds,
                            "scheduler_new_assignment": execution_lease.new_assignment,
                            "scheduler_assignment_key": execution_lease.assignment_key,
                        }
                    )
            if not isinstance(quantizer_metrics, dict):
                raise RuntimeError(
                    f"EXL3 quantizer returned no structured error metrics for `{module.full_name}`."
                )
            try:
                validate_exl3_hessian_metrics(
                    quantizer_metrics,
                    sample_count=capture.nsamples,
                    sigma_reg=float(quant_args["sigma_reg"]),
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{error} for `{module.full_name}`"
                ) from error
            if isinstance(proxy_err, torch.Tensor):
                proxy_err = proxy_err.item()
            encoded_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in out_tensors.values()
                if isinstance(tensor, torch.Tensor)
            )
            ledger_record = build_projection_record(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                bits=int(quant_args["K"]),
                codebook=module_qcfg.codebook,
                sample_count=capture.nsamples,
                duration_seconds=duration,
                encoded_bytes=encoded_bytes,
                device_names=device_names,
                quantizer_metrics=quantizer_metrics,
                provenance=projection_provenance,
                route_evidence=task_entry.get("route_evidence"),
                zero_route_recovery=task_entry.get("zero_route_recovery"),
            )
            checkpoint_result = {
                "duration_seconds": duration,
                "proxy_error": proxy_err,
                "device_names": device_names,
                "quantizer_metrics": quantizer_metrics,
                "ledger_record": ledger_record,
                "execution_contract": execution_contract,
                "execution_result": execution_result,
            }
            if checkpoint_store is not None:
                checkpoint_store.commit(
                    checkpoint_request,
                    out_tensors,
                    checkpoint_result,
                )
        else:
            out_tensors, checkpoint_result = loaded_checkpoint
            duration = checkpoint_result.get("duration_seconds")
            proxy_err = checkpoint_result.get("proxy_error")
            device_names = checkpoint_result.get("device_names")
            quantizer_metrics = checkpoint_result.get("quantizer_metrics")
            ledger_record = checkpoint_result.get("ledger_record")
            stored_execution_contract = checkpoint_result.get("execution_contract")
            execution_result = checkpoint_result.get("execution_result")
            encoded_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in out_tensors.values()
                if isinstance(tensor, torch.Tensor)
            )
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not isinstance(device_names, list)
                or not all(isinstance(value, str) for value in device_names)
                or not isinstance(quantizer_metrics, dict)
                or not isinstance(ledger_record, dict)
                or stored_execution_contract != execution_contract
                or (remote_client is not None and not isinstance(execution_result, dict))
            ):
                raise ValueError("EXL3 projection checkpoint result is malformed")
            expected_ledger_record = build_projection_record(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                bits=int(quant_args["K"]),
                codebook=module_qcfg.codebook,
                sample_count=capture.nsamples,
                duration_seconds=duration,
                encoded_bytes=encoded_bytes,
                device_names=device_names,
                quantizer_metrics=quantizer_metrics,
                provenance=projection_provenance,
                route_evidence=task_entry.get("route_evidence"),
                zero_route_recovery=task_entry.get("zero_route_recovery"),
            )
            if ledger_record != expected_ledger_record:
                legacy_ledger_record = build_projection_record(
                    module_full_name=module.full_name,
                    layer_index=module.layer_index,
                    bits=int(quant_args["K"]),
                    codebook=module_qcfg.codebook,
                    sample_count=capture.nsamples,
                    duration_seconds=duration,
                    encoded_bytes=encoded_bytes,
                    device_names=device_names,
                    quantizer_metrics=quantizer_metrics,
                    provenance=projection_provenance,
                    route_evidence=task_entry.get("route_evidence"),
                    zero_route_recovery=task_entry.get("zero_route_recovery"),
                    compact_provenance=False,
                )
                if ledger_record != legacy_ledger_record:
                    raise ValueError(
                        "EXL3 projection checkpoint ledger is inconsistent"
                    )
            checkpoint_hit = True

        if execution_lease is not None and isinstance(execution_slot, RemoteEndpoint):
            # The Spark is free as soon as the packed result is durable. Do not
            # strand it while the coordinator reconstructs/stages that result.
            execution_lease.release()

        # The packed result and its exact ledger are now durable when the run
        # opted into projection checkpoints. The journal remains the ordered
        # coordinator commit barrier before tensors enter async save state.
        journal_path = (
            self.inline_mixed_candidate_journal_path
            if candidate_pass
            else self.error_journal_path
        )
        with self._stats_lock:
            ledger_record_sha256 = append_exl3_error_journal(
                journal_path,
                ledger_record,
            )
        publication_ledger_record = compact_projection_record(ledger_record)

        stream_payload = dict(out_tensors)
        if module.bias is not None:
            stream_payload["bias"] = module.bias.detach()
        module.stream_state_payload_to_cpu(stream_payload)

        self._stage_runtime_weight(
            module=module,
            out_tensors=out_tensors,
            target_device=target_device,
        )

        workspace_summary = getattr(capture, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(capture, "_borrow_workspace_totals", None)

        if isinstance(proxy_err, str):
            loss_display = proxy_err
        else:
            loss_display = (
                f"{proxy_err:.10f}"
                if isinstance(proxy_err, (int, float))
                else "unknown"
            )

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
            QUANT_LOG_LOSS_KIND: quantizer_metrics["reported_metric_kind"],
            "exl3_error_ledger_record": publication_ledger_record,
            "exl3_error_journal": journal_path,
            "exl3_error_record_sha256": ledger_record_sha256,
            "exl3_projection_checkpoint": (
                checkpoint_request.get("request_sha256")
                if checkpoint_request is not None
                else None
            ),
            "exl3_projection_checkpoint_hit": checkpoint_hit,
            "exl3_execution_contract": execution_contract,
            "exl3_execution_result": execution_result,
            "exl3_inline_mixed": copy.deepcopy(inline_context),
        }

        if workspace_summary:
            requests = int(workspace_summary.get("requests", 0) or 0)
            if requests:
                hit_rate = float(workspace_summary.get("hit_rate", 0.0) or 0.0)
                chunk_rows = workspace_summary.get("chunk_rows")
                stat["workspace_cache_requests"] = str(requests)
                stat["workspace_cache_hit_rate"] = f"{hit_rate:.1%}"
                stat["workspace_stage_dtype"] = workspace_summary.get(
                    "staging_dtype", ""
                )
                if chunk_rows is not None:
                    stat["workspace_chunk_rows"] = str(chunk_rows)
        if workspace_totals:
            total_requests = int(workspace_totals.get("requests", 0) or 0)
            if total_requests:
                cumulative_hit_rate = (
                    float(workspace_totals.get("materialized_hits", 0) or 0.0)
                    / total_requests
                )
                stat["workspace_total_requests"] = str(total_requests)
                stat["workspace_total_hit_rate"] = f"{cumulative_hit_rate:.1%}"

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self._stats_lock:
            if defer_stats:
                task_entry["inline_candidate_stat"] = stat
                task_entry["inline_candidate_ledger_record"] = ledger_record
            else:
                self.durations.append(duration)
                if isinstance(proxy_err, (int, float)):
                    self.avg_losses.append(proxy_err)
                self.module_names.append(
                    f"layer-{module.layer_index}-{module.name}"
                )
                self.log.append(stat)

        self.log_new_row(stat)

        capture.free()
        del input_weight, out_tensors, stream_payload
        return stat

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Builds and installs the ExLlamaV3 module from the staged tensors."""

        del kwargs

        module.stream_sync()

        finalize_marker = module.state.pop(_DEFERRED_FINALIZE_WEIGHT_STATE, None)
        if finalize_marker is not None:
            weight = getattr(module.module, "weight", None)
            if (
                not isinstance(finalize_marker, dict)
                or not isinstance(weight, torch.Tensor)
                or weight.device.type != "cpu"
                or weight.numel() != 0
                or str(weight.dtype) != finalize_marker.get("dtype")
            ):
                raise RuntimeError(
                    "EXL3 terminal finalize sentinel is invalid for "
                    f"`{module.full_name}`"
                )

        tensors: Dict[str, torch.Tensor] = {}
        with self._stats_lock:
            module.state.pop("w", None)
            for tensor_name in (
                "trellis",
                "suh",
                "svh",
                "su",
                "sv",
                "bias",
                "mcg",
                "mul1",
            ):
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

    def completed_layer_checkpoint_entries(
        self, layer_index: int
    ) -> list[dict[str, str]]:
        """Return the packed/error identities committed for one decoder layer."""

        entries: dict[str, dict[str, str]] = {}
        with self._stats_lock:
            log_snapshot = list(self.log)
        for stat in log_snapshot:
            if stat.get(PROCESS_LOG_LAYER) != layer_index:
                continue
            record = stat.get("exl3_error_ledger_record")
            request_sha256 = stat.get("exl3_projection_checkpoint")
            record_sha256 = stat.get("exl3_error_record_sha256")
            module_name = record.get("module") if isinstance(record, dict) else None
            if not all(
                isinstance(value, str) and value
                for value in (module_name, request_sha256, record_sha256)
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has an uncommitted projection result"
                )
            entry = {
                "module": module_name,
                "request_sha256": request_sha256,
                "record_sha256": record_sha256,
            }
            previous = entries.setdefault(module_name, entry)
            if previous != entry:
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has conflicting projection identities"
                )
        return [entries[name] for name in sorted(entries)]

    def _completed_checkpoint_tree_entries(
        self,
        *,
        block_namespace: str,
        layer_count: int,
        experts_per_layer: int,
    ) -> list[list[dict[str, str]]] | None:
        """Discover an authoritative complete packed tree without tensor I/O.

        The manifest scan is deliberately read-only.  Missing projections or
        selected higher-tier replacements mean the tree is incomplete and the
        normal
        quantization resume path must run.  Any committed identity, policy, or
        tier-plan contradiction is corruption and raises instead of silently
        rebuilding a different model.
        """

        if block_namespace not in {"base", "mtp"}:
            raise ValueError("EXL3 checkpoint restore namespace is invalid")
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count <= 0
            or isinstance(experts_per_layer, bool)
            or not isinstance(experts_per_layer, int)
            or experts_per_layer <= 0
        ):
            raise ValueError("EXL3 checkpoint restore geometry is invalid")
        provenance = self._ledger_provenance()
        checkpoint_root = checkpoint_root_from_provenance(provenance)
        if checkpoint_root is None:
            return None
        family_join = (
            provenance.get("family_join")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(family_join, dict):
            raise ValueError("EXL3 checkpoint restore has no family join")
        store = EXL3ProjectionCheckpointStore(checkpoint_root)
        policy = self._inline_mixed_policy(namespace=block_namespace)
        if policy is not None and policy.namespace != block_namespace:
            policy = None

        projection_names = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }
        block_prefix = "model.layers" if block_namespace == "base" else "mtp"
        expected: dict[str, dict[str, Any]] = {
            (
                f"{block_prefix}.{layer}.mlp.experts.{expert}."
                f"{projection_names[projection]}"
            ): {
                "logical_layer": layer,
                "expert": expert,
                "projection": projection,
            }
            for layer in range(layer_count)
            for expert in range(experts_per_layer)
            for projection in PROJECTION_ORDER
        }
        inventory: dict[
            str, dict[str, tuple[dict[str, Any], dict[str, Any], str]]
        ] = {}
        relevant_count = 0
        for request, result in store.inspect_committed_manifests():
            module = request.get("module")
            if not isinstance(module, str):
                raise ValueError("EXL3 checkpoint restore found no module identity")
            identity = self._routed_expert_identity(module)
            if identity is None:
                if module.startswith(f"{block_prefix}."):
                    raise ValueError(
                        f"EXL3 checkpoint restore found malformed `{module}`"
                    )
                continue
            if identity["block_namespace"] != block_namespace:
                continue
            relevant_count += 1
            expected_identity = expected.get(module)
            if expected_identity != {
                key: identity[key]
                for key in ("logical_layer", "expert", "projection")
            }:
                raise ValueError(
                    f"EXL3 checkpoint restore found unexpected `{module}`"
                )
            layer_index = identity["logical_layer"]
            contract = request.get("quantizer_contract")
            ledger_record = result.get("ledger_record")
            ledger_provenance = (
                ledger_record.get("provenance")
                if isinstance(ledger_record, dict)
                else None
            )
            if (
                request.get("processor_layer_index") != layer_index
                or request.get("family_join") != family_join
                or not isinstance(contract, dict)
                or not isinstance(ledger_record, dict)
                or ledger_record.get("module") != module
                or ledger_record.get("processor_layer_index") != layer_index
                or any(
                    ledger_record.get(key) != value
                    for key, value in {
                        "block_namespace": block_namespace,
                        **expected_identity,
                    }.items()
                )
                or ledger_record.get("codebook") != contract.get("codebook")
                or ledger_record.get("bits") != contract.get("bits")
                or not isinstance(ledger_provenance, dict)
                or ledger_provenance.get("family_join") != family_join
            ):
                raise ValueError(
                    f"EXL3 checkpoint restore identity differs for `{module}`"
                )
            _module, role = store._module_request_key(request)
            if _module != module:
                raise ValueError("EXL3 checkpoint restore module key differs")
            record_sha256 = sha256_bytes(canonical_json_bytes(ledger_record))
            roles = inventory.setdefault(module, {})
            if role in roles:
                raise ValueError(
                    f"EXL3 checkpoint restore has duplicate {role} for `{module}`"
                )
            roles[role] = (request, result, record_sha256)

        if relevant_count == 0:
            return None
        candidate_role = (
            _inline_candidate_role(policy) if policy is not None else None
        )
        selected_role = _inline_selected_role(policy) if policy is not None else None
        if policy is not None and any(
            selected_role in roles and candidate_role not in roles
            for roles in inventory.values()
        ):
            raise ValueError("EXL3 selected checkpoint has no base-tier candidate")
        # Missing base-tier/uniform projections are ordinary partial-run state. Do
        # not install even one packed module until the complete tree is proven.
        required_primary = candidate_role if policy is not None else "uniform"
        if any(required_primary not in inventory.get(module, {}) for module in expected):
            return None

        entries_by_layer: list[list[dict[str, str]]] = []
        for layer_index in range(layer_count):
            selected: dict[str, dict[str, Any]] = {}
            tier_plan_sha256: str | None = None
            if policy is not None:
                tier_plan = InlineMixedTierPlanStore(policy).load(
                    layer_index=layer_index,
                    layer_count=layer_count,
                    experts_per_layer=experts_per_layer,
                )
                if tier_plan is None:
                    return None
                tier_plan_sha256 = tier_plan["tier_plan_sha256"]
                for selection in tier_plan["selected"]:
                    module = selection["module"]
                    expected_identity = expected.get(module)
                    if (
                        expected_identity is None
                        or expected_identity["logical_layer"] != layer_index
                        or selection["expert"] != expected_identity["expert"]
                        or selection["projection"] != expected_identity["projection"]
                        or module in selected
                    ):
                        raise ValueError(
                            "EXL3 checkpoint restore tier selection differs from modules"
                        )
                    candidate = inventory[module][candidate_role]
                    candidate_record = candidate[1]["ledger_record"]
                    if (
                        selection["candidate_record_sha256"] != candidate[2]
                        or float(selection["score"])
                        != projection_score(candidate_record)
                    ):
                        raise ValueError(
                            f"EXL3 checkpoint restore tier score differs for `{module}`"
                        )
                    selected[module] = selection

            layer_entries: list[dict[str, str]] = []
            for module, expected_identity in expected.items():
                if expected_identity["logical_layer"] != layer_index:
                    continue
                roles = inventory[module]
                if policy is None:
                    if set(roles) != {"uniform"}:
                        raise ValueError(
                            f"EXL3 uniform checkpoint roles differ for `{module}`"
                        )
                    request, result, record_sha256 = roles["uniform"]
                    if request["quantizer_contract"].get("bits") != self.qcfg.bits:
                        raise ValueError(
                            f"EXL3 uniform checkpoint bits differ for `{module}`"
                        )
                else:
                    candidate_request = roles[candidate_role][0]
                    candidate_inline = candidate_request["quantizer_contract"].get(
                        "inline_mixed"
                    )
                    expected_candidate_inline = {
                        "base_bits": policy.base_bits,
                        "upgrade_bits": policy.upgrade_bits,
                        "policy_sha256": policy.policy_sha256,
                        "role": candidate_role,
                    }
                    if (
                        candidate_inline != expected_candidate_inline
                        or candidate_request["quantizer_contract"].get("bits")
                        != policy.base_bits
                    ):
                        raise ValueError(
                            f"EXL3 candidate checkpoint policy differs for `{module}`"
                        )
                    upgraded = module in selected
                    expected_roles = {candidate_role}
                    if upgraded:
                        expected_roles.add(selected_role)
                    if not expected_roles.issubset(roles):
                        return None
                    if set(roles) != expected_roles:
                        raise ValueError(
                            f"EXL3 selected checkpoint roles differ for `{module}`"
                        )
                    if upgraded:
                        request, result, record_sha256 = roles[selected_role]
                        expected_selected_inline = {
                            **expected_candidate_inline,
                            "role": selected_role,
                            "candidate_request_sha256": candidate_request[
                                "request_sha256"
                            ],
                            "tier_plan_sha256": tier_plan_sha256,
                        }
                        if (
                            request["quantizer_contract"].get("inline_mixed")
                            != expected_selected_inline
                            or request["quantizer_contract"].get("bits")
                            != policy.upgrade_bits
                        ):
                            raise ValueError(
                                f"EXL3 selected checkpoint binding differs for `{module}`"
                            )
                    else:
                        request, result, record_sha256 = roles[candidate_role]
                layer_entries.append(
                    {
                        "module": module,
                        "request_sha256": request["request_sha256"],
                        "record_sha256": record_sha256,
                    }
                )
            entries_by_layer.append(sorted(layer_entries, key=lambda item: item["module"]))
        return entries_by_layer

    @classmethod
    def restore_completed_checkpoint_tree_if_complete(
        cls,
        *,
        model: BaseQModel,
        block_namespace: str,
        layer_count: int,
        experts_per_layer: int,
        error_journal_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        """Install a complete durable packed tree and skip corpus replay."""

        processor = cls.__new__(cls)
        processor.qcfg = model.quantize_config
        processor.error_journal_path = os.fspath(
            error_journal_path
            or os.getenv(JOURNAL_ENV)
            or Path("gptq_log") / "exl3-error-ledger.jsonl"
        )
        processor._stats_lock = threading.Lock()
        processor.durations = []
        processor.avg_losses = []
        processor.module_names = []
        processor.log = []
        entries_by_layer = processor._completed_checkpoint_tree_entries(
            block_namespace=block_namespace,
            layer_count=layer_count,
            experts_per_layer=experts_per_layer,
        )
        if entries_by_layer is None:
            return False
        for layer_index, entries in enumerate(entries_by_layer):
            processor.restore_completed_layer_checkpoints(
                model=model,
                layer_index=layer_index,
                projection_entries=entries,
            )
        model.quant_log = list(processor.log)
        model.quantized = True
        model.quantize_config.method = METHOD.EXL3
        model.quantize_config.format = FORMAT.EXL3
        model.qlinear_kernel = ExllamaV3Linear
        return True

    def restore_completed_layer_checkpoints(
        self,
        *,
        model: BaseQModel,
        layer_index: int,
        projection_entries: list[dict[str, str]],
        materialize_device: torch.device | str | None = None,
    ) -> None:
        """Install packed modules directly, without Hessian/corpus reconstruction."""

        ledger_provenance = self._ledger_provenance()
        checkpoint_root = checkpoint_root_from_provenance(ledger_provenance)
        if checkpoint_root is None:
            raise RuntimeError("EXL3 layer restore requires projection checkpoints")
        checkpoint_store = EXL3ProjectionCheckpointStore(checkpoint_root)
        family_join = (
            ledger_provenance.get("family_join")
            if isinstance(ledger_provenance, dict)
            else None
        )
        offload_path = getattr(self.qcfg, "offload_to_disk_path", None)
        if not getattr(self.qcfg, "offload_to_disk", False) or not offload_path:
            raise RuntimeError("EXL3 layer restore requires durable disk offload")

        # A newly completed layer is already represented in ``self.log``.  The
        # layer-boundary controller may nevertheless restore its packed modules
        # from checkpoints during deferred publication.  Treat that overlap as
        # an idempotent module materialization, not as a second publication of
        # the same projection.  Conflicting or already-duplicated history must
        # fail closed before any module is replaced.
        existing_stats: dict[str, dict[str, Any]] = {}
        with self._stats_lock:
            log_snapshot = list(self.log)
        for stat in log_snapshot:
            if stat.get(PROCESS_LOG_LAYER) != layer_index:
                continue
            record = stat.get("exl3_error_ledger_record")
            module_name = record.get("module") if isinstance(record, dict) else None
            request_sha256 = stat.get("exl3_projection_checkpoint")
            record_sha256 = stat.get("exl3_error_record_sha256")
            if not all(
                isinstance(value, str) and value
                for value in (module_name, request_sha256, record_sha256)
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has an uncommitted projection result"
                )
            if module_name in existing_stats:
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has duplicate projection history "
                    f"for `{module_name}`"
                )
            existing_stats[module_name] = stat

        restored_stats: list[dict[str, Any]] = []
        seen_modules: set[str] = set()
        for entry in sorted(projection_entries, key=lambda item: item.get("module", "")):
            module_name = entry.get("module")
            request_sha256 = entry.get("request_sha256")
            expected_record_sha256 = entry.get("record_sha256")
            if (
                not isinstance(module_name, str)
                or module_name in seen_modules
                or not isinstance(request_sha256, str)
                or not isinstance(expected_record_sha256, str)
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} restore index is malformed"
                )
            loaded = checkpoint_store.load_committed(request_sha256)
            if loaded is None:
                raise RuntimeError(
                    f"EXL3 packed checkpoint disappeared for `{module_name}`"
                )
            request, tensors, result = loaded
            ledger_record = result.get("ledger_record")
            calculated_record_sha256 = (
                sha256_bytes(canonical_json_bytes(ledger_record))
                if isinstance(ledger_record, dict)
                else None
            )
            if (
                request.get("module") != module_name
                or request.get("processor_layer_index") != layer_index
                or request.get("family_join") != family_join
                or calculated_record_sha256 != expected_record_sha256
                or not isinstance(ledger_record, dict)
                or ledger_record.get("module") != module_name
                or ledger_record.get("processor_layer_index") != layer_index
            ):
                raise RuntimeError(
                    f"EXL3 packed checkpoint identity differs for `{module_name}`"
                )
            actual_record_sha256 = append_exl3_error_journal(
                self.error_journal_path, ledger_record
            )
            publication_ledger_record = compact_projection_record(ledger_record)
            existing_stat = existing_stats.get(module_name)
            if existing_stat is not None and (
                existing_stat.get("exl3_projection_checkpoint") != request_sha256
                or existing_stat.get("exl3_error_record_sha256")
                != actual_record_sha256
                or existing_stat.get("exl3_error_ledger_record")
                != publication_ledger_record
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has conflicting projection history "
                    f"for `{module_name}`"
                )
            try:
                source_module = model.model.get_submodule(module_name)
            except AttributeError as error:
                raise RuntimeError(
                    f"EXL3 restore target is absent: `{module_name}`"
                ) from error
            named_source = source_module
            if isinstance(source_module, ExllamaV3Linear):
                trellis = getattr(source_module, "trellis", None)
                if trellis is None or trellis.device.type != "meta":
                    raise RuntimeError(
                        f"EXL3 restore target is already packed: `{module_name}`"
                    )
                named_source = torch.nn.Linear(
                    source_module.in_features,
                    source_module.out_features,
                    bias=False,
                    dtype=source_module.out_dtype,
                    device="meta",
                )
            bias = getattr(source_module, "bias", None)
            checkpoint_includes_bias = "bias" in tensors
            if bias is not None:
                if getattr(bias, "is_meta", False):
                    raise RuntimeError(
                        f"EXL3 restore cannot recover META bias for `{module_name}`"
                    )
                tensors = {**tensors, "bias": bias.detach().to(device="cpu")}
            relative_name = module_name.removeprefix(
                f"model.layers.{layer_index}."
            )
            named = NamedModule(
                named_source,
                name=relative_name,
                full_name=module_name,
                layer_index=layer_index,
            )
            packed = create_exllamav3_module(
                module_root=model.model,
                name=module_name,
                submodule=named,
                tensors=tensors,
            )
            if materialize_device is None:
                # Preserve scalar multiplier values before replacing every
                # packed buffer with META storage. The saver streams directly
                # from the already authenticated projection checkpoint, so a
                # complete run does not create a second packed-weight copy.
                packed.tensor_storage = packed.tensor_storage_entry()
                if bias is None or checkpoint_includes_bias:
                    offload_to_safetensors_reference(
                        model=model.model,
                        module=packed,
                        source_path=checkpoint_store.committed_tensor_path(
                            request_sha256
                        ),
                        disk_path=offload_path,
                        module_name=module_name,
                    )
                else:
                    offload_to_disk(
                        model=model.model,
                        module=packed,
                        disk_path=offload_path,
                    )
            else:
                target_device = torch.device(materialize_device)
                if target_device.type == "cpu":
                    raise RuntimeError(
                        "EXL3 catch-up replay requires an accelerator device"
                    )
                packed.to(device=target_device)
                setattr(packed, "target_device", target_device)

            duration = result.get("duration_seconds")
            proxy_error = result.get("proxy_error")
            quantizer_metrics = result.get("quantizer_metrics")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not isinstance(quantizer_metrics, dict)
            ):
                raise RuntimeError(
                    f"EXL3 packed checkpoint result is malformed for `{module_name}`"
                )
            if existing_stat is None:
                restored_stats.append(
                    {
                        PROCESS_LOG_NAME: self.name(),
                        PROCESS_LOG_LAYER: layer_index,
                        PROCESS_LOG_MODULE: relative_name,
                        MODULE_FEATURE_COLUMN: self.module_feature_summary(named),
                        DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named),
                        QUANT_LOG_LOSS: (
                            f"{proxy_error:.10f}"
                            if isinstance(proxy_error, (int, float))
                            else str(proxy_error)
                        ),
                        QUANT_LOG_NSAMPLES: str(request.get("sample_count")),
                        QUANT_LOG_DAMP: f"{_EXL3_SIGMA_REG:.5f}",
                        PROCESS_LOG_TIME: f"{float(duration):.3f}",
                        PROCESS_LOG_FWD_TIME: "0.000",
                        PROCESS_USED_MEMORY: "restored",
                        QUANT_LOG_LOSS_KIND: quantizer_metrics.get(
                            "reported_metric_kind", "unknown"
                        ),
                        "exl3_error_ledger_record": publication_ledger_record,
                        "exl3_error_journal": self.error_journal_path,
                        "exl3_error_record_sha256": actual_record_sha256,
                        "exl3_projection_checkpoint": request_sha256,
                        "exl3_projection_checkpoint_hit": True,
                        "exl3_execution_contract": result.get(
                            "execution_contract"
                        ),
                        "exl3_execution_result": result.get("execution_result"),
                        "exl3_layer_boundary_restore": True,
                    }
                )
            seen_modules.add(module_name)

        with self._stats_lock:
            for stat in restored_stats:
                self.log.append(stat)
                self.module_names.append(
                    f"layer-{layer_index}-{stat[PROCESS_LOG_MODULE]}"
                )
                proxy_error = stat[QUANT_LOG_LOSS]
                try:
                    self.avg_losses.append(float(proxy_error))
                except (TypeError, ValueError):
                    pass
                self.durations.append(float(stat[PROCESS_LOG_TIME]))

    def defer_completed_layer_checkpoints(
        self,
        *,
        model: BaseQModel,
        layer_index: int,
        projection_entries: list[dict[str, str]],
    ) -> None:
        """Replace a durable packed layer with metadata-only EXL3 shells.

        Projection checkpoints already own every packed tensor and the layer
        boundary owns the outputs needed by the next decoder block. Keeping a
        second in-memory copy until publication makes resident memory grow by
        one complete routed layer per iteration.
        """

        expected = {
            entry.get("module")
            for entry in projection_entries
            if isinstance(entry, dict)
        }
        if len(expected) != len(projection_entries) or not all(
            isinstance(name, str) and name for name in expected
        ):
            raise RuntimeError(
                f"EXL3 layer {layer_index} deferral index is malformed"
            )
        completed = {
            entry["module"]
            for entry in self.completed_layer_checkpoint_entries(layer_index)
        }
        if completed != expected:
            raise RuntimeError(
                f"EXL3 layer {layer_index} deferral differs from durable checkpoints"
            )

        for module_name in sorted(expected):
            packed = model.model.get_submodule(module_name)
            if not isinstance(packed, ExllamaV3Linear):
                raise RuntimeError(
                    f"EXL3 deferral target is not packed: `{module_name}`"
                )
            trellis = getattr(packed, "trellis", None)
            if trellis is None:
                raise RuntimeError(
                    f"EXL3 deferral target has no Trellis storage: `{module_name}`"
                )
            # Packed catch-up replay disk-offloads the restored module before
            # boundary commit.  Accelerate consequently leaves a valid EXL3
            # module with META buffers and an execution hook here.  Deferral
            # is deliberately idempotent: normalize either that state or a
            # freshly materialized packed module to our hook-free metadata
            # shell.  The completed-set equality above still binds this shell
            # to a committed projection checkpoint.
            tensor_storage = (
                packed.tensor_storage_entry()
                if trellis.device.type != "meta"
                else packed.tensor_storage
            )
            if (
                not isinstance(tensor_storage, dict)
                or tensor_storage.get("quant_format") != "exl3"
                or not tensor_storage.get("stored_tensors")
            ):
                raise RuntimeError(
                    f"EXL3 deferral target has no durable tensor metadata: `{module_name}`"
                )
            placeholder = ExllamaV3Linear(
                in_features=packed.in_features,
                out_features=packed.out_features,
                name=module_name,
                tensor_storage=tensor_storage,
                out_dtype=packed.out_dtype,
            )
            recurse_setattr(model.model, module_name, placeholder)

    def offload_restored_layer_checkpoints(
        self,
        *,
        model: BaseQModel,
        layer_index: int,
    ) -> None:
        """Offload a replayed packed layer before its boundary is promoted."""

        offload_path = getattr(self.qcfg, "offload_to_disk_path", None)
        if not getattr(self.qcfg, "offload_to_disk", False) or not offload_path:
            raise RuntimeError("EXL3 layer restore requires durable disk offload")
        entries = self.completed_layer_checkpoint_entries(layer_index)
        if not entries:
            raise RuntimeError(
                f"EXL3 restored layer {layer_index} contains no projections"
            )
        checkpoint_root = checkpoint_root_from_provenance(
            self._ledger_provenance()
        )
        if checkpoint_root is None:
            raise RuntimeError("EXL3 layer restore requires projection checkpoints")
        checkpoint_store = EXL3ProjectionCheckpointStore(checkpoint_root)
        for entry in entries:
            module_name = entry["module"]
            packed = model.model.get_submodule(module_name)
            if not isinstance(packed, ExllamaV3Linear):
                raise RuntimeError(
                    f"EXL3 restored layer target is not packed: `{module_name}`"
                )
            packed.tensor_storage = packed.tensor_storage_entry()
            if getattr(packed, "bias", None) is None:
                offload_to_safetensors_reference(
                    model=model.model,
                    module=packed,
                    source_path=checkpoint_store.committed_tensor_path(
                        entry["request_sha256"]
                    ),
                    disk_path=offload_path,
                    module_name=module_name,
                )
            else:
                offload_to_disk(
                    model=model.model,
                    module=packed,
                    disk_path=offload_path,
                )

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
