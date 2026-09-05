# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Production A41/R0 grouped V2B2-P32 execution on Hopper.

The architecture layer declares ordered same-input projection names.  R0
accepts only groups whose canonical QVQ state proves that the complete input
transform is identical.  The first sibling then performs one ``SU -> H``
transform and one segmented Hopper launch; output recovery stays child-local.

This module owns transient inference state only.  Checkpoint buffers remain on
the original :class:`QVQLinear` children and every unsupported runtime case
uses the unmodified child forward method.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import median
from threading import RLock
from types import MethodType
from typing import Any
from weakref import ref

import torch
from torch import nn

from ..quantization.qvq import (
    pack_qvq_binary_bank_ids,
    repack_p32_planar_to_window,
    unpack_qvq_binary_bank_ids,
)
from ..quantization.qvq_activation import quantize_qvq_fp8_activation
from ..quantization.qvq_rates import qvq_transition_bits, qvq_words_per_tile
from ..utils.qvq_wgmma_cuda import (
    QVQHopperGroupedP32Payload,
    qvq_fp16_to_fp8_e5m2_clamped,
    qvq_h100_grouped_ordered_split_counts,
    qvq_p32_window_wgmma_group_plan,
    qvq_p32_window_wgmma_grouped_ordered_packed,
    qvq_p32_window_wgmma_grouped_ordered_partials_packed,
    qvq_p32_window_wgmma_grouped_packed,
    qvq_p32_window_wgmma_grouped_reuse2_packed,
    qvq_p32_window_wgmma_grouped_reuse4_packed,
    qvq_p32_window_wgmma_grouped_reuse8_packed,
    qvq_p32_window_wgmma_grouped_reuse11_packed,
)
from .qlinear.qvq import QVQLinear

_QKV_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("wq", "wk", "wv"),
    ("query", "key", "value"),
    ("q", "k", "v"),
)
_GATE_UP_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("gate_proj", "up_proj"),
    ("w2", "w1"),
    ("w1", "w3"),
    ("gate", "up"),
)
_H100_LARGE_M_CHUNK_CANDIDATES = (512, 1024, 2048, 4096)


class _R0Fallback(RuntimeError):
    """An expected exactness/dispatch rejection, not a kernel failure."""


@dataclass(frozen=True)
class _H100FP8PrefillPayload:
    """Transient folded weight used only by the measured large-M QKV path."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor
    bias: torch.Tensor | None
    source_key: tuple[Any, ...]


@dataclass(frozen=True)
class _H100OnDemandFP8Scales:
    """Tiny source-versioned scales retained by on-demand FP8 prefill."""

    weight_scale: torch.Tensor
    input_scale: torch.Tensor
    source_key: tuple[Any, ...]


def _is_exact_silu_activation(act_fn: Any) -> bool:
    """Return whether ``act_fn`` has PyTorch's standard non-inplace SiLU contract."""

    if act_fn is torch.nn.functional.silu:
        return True
    if isinstance(act_fn, nn.SiLU):
        return not act_fn.inplace
    try:
        from transformers.activations import SiLUActivation
    except ImportError:
        return False
    return isinstance(act_fn, SiLUActivation)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors do not expose mutation counters.  Canonical QVQ
        # buffers are made versioned by post_init(), but activation tensors may
        # legitimately be inference tensors.  Identity still prevents sibling
        # cycles from crossing activation objects.
        return None


def _same_tensor_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


def _source_key(children: Sequence[QVQLinear]) -> tuple[Any, ...]:
    key: list[Any] = []
    for child in children:
        for name in ("trellis", "bank_ids", "bank_alt_id", "SU"):
            tensor = getattr(child, name, None)
            key.extend(
                (id(tensor), None if tensor is None else _tensor_version(tensor))
            )
        key.extend(
            (
                float(child.bits),
                int(child.in_features),
                int(child.out_features),
                str(child.codebook_version).strip().lower(),
                int(child.vector_size),
                int(child.trellis_window),
                int(child.bank_count),
                bool(child.v2b2_p32),
                bool(child.input_hadamard),
                bool(child.output_hadamard),
                (
                    None
                    if child.activation is None
                    else (
                        int(child.activation.bits),
                        child.activation.format,
                        child.activation.scale_method,
                        child.activation.target,
                        child.activation.kernel_mode,
                        int(child.activation.replay_passes),
                        int(child.activation.replay_max_rows),
                        float(child.activation.replay_validation_fraction),
                    )
                ),
            )
        )
    return tuple(key)


def _validate_static_group(
    children: Sequence[nn.Module],
    *,
    allow_installed: bool = False,
) -> tuple[QVQLinear, ...]:
    if len(children) not in (2, 3) or not all(
        isinstance(child, QVQLinear) for child in children
    ):
        raise _R0Fallback("members are not a two- or three-child QVQLinear group")
    resolved = tuple(children)
    first = resolved[0]
    if not allow_installed and any(
        getattr(child, "_gptqmodel_qvq_grouped_runtime", None) is not None
        for child in resolved
    ):
        raise _R0Fallback("group is already installed")
    if any(
        getattr(child, "_gptqmodel_fused_group", None) is not None for child in resolved
    ):
        raise _R0Fallback("child already belongs to another fused group")
    if any(
        child.training or getattr(child, "adapter", None) is not None
        for child in resolved
    ):
        raise _R0Fallback("training and adapters retain ordinary per-child execution")
    if any(
        not child.v2b2_p32
        or child.vector_size != 2
        or child.trellis_window != 16
        or child.bank_count != 2
        for child in resolved
    ):
        raise _R0Fallback("group requires V2B2-P32 vector-size-2 children")
    if qvq_transition_bits(first.bits, vector_size=2) not in (4, 5, 6, 7):
        raise _R0Fallback("grouped Hopper supports W2 through W3.5")
    if any(
        child.in_features != first.in_features
        or float(child.bits) != float(first.bits)
        or child.codebook_version != first.codebook_version
        or child.trellis.device != first.trellis.device
        for child in resolved[1:]
    ):
        raise _R0Fallback("children disagree on K, rate, codebook, or device")
    if first.in_features <= 0 or first.in_features % 256:
        raise _R0Fallback("grouped Hopper requires K divisible by 256")
    if any(child.out_features <= 0 or child.out_features % 256 for child in resolved):
        raise _R0Fallback("grouped Hopper requires every child N divisible by 256")
    if any(
        child.trellis.device.type == "meta"
        or child.bank_ids is None
        or child.bank_ids.device != child.trellis.device
        or child.bank_alt_id is None
        or child.bank_alt_id.device != child.trellis.device
        for child in resolved
    ):
        raise _R0Fallback("grouped Hopper requires concrete co-located P32 payloads")
    if any(not _same_tensor_bits(first.SU, child.SU) for child in resolved[1:]):
        raise _R0Fallback("R0 requires bit-identical SU vectors")
    if any(child.input_hadamard != first.input_hadamard for child in resolved[1:]):
        raise _R0Fallback("R0 requires identical input-Hadamard state")
    if any(
        child.activation != first.activation
        for child in resolved[1:]
    ):
        raise _R0Fallback("R0 requires identical activation-quantization state")
    return resolved


@dataclass
class QVQGroupedRuntimeTelemetry:
    category: str
    members: tuple[str, ...]
    grouped_launches: int = 0
    grouped_a8_launches: int = 0
    fp8_independent_child_launches: int = 0
    shared_fp8_quantizations: int = 0
    sibling_cache_hits: int = 0
    plain_fallbacks: int = 0
    payload_builds: int = 0
    payload_drops: int = 0
    stale_cycles: int = 0
    grouped_window_bytes: int = 0
    grouped_selector_bytes: int = 0
    child_window_bytes_avoided: int = 0
    paired_recovery_launches: int = 0
    h100_multiblock_recovery_launches: int = 0
    h100_warp_recovery_low_launches: int = 0
    h100_fused_recovery_precondition_launches: int = 0
    h100_paired_recovery_tiles_launches: int = 0
    h100_bounded_recovery_rounding_launches: int = 0
    h100_packed_gate_up_recovery_launches: int = 0
    h100_multiblock_precondition_launches: int = 0
    h100_half2_precondition_high_launches: int = 0
    h100_fused_silu_precondition_low_launches: int = 0
    h100_half2_precondition_low_launches: int = 0
    h100_direct_padded_precondition_launches: int = 0
    h100_direct_padded_input_launches: int = 0
    h100_multiblock_input_hadamard_launches: int = 0
    h100_fp16_recovery_store_launches: int = 0
    h100_fused_down_reduction_recovery_launches: int = 0
    h100_multiblock_down_recovery_launches: int = 0
    h100_w25_n128_gate_up_launches: int = 0
    h100_wide_reuse_gate_up_launches: int = 0
    h100_reuse8_gate_up_launches: int = 0
    h100_reuse11_gate_up_launches: int = 0
    h100_large_m_chunk_autotunes: int = 0
    h100_large_m_chunked_mlp_launches: int = 0
    h100_large_m_chunk_rows: int = 0
    h100_large_m_chunked_group_launches: int = 0
    h100_large_m_group_chunk_rows: int = 0
    h100_fp8_prefill_launches: int = 0
    h100_fp8_prefill_bytes: int = 0
    h100_fp8_mlp_down_launches: int = 0
    h100_fp8_mlp_down_bytes: int = 0
    h100_fp8_mlp_fused_silu_launches: int = 0
    h100_fp8_ondemand_launches: int = 0
    h100_fp8_ondemand_retained_bytes: int = 0
    h100_fp8_ondemand_scratch_bytes: int = 0
    h100_fp16_prefill_launches: int = 0
    h100_fp16_prefill_temporary_bytes: int = 0
    h100_fp16_prefill_native_launches: int = 0
    h100_fp16_prefill_native_scratch_bytes: int = 0
    h100_folded_qwen_mlp_launches: int = 0
    h100_folded_qwen_fused_precondition_launches: int = 0
    h100_folded_qwen_fused_ordered_reduction_launches: int = 0
    h100_qwen_w3_ordered_decode_prefetch_launches: int = 0
    h100_qwen_ordered_decode_prefetch_launches: int = 0
    h100_qwen_w3_down_decode_prefetch_launches: int = 0
    h100_qwen_down_decode_prefetch_launches: int = 0
    h100_qwen_fixed_ordered_grid_launches: int = 0
    h100_qwen_fixed_linear_grid_launches: int = 0
    h100_qwen_linear_decode_prefetch_launches: int = 0
    h100_qwen_composite_down_recovery_launches: int = 0
    h100_qwen_ordered_composite_down_recovery_launches: int = 0
    h100_qwen_linear_composite_recovery_launches: int = 0
    h100_qwen_linear_multiblock_recovery_launches: int = 0
    h100_qwen_composite_input_launches: int = 0
    h100_qwen_large_m_direct_down_launches: int = 0
    h100_qwen_m32_fused_tiles: int = 0
    h100_qwen_large_m_unsplit_gate_up_launches: int = 0
    independent_recovery_children: int = 0
    fused_mlp_launches: int = 0
    fused_mlp_fallbacks: int = 0
    ordered_split_launches: int = 0
    active_split_counts: tuple[int, ...] = ()
    last_fallback_reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "members": self.members,
            "grouped_launches": self.grouped_launches,
            "grouped_a8_launches": self.grouped_a8_launches,
            "fp8_independent_child_launches": self.fp8_independent_child_launches,
            "shared_fp8_quantizations": self.shared_fp8_quantizations,
            "sibling_cache_hits": self.sibling_cache_hits,
            "plain_fallbacks": self.plain_fallbacks,
            "payload_builds": self.payload_builds,
            "payload_drops": self.payload_drops,
            "stale_cycles": self.stale_cycles,
            "grouped_window_bytes": self.grouped_window_bytes,
            "grouped_selector_bytes": self.grouped_selector_bytes,
            "child_window_bytes_avoided": self.child_window_bytes_avoided,
            "paired_recovery_launches": self.paired_recovery_launches,
            "h100_multiblock_recovery_launches": self.h100_multiblock_recovery_launches,
            "h100_warp_recovery_low_launches": self.h100_warp_recovery_low_launches,
            "h100_fused_recovery_precondition_launches": (
                self.h100_fused_recovery_precondition_launches
            ),
            "h100_paired_recovery_tiles_launches": (
                self.h100_paired_recovery_tiles_launches
            ),
            "h100_bounded_recovery_rounding_launches": (
                self.h100_bounded_recovery_rounding_launches
            ),
            "h100_packed_gate_up_recovery_launches": (
                self.h100_packed_gate_up_recovery_launches
            ),
            "h100_multiblock_precondition_launches": self.h100_multiblock_precondition_launches,
            "h100_half2_precondition_high_launches": self.h100_half2_precondition_high_launches,
            "h100_fused_silu_precondition_low_launches": self.h100_fused_silu_precondition_low_launches,
            "h100_half2_precondition_low_launches": self.h100_half2_precondition_low_launches,
            "h100_direct_padded_precondition_launches": self.h100_direct_padded_precondition_launches,
            "h100_direct_padded_input_launches": self.h100_direct_padded_input_launches,
            "h100_multiblock_input_hadamard_launches": self.h100_multiblock_input_hadamard_launches,
            "h100_fp16_recovery_store_launches": self.h100_fp16_recovery_store_launches,
            "h100_fused_down_reduction_recovery_launches": self.h100_fused_down_reduction_recovery_launches,
            "h100_multiblock_down_recovery_launches": self.h100_multiblock_down_recovery_launches,
            "h100_w25_n128_gate_up_launches": self.h100_w25_n128_gate_up_launches,
            "h100_wide_reuse_gate_up_launches": self.h100_wide_reuse_gate_up_launches,
            "h100_reuse8_gate_up_launches": self.h100_reuse8_gate_up_launches,
            "h100_reuse11_gate_up_launches": self.h100_reuse11_gate_up_launches,
            "h100_large_m_chunk_autotunes": self.h100_large_m_chunk_autotunes,
            "h100_large_m_chunked_mlp_launches": self.h100_large_m_chunked_mlp_launches,
            "h100_large_m_chunk_rows": self.h100_large_m_chunk_rows,
            "h100_large_m_chunked_group_launches": self.h100_large_m_chunked_group_launches,
            "h100_large_m_group_chunk_rows": self.h100_large_m_group_chunk_rows,
            "h100_fp8_prefill_launches": self.h100_fp8_prefill_launches,
            "h100_fp8_prefill_bytes": self.h100_fp8_prefill_bytes,
            "h100_fp8_mlp_down_launches": self.h100_fp8_mlp_down_launches,
            "h100_fp8_mlp_down_bytes": self.h100_fp8_mlp_down_bytes,
            "h100_fp8_mlp_fused_silu_launches": (
                self.h100_fp8_mlp_fused_silu_launches
            ),
            "h100_fp8_ondemand_launches": self.h100_fp8_ondemand_launches,
            "h100_fp8_ondemand_retained_bytes": (
                self.h100_fp8_ondemand_retained_bytes
            ),
            "h100_fp8_ondemand_scratch_bytes": (
                self.h100_fp8_ondemand_scratch_bytes
            ),
            "h100_fp16_prefill_launches": self.h100_fp16_prefill_launches,
            "h100_fp16_prefill_temporary_bytes": (
                self.h100_fp16_prefill_temporary_bytes
            ),
            "h100_fp16_prefill_native_launches": (
                self.h100_fp16_prefill_native_launches
            ),
            "h100_fp16_prefill_native_scratch_bytes": (
                self.h100_fp16_prefill_native_scratch_bytes
            ),
            "h100_folded_qwen_mlp_launches": self.h100_folded_qwen_mlp_launches,
            "h100_folded_qwen_fused_precondition_launches": self.h100_folded_qwen_fused_precondition_launches,
            "h100_folded_qwen_fused_ordered_reduction_launches": self.h100_folded_qwen_fused_ordered_reduction_launches,
            "h100_qwen_w3_ordered_decode_prefetch_launches": self.h100_qwen_w3_ordered_decode_prefetch_launches,
            "h100_qwen_ordered_decode_prefetch_launches": self.h100_qwen_ordered_decode_prefetch_launches,
            "h100_qwen_w3_down_decode_prefetch_launches": self.h100_qwen_w3_down_decode_prefetch_launches,
            "h100_qwen_down_decode_prefetch_launches": self.h100_qwen_down_decode_prefetch_launches,
            "h100_qwen_fixed_ordered_grid_launches": self.h100_qwen_fixed_ordered_grid_launches,
            "h100_qwen_fixed_linear_grid_launches": self.h100_qwen_fixed_linear_grid_launches,
            "h100_qwen_linear_decode_prefetch_launches": self.h100_qwen_linear_decode_prefetch_launches,
            "h100_qwen_composite_down_recovery_launches": self.h100_qwen_composite_down_recovery_launches,
            "h100_qwen_ordered_composite_down_recovery_launches": self.h100_qwen_ordered_composite_down_recovery_launches,
            "h100_qwen_linear_composite_recovery_launches": self.h100_qwen_linear_composite_recovery_launches,
            "h100_qwen_linear_multiblock_recovery_launches": self.h100_qwen_linear_multiblock_recovery_launches,
            "h100_qwen_composite_input_launches": self.h100_qwen_composite_input_launches,
            "h100_qwen_large_m_direct_down_launches": self.h100_qwen_large_m_direct_down_launches,
            "h100_qwen_m32_fused_tiles": self.h100_qwen_m32_fused_tiles,
            "h100_qwen_large_m_unsplit_gate_up_launches": self.h100_qwen_large_m_unsplit_gate_up_launches,
            "independent_recovery_children": self.independent_recovery_children,
            "fused_mlp_launches": self.fused_mlp_launches,
            "fused_mlp_fallbacks": self.fused_mlp_fallbacks,
            "ordered_split_launches": self.ordered_split_launches,
            "active_split_counts": self.active_split_counts,
            "last_fallback_reason": self.last_fallback_reason,
        }


class QVQHopperGroupedRuntime:
    """One ordered, fail-closed production sibling group."""

    def __init__(
        self,
        children: Sequence[QVQLinear],
        member_names: Sequence[str],
        *,
        category: str,
    ) -> None:
        self._refs = tuple(ref(child) for child in children)
        self.member_names = tuple(member_names)
        self.category = str(category)
        self.telemetry = QVQGroupedRuntimeTelemetry(self.category, self.member_names)
        self._payload: QVQHopperGroupedP32Payload | None = None
        self._h100_qwen_large_m_unsplit_payload: QVQHopperGroupedP32Payload | None = None
        self._payload_source_key: tuple[Any, ...] | None = None
        self._h100_multiblock_intermediate_enabled = False
        self._h100_direct_padded_input_enabled = False
        self._h100_multiblock_input_hadamard_enabled = False
        self._h100_fp16_recovery_store_enabled = False
        self._h100_w25_n128_gate_up_enabled = False
        self._h100_bounded_recovery_rounding_enabled = False
        self._h100_packed_gate_up_recovery_enabled = False
        self._large_m_chunk_cache: dict[tuple[Any, ...], int] = {}
        self._large_m_chunk_cache_lock = RLock()
        self._h100_fp8_prefill_payload: _H100FP8PrefillPayload | None = None
        self._h100_fp8_mlp_down_payload: _H100FP8PrefillPayload | None = None
        self._h100_fp8_ondemand_scales: _H100OnDemandFP8Scales | None = None
        self._input: torch.Tensor | None = None
        self._input_version: int | None = None
        self._outputs: tuple[torch.Tensor, ...] | None = None
        self._next_index = 0
        self._mlp_parent_ref: Any = None
        self._mlp_down_ref: Any = None
        self._mlp_act_fn: Any = None
        self._mlp_activation_is_exact_silu = False

    def _children(self) -> tuple[QVQLinear, ...]:
        children = tuple(child_ref() for child_ref in self._refs)
        if any(child is None for child in children):
            raise RuntimeError("QVQ grouped runtime lost a child module")
        return children  # type: ignore[return-value]

    def _clear_cycle(self) -> None:
        self._input = None
        self._input_version = None
        self._outputs = None
        self._next_index = 0

    def invalidate(self) -> None:
        """Release all transient state after source/device ownership changes."""

        self._clear_cycle()
        if self._payload is not None:
            self.telemetry.payload_drops += 1
        self._payload = None
        self._h100_qwen_large_m_unsplit_payload = None
        self._payload_source_key = None
        self._h100_multiblock_intermediate_enabled = False
        self._h100_direct_padded_input_enabled = False
        self._h100_multiblock_input_hadamard_enabled = False
        self._h100_fp16_recovery_store_enabled = False
        self._h100_w25_n128_gate_up_enabled = False
        self._h100_bounded_recovery_rounding_enabled = False
        self._h100_packed_gate_up_recovery_enabled = False
        with self._large_m_chunk_cache_lock:
            self._large_m_chunk_cache.clear()
        self._h100_fp8_prefill_payload = None
        self._h100_fp8_mlp_down_payload = None
        self._h100_fp8_ondemand_scales = None
        self.telemetry.grouped_window_bytes = 0
        self.telemetry.grouped_selector_bytes = 0
        self.telemetry.child_window_bytes_avoided = 0
        self.telemetry.active_split_counts = ()
        self.telemetry.h100_fp8_prefill_bytes = 0
        self.telemetry.h100_fp8_mlp_down_bytes = 0
        self.telemetry.h100_fp8_ondemand_retained_bytes = 0
        self.telemetry.h100_fp8_ondemand_scratch_bytes = 0
        self.telemetry.h100_fp16_prefill_temporary_bytes = 0
        self.telemetry.h100_fp16_prefill_native_scratch_bytes = 0

    def _fallback(
        self, member_index: int, x: torch.Tensor, reason: str
    ) -> torch.Tensor:
        self._drop_payload_before_plain()
        self.telemetry.plain_fallbacks += 1
        self.telemetry.last_fallback_reason = reason
        child = self._children()[member_index]
        original = child._gptqmodel_qvq_grouped_original_forward
        return original(x)

    def _drop_payload_before_plain(self) -> None:
        # A plain child may lazily build its own continuous-window cache.  Drop
        # the grouped window first so an unsupported prefill cannot make both
        # representations persistent at once.
        if (
            self._payload is not None
            and not (
                self._payload.trellis.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            self.invalidate()

    def _runtime_eligible(
        self, x: torch.Tensor, *, maximum_rows: int | None = None
    ) -> str | None:
        children = self._children()
        if not isinstance(x, torch.Tensor):
            return "input is not a tensor"
        if x.requires_grad or any(child.training for child in children):
            return "autograd/training requires the original forward"
        if any(getattr(child, "adapter", None) is not None for child in children):
            return "an attached adapter requires the original forward"
        if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16):
            return "grouped Hopper requires FP16 or BF16 CUDA activations"
        if x.shape[-1] != children[0].in_features or x.numel() == 0:
            return "input shape is unsupported"
        rows = x.numel() // children[0].in_features
        if rows < 1 or (maximum_rows is not None and rows > maximum_rows):
            if maximum_rows is None:
                return "grouped Hopper execution requires at least one row"
            return f"grouped Hopper execution currently requires one through {maximum_rows} rows"
        if any(child.trellis.device != x.device for child in children):
            return "activation and grouped payload devices differ"
        properties = torch.cuda.get_device_properties(x.device)
        if (properties.major, properties.minor) != (9, 0):
            return "grouped runtime requires Hopper SM90"
        if rows > 4096 and properties.name != "NVIDIA H100":
            return "row multiplexing above 4096 is measured only on H100"
        return None

    def _packed_selectors(self, child: QVQLinear) -> torch.Tensor:
        tile_count = (child.in_features // 16) * (child.out_features // 16)
        return pack_qvq_binary_bank_ids(
            unpack_qvq_binary_bank_ids(child.bank_ids, tile_count * 8)
        ).to(device=child.trellis.device)

    def _build_payload(
        self,
        children: tuple[QVQLinear, ...],
        source_key: tuple[Any, ...],
    ) -> QVQHopperGroupedP32Payload:
        # Re-run R0 only when a canonical source identity/version changed.  The
        # equality check may synchronize and therefore never occurs in the
        # warmed CUDA-graph capture path.
        _validate_static_group(children, allow_installed=True)
        device = children[0].trellis.device
        placeholder = torch.empty(
            (16, children[0].in_features), device=device, dtype=torch.float16
        )
        selectors = tuple(self._packed_selectors(child) for child in children)
        alt_ids = tuple(int(child.bank_alt_id.detach().item()) for child in children)
        if any(not 1 <= alt_id <= 3 for alt_id in alt_ids):
            raise _R0Fallback("V2B2-P32 alternative bank IDs must remain in [1, 3]")
        from ..utils.qvq_cuda import _pgc16_levels

        properties = torch.cuda.get_device_properties(device)
        self._h100_multiblock_intermediate_enabled = (
            self.category == "gate_up"
            and len(children) == 2
            and all(child.out_features == 8192 for child in children)
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
        )
        self._h100_direct_padded_input_enabled = (
            children[0].input_hadamard
            and children[0].in_features == 2048
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
        )
        self._h100_multiblock_input_hadamard_enabled = (
            self._h100_direct_padded_input_enabled
        )
        self._h100_fp16_recovery_store_enabled = (
            properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
        )
        self._h100_w25_n128_gate_up_enabled = (
            self.category == "gate_up"
            and len(children) == 2
            and children[0].in_features == 2048
            and all(child.out_features == 8192 for child in children)
            and qvq_transition_bits(
                children[0].bits, vector_size=children[0].vector_size
            )
            == 5
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
        )
        self._h100_bounded_recovery_rounding_enabled = (
            self._h100_multiblock_intermediate_enabled
        )
        self._h100_packed_gate_up_recovery_enabled = (
            self._h100_multiblock_intermediate_enabled
        )
        measured_splits = qvq_h100_grouped_ordered_split_counts(
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            in_features=children[0].in_features,
            out_features=tuple(child.out_features for child in children),
            transition_bits=qvq_transition_bits(
                children[0].bits, vector_size=children[0].vector_size
            ),
        )

        plan = qvq_p32_window_wgmma_group_plan(
            placeholder,
            tuple(child.trellis for child in children),
            _pgc16_levels(device, children[0].codebook_version),
            selectors,
            children[0].bits,
            out_features=tuple(child.out_features for child in children),
            bank_alt_ids=alt_ids,
            split_counts=measured_splits,
        )
        if measured_splits is None and any(
            segment.split_count != 1 for segment in plan.segments
        ):
            raise _R0Fallback(
                "a child requires an unvalidated grouped split-K schedule"
            )

        k_tiles = children[0].in_features // 16
        words_per_tile = qvq_words_per_tile(
            children[0].bits, weight_count=256, vector_size=2
        )
        planar = torch.cat(
            tuple(
                child.trellis.reshape(k_tiles, child.out_features // 16, words_per_tile)
                for child in children
            ),
            dim=1,
        ).reshape(-1, words_per_tile)
        grouped_window = repack_p32_planar_to_window(planar, bits=children[0].bits)
        grouped_selectors = (
            torch.cat(
                tuple(
                    child_selectors.reshape(k_tiles, child.out_features // 16)
                    for child, child_selectors in zip(children, selectors, strict=True)
                ),
                dim=1,
            )
            .reshape(-1)
            .contiguous()
        )
        if source_key != _source_key(children):
            raise RuntimeError("QVQ grouped canonical payload changed during repack")
        payload = QVQHopperGroupedP32Payload(
            trellis=grouped_window,
            bank_ids=grouped_selectors,
            plan=plan,
        )
        if (
            properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
            and self.category == "gate_up"
            and children[0].in_features == 5120
            and tuple(child.out_features for child in children)
            == (17408, 17408)
            and all(segment.split_count == 5 for segment in plan.segments)
        ):
            unsplit_plan = replace(
                plan,
                segments=tuple(
                    replace(segment, split_count=1)
                    for segment in plan.segments
                ),
            )
            self._h100_qwen_large_m_unsplit_payload = replace(
                payload, plan=unsplit_plan
            )

        # A grouped window has exactly the same number of words as the child
        # windows it replaces.  Clear any plain-path windows/selectors left by
        # an earlier prefill before publishing the grouped payload.
        avoided = 0
        for child in children:
            cached = child._qvq_cuda_window_cache
            if cached is not None:
                avoided += cached[3].numel() * cached[3].element_size()
            else:
                avoided += child.trellis.numel() * child.trellis.element_size()
            with child._qvq_cuda_bank_cache_lock:
                child._qvq_cuda_window_cache = None
                child._qvq_cuda_bank_cache = None
        self.telemetry.payload_builds += 1
        self.telemetry.grouped_window_bytes = (
            grouped_window.numel() * grouped_window.element_size()
        )
        self.telemetry.grouped_selector_bytes = (
            grouped_selectors.numel() * grouped_selectors.element_size()
        )
        self.telemetry.child_window_bytes_avoided = avoided
        self.telemetry.active_split_counts = tuple(
            segment.split_count for segment in plan.segments
        )
        return payload

    def _ensure_payload(self) -> QVQHopperGroupedP32Payload:
        children = self._children()
        source_key = _source_key(children)
        if self._payload is not None and source_key == self._payload_source_key:
            return self._payload
        if (
            children[0].trellis.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            # R0 validation compares canonical tensors and payload construction
            # repacks them.  Neither operation belongs in capture.  A group
            # whose exact payload was not successfully warmed must fail closed
            # to the original graph-safe child path for this capture.
            raise _R0Fallback(
                "grouped P32 payload must be successfully warmed before CUDA Graph capture"
            )
        self.invalidate()
        payload = self._build_payload(children, source_key)
        self._payload = payload
        self._payload_source_key = source_key
        return payload

    def _execute(
        self,
        x: torch.Tensor,
        *,
        recover: bool = True,
        return_ordered_partials: bool = False,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        if return_ordered_partials and recover:
            raise ValueError("ordered partial execution cannot recover child outputs")
        children = self._children()
        rows = x.numel() // children[0].in_features
        if (
            recover
            and not return_ordered_partials
            and self._h100_fp8_ondemand_eligible(x, rows)
        ):
            scales = self._ensure_h100_fp8_ondemand_scales()
            if scales is not None:
                self.telemetry.h100_fp8_ondemand_launches += 1
                return self._execute_h100_fp8_ondemand(x, scales)
        if (
            recover
            and not return_ordered_partials
            and self._h100_fp8_prefill_eligible(x, rows)
        ):
            prefill_payload = self._ensure_h100_fp8_prefill_payload()
            if prefill_payload is not None:
                self.telemetry.h100_fp8_prefill_launches += 1
                return self._execute_h100_fp8_prefill(x, prefill_payload)
        if (
            recover
            and not return_ordered_partials
            and self._h100_fp16_prefill_eligible(x, rows)
        ):
            self.telemetry.h100_fp16_prefill_launches += 1
            return self._execute_h100_fp16_prefill(x)
        if rows > 4096:
            if return_ordered_partials or not recover:
                raise _R0Fallback(
                    "raw grouped output above 4096 requires child-local reduction"
                )
            chunk_rows = self._autotune_large_m_chunk_rows(
                x, rows, scope="group"
            )
            self.telemetry.h100_large_m_chunked_group_launches += 1
            self.telemetry.h100_large_m_group_chunk_rows = chunk_rows
            return self._execute_group_chunked(x, chunk_rows, recover=recover)
        native_fp8_children = (
            recover
            and not return_ordered_partials
            and children[0].activation is not None
            and children[0].activation.target == "p32_operand"
        )
        # Payload construction also resolves physical-device feature gates.
        # Native FP8 dispatch returns through independent child kernels and
        # must not build a grouped P32 window it cannot consume.
        payload = None if native_fp8_children else self._ensure_payload()
        if (
            payload is not None
            and rows >= 128
            and self._h100_qwen_large_m_unsplit_payload is not None
        ):
            payload = self._h100_qwen_large_m_unsplit_payload
            self.telemetry.h100_qwen_large_m_unsplit_gate_up_launches += 1
        # Preserve BF16 until activation fake-quantization so its explicit
        # rounding contract matches ordinary child execution. The shared
        # input transform narrows the resulting operand to FP16 for WGMMA.
        x_2d = x.reshape(rows, children[0].in_features)
        padded_rows = (
            16
            if rows <= 16
            else 32
            if rows <= 32
            else ((rows + 63) // 64) * 64
        )
        direct_pad = self._h100_direct_padded_input_enabled and rows < 16
        use_qwen_composite_input = (
            self._h100_fp16_recovery_store_enabled
            and x.dtype == torch.float16
            and children[0].activation is None
            and children[0].input_hadamard
            and children[0].in_features == 5120
            and rows <= 16
            # The single native H40 x H128 launch wins consistently once
            # eight logical rows amortize its wider block-local transform.
            # Decode-sized M1/M2/M4 retains the lower-latency staged path.
            and rows >= 8
        )
        if use_qwen_composite_input:
            from ..quantization.rotation.hadamard_utils import _get_hadK_on
            from ..utils.qvq_cuda import qvq_cuda_qwen_composite_input_fp16_padded

            input_scale = children[0]._cached_cast("SU", torch.float16)
            base, base_width = _get_hadK_on(input_scale, False)
            if base is None or base_width != 40:
                raise _R0Fallback("Qwen 5120 input requires its canonical H40 base")
            padded = qvq_cuda_qwen_composite_input_fp16_padded(
                x_2d,
                base=base,
                pre_scale=input_scale,
            )
            self.telemetry.h100_qwen_composite_input_launches += 1
        elif (
            self._h100_multiblock_input_hadamard_enabled
            and rows <= 4096
            and x.dtype == torch.float16
            and children[0].activation is None
        ):
            from ..utils.qvq_cuda import (
                qvq_cuda_hadamard_input_fp16_padded_multiblock,
            )

            padded = qvq_cuda_hadamard_input_fp16_padded_multiblock(
                x_2d,
                pre_scale=children[0]._cached_cast("SU", torch.float16),
            )
            self.telemetry.h100_multiblock_input_hadamard_launches += 1
            if direct_pad:
                self.telemetry.h100_direct_padded_input_launches += 1
        else:
            transform_dtype = (
                torch.float32
                if x.dtype == torch.bfloat16
                and children[0].activation is not None
                and children[0].activation.target == "p32_operand"
                else torch.float16
            )
            transformed = children[0]._qvq_prepare_inference_input(
                x_2d,
                transform_dtype,
                pad_to_16=direct_pad,
            )
            if (
                children[0].activation is not None
                and children[0].activation.target == "linear_input"
            ):
                self.telemetry.shared_fp8_quantizations += 1
            if (
                children[0].activation is not None
                and children[0].activation.target == "p32_operand"
            ):
                if return_ordered_partials or not recover:
                    raise _R0Fallback("P32 FP8 grouped split-partial execution is not implemented")
                config = children[0].activation
                quantized, scale = quantize_qvq_fp8_activation(
                    transformed[:rows],
                    format=config.format,
                    scale_method=config.scale_method,
                    validate=False,
                )
                outputs = tuple(
                    child.forward_prequantized_fp8(
                        quantized,
                        scale,
                        output_dtype=x.dtype,
                    )
                    for child in children
                )
                self.telemetry.grouped_a8_launches += 1
                self.telemetry.shared_fp8_quantizations += 1
                self.telemetry.fp8_independent_child_launches += len(children)
                return outputs
            if direct_pad:
                padded = transformed
                self.telemetry.h100_direct_padded_input_launches += 1
            elif rows == padded_rows:
                padded = transformed.contiguous()
            else:
                padded = torch.zeros(
                    (padded_rows, children[0].in_features),
                    device=x.device,
                    dtype=torch.float16,
                )
                padded[:rows].copy_(transformed)
        if payload is None:
            payload = self._ensure_payload()
        from ..utils.qvq_cuda import _pgc16_levels

        if return_ordered_partials:
            partials = qvq_p32_window_wgmma_grouped_ordered_partials_packed(
                padded,
                payload,
                _pgc16_levels(x.device, children[0].codebook_version),
            )
            self.telemetry.ordered_split_launches += 1
            return partials

        use_h100_reuse11_gate_up = (
            rows in (512, 1024, 2048, 4096)
            and self._h100_multiblock_intermediate_enabled
            and children[0].in_features == 2048
            and all(segment.split_count == 1 for segment in payload.plan.segments)
        )
        if use_h100_reuse11_gate_up:
            reuse11_rows = rows + rows // 32
            reuse11_padded = torch.zeros(
                (reuse11_rows, children[0].in_features),
                device=padded.device,
                dtype=padded.dtype,
            )
            reuse11_padded[:rows].copy_(padded[:rows])
            padded = reuse11_padded
        use_h100_reuse8_gate_up = (
            padded.shape[0] >= 128
            and padded.shape[0] % 128 == 0
            and self._h100_multiblock_intermediate_enabled
            and children[0].in_features == 2048
            and all(segment.split_count == 1 for segment in payload.plan.segments)
        )
        if use_h100_reuse11_gate_up:
            grouped_inner = qvq_p32_window_wgmma_grouped_reuse11_packed
        elif use_h100_reuse8_gate_up:
            grouped_inner = qvq_p32_window_wgmma_grouped_reuse8_packed
        elif padded.shape[0] >= 64 and padded.shape[0] % 64 == 0:
            grouped_inner = qvq_p32_window_wgmma_grouped_reuse4_packed
        elif padded.shape[0] >= 32 and padded.shape[0] % 32 == 0:
            grouped_inner = qvq_p32_window_wgmma_grouped_reuse2_packed
        else:
            grouped_inner = (
                qvq_p32_window_wgmma_grouped_ordered_packed
                if any(segment.split_count != 1 for segment in payload.plan.segments)
                else qvq_p32_window_wgmma_grouped_packed
            )
        inner_outputs = grouped_inner(
            padded,
            payload,
            _pgc16_levels(x.device, children[0].codebook_version),
        )
        if children[0].activation is not None:
            self.telemetry.grouped_a8_launches += 1
        if (
            grouped_inner in (
                qvq_p32_window_wgmma_grouped_reuse4_packed,
                qvq_p32_window_wgmma_grouped_reuse8_packed,
                qvq_p32_window_wgmma_grouped_reuse11_packed,
            )
            and padded.shape[0] >= 128
            and self._h100_multiblock_intermediate_enabled
            and children[0].in_features == 2048
            and all(segment.split_count == 1 for segment in payload.plan.segments)
        ):
            self.telemetry.h100_wide_reuse_gate_up_launches += 1
        if grouped_inner is qvq_p32_window_wgmma_grouped_reuse8_packed:
            self.telemetry.h100_reuse8_gate_up_launches += 1
        if grouped_inner is qvq_p32_window_wgmma_grouped_reuse11_packed:
            self.telemetry.h100_reuse11_gate_up_launches += 1
        if grouped_inner in (
            qvq_p32_window_wgmma_grouped_ordered_packed,
            qvq_p32_window_wgmma_grouped_reuse2_packed,
            qvq_p32_window_wgmma_grouped_reuse4_packed,
        ) and any(segment.split_count != 1 for segment in payload.plan.segments):
            self.telemetry.ordered_split_launches += 1
            if (
                self._h100_fp16_recovery_store_enabled
                and self.category == "qkv"
                and children[0].in_features == 5120
                and tuple(child.out_features for child in children)
                == (10240, 6144)
                and qvq_transition_bits(
                    children[0].bits, vector_size=children[0].vector_size
                )
                <= 6
            ):
                self.telemetry.h100_qwen_fixed_linear_grid_launches += 1
                if qvq_transition_bits(
                    children[0].bits, vector_size=children[0].vector_size
                ) != 5:
                    self.telemetry.h100_qwen_linear_decode_prefetch_launches += 1
        if self._h100_w25_n128_gate_up_enabled:
            self.telemetry.h100_w25_n128_gate_up_launches += 1
        if not recover:
            return tuple(inner[:rows] for inner in inner_outputs)

        # Gate and up have equal-width, independent output transforms.  One
        # grid schedules both row sets concurrently, applies each child's own
        # SV/bias, and performs the final FP16 cast at the store.  Query/key/
        # value and unusual gate/up geometries retain the established exact
        # child-local path.
        if (
            self.category == "gate_up"
            and len(children) == 2
            and children[0].out_features == children[1].out_features
            and children[0].out_features <= 16384
            and children[0].out_features & (children[0].out_features - 1) == 0
        ):
            from ..utils.qvq_cuda import (
                qvq_cuda_hadamard_pair_fp32_to_fp16,
                qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock,
            )

            output_dtype = inner_outputs[0].dtype
            # The N=8192 factorization is promoted only on the physical H100
            # where its launch geometry was validated.  H200 and other Hopper
            # products keep the established single-CTA path until separately
            # measured; device identity comes from CUDA properties, never a
            # visible-device index.
            use_h100_multiblock = (
                self._h100_multiblock_intermediate_enabled and rows <= 16
            )
            recovery = (
                qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock
                if use_h100_multiblock
                else qvq_cuda_hadamard_pair_fp32_to_fp16
            )
            recovered_pair = recovery(
                inner_outputs[0][:rows],
                inner_outputs[1][:rows],
                post_scale0=children[0]._cached_cast("SV", torch.float16, output_dtype),
                post_scale1=children[1]._cached_cast("SV", torch.float16, output_dtype),
                bias0=children[0]._cached_cast("bias", torch.float16, output_dtype),
                bias1=children[1]._cached_cast("bias", torch.float16, output_dtype),
                scale_mode=3 if children[0].out_features >= 2048 else 4,
                **({"warp_low": True} if use_h100_multiblock else {}),
            )
            self.telemetry.paired_recovery_launches += 1
            if use_h100_multiblock:
                self.telemetry.h100_multiblock_recovery_launches += 1
                self.telemetry.h100_warp_recovery_low_launches += 1
            return tuple(
                recovered.reshape(*x.shape[:-1], child.out_features).to(x.dtype)
                for child, recovered in zip(children, recovered_pair, strict=True)
            )

        outputs = []
        for child, inner in zip(children, inner_outputs, strict=True):
            use_qwen_linear_composite_recovery = (
                self._h100_fp16_recovery_store_enabled
                and self.category == "qkv"
                and children[0].in_features == 5120
                and tuple(member.out_features for member in children)
                == (10240, 6144)
                and rows <= 16
                and child.output_hadamard
                and inner.dtype == torch.float32
            )
            if use_qwen_linear_composite_recovery:
                from ..quantization.rotation.hadamard_utils import _get_hadK_on
                from ..utils.qvq_cuda import (
                    qvq_cuda_qwen_composite_recovery_fp32_to_fp16,
                )

                base, base_width = _get_hadK_on(
                    child._cached_cast("SV", torch.float16), False
                )
                expected_base_width = 40 if child.out_features == 10240 else 12
                if base is None or base_width != expected_base_width:
                    raise _R0Fallback(
                        f"Qwen {child.out_features} output requires its canonical "
                        f"H{expected_base_width} base"
                    )
                recovered = qvq_cuda_qwen_composite_recovery_fp32_to_fp16(
                    inner[:rows].contiguous(),
                    base=base,
                    post_scale=child._cached_cast(
                        "SV", torch.float16, torch.float32
                    ),
                    bias=child._cached_cast(
                        "bias", torch.float16, torch.float32
                    ),
                )
                self.telemetry.h100_qwen_linear_composite_recovery_launches += 1
                self.telemetry.h100_qwen_linear_multiblock_recovery_launches += 1
                self.telemetry.h100_fp16_recovery_store_launches += 1
                outputs.append(
                    recovered.reshape(*x.shape[:-1], child.out_features).to(x.dtype)
                )
                continue
            fp16_store = (
                self._h100_fp16_recovery_store_enabled
                and child.output_hadamard
                and inner.dtype == torch.float32
                and child.out_features <= 16384
                and child.out_features & (child.out_features - 1) == 0
                and rows <= 16
            )
            recovered = child._qvq_recover_inference_output(
                inner[:rows],
                torch.float16,
                output_fp16=fp16_store,
            )
            if fp16_store:
                self.telemetry.h100_fp16_recovery_store_launches += 1
            outputs.append(
                recovered.reshape(*x.shape[:-1], child.out_features).to(x.dtype)
            )
        self.telemetry.independent_recovery_children += len(children)
        return tuple(outputs)

    def _configure_mlp_fusion(
        self,
        parent: nn.Module,
        down: QVQLinear,
        act_fn: Any,
    ) -> None:
        self._mlp_parent_ref = ref(parent)
        self._mlp_down_ref = ref(down)
        self._mlp_act_fn = act_fn
        self._mlp_activation_is_exact_silu = _is_exact_silu_activation(act_fn)

    def _clear_mlp_fusion(self) -> None:
        self._mlp_parent_ref = None
        self._mlp_down_ref = None
        self._mlp_act_fn = None
        self._mlp_activation_is_exact_silu = False

    def _mlp_rejection(self, x: torch.Tensor) -> str | None:
        rejection = self._runtime_eligible(x, maximum_rows=None)
        if rejection is not None:
            return rejection
        rows = x.numel() // self._children()[0].in_features
        if rows > 4096:
            properties = torch.cuda.get_device_properties(x.device)
            if properties.name != "NVIDIA H100":
                return "MLP row multiplexing above 4096 is measured only on H100"
        down = None if self._mlp_down_ref is None else self._mlp_down_ref()
        if not isinstance(down, QVQLinear):
            return "fused MLP lost its QVQ down projection"
        children = self._children()
        if (
            len(children) != 2
            or children[0].out_features != children[1].out_features
            or children[0].out_features != down.in_features
            or down.out_features != children[0].in_features
        ):
            return "fused MLP projection geometry changed"
        if down.training or getattr(down, "adapter", None) is not None:
            return "fused MLP down projection requires the original path"
        if down.trellis.device != x.device:
            return "fused MLP down payload and activation devices differ"
        if not callable(self._mlp_act_fn):
            return "fused MLP activation is unavailable"
        return None

    @staticmethod
    def _large_m_chunk_autotune_enabled() -> bool:
        return os.environ.get("QVQ_HOPPER_LARGE_M_AUTOTUNE", "1").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    @staticmethod
    def _h100_fp8_prefill_enabled() -> bool:
        return os.environ.get("QVQ_HOPPER_FP8_PREFILL", "1").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    @staticmethod
    def _h100_fp8_mlp_prefill_enabled() -> bool:
        """Gate the approximate persistent MLP cache independently of QKV."""

        return os.environ.get("QVQ_HOPPER_FP8_MLP_PREFILL", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    @staticmethod
    def _h100_fp8_ondemand_enabled() -> bool:
        return os.environ.get("QVQ_HOPPER_FP8_PREFILL_ON_DEMAND", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    @staticmethod
    def _h100_fp16_prefill_enabled() -> bool:
        return os.environ.get("QVQ_HOPPER_FP16_PREFILL", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    @staticmethod
    def _h100_fp16_prefill_native_enabled() -> bool:
        return os.environ.get("QVQ_HOPPER_FP16_PREFILL_NATIVE", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
            "",
        }

    def _h100_fp16_prefill_eligible(self, x: torch.Tensor, rows: int) -> bool:
        children = self._children()
        if not self._h100_fp16_prefill_enabled() or rows < 512:
            return False
        properties = torch.cuda.get_device_properties(x.device)
        return (
            self.category == "qkv"
            and len(children) == 3
            and children[0].in_features == 2048
            and tuple(child.out_features for child in children)
            == (2048, 512, 512)
            and all(child.bias is None for child in children)
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
        )

    def _execute_h100_fp16_prefill(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        from ..utils.qvq_cuda import _pgc16_levels, qvq_cuda_hadamard
        from ..utils.qvq_wgmma_cuda import qvq_p32_window_decode_grouped_fp16_packed

        children = self._children()
        rows = x.numel() // children[0].in_features
        x_2d = x.reshape(rows, children[0].in_features).to(torch.float16)
        payload = self._ensure_payload()
        levels = _pgc16_levels(x.device, children[0].codebook_version)
        if self._h100_fp16_prefill_native_enabled():
            from ..utils.qvq_wgmma_cuda import (
                qvq_p32_window_prepare_grouped_fp16_packed,
            )

            folded_weight = qvq_p32_window_prepare_grouped_fp16_packed(
                payload,
                levels,
                children[0]._cached_cast("SU", torch.float32, torch.float32),
                [
                    child._cached_cast("SV", torch.float32, torch.float32)
                    for child in children
                ],
                [child.output_hadamard for child in children],
            )
            self.telemetry.h100_fp16_prefill_native_launches += 1
            # Decoder FP16 + two FP32 transform planes + final FP16 output.
            self.telemetry.h100_fp16_prefill_native_scratch_bytes = (
                children[0].in_features
                * sum(child.out_features for child in children)
                * (torch.float16.itemsize * 2 + torch.float32.itemsize * 2)
            )
        else:
            decoded = qvq_p32_window_decode_grouped_fp16_packed(payload, levels)
            folded_transposed = decoded.float().t().contiguous()
            if children[0].input_hadamard:
                folded_transposed = qvq_cuda_hadamard(
                    folded_transposed,
                    post_scale=children[0]._cached_cast(
                        "SU", torch.float32, torch.float32
                    ),
                    scale_mode=1,
                )
            else:
                folded_transposed = folded_transposed * children[0]._cached_cast(
                    "SU", torch.float32, torch.float32
                )
            folded = folded_transposed.t().contiguous()
            folded_children = []
            offset = 0
            for child in children:
                child_weight = folded[
                    :, offset : offset + child.out_features
                ].contiguous()
                post_scale = child._cached_cast("SV", torch.float32, torch.float32)
                if child.output_hadamard:
                    child_weight = qvq_cuda_hadamard(
                        child_weight,
                        post_scale=post_scale,
                        scale_mode=1,
                    )
                else:
                    child_weight = child_weight * post_scale
                folded_children.append(child_weight)
                offset += child.out_features
            folded_weight = torch.cat(folded_children, dim=1).half()
        output = torch.mm(x_2d.contiguous(), folded_weight, out_dtype=torch.float32)
        inner_outputs = torch.split(
            output, [child.out_features for child in children], dim=1
        )
        self.telemetry.h100_fp16_prefill_temporary_bytes = (
            children[0].in_features
            * sum(child.out_features for child in children)
            * torch.float16.itemsize
        )
        outputs = tuple(
            inner.reshape(*x.shape[:-1], child.out_features).to(x.dtype)
            for child, inner in zip(children, inner_outputs, strict=True)
        )
        return outputs

    def _h100_fp8_prefill_eligible(self, x: torch.Tensor, rows: int) -> bool:
        children = self._children()
        if not self._h100_fp8_prefill_enabled():
            return False
        properties = torch.cuda.get_device_properties(x.device)
        llama_qkv = (
            rows >= 8192
            and self.category == "qkv"
            and len(children) == 3
            and children[0].in_features == 2048
            and tuple(child.out_features for child in children)
            == (2048, 512, 512)
        )
        llama_gate_up = (
            rows >= 512
            and self._h100_fp8_mlp_prefill_enabled()
            and self.category == "gate_up"
            and len(children) == 2
            and children[0].in_features == 2048
            and tuple(child.out_features for child in children)
            == (8192, 8192)
        )
        return (
            (llama_qkv or llama_gate_up)
            and all(child.bias is None for child in children)
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
            and hasattr(torch, "_scaled_mm")
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "float8_e5m2")
        )

    def _h100_fp8_ondemand_eligible(self, x: torch.Tensor, rows: int) -> bool:
        children = self._children()
        if not self._h100_fp8_ondemand_enabled() or rows < 16384:
            return False
        properties = torch.cuda.get_device_properties(x.device)
        return (
            self.category == "qkv"
            and len(children) == 3
            and children[0].in_features == 2048
            and tuple(child.out_features for child in children)
            == (2048, 512, 512)
            and all(child.bias is None for child in children)
            and properties.name == "NVIDIA H100"
            and (properties.major, properties.minor) == (9, 0)
            and hasattr(torch, "_scaled_mm")
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "float8_e5m2")
        )

    def _build_h100_fp8_ondemand_scales(
        self, source_key: tuple[Any, ...]
    ) -> _H100OnDemandFP8Scales:
        from ..utils.qvq_cuda import _pgc16_levels
        from ..utils.qvq_wgmma_cuda import (
            qvq_p32_window_prepare_grouped_fp16_packed,
        )

        children = self._children()
        payload = self._ensure_payload()
        levels = _pgc16_levels(children[0].trellis.device, children[0].codebook_version)
        folded = qvq_p32_window_prepare_grouped_fp16_packed(
            payload,
            levels,
            children[0]._cached_cast("SU", torch.float32, torch.float32),
            [
                child._cached_cast("SV", torch.float32, torch.float32)
                for child in children
            ],
            [child.output_hadamard for child in children],
        )
        weight_scale = (folded.abs().amax() / 448.0).float()
        weight_scale = torch.where(
            weight_scale > 0, weight_scale, torch.ones_like(weight_scale)
        )
        scales = _H100OnDemandFP8Scales(
            weight_scale=weight_scale,
            input_scale=torch.ones_like(weight_scale),
            source_key=source_key,
        )
        if source_key != self._h100_fp8_prefill_source_key():
            raise RuntimeError(
                "QVQ canonical state changed during on-demand FP8 scale calibration"
            )
        self.telemetry.h100_fp8_ondemand_retained_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (scales.weight_scale, scales.input_scale)
        )
        return scales

    def _ensure_h100_fp8_ondemand_scales(
        self,
    ) -> _H100OnDemandFP8Scales | None:
        source_key = self._h100_fp8_prefill_source_key()
        cached = self._h100_fp8_ondemand_scales
        if cached is not None and cached.source_key == source_key:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return None
        self._h100_fp8_ondemand_scales = self._build_h100_fp8_ondemand_scales(
            source_key
        )
        return self._h100_fp8_ondemand_scales

    def _execute_h100_fp8_ondemand(
        self, x: torch.Tensor, scales: _H100OnDemandFP8Scales
    ) -> tuple[torch.Tensor, ...]:
        from ..utils.qvq_cuda import _pgc16_levels
        from ..utils.qvq_wgmma_cuda import (
            qvq_p32_window_prepare_grouped_fp8_packed,
        )

        children = self._children()
        rows = x.numel() // children[0].in_features
        payload = self._ensure_payload()
        levels = _pgc16_levels(x.device, children[0].codebook_version)
        folded_fp8 = qvq_p32_window_prepare_grouped_fp8_packed(
            payload,
            levels,
            children[0]._cached_cast("SU", torch.float32, torch.float32),
            [
                child._cached_cast("SV", torch.float32, torch.float32)
                for child in children
            ],
            [child.output_hadamard for child in children],
            scales.weight_scale,
            half_fold=True,
        )
        x_fp8 = qvq_fp16_to_fp8_e5m2_clamped(
            x.reshape(rows, children[0].in_features).contiguous()
        )
        output = torch._scaled_mm(
            x_fp8,
            folded_fp8,
            scale_a=scales.input_scale,
            scale_b=scales.weight_scale,
            bias=None,
            out_dtype=torch.float16,
            use_fast_accum=True,
        )
        # Decoder FP16 + two FP16 transform planes + final E4M3 output.
        self.telemetry.h100_fp8_ondemand_scratch_bytes = (
            children[0].in_features
            * sum(child.out_features for child in children)
            * (torch.float16.itemsize * 3 + 1)
        )
        widths = tuple(child.out_features for child in children)
        return tuple(
            child_output.reshape(*x.shape[:-1], width)
            for child_output, width in zip(
                torch.split(output, widths, dim=1), widths, strict=True
            )
        )

    def _h100_fp8_prefill_source_key(self) -> tuple[Any, ...]:
        key = list(_source_key(self._children()))
        for child in self._children():
            for name in ("SV", "bias"):
                tensor = getattr(child, name, None)
                key.extend(
                    (id(tensor), None if tensor is None else _tensor_version(tensor))
                )
        return tuple(key)

    def _build_h100_fp8_prefill_payload(
        self, source_key: tuple[Any, ...]
    ) -> _H100FP8PrefillPayload:
        from ..quantization.rotation.hadamard_utils import matmul_hadU

        children = self._children()
        folded_weights = []
        for child in children:
            weight = child.get_inner_weight_tensor(torch.float32)
            if child.input_hadamard:
                weight = matmul_hadU(weight.t()).t()
            weight = weight * child.SU.float().unsqueeze(1)
            if child.output_hadamard:
                weight = matmul_hadU(weight)
            weight = weight * child.SV.float().unsqueeze(0)
            folded_weights.append(weight.half())
        folded = torch.cat(folded_weights, dim=1)
        # QKV uses the previously promoted tensorwise E4M3-weight/E5M2-input
        # contract.  The MLP traverses two FP8 GEMMs around SiLU, so preserve
        # substantially more precision there with one scale per weight output
        # column and one dynamic E4M3 scale per activation row.
        scale = (
            folded.abs().amax(dim=0, keepdim=True) / 448.0
            if self.category == "gate_up"
            else folded.abs().amax() / 448.0
        ).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        # cuBLASLt FP8 expects B in column-major order.  Quantize the
        # transposed contiguous view, then transpose back without copying.
        divisor = scale.t() if scale.ndim == 2 else scale
        weight = (folded.t().contiguous() / divisor).to(torch.float8_e4m3fn).t()
        payload = _H100FP8PrefillPayload(
            weight=weight,
            weight_scale=scale,
            input_scale=torch.ones((), device=weight.device, dtype=torch.float32),
            bias=None,
            source_key=source_key,
        )
        if source_key != self._h100_fp8_prefill_source_key():
            raise RuntimeError("QVQ canonical state changed during FP8 prefill folding")
        self.telemetry.h100_fp8_prefill_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                payload.weight,
                payload.weight_scale,
                payload.input_scale,
            )
        )
        return payload

    def _ensure_h100_fp8_prefill_payload(
        self,
    ) -> _H100FP8PrefillPayload | None:
        source_key = self._h100_fp8_prefill_source_key()
        cached = self._h100_fp8_prefill_payload
        if cached is not None and cached.source_key == source_key:
            return cached
        if torch.cuda.is_current_stream_capturing():
            # Cache construction reconstructs and folds the canonical weight.
            # A cold capture keeps using exact row multiplexing; an eager
            # warmup can build this measured prefill representation later.
            return None
        self._h100_fp8_prefill_payload = self._build_h100_fp8_prefill_payload(
            source_key
        )
        return self._h100_fp8_prefill_payload

    def _execute_h100_fp8_prefill(
        self, x: torch.Tensor, payload: _H100FP8PrefillPayload
    ) -> tuple[torch.Tensor, ...]:
        from ..utils.qvq_cuda import qvq_cuda_quantize_fp8_per_row

        children = self._children()
        rows = x.numel() // children[0].in_features
        x_2d = x.reshape(rows, children[0].in_features).contiguous()
        if payload.weight_scale.ndim == 2:
            x_fp8, input_scale = qvq_cuda_quantize_fp8_per_row(x_2d)
        else:
            x_fp8 = qvq_fp16_to_fp8_e5m2_clamped(x_2d)
            input_scale = payload.input_scale
        output = torch._scaled_mm(
            x_fp8,
            payload.weight,
            scale_a=input_scale,
            scale_b=payload.weight_scale,
            bias=payload.bias,
            out_dtype=torch.float16,
            use_fast_accum=True,
        )
        widths = tuple(child.out_features for child in children)
        return tuple(
            child_output.reshape(*x.shape[:-1], width)
            for child_output, width in zip(
                torch.split(output, widths, dim=1), widths, strict=True
            )
        )

    def _h100_fp8_mlp_down_source_key(self) -> tuple[Any, ...]:
        down = self._mlp_down_ref()
        key = list(_source_key((down,)))
        for name in ("SV", "bias"):
            tensor = getattr(down, name, None)
            key.extend((id(tensor), None if tensor is None else _tensor_version(tensor)))
        return tuple(key)

    def _ensure_h100_fp8_mlp_down_payload(
        self,
    ) -> _H100FP8PrefillPayload | None:
        from ..quantization.rotation.hadamard_utils import matmul_hadU

        down = self._mlp_down_ref()
        source_key = self._h100_fp8_mlp_down_source_key()
        cached = self._h100_fp8_mlp_down_payload
        if cached is not None and cached.source_key == source_key:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return None
        weight = down.get_inner_weight_tensor(torch.float32)
        if down.input_hadamard:
            weight = matmul_hadU(weight.t()).t()
        weight = weight * down.SU.float().unsqueeze(1)
        if down.output_hadamard:
            weight = matmul_hadU(weight)
        weight = weight * down.SV.float().unsqueeze(0)
        folded = weight.half()
        scale = (folded.abs().amax(dim=0, keepdim=True) / 448.0).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        fp8_weight = (
            folded.t().contiguous() / scale.t()
        ).to(torch.float8_e4m3fn).t()
        payload = _H100FP8PrefillPayload(
            weight=fp8_weight,
            weight_scale=scale,
            input_scale=torch.ones((), device=fp8_weight.device, dtype=torch.float32),
            bias=None,
            source_key=source_key,
        )
        if source_key != self._h100_fp8_mlp_down_source_key():
            raise RuntimeError("QVQ down state changed during FP8 prefill folding")
        self._h100_fp8_mlp_down_payload = payload
        self.telemetry.h100_fp8_mlp_down_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                payload.weight,
                payload.weight_scale,
                payload.input_scale,
            )
        )
        return payload

    def _execute_h100_fp8_mlp_down(
        self, x: torch.Tensor, payload: _H100FP8PrefillPayload
    ) -> torch.Tensor:
        from ..utils.qvq_cuda import qvq_cuda_quantize_fp8_per_row

        x_fp8, input_scale = qvq_cuda_quantize_fp8_per_row(x.contiguous())
        return torch._scaled_mm(
            x_fp8,
            payload.weight,
            scale_a=input_scale,
            scale_b=payload.weight_scale,
            bias=payload.bias,
            out_dtype=torch.float16,
            use_fast_accum=True,
        )

    @staticmethod
    def _execute_h100_fp8_mlp_down_quantized(
        x_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        payload: _H100FP8PrefillPayload,
    ) -> torch.Tensor:
        return torch._scaled_mm(
            x_fp8,
            payload.weight,
            scale_a=input_scale,
            scale_b=payload.weight_scale,
            bias=payload.bias,
            out_dtype=torch.float16,
            use_fast_accum=True,
        )

    @staticmethod
    def _large_m_chunk_candidates() -> tuple[int, ...]:
        configured = os.environ.get("QVQ_HOPPER_LARGE_M_CHUNK_CANDIDATES")
        if configured is None:
            return _H100_LARGE_M_CHUNK_CANDIDATES
        candidates: list[int] = []
        for token in configured.split(","):
            try:
                candidate = int(token.strip())
            except ValueError:
                continue
            if (
                candidate in _H100_LARGE_M_CHUNK_CANDIDATES
                and candidate not in candidates
            ):
                candidates.append(candidate)
        return tuple(candidates) or _H100_LARGE_M_CHUNK_CANDIDATES

    def _large_m_chunk_cache_key(
        self, x: torch.Tensor, rows: int, *, scope: str
    ) -> tuple[Any, ...]:
        children = self._children()
        down = None if self._mlp_down_ref is None else self._mlp_down_ref()
        row_bucket = 1 << (rows - 1).bit_length()
        return (
            x.get_device(),
            x.dtype,
            scope,
            row_bucket,
            children[0].in_features,
            tuple(child.out_features for child in children),
            None if down is None else down.out_features,
            qvq_transition_bits(
                children[0].bits, vector_size=children[0].vector_size
            ),
        )

    def _execute_mlp_chunked(
        self, x: torch.Tensor, chunk_rows: int
    ) -> torch.Tensor:
        children = self._children()
        down = self._mlp_down_ref()
        rows = x.numel() // children[0].in_features
        x_2d = x.reshape(rows, children[0].in_features)
        outputs = [
            self._execute_mlp(x_2d[begin : min(begin + chunk_rows, rows)])
            for begin in range(0, rows, chunk_rows)
        ]
        return torch.cat(outputs, dim=0).reshape(
            *x.shape[:-1], down.out_features
        )

    def _execute_group_chunked(
        self, x: torch.Tensor, chunk_rows: int, *, recover: bool
    ) -> tuple[torch.Tensor, ...]:
        children = self._children()
        rows = x.numel() // children[0].in_features
        x_2d = x.reshape(rows, children[0].in_features)
        chunks = [
            self._execute(
                x_2d[begin : min(begin + chunk_rows, rows)], recover=recover
            )
            for begin in range(0, rows, chunk_rows)
        ]
        return tuple(
            torch.cat(child_chunks, dim=0).reshape(
                *x.shape[:-1], child.out_features
            )
            for child, child_chunks in zip(children, zip(*chunks), strict=True)
        )

    def _autotune_large_m_chunk_rows(
        self, x: torch.Tensor, rows: int, *, scope: str
    ) -> int:
        if scope not in ("group", "mlp"):
            raise ValueError("large-M autotune scope must be group or mlp")
        key = self._large_m_chunk_cache_key(x, rows, scope=scope)
        with self._large_m_chunk_cache_lock:
            cached = self._large_m_chunk_cache.get(key)
            if cached is not None:
                return cached

            # Event creation and synchronization are illegal inside capture.
            # A cold graph gets the conservative largest chunk without
            # poisoning the cache; ordinary eager warmup can still tune later.
            if (
                torch.cuda.is_current_stream_capturing()
                or not self._large_m_chunk_autotune_enabled()
            ):
                return 4096

            candidates = self._large_m_chunk_candidates()
            warmup = max(
                0, int(os.environ.get("QVQ_HOPPER_LARGE_M_AUTOTUNE_WARMUP", "1"))
            )
            iterations = max(
                1,
                int(os.environ.get("QVQ_HOPPER_LARGE_M_AUTOTUNE_ITERATIONS", "5")),
            )
            repeats = max(
                1, int(os.environ.get("QVQ_HOPPER_LARGE_M_AUTOTUNE_REPEATS", "3"))
            )
            stream = torch.cuda.current_stream(x.device)
            best_chunk = 4096
            best_ms = float("inf")
            telemetry_before = self.telemetry.__dict__.copy()
            execute = (
                self._execute_mlp_chunked
                if scope == "mlp"
                else lambda value, chunk: self._execute_group_chunked(
                    value, chunk, recover=True
                )
            )
            for candidate in candidates:
                try:
                    for _ in range(warmup):
                        output = execute(x, candidate)
                        del output
                    torch.cuda.synchronize(x.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = execute(x, candidate)
                    for _ in range(warmup):
                        graph.replay()
                    torch.cuda.synchronize(x.device)
                    samples = []
                    for _ in range(repeats):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        for _ in range(iterations):
                            graph.replay()
                        end.record(stream)
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / iterations)
                    elapsed_ms = median(samples)
                    if elapsed_ms < best_ms:
                        best_ms = elapsed_ms
                        best_chunk = candidate
                    del captured, graph
                except (RuntimeError, ValueError):
                    continue
            for name, value in telemetry_before.items():
                setattr(self.telemetry, name, value)
            self.telemetry.h100_large_m_chunk_autotunes += 1
            self._large_m_chunk_cache[key] = best_chunk
            return best_chunk

    def _execute_mlp(self, x: torch.Tensor) -> torch.Tensor:
        down = self._mlp_down_ref()
        from ..utils.qvq_cuda import (
            qvq_cuda_folded_swiglu_precondition_fp32,
            qvq_cuda_folded_swiglu_precondition_ordered_fp32,
            qvq_cuda_hadamard_fp32_to_fp16_multiblock,
            qvq_cuda_hadamard_ordered_split16_fp32_to_fp16,
            qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
            qvq_cuda_swiglu_precondition,
            qvq_cuda_swiglu_precondition_multiblock,
        )

        rows = x.numel() // self._children()[0].in_features
        if rows > 4096:
            chunk_rows = self._autotune_large_m_chunk_rows(
                x, rows, scope="mlp"
            )
            self.telemetry.h100_large_m_chunked_mlp_launches += 1
            self.telemetry.h100_large_m_chunk_rows = chunk_rows
            return self._execute_mlp_chunked(x, chunk_rows)
        children = self._children()
        qwen_folded_intermediate = (
            self._mlp_activation_is_exact_silu
            and not children[0].output_hadamard
            and not children[1].output_hadamard
            and not down.input_hadamard
        )
        # The two exact M16 tiles beat native M32 for W2.5-W3.5, while W2's
        # smaller decoder remains faster in the native direct-down path.
        if (
            rows == 32
            and qwen_folded_intermediate
            and float(children[0].bits) != 2.0
        ):
            self.telemetry.h100_qwen_m32_fused_tiles += 1
            return self._execute_mlp_chunked(x, 16)
        fp8_prefill = (
            self._ensure_h100_fp8_prefill_payload()
            if self._h100_fp8_prefill_eligible(x, rows)
            else None
        )
        if fp8_prefill is not None:
            self.telemetry.h100_fp8_prefill_launches += 1
            gate, up = self._execute_h100_fp8_prefill(x, fp8_prefill)
            # Keep the two split views strided.  SiLU and multiplication can
            # consume them directly, avoiding two full Mx8192 materializations.
            use_fp8_down = (
                (down.in_features, down.out_features) == (8192, 2048)
                and down.bias is None
                and down.v2b2_p32
                and down.vector_size == 2
                and down.trellis_window == 16
            )
            fp8_down = (
                self._ensure_h100_fp8_mlp_down_payload()
                if use_fp8_down
                else None
            )
            if fp8_down is not None:
                if self._mlp_activation_is_exact_silu:
                    from .triton_utils.kernels import fused_silu_mul_quant_fp8

                    intermediate_fp8, intermediate_scale = (
                        fused_silu_mul_quant_fp8(gate, up)
                    )
                    self.telemetry.h100_fp8_mlp_fused_silu_launches += 1
                    recovered = self._execute_h100_fp8_mlp_down_quantized(
                        intermediate_fp8,
                        intermediate_scale,
                        fp8_down,
                    )
                else:
                    intermediate = self._mlp_act_fn(gate) * up
                    recovered = self._execute_h100_fp8_mlp_down(
                        intermediate, fp8_down
                    )
                self.telemetry.h100_fp8_mlp_down_launches += 1
                return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)
            activated_gate = self._mlp_act_fn(gate)
            transformed = qvq_cuda_swiglu_precondition_multiblock(
                activated_gate.reshape(rows, down.in_features),
                up.reshape(rows, down.in_features),
                down._cached_cast("SU", torch.float16),
                half2_high=True,
            )
            self.telemetry.h100_multiblock_precondition_launches += 1
            self.telemetry.h100_half2_precondition_high_launches += 1
        elif rows > 16 and (
            qwen_folded_intermediate
            or not self._h100_multiblock_intermediate_enabled
        ):
            # Large-M uses the exact generic module boundaries while sharing
            # gate/up input preparation and P32 decode. The ordinary down
            # module now owns the same M32/M64 row-reuse dispatch, so this
            # route is graph-safe and avoids the planar GEMV fallback without
            # extending decode-only fused transform kernels beyond their
            # measured geometry.
            gate, up = self._execute(x)
            activated_gate = self._mlp_act_fn(gate)
            intermediate = activated_gate * up
            if qwen_folded_intermediate:
                # Qwen's architecture contract removes the 17,408-wide
                # gate/up output transforms and the down input transform.
                # Execute the remaining down transform directly so CUDA Graph
                # capture does not also record QVQLinear's conservative BF16
                # overflow retry for a composite input transform that is not
                # present. The P32 inner result and output recovery stay FP32
                # until the ordinary final model-dtype cast.
                transformed = down._qvq_prepare_inference_input(
                    intermediate.reshape(rows, down.in_features),
                    torch.float16,
                )
                recovered = down._forward_pretransformed_compute_dtype(
                    transformed,
                    torch.float16,
                )
                self.telemetry.h100_qwen_large_m_direct_down_launches += 1
                return recovered.reshape(
                    *x.shape[:-1], down.out_features
                ).to(x.dtype)
            return down(intermediate)
        elif qwen_folded_intermediate:
            # Qwen3.8-27B has a 17*1024 intermediate width, for which no exact
            # composite Hadamard base exists.  Its model definition therefore
            # quantizes gate/up without output H and down without input H.
            # Preserve the ordinary module boundary exactly: recover each
            # FP32 child, round it to the model FP16 dtype, execute SiLU and
            # product, then apply down.SU.  No transform is commuted through
            # the nonlinearity, and every operation is CUDA Graph capturable.
            payload = self._ensure_payload()
            use_h100_folded_fusion = (
                self._h100_fp16_recovery_store_enabled
                and x.dtype == torch.float16
                and children[0].in_features == 5120
                and down.in_features == 17408
                and down.out_features == 5120
            )
            use_ordered_reduction_fusion = (
                use_h100_folded_fusion
                and len({segment.split_count for segment in payload.plan.segments}) == 1
                and payload.plan.segments[0].split_count in (5, 10)
            )
            if use_ordered_reduction_fusion:
                gate_up_split_count = payload.plan.segments[0].split_count
                partials = self._execute(
                    x,
                    recover=False,
                    return_ordered_partials=True,
                )
                transformed = qvq_cuda_folded_swiglu_precondition_ordered_fp32(
                    partials,
                    gate_scale=children[0]._cached_cast("SV", torch.float16, torch.float32),
                    up_scale=children[1]._cached_cast("SV", torch.float16, torch.float32),
                    gate_bias=children[0]._cached_cast("bias", torch.float16, torch.float32),
                    up_bias=children[1]._cached_cast("bias", torch.float16, torch.float32),
                    down_scale=down._cached_cast("SU", torch.float16),
                    split_count=gate_up_split_count,
                    logical_rows=rows,
                )
                self.telemetry.h100_folded_qwen_fused_precondition_launches += 1
                self.telemetry.h100_folded_qwen_fused_ordered_reduction_launches += 1
                if gate_up_split_count == 5:
                    self.telemetry.h100_qwen_fixed_ordered_grid_launches += 1
                self.telemetry.h100_qwen_ordered_decode_prefetch_launches += 1
                if children[0].bits == 3:
                    self.telemetry.h100_qwen_w3_ordered_decode_prefetch_launches += 1
            elif use_h100_folded_fusion:
                inner_gate, inner_up = self._execute(x, recover=False)
                transformed = qvq_cuda_folded_swiglu_precondition_fp32(
                    inner_gate,
                    inner_up,
                    gate_scale=children[0]._cached_cast("SV", torch.float16, torch.float32),
                    up_scale=children[1]._cached_cast("SV", torch.float16, torch.float32),
                    gate_bias=children[0]._cached_cast("bias", torch.float16, torch.float32),
                    up_bias=children[1]._cached_cast("bias", torch.float16, torch.float32),
                    down_scale=down._cached_cast("SU", torch.float16),
                )
                self.telemetry.h100_folded_qwen_fused_precondition_launches += 1
            else:
                inner_gate, inner_up = self._execute(x, recover=False)
                gate = children[0]._qvq_recover_inference_output(
                    inner_gate, torch.float16
                ).reshape(rows, down.in_features).to(x.dtype)
                up = children[1]._qvq_recover_inference_output(
                    inner_up, torch.float16
                ).reshape(rows, down.in_features).to(x.dtype)
                activated_gate = self._mlp_act_fn(gate)
                transformed = down._qvq_prepare_inference_input(
                    activated_gate * up,
                    torch.float16,
                )
            self.telemetry.independent_recovery_children += 2
            self.telemetry.h100_folded_qwen_mlp_launches += 1
        elif (
            self._h100_multiblock_intermediate_enabled
            and self._mlp_activation_is_exact_silu
        ):
            direct_pad = rows < 16
            inner_gate, inner_up = self._execute(x, recover=False)
            transformed = qvq_cuda_hadamard_pair_swiglu_precondition_multiblock(
                inner_gate,
                inner_up,
                post_scale0=children[0]._cached_cast(
                    "SV", torch.float16, torch.float32
                ),
                post_scale1=children[1]._cached_cast(
                    "SV", torch.float16, torch.float32
                ),
                bias0=children[0]._cached_cast(
                    "bias", torch.float16, torch.float32
                ),
                bias1=children[1]._cached_cast(
                    "bias", torch.float16, torch.float32
                ),
                pre_scale=down._cached_cast("SU", torch.float16),
                scale_mode=3,
                pad_to_16=direct_pad,
                pair_tiles=rows >= 16,
                bounded_rounding=(
                    self._h100_bounded_recovery_rounding_enabled and rows == 16
                ),
                packed_gate_up=(
                    self._h100_bounded_recovery_rounding_enabled
                    and self._h100_packed_gate_up_recovery_enabled
                    and rows == 16
                ),
            )
            self.telemetry.paired_recovery_launches += 1
            self.telemetry.h100_multiblock_recovery_launches += 1
            self.telemetry.h100_warp_recovery_low_launches += 1
            self.telemetry.h100_fused_recovery_precondition_launches += 1
            if rows >= 16:
                self.telemetry.h100_paired_recovery_tiles_launches += 1
            if self._h100_bounded_recovery_rounding_enabled and rows == 16:
                self.telemetry.h100_bounded_recovery_rounding_launches += 1
            if self._h100_packed_gate_up_recovery_enabled and rows == 16:
                self.telemetry.h100_packed_gate_up_recovery_launches += 1
            self.telemetry.h100_multiblock_precondition_launches += 1
            self.telemetry.h100_half2_precondition_high_launches += 1
            self.telemetry.h100_fused_silu_precondition_low_launches += 1
            self.telemetry.h100_half2_precondition_low_launches += 1
            if direct_pad:
                self.telemetry.h100_direct_padded_precondition_launches += 1
        else:
            gate, up = self._execute(x)
            if (
                not isinstance(gate, torch.Tensor)
                or gate.shape != up.shape
                or gate.dtype != torch.float16
                or not gate.is_contiguous()
                or not up.is_contiguous()
            ):
                raise _R0Fallback(
                    "fused MLP recovery must return contiguous FP16 gate/up geometry"
                )
            activated_gate = self._mlp_act_fn(gate)
            if (
                not isinstance(activated_gate, torch.Tensor)
                or activated_gate.shape != up.shape
                or activated_gate.dtype != torch.float16
                or not activated_gate.is_contiguous()
            ):
                raise _R0Fallback(
                    "fused MLP activation must return contiguous FP16 gate geometry"
                )
            if self._h100_multiblock_intermediate_enabled:
                transformed = qvq_cuda_swiglu_precondition_multiblock(
                    activated_gate.reshape(rows, down.in_features),
                    up.reshape(rows, down.in_features),
                    down._cached_cast("SU", torch.float16),
                    half2_high=True,
                )
                self.telemetry.h100_multiblock_precondition_launches += 1
                self.telemetry.h100_half2_precondition_high_launches += 1
            else:
                transformed = qvq_cuda_swiglu_precondition(
                    activated_gate.reshape(rows, down.in_features),
                    up.reshape(rows, down.in_features),
                    down._cached_cast("SU", torch.float16),
                )
        fused_down_recovery = (
            rows <= 16
            and self._h100_multiblock_intermediate_enabled
            and down.output_hadamard
            and (down.in_features, down.out_features) == (8192, 2048)
        )
        if fused_down_recovery:
            partials = down._inner_forward(
                transformed,
                return_ordered_partials=True,
            )
            recovered = qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
                partials,
                post_scale=down._cached_cast(
                    "SV", torch.float16, torch.float32
                ),
                bias=down._cached_cast("bias", torch.float16, torch.float32),
                scale_mode=3,
                logical_rows=rows,
                multiblock=True,
            )
            self.telemetry.h100_fused_down_reduction_recovery_launches += 1
            self.telemetry.h100_multiblock_down_recovery_launches += 1
            self.telemetry.h100_fp16_recovery_store_launches += 1
            return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)

        qwen_transition_bits = qvq_transition_bits(
            down.bits, vector_size=down.vector_size
        )
        use_qwen_ordered_composite_recovery = (
            rows <= 16
            and self._h100_fp16_recovery_store_enabled
            and down.output_hadamard
            and (down.in_features, down.out_features) == (17408, 5120)
            and qwen_transition_bits == 6
        )
        if use_qwen_ordered_composite_recovery:
            from ..quantization.rotation.hadamard_utils import _get_hadK_on
            from ..utils.qvq_cuda import (
                qvq_cuda_qwen_composite_ordered_recovery_fp32_to_fp16,
            )

            split_count = 17
            partials = down._inner_forward(
                transformed,
                return_ordered_partials=True,
                ordered_split_count=split_count,
            )
            base, base_width = _get_hadK_on(
                down._cached_cast("SV", torch.float16), False
            )
            if base is None or base_width != 40:
                raise _R0Fallback("Qwen 5120 output requires its canonical H40 base")
            recovered = qvq_cuda_qwen_composite_ordered_recovery_fp32_to_fp16(
                partials,
                base=base,
                post_scale=down._cached_cast("SV", torch.float16, torch.float32),
                bias=down._cached_cast("bias", torch.float16, torch.float32),
                split_count=split_count,
                logical_rows=rows,
            )
            self.telemetry.h100_qwen_composite_down_recovery_launches += 1
            self.telemetry.h100_qwen_ordered_composite_down_recovery_launches += 1
            self.telemetry.h100_qwen_down_decode_prefetch_launches += 1
            self.telemetry.h100_qwen_w3_down_decode_prefetch_launches += 1
            self.telemetry.h100_fp16_recovery_store_launches += 1
            return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)

        inner = down._inner_forward(transformed)
        use_large_m_multiblock_down_recovery = (
            rows > 16
            and self._h100_multiblock_intermediate_enabled
            and down.output_hadamard
            and inner.dtype == torch.float32
            and (down.in_features, down.out_features) == (8192, 2048)
        )
        if use_large_m_multiblock_down_recovery:
            recovered = qvq_cuda_hadamard_fp32_to_fp16_multiblock(
                inner[:rows].contiguous(),
                post_scale=down._cached_cast("SV", torch.float16, torch.float32),
                bias=down._cached_cast("bias", torch.float16, torch.float32),
                scale_mode=3,
            )
            self.telemetry.h100_multiblock_down_recovery_launches += 1
            self.telemetry.h100_fp16_recovery_store_launches += 1
            return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)
        if (
            self._h100_fp16_recovery_store_enabled
            and (down.in_features, down.out_features) == (17408, 5120)
            and qwen_transition_bits in (4, 5)
        ):
            self.telemetry.h100_qwen_down_decode_prefetch_launches += 1
        use_qwen_composite_recovery = (
            self._h100_fp16_recovery_store_enabled
            and down.output_hadamard
            and inner.dtype == torch.float32
            and (down.in_features, down.out_features) == (17408, 5120)
        )
        if use_qwen_composite_recovery:
            from ..quantization.rotation.hadamard_utils import _get_hadK_on
            from ..utils.qvq_cuda import (
                qvq_cuda_qwen_composite_recovery_fp32_to_fp16,
            )

            base, base_width = _get_hadK_on(
                down._cached_cast("SV", torch.float16), False
            )
            if base is None or base_width != 40:
                raise _R0Fallback("Qwen 5120 output requires its canonical H40 base")
            recovered = qvq_cuda_qwen_composite_recovery_fp32_to_fp16(
                inner[:rows].contiguous(),
                base=base,
                post_scale=down._cached_cast("SV", torch.float16, torch.float32),
                bias=down._cached_cast("bias", torch.float16, torch.float32),
            )
            self.telemetry.h100_qwen_composite_down_recovery_launches += 1
            self.telemetry.h100_fp16_recovery_store_launches += 1
            return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)
        fp16_store = (
            self._h100_fp16_recovery_store_enabled
            and down.output_hadamard
            and inner.dtype == torch.float32
            and down.out_features <= 16384
            and down.out_features & (down.out_features - 1) == 0
        )
        recovered = down._qvq_recover_inference_output(
            inner[:rows],
            torch.float16,
            output_fp16=fp16_store,
        )
        if fp16_store:
            self.telemetry.h100_fp16_recovery_store_launches += 1
        return recovered.reshape(*x.shape[:-1], down.out_features).to(x.dtype)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Execute exact gate/up, activation, precondition, and down projection."""

        if self._outputs is not None:
            self.telemetry.stale_cycles += 1
            self._clear_cycle()
        rejection = self._mlp_rejection(x)
        if rejection is not None:
            self.telemetry.fused_mlp_fallbacks += 1
            self.telemetry.last_fallback_reason = rejection
            parent = self._mlp_parent_ref()
            return parent._gptqmodel_qvq_fused_mlp_original_forward(x)
        try:
            output = self._execute_mlp(x)
        except _R0Fallback as exc:
            self.telemetry.fused_mlp_fallbacks += 1
            self.telemetry.last_fallback_reason = str(exc)
            parent = self._mlp_parent_ref()
            return parent._gptqmodel_qvq_fused_mlp_original_forward(x)
        self.telemetry.grouped_launches += 1
        self.telemetry.fused_mlp_launches += 1
        self.telemetry.last_fallback_reason = None
        return output

    def forward(self, member_index: int, x: torch.Tensor) -> torch.Tensor:
        if not 0 <= member_index < len(self._refs):
            raise IndexError("QVQ grouped runtime member index is out of range")

        if member_index == 0:
            if self._outputs is not None:
                self.telemetry.stale_cycles += 1
                self._clear_cycle()
            rejection = self._runtime_eligible(x)
            if rejection is not None:
                return self._fallback(member_index, x, rejection)
            try:
                outputs = self._execute(x)
            except _R0Fallback as exc:
                return self._fallback(member_index, x, str(exc))
            self.telemetry.grouped_launches += 1
            self.telemetry.last_fallback_reason = None
            self._input = x
            self._input_version = _tensor_version(x)
            self._outputs = outputs
            self._next_index = 1
            return outputs[0]

        if self._outputs is None:
            return self._fallback(
                member_index, x, "sibling arrived without an active primary"
            )
        if (
            member_index != self._next_index
            or self._input is not x
            or self._input_version != _tensor_version(x)
        ):
            self.telemetry.stale_cycles += 1
            self._clear_cycle()
            return self._fallback(
                member_index, x, "sibling order, identity, or mutation version changed"
            )

        output = self._outputs[member_index]
        self.telemetry.sibling_cache_hits += 1
        self._next_index += 1
        if self._next_index == len(self._refs):
            self._clear_cycle()
        return output


@torch._dynamo.disable
def _qvq_grouped_projection_forward(self: QVQLinear, x: torch.Tensor) -> torch.Tensor:
    runtime = self._gptqmodel_qvq_grouped_runtime
    return runtime.forward(self._gptqmodel_qvq_grouped_index, x)


@torch._dynamo.disable
def _qvq_fused_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    runtime = self._gptqmodel_qvq_fused_mlp_runtime
    return runtime.forward_mlp(x)


def _maybe_install_qvq_mlp_fusion(
    parent: nn.Module,
    parent_name: str,
    children: tuple[QVQLinear, ...],
    runtime: QVQHopperGroupedRuntime,
) -> bool:
    """Install only after exact one-row parity proves the parent structure."""

    if len(children) != 2 or children[0].out_features != children[1].out_features:
        return False
    from .fused_quant_linear import (
        _detect_mlp_activation,
        _find_gate_up_down_module,
        _is_safe_mlp_parent,
    )

    down = _find_gate_up_down_module(
        parent,
        list(children),
        children[0].out_features,
        children[0].in_features,
    )
    act_fn = _detect_mlp_activation(parent)
    if (
        not isinstance(down, QVQLinear)
        or act_fn is None
        or not _is_safe_mlp_parent(parent, parent_name)
        or children[0].trellis.device.type != "cuda"
    ):
        return False
    properties = torch.cuda.get_device_properties(children[0].trellis.device)
    if (properties.major, properties.minor) != (9, 0):
        return False

    runtime._configure_mlp_fusion(parent, down, act_fn)
    sample = torch.linspace(
        -0.01,
        0.01,
        children[0].in_features,
        device=children[0].trellis.device,
        dtype=torch.float16,
    ).reshape(1, -1)
    try:
        with torch.inference_mode():
            expected = parent.forward(sample)
            actual = runtime._execute_mlp(sample)
        exact = torch.equal(actual, expected)
    except (AttributeError, RuntimeError, TypeError, ValueError, _R0Fallback):
        exact = False
    runtime.invalidate()
    runtime.telemetry = QVQGroupedRuntimeTelemetry(
        runtime.category, runtime.member_names
    )
    if not exact:
        runtime._clear_mlp_fusion()
        return False

    parent._gptqmodel_qvq_fused_mlp_runtime = runtime
    parent._gptqmodel_qvq_fused_mlp_original_forward = parent.forward
    parent.forward = MethodType(_qvq_fused_mlp_forward, parent)
    return True


def _install_candidates(
    model: nn.Module,
    candidates: Sequence[tuple[str, ...]],
    *,
    category: str,
    fuse_activation: bool = False,
) -> int:
    installed = 0
    # Materialize the traversal before replacing methods.  The runtime object
    # is deliberately not an nn.Module, so it never alters the model tree.
    for parent_name, parent in tuple(model.named_modules()):
        for member_names in candidates:
            members = tuple(getattr(parent, name, None) for name in member_names)
            try:
                children = _validate_static_group(members)
            except (AttributeError, TypeError, ValueError, _R0Fallback):
                continue
            runtime = QVQHopperGroupedRuntime(
                children,
                member_names,
                category=category,
            )
            for index, child in enumerate(children):
                child._gptqmodel_qvq_grouped_runtime = runtime
                child._gptqmodel_qvq_grouped_index = index
                child._gptqmodel_qvq_grouped_original_forward = child.forward
                child.forward = MethodType(_qvq_grouped_projection_forward, child)
            if category == "gate_up" and fuse_activation:
                _maybe_install_qvq_mlp_fusion(
                    parent,
                    parent_name,
                    children,
                    runtime,
                )
            installed += 1
    return installed


def install_qvq_hopper_groups(
    model: nn.Module,
    *,
    qkv_candidates: Sequence[tuple[str, ...]] | None = None,
    gate_up_candidates: Sequence[tuple[str, ...]] | None = None,
    qkv: bool = True,
    gate_up: bool = True,
    gate_up_activation: bool = True,
) -> dict[str, int]:
    """Install exact architecture-declared QVQ groups for production inference."""

    counts: dict[str, int] = {}
    if qkv:
        counts["qkv"] = _install_candidates(
            model,
            _QKV_CANDIDATES if qkv_candidates is None else qkv_candidates,
            category="qkv",
        )
    if gate_up:
        counts["gate_up"] = _install_candidates(
            model,
            _GATE_UP_CANDIDATES if gate_up_candidates is None else gate_up_candidates,
            category="gate_up",
            fuse_activation=gate_up_activation,
        )
    return counts


def qvq_grouped_runtime_telemetry(model: nn.Module) -> list[dict[str, Any]]:
    """Return one telemetry snapshot for every unique installed QVQ group."""

    seen: set[int] = set()
    snapshots = []
    for module in model.modules():
        runtime = getattr(module, "_gptqmodel_qvq_grouped_runtime", None)
        if runtime is None or id(runtime) in seen:
            continue
        seen.add(id(runtime))
        snapshots.append(runtime.telemetry.snapshot())
    return snapshots


def uninstall_qvq_hopper_groups(model: nn.Module) -> int:
    """Restore original child forwards and release grouped transient buffers."""

    seen: set[int] = set()
    removed = 0
    for module in model.modules():
        runtime = getattr(module, "_gptqmodel_qvq_grouped_runtime", None)
        if runtime is None:
            continue
        if id(runtime) not in seen:
            parent = (
                None if runtime._mlp_parent_ref is None else runtime._mlp_parent_ref()
            )
            if parent is not None:
                original_parent_forward = getattr(
                    parent,
                    "_gptqmodel_qvq_fused_mlp_original_forward",
                    None,
                )
                if original_parent_forward is not None:
                    parent.forward = original_parent_forward
                for parent_attr in (
                    "_gptqmodel_qvq_fused_mlp_runtime",
                    "_gptqmodel_qvq_fused_mlp_original_forward",
                ):
                    if hasattr(parent, parent_attr):
                        delattr(parent, parent_attr)
                runtime._clear_mlp_fusion()
            runtime.invalidate()
            seen.add(id(runtime))
            removed += 1
        original = getattr(module, "_gptqmodel_qvq_grouped_original_forward", None)
        if original is not None:
            module.forward = original
        for name in (
            "_gptqmodel_qvq_grouped_runtime",
            "_gptqmodel_qvq_grouped_index",
            "_gptqmodel_qvq_grouped_original_forward",
        ):
            if hasattr(module, name):
                delattr(module, name)
    return removed


__all__ = [
    "QVQGroupedRuntimeTelemetry",
    "QVQHopperGroupedRuntime",
    "install_qvq_hopper_groups",
    "qvq_grouped_runtime_telemetry",
    "uninstall_qvq_hopper_groups",
]
