# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Grouped GEMM MoE dispatch for post-quantization inference.

This module provides a `linear_loop` experts implementation that uses
`torch.nn.functional.grouped_mm` to dispatch active experts with one kernel
per projection (gate, up, down) instead of looping over every expert.  It is
intended to be registered by GPTQModel after a quantized model has been loaded;
the plain per-expert fallback lives in Defuser and is used during quantization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.utils import log
from gptqmodel.utils.env import env_flag
from gptqmodel.utils.marlin_moe import (
    marlin_moe_runtime_available,
    moe_wna16_marlin_gemm,
)

DEBUG_ON = env_flag("DEBUG")

LINEAR_LOOP_IMPL = "linear_loop"

# Per-experts-module flag that enables the grouped GEMM fast path.
# Only models loaded for post-quantization inference should set this; quantization
# still needs the defused per-expert forward for per-module calibration hooks.
GROUPED_DISPATCH_FLAG = "_gptqmodel_grouped_dispatch_enabled"

try:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    HAS_EXPERTS_INTERFACE = True
except Exception:
    ALL_EXPERTS_FUNCTIONS = None  # type: ignore[assignment]
    HAS_EXPERTS_INTERFACE = False

# Global cache for weight orientation decisions per projection class.
# True  -> raw weight must be transposed before grouped_mm (nn.Linear convention).
# False -> raw weight is already in [in_features, out_features] layout.
# None  -> class was probed and does not support dense grouped-mm.
_ORIENTATION_CACHE: dict[int, bool | None] = {}

# Cache for the batched/offset Marlin MoE mega-kernel availability check.
# None -> not probed yet; True/False -> result.
_MARLIN_MOE_AVAILABLE: bool | None = None


def _grouped_mm_min_tokens() -> int:
    """Minimum token-expert pairs before the grouped dispatch path is attempted.

    The batched/offset Marlin MoE kernel and grouped ``grouped_mm`` path both pay
    off for prefill and moderate-to-large batches, but for very small decode
    batches the per-call overhead of building the block-aligned token-expert index
    and stacking active-expert weights can dominate.  The default (1) keeps the
    grouped path enabled for all shapes; raise it via the environment variable when
    profiling shows the fallback is faster for a given model/workload.
    """
    import os

    try:
        return int(os.environ.get("GPTQMODEL_GROUPED_MM_MIN_TOKENS", "1"))
    except ValueError:
        return 1


def _marlin_moe_available() -> bool:
    """Return whether the batched/offset Marlin MoE mega-kernel can be loaded."""
    global _MARLIN_MOE_AVAILABLE
    if _MARLIN_MOE_AVAILABLE is None:
        _MARLIN_MOE_AVAILABLE = marlin_moe_runtime_available()
    return _MARLIN_MOE_AVAILABLE


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group token-expert pairs by active expert and pad each group to ``block_size``.

    Returns ``(active_experts, sorted_token_ids, expert_ids, num_tokens_post_padded)``.
    ``sorted_token_ids`` values are global token-expert indices (0 .. num_tokens*top_k-1),
    followed by ``num_tokens*top_k`` as padding fill.  ``expert_ids`` are local indices
    (0 .. num_active-1) into ``active_experts`` and therefore into a stacked weight
    tensor built in the same order.

    This implementation is fully vectorized except for a small loop over the active
    experts; it avoids per-expert ``nonzero`` and ``tolist`` calls that force
    device-to-host synchronization on the hot decode path.
    """
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_indices = torch.argsort(flat, stable=True)
    sorted_flat = flat[sorted_indices]
    active, counts = torch.unique_consecutive(sorted_flat, return_counts=True)

    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded = int(padded_counts.sum().item())

    sorted_token_ids = torch.full(
        (total_padded,), numel, dtype=torch.int32, device=flat.device
    )
    expert_ids_tensor = torch.empty(
        (total_padded // block_size,), dtype=torch.int32, device=flat.device
    )
    out_pos = 0
    exp_pos = 0
    start = 0
    for i, c in enumerate(counts.tolist()):
        if c:
            sorted_token_ids[out_pos : out_pos + c] = sorted_indices[start : start + c]
            blocks = (c + block_size - 1) // block_size
            expert_ids_tensor[exp_pos : exp_pos + blocks] = i
            out_pos += blocks * block_size
            exp_pos += blocks
        start += c

    num_tokens_post_padded = torch.tensor(
        total_padded, dtype=torch.int32, device=flat.device
    )
    return active, sorted_token_ids, expert_ids_tensor, num_tokens_post_padded


def _moe_block_size(num_tokens: int, top_k: int) -> int:
    """Pick a MoE block size that keeps block count modest without excessive padding."""
    total = num_tokens * top_k
    if total <= 32:
        return 8
    if total <= 128:
        return 16
    if total <= 384:
        return 32
    return 64


def _has_extra_transform(proj: nn.Module) -> bool:
    """Return True when `proj` applies an adapter or online rotation to its output.

    Projections with these transforms cannot be reproduced by a plain dense
    GEMM of the raw packed weights, so the grouped dispatch must fall back to
    the per-expert loop. Mirrors `_has_active_rotation` in fused_quant_linear.
    """
    return (
        getattr(proj, "adapter", None) is not None
        or getattr(proj, "online_full_had", False)
        or getattr(proj, "online_partial_had", False)
        or getattr(proj, "had_K", None) is not None
    )


def _grouped_mm_available() -> bool:
    """Check whether the PyTorch grouped_mm helper is available on CUDA."""
    return hasattr(torch.nn.functional, "grouped_mm") and torch.cuda.is_available()


def _extract_expert_bias(
    proj: nn.Module,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a contiguous (out_features,) bias tensor, or None."""
    bias = getattr(proj, "bias", None)
    if bias is None:
        return None
    bias = bias.to(device=device, dtype=dtype)
    if bias.numel() != out_features:
        return None
    return bias.contiguous()


def _is_marlin_packed(proj: nn.Module) -> bool:
    """Return True when ``proj`` is a Marlin-packed quantized linear layer."""
    try:
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    except Exception:
        return False
    return isinstance(proj, MarlinLinear)


def _try_dequant_gptq_weight(
    proj: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Dequantize a standard int32-packed GPTQ projection directly to ``dtype``.

    This bypasses ``proj.dequantize_weight()`` which may return ``float16`` and
    then get cast to ``bfloat16``, introducing small numerical drift that makes
    the grouped GEMM probe fail for TritonV2-backed experts.  It is only valid
    for the standard TritonV2 GPTQ pack layout; Marlin-packed weights must use
    the Marlin kernel's own dequantization path.
    """
    if _is_marlin_packed(proj):
        return None
    if not (
        hasattr(proj, "qweight")
        and hasattr(proj, "scales")
        and hasattr(proj, "qzeros")
        and hasattr(proj, "g_idx")
        and hasattr(proj, "bits")
        and hasattr(proj, "pack_dtype_bits")
        and hasattr(proj, "maxq")
    ):
        return None
    bits = proj.bits
    pack_dtype_bits = proj.pack_dtype_bits
    if bits not in (2, 3, 4, 8) or pack_dtype_bits != 32:
        return None

    try:
        from gptqmodel.nn_modules.triton_utils.dequant import dequant

        w = dequant(dtype, proj.qweight, proj.scales, proj.qzeros, proj.g_idx, bits, pack_dtype_bits, proj.maxq)
    except Exception:
        return None

    expected_rows = proj.in_features
    expected_cols = proj.out_features
    if w.shape != (expected_rows, expected_cols):
        return None
    return w.to(device=device, dtype=dtype).contiguous()


def _probe_weight_orientation(
    proj: nn.Module,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool | None:
    """Probe a projection to discover whether its raw weight needs a transpose.

    We compare the projection's own forward() against x @ raw_weight for both
    orientations on a tiny random input.  This handles mixed conventions
    (nn.Linear uses [out, in]; GPTQ dequantize_weight returns [in, out]) and
    automatically excludes backends whose native kernel output cannot be
    reproduced by a dense fp16/bf16 GEMM.
    """
    cls_id = id(proj.__class__)
    if cls_id in _ORIENTATION_CACHE:
        return _ORIENTATION_CACHE[cls_id]

    x = torch.randn(2, in_features, device=device, dtype=dtype)
    with torch.inference_mode():  # keep the probe out of any autograd graph
        try:
            y_ref = proj(x)
        except Exception:
            _ORIENTATION_CACHE[cls_id] = None
            return None
        if y_ref is None or y_ref.shape[-1] != out_features:
            _ORIENTATION_CACHE[cls_id] = None
            return None
        y_ref = y_ref.reshape(2, out_features)

        # Try the direct GPTQ dequant path first; it is exact for TritonV2 experts
        # and avoids fp16->bf16 casting drift in proj.dequantize_weight().
        w_gptq = _try_dequant_gptq_weight(proj, dtype, device)
        if w_gptq is not None:
            bias = _extract_expert_bias(proj, out_features, dtype, device)
            y = x @ w_gptq
            if bias is not None:
                y = y + bias
            if torch.allclose(y, y_ref, atol=1e-3, rtol=1e-2):
                _ORIENTATION_CACHE[cls_id] = False
                return False

        if hasattr(proj, "dequantize_weight"):
            try:
                w = proj.dequantize_weight()
            except Exception:
                _ORIENTATION_CACHE[cls_id] = None
                return None
        else:
            w = getattr(proj, "weight", None)
        if not isinstance(w, torch.Tensor):
            _ORIENTATION_CACHE[cls_id] = None
            return None
        w = w.to(device=device, dtype=dtype)
        bias = _extract_expert_bias(proj, out_features, dtype, device)

        for needs_t in (False, True):
            wt = w.t() if needs_t else w
            if wt.shape != (in_features, out_features):
                continue
            y = x @ wt
            if bias is not None:
                y = y + bias
            if torch.allclose(y, y_ref, atol=1e-3, rtol=1e-2):
                _ORIENTATION_CACHE[cls_id] = needs_t
                return needs_t

        _ORIENTATION_CACHE[cls_id] = None
        return None


def _extract_expert_dense_weight(
    proj: nn.Module,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a dense (in_features, out_features) weight tensor for grouped GEMM."""
    # Fast path: directly dequantize standard GPTQ packed weights to the activation dtype.
    # Only safe when the projection is a plain dequant + matmul; adapters, rotations,
    # or any other per-forward transform require the per-expert loop.
    if not _has_extra_transform(proj):
        w_gptq = _try_dequant_gptq_weight(proj, dtype, device)
        if w_gptq is not None:
            return w_gptq

    needs_t = _probe_weight_orientation(proj, in_features, out_features, dtype, device)
    if needs_t is None:
        return None

    if hasattr(proj, "dequantize_weight"):
        w = proj.dequantize_weight()
    else:
        w = getattr(proj, "weight", None)
    if not isinstance(w, torch.Tensor):
        return None
    w = w.to(device=device, dtype=dtype)
    if needs_t:
        w = w.t()
    if w.shape != (in_features, out_features):
        return None
    return w.contiguous()


def _can_extract_dense_expert(
    expert0: nn.Module,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return True when every projection needed for grouped dispatch has a dense equivalent."""
    if _expert_gateup_is_fused(expert0):
        if _extract_fused_gateup_dense_weight(expert0, dtype, device) is None:
            return False
        if _extract_expert_dense_weight(
            expert0.down_proj, intermediate_dim, hidden_dim, dtype, device
        ) is None:
            return False
    else:
        for proj, in_f, out_f in (
            (expert0.gate_proj, hidden_dim, intermediate_dim),
            (expert0.up_proj, hidden_dim, intermediate_dim),
            (expert0.down_proj, intermediate_dim, hidden_dim),
        ):
            if _extract_expert_dense_weight(proj, in_f, out_f, dtype, device) is None:
                return False
    return True


def _is_marlin_expert(expert: nn.Module) -> bool:
    """Return True when all three expert projections are Marlin-packed GPTQ layers."""
    try:
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    except Exception:
        return False
    return (
        isinstance(getattr(expert, "gate_proj", None), MarlinLinear)
        and isinstance(getattr(expert, "up_proj", None), MarlinLinear)
        and isinstance(getattr(expert, "down_proj", None), MarlinLinear)
    )


def _probe_marlin_expert(
    self: nn.Module,
    expert: nn.Module,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Probe a Marlin expert to verify gate/up/down forward shapes work."""
    x = torch.randn(2, hidden_dim, device=device, dtype=dtype)
    with torch.inference_mode():
        try:
            gate = expert.gate_proj(x)
            up = expert.up_proj(x)
        except Exception:
            return False
        if (
            not isinstance(gate, torch.Tensor)
            or not isinstance(up, torch.Tensor)
            or gate.shape != (2, intermediate_dim)
            or up.shape != (2, intermediate_dim)
        ):
            return False

        act_fn = getattr(self, "act_fn", F.silu)
        try:
            gated = act_fn(gate) * up
            down = expert.down_proj(gated)
        except Exception:
            return False
        return isinstance(down, torch.Tensor) and down.shape == (2, hidden_dim)


def _apply_expert_gate(
    module: nn.Module,
    gate_out: torch.Tensor | None,
    up_out: torch.Tensor,
) -> torch.Tensor:
    """Apply the expert activation path for gated and non-gated expert MLPs."""
    if gate_out is None:
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            raise AttributeError(f"{module.__class__.__name__} must define `act_fn` for non-gated experts.")
        return act_fn(up_out)

    if hasattr(module, "_apply_gate"):
        return module._apply_gate(torch.cat([gate_out, up_out], dim=-1))

    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        raise AttributeError(f"{module.__class__.__name__} must define either `_apply_gate` or `act_fn`.")
    return act_fn(gate_out) * up_out


def _expert_gateup_is_fused(expert: nn.Module) -> bool:
    """Return True when an expert's gate_proj and up_proj share a fused group."""
    gate_proj = getattr(expert, "gate_proj", None)
    up_proj = getattr(expert, "up_proj", None)
    group = getattr(gate_proj, "_gptqmodel_fused_group", None) if gate_proj is not None else None
    return (
        group is not None
        and group is getattr(up_proj, "_gptqmodel_fused_group", None)
        and getattr(group, "dequantize_weight", None) is not None
    )


def _extract_fused_gateup_dense_weight(
    expert: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the dense (hidden, 2 * intermediate) weight for a fused gate/up expert."""
    gate_proj = getattr(expert, "gate_proj", None)
    up_proj = getattr(expert, "up_proj", None)
    group = getattr(gate_proj, "_gptqmodel_fused_group", None) if gate_proj is not None else None
    if group is None or group is not getattr(up_proj, "_gptqmodel_fused_group", None):
        return None

    dequant_fn = getattr(group, "dequantize_weight", None)
    if dequant_fn is None:
        return None

    try:
        w = dequant_fn(dtype)
    except Exception:
        return None

    hidden_dim = getattr(gate_proj, "in_features", None)
    gate_dim = getattr(gate_proj, "out_features", None)
    up_dim = getattr(up_proj, "out_features", None)
    if None in (hidden_dim, gate_dim, up_dim) or w is None or w.shape != (hidden_dim, gate_dim + up_dim):
        return None
    return w.to(device=device, dtype=dtype).contiguous()


def _can_use_grouped_mm(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor | None = None,
) -> bool:
    """Return True when this experts module can use the fast MoE dispatch path."""
    if getattr(self, "_grouped_mm_ok", False):
        return True
    if getattr(self, "_grouped_mm_failed", False):
        return False

    if hidden_states.device.type != "cuda":
        return False
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return False

    # Avoid the grouped dispatch overhead for very small token counts (typical
    # single-token decode).  Stacking active-expert weights and building the
    # block-aligned index costs more than the per-expert loop at this scale.
    total_pairs = hidden_states.size(0)
    if top_k_index is not None:
        total_pairs *= top_k_index.size(-1)
    if total_pairs < _grouped_mm_min_tokens():
        return False

    expert0 = getattr(self, "0", None)
    if expert0 is None:
        return False
    if not (hasattr(expert0, "gate_proj") and hasattr(expert0, "up_proj") and hasattr(expert0, "down_proj")):
        return False

    hidden_dim = hidden_states.size(-1)
    intermediate_dim = getattr(getattr(expert0, "gate_proj", None), "out_features", None)
    if not isinstance(intermediate_dim, int) or intermediate_dim <= 0:
        return False
    if hidden_dim % 8 != 0 or intermediate_dim % 8 != 0:
        return False

    is_marlin = _is_marlin_expert(expert0)
    is_marlin_fused = is_marlin and _expert_gateup_is_fused(expert0)
    if is_marlin:
        # The per-expert Marlin path must at least be able to run; if it cannot,
        # grouped_mm cannot save us.
        if not _probe_marlin_expert(
            self, expert0, hidden_dim, intermediate_dim, hidden_states.dtype, hidden_states.device
        ):
            self._grouped_mm_failed = True
            return False

    # Prefer the native batched/offset Marlin MoE mega-kernel for Marlin-packed
    # experts.  It removes per-expert launch overhead by grouping tokens by expert
    # and jumping directly to the packed expert weight in a single CUDA kernel.
    if is_marlin and _batched_marlin_moe_supported(
        self, expert0, hidden_dim, intermediate_dim, hidden_states.dtype
    ):
        self._grouped_mm_ok = True
        self._moe_dispatch_backend = "marlin_moe"
        return True

    # For Marlin-packed experts the dense grouped_mm path is only used when the
    # gate and up projections have already been fused.  In that case the fused
    # `dequantize_weight` (which dequantizes the concatenated Marlin-packed weight
    # as one GEMM) matches the packed Marlin kernel output better than per-module
    # dequantization.  Non-fused Marlin experts stay on the per-expert Marlin path.
    if is_marlin and not is_marlin_fused:
        self._grouped_mm_ok = True
        self._moe_dispatch_backend = "marlin"
        return True

    # Whether TritonV2 or fused Marlin, the grouped_mm path only works if we can
    # extract a dense (in, out) weight for every active-expert projection.
    grouped_dense_ok = _grouped_mm_available() and _can_extract_dense_expert(
        expert0, hidden_dim, intermediate_dim, hidden_states.dtype, hidden_states.device
    )

    if is_marlin_fused:
        # The batched/offset kernel is not usable (usually heterogeneous packed
        # shapes in this checkpoint).  Fall back to one Marlin kernel launch per
        # active expert/per projection; this still removes the CPU-side defuser
        # loop overhead and keeps the fast packed Marlin path.
        self._grouped_mm_ok = True
        self._moe_dispatch_backend = "marlin"
        return True

    if not grouped_dense_ok:
        self._grouped_mm_failed = True
        return False

    self._grouped_mm_ok = True
    self._moe_dispatch_backend = "grouped_mm"
    return True


def _project_fused_gateup(
    self: nn.Module,
    active_experts: torch.Tensor,
    intermediate_dim: int,
    x: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the fused gate+up weight for active experts, stack, and grouped_mm."""
    weights: list[torch.Tensor] = []
    for expert_idx in active_experts.tolist():
        expert = getattr(self, str(expert_idx))
        w = _extract_fused_gateup_dense_weight(expert, dtype, device)
        if w is None:
            raise RuntimeError(
                f"Unable to extract fused gate/up weight for expert {expert_idx} in {self.__class__.__name__}"
            )
        weights.append(w)

    stack = torch.stack(weights, dim=0)
    del weights
    gateup_out = torch.nn.functional.grouped_mm(x, stack, offs=offsets)
    del stack
    gate = gateup_out[..., :intermediate_dim]
    up = gateup_out[..., intermediate_dim:]
    return gate, up


def _marlin_experts_project(
    self: nn.Module,
    active_experts: torch.Tensor,
    offsets: torch.Tensor,
    selected_hidden: torch.Tensor,
    intermediate_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Run the per-expert Marlin kernels for gate/up/down over sorted token blocks."""
    down_blocks: list[torch.Tensor] = []
    prev = 0
    for i, expert_idx in enumerate(active_experts.tolist()):
        end = int(offsets[i].item())
        x_block = selected_hidden[prev:end]
        prev = end
        if x_block.size(0) == 0:
            continue

        expert = getattr(self, str(expert_idx))
        # gate_proj and up_proj may be a fused _FusedQuantGroup (one Marlin kernel)
        # or two separate MarlinLinear modules (two kernels). Either way, the
        # forward calls produce the gate and up activations for this block.
        gate_group = getattr(expert.gate_proj, "_gptqmodel_fused_group", None)
        up_group = getattr(expert.up_proj, "_gptqmodel_fused_group", None)
        if gate_group is not None and gate_group is up_group:
            gate_up_out = gate_group._compute(x_block)
            gate_out = gate_up_out[..., gate_group.slices[0][0] : gate_group.slices[0][1]]
            up_out = gate_up_out[..., gate_group.slices[1][0] : gate_group.slices[1][1]]
        else:
            gate_out = expert.gate_proj(x_block)
            up_out = expert.up_proj(x_block)
        gated = _apply_expert_gate(self, gate_out, up_out)
        del gate_out, up_out
        down_block = expert.down_proj(gated)
        down_blocks.append(down_block)

    return torch.cat(down_blocks, dim=0) if down_blocks else torch.empty(
        0, hidden_dim, device=selected_hidden.device, dtype=selected_hidden.dtype
    )


def _marlin_moe_target(proj: nn.Module) -> object:
    """Resolve the packed-weight target for a Marlin-backed projection."""
    target = getattr(proj, "kernel", proj)
    group = getattr(proj, "_gptqmodel_fused_group", None)
    if group is not None and hasattr(group, "kernel"):
        target = group.kernel
    return target


def _batched_marlin_moe_supported(
    self: nn.Module,
    expert0: nn.Module,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
) -> bool:
    """Return True when the Marlin-packed experts can use the batched/offset MoE kernel."""
    if not _is_marlin_expert(expert0):
        return False
    if not _marlin_moe_available():
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if hidden_dim % 8 != 0 or intermediate_dim % 8 != 0:
        return False

    num_experts = getattr(self, "num_experts", None)
    if num_experts is None:
        return False

    for attr in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(expert0, attr, None)
        if proj is None or _has_extra_transform(proj):
            return False
        target = _marlin_moe_target(proj)
        if not all(
            hasattr(target, name)
            for name in (
                "qweight",
                "scales",
                "padded_in_features",
                "padded_out_features",
                "weight_type",
                "is_k_full",
            )
        ):
            return False
        wt = getattr(target, "weight_type", None)
        if wt is None or getattr(wt, "id", None) is None:
            return False

        ref_shape_qw = tuple(target.qweight.shape)
        ref_shape_sc = tuple(target.scales.shape)
        for i in range(1, num_experts):
            other_proj = getattr(getattr(self, str(i)), attr)
            other_target = _marlin_moe_target(other_proj)
            if tuple(other_target.qweight.shape) != ref_shape_qw or tuple(other_target.scales.shape) != ref_shape_sc:
                return False

    gate = expert0.gate_proj
    up = expert0.up_proj

    # Gate and up are launched together (fused or as separate batched kernels).
    # They must share the same bit packing; their group_size must match when
    # fused into a single weight tensor.  Down is a separate launch and is
    # allowed to have a different group_size.
    if gate.bits != up.bits:
        return False

    group_gate = getattr(gate, "_gptqmodel_fused_group", None)
    group_up = getattr(up, "_gptqmodel_fused_group", None)
    if group_gate is not group_up:
        return False

    if group_gate is not None and gate.group_size != up.group_size:
        return False

    return True


def _marlin_packed(obj: nn.Module) -> tuple:
    """Return packed Marlin buffers, either from a _FusedQuantGroup or a MarlinLinear."""
    target = getattr(obj, "kernel", obj)
    return (
        target.qweight,
        target.scales,
        target.weight_type,
        target.padded_in_features,
        target.padded_out_features,
        target.is_k_full,
    )


def _set_stacked_tensor(target: object, name: str, tensor: torch.Tensor) -> None:
    """Assign a view of a stacked tensor, handling nn.Parameter attributes."""
    if isinstance(target, nn.Module) and name in target._parameters:
        setattr(target, name, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(target, name, tensor)


def _prestack_marlin_moe_weights(self: nn.Module, expert0: nn.Module) -> dict[str, tuple]:
    """Pre-stack all expert Marlin qweight/scales once so decode paths avoid per-call copies.

    The per-expert projection objects (either ``MarlinLinear`` or a
    ``_FusedQuantGroup``'s ``_FusedMarlinKernel``) have their ``qweight`` and
    ``scales`` replaced by views into the stacked tensor.  This keeps total
    memory roughly flat while removing the ``active.tolist()``/``torch.stack``
    overhead from every forward.  Returns a dict keyed by ``"gate"``, ``"up"``,
    ``"down"`` or ``"gateup"``.
    """
    num_experts = self.num_experts
    fused_gateup = _expert_gateup_is_fused(expert0)

    proj_specs: list[tuple[str, str]] = []
    if fused_gateup:
        proj_specs.append(("gateup", "gate_proj"))
    else:
        proj_specs.append(("gate", "gate_proj"))
        proj_specs.append(("up", "up_proj"))
    proj_specs.append(("down", "down_proj"))

    stacked: dict[str, tuple] = {}
    for proj_key, attr_name in proj_specs:
        targets: list[object] = []
        for i in range(num_experts):
            expert = getattr(self, str(i))
            obj = getattr(expert, attr_name)
            if hasattr(obj, "_gptqmodel_fused_group"):
                obj = obj._gptqmodel_fused_group
            target = getattr(obj, "kernel", obj)
            targets.append(target)

        q0 = targets[0].qweight
        s0 = targets[0].scales
        device = q0.device
        stacked_qw = torch.empty((num_experts, *q0.shape), dtype=q0.dtype, device=device)
        stacked_sc = torch.empty((num_experts, *s0.shape), dtype=s0.dtype, device=device)

        for i, target in enumerate(targets):
            stacked_qw[i].copy_(target.qweight)
            stacked_sc[i].copy_(target.scales)
            _set_stacked_tensor(target, "qweight", stacked_qw[i])
            _set_stacked_tensor(target, "scales", stacked_sc[i])

        stacked[proj_key] = (
            stacked_qw,
            stacked_sc,
            targets[0].weight_type,
            targets[0].padded_in_features,
            targets[0].padded_out_features,
            targets[0].is_k_full,
        )

    self._marlin_moe_stacked = stacked
    return stacked


def _batched_marlin_moe_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Batched/offset Marlin MoE mega-kernel dispatch.

    Uses pre-stacked per-expert Marlin packed weights and a vectorized
    token-expert alignment so each decode forward only launches one (or two, for
    non-fused gate/up) batched Marlin GEMMs per projection without Python loops
    or per-call ``torch.stack`` copies.
    """
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)

    device = hidden_states.device
    dtype = hidden_states.dtype
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.num_experts

    topk_ids = top_k_index.to(torch.int32)
    topk_w = top_k_weights.to(torch.float32)
    sentinel_mask = topk_ids >= num_experts
    if sentinel_mask.any():
        if topk_w is top_k_weights:
            topk_w = topk_w.clone()
        topk_w = topk_w.masked_fill(sentinel_mask, 0.0)
        topk_ids = topk_ids.clamp(max=num_experts - 1)

    expert0 = getattr(self, "0")
    gate_proj = expert0.gate_proj
    up_proj = expert0.up_proj
    gate_out_features = gate_proj.out_features
    up_out_features = up_proj.out_features

    # Pre-stack expert packed weights once and use global expert IDs in the kernel.
    stacked = getattr(self, "_marlin_moe_stacked", None)
    if stacked is None:
        stacked = _prestack_marlin_moe_weights(self, expert0)

    block_size = _moe_block_size(num_tokens, num_top_k)
    active, sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids, block_size
    )

    workspace = getattr(self, "_marlin_moe_workspace", None)
    if workspace is None or workspace.device != device:
        from gptqmodel.utils.marlin import marlin_make_workspace_new

        workspace = marlin_make_workspace_new(device, max_blocks_per_sm=4)
        self._marlin_moe_workspace = workspace

    topk_weights_flat = topk_w.reshape(-1).contiguous()
    fused_gateup = _expert_gateup_is_fused(expert0)

    # Map local expert_ids to the original (global) expert indices.  The kernel
    # can then index into the pre-stacked [num_experts, ...] weight tensor.
    global_expert_ids = active.long()[expert_ids.long()].to(torch.int32)

    def _maybe_pad(x: torch.Tensor, padded_in: int) -> torch.Tensor:
        if x.size(-1) == padded_in:
            return x
        pad = padded_in - x.size(-1)
        return F.pad(x, (0, pad))

    # Gate / up projection (one fused GEMM when gate/up are fused, two otherwise).
    if fused_gateup:
        fused_qw, fused_sc, weight_type, pin, pout, is_k_full = stacked["gateup"]
        group = getattr(gate_proj, "_gptqmodel_fused_group")
        gate_slice = group.slices[0]
        up_slice = group.slices[1]

        x_padded = _maybe_pad(hidden_states, pin).contiguous()
        fused_out = moe_wna16_marlin_gemm(
            x_padded,
            fused_qw,
            fused_sc,
            workspace,
            sorted_token_ids,
            global_expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            weight_type,
            moe_block_size=block_size,
            top_k=num_top_k,
            size_m=num_tokens,
            size_n=pout,
            size_k=pin,
            is_k_full=is_k_full,
            use_fp32_reduce=True,
            use_atomic_add=False,
            mul_topk_weights=False,
        )
        gate_out = fused_out[:, gate_slice[0] : gate_slice[1]]
        up_out = fused_out[:, up_slice[0] : up_slice[1]]
    else:
        gate_qw, gate_sc, gate_wt, gate_pin, gate_pout, gate_is_k = stacked["gate"]
        up_qw, up_sc, up_wt, up_pin, up_pout, up_is_k = stacked["up"]

        x_padded = _maybe_pad(hidden_states, gate_pin).contiguous()
        gate_out = moe_wna16_marlin_gemm(
            x_padded,
            gate_qw,
            gate_sc,
            workspace,
            sorted_token_ids,
            global_expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            gate_wt,
            moe_block_size=block_size,
            top_k=num_top_k,
            size_m=num_tokens,
            size_n=gate_pout,
            size_k=gate_pin,
            is_k_full=gate_is_k,
            use_fp32_reduce=True,
            use_atomic_add=False,
            mul_topk_weights=False,
        )
        gate_out = gate_out[:, :gate_out_features]

        up_out = moe_wna16_marlin_gemm(
            x_padded,
            up_qw,
            up_sc,
            workspace,
            sorted_token_ids,
            global_expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            up_wt,
            moe_block_size=block_size,
            top_k=num_top_k,
            size_m=num_tokens,
            size_n=up_pout,
            size_k=up_pin,
            is_k_full=up_is_k,
            use_fp32_reduce=True,
            use_atomic_add=False,
            mul_topk_weights=False,
        )
        up_out = up_out[:, :up_out_features]

    act = _apply_expert_gate(self, gate_out, up_out)
    del gate_out, up_out

    # Pad activation to the down projection's padded input dimension.
    down_qw, down_sc, down_wt, down_pin, down_pout, down_is_k = stacked["down"]
    act_padded = _maybe_pad(act, down_pin).contiguous()
    del act

    down_out = moe_wna16_marlin_gemm(
        act_padded,
        down_qw,
        down_sc,
        workspace,
        sorted_token_ids,
        global_expert_ids,
        num_tokens_post_padded,
        topk_weights_flat,
        down_wt,
        moe_block_size=block_size,
        top_k=1,
        size_m=num_tokens * num_top_k,
        size_n=down_pout,
        size_k=down_pin,
        is_k_full=down_is_k,
        use_fp32_reduce=True,
        use_atomic_add=False,
        mul_topk_weights=True,
    )

    final_hidden_states = (
        down_out[:, :hidden_dim]
        .view(num_tokens, num_top_k, hidden_dim)
        .to(torch.float32)
        .sum(dim=1)
        .to(dtype)
    )

    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states


def _grouped_mm_dequant_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Forward using one grouped GEMM per projection over the active experts.

    This path dequantizes only the active experts, sorts tokens by expert,
    and launches three grouped GEMM kernels (gate, up, down) instead of
    looping over experts.  It falls back to the per-expert loop if any
    projection cannot provide a dense weight.
    """
    if getattr(self, "_moe_dispatch_backend", None) == "marlin_moe":
        return _batched_marlin_moe_forward(self, hidden_states, top_k_index, top_k_weights)

    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.num_experts

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1)

    sentinel_mask = expert_ids >= num_experts
    if sentinel_mask.any():
        expert_ids = expert_ids.clamp(max=num_experts - 1)

    if expert_ids.numel() == 0:
        final_hidden_states = torch.zeros(num_tokens, hidden_dim, device=device, dtype=hidden_states.dtype)
        if batch_size is not None:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states

    active_experts = torch.unique(expert_ids, sorted=True)
    local_ids = torch.searchsorted(active_experts, expert_ids)
    local_sorted, perm = torch.sort(local_ids)
    token_idx_sorted = token_idx[perm]
    sample_weights_sorted = sample_weights[perm]
    selected_hidden = hidden_states[token_idx_sorted]

    e_active = active_experts.numel()
    counts_local = torch.bincount(local_sorted, minlength=e_active)
    offsets = torch.cumsum(counts_local, dim=0, dtype=torch.int64).to(torch.int32)

    expert0 = getattr(self, "0")
    intermediate_dim = expert0.gate_proj.out_features

    # Defer the grouped_mm lookup until the grouped-mm branch is actually used.
    # The marlin and marlin_moe branches do not need it, so an environment without
    # torch.nn.functional.grouped_mm should still be able to take those paths.
    grouped_mm = getattr(torch.nn.functional, "grouped_mm", None)

    def _project(expert_attr: str, in_f: int, out_f: int, x: torch.Tensor) -> torch.Tensor:
        """Dequantize one projection for the active experts, stack, and grouped_mm."""
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor | None] = []
        for expert_idx in active_experts.tolist():
            expert = getattr(self, str(expert_idx))
            proj = getattr(expert, expert_attr)
            w = _extract_expert_dense_weight(proj, in_f, out_f, hidden_states.dtype, device)
            if w is None:
                raise RuntimeError(
                    f"Unable to extract dense weight for {expert_attr} of expert {expert_idx} in {self.__class__.__name__}"
                )
            weights.append(w)
            biases.append(_extract_expert_bias(proj, out_f, hidden_states.dtype, device))

        stack = torch.stack(weights, dim=0)
        del weights
        out = grouped_mm(x, stack, offs=offsets)
        del stack

        if any(b is not None for b in biases):
            bias_stack = torch.stack([
                b if b is not None else torch.zeros(out_f, device=device, dtype=hidden_states.dtype)
                for b in biases
            ], dim=0)
            out = out + bias_stack[local_sorted]
            del bias_stack
        return out

    if getattr(self, "_moe_dispatch_backend", None) == "marlin":
        down_out = _marlin_experts_project(
            self, active_experts, offsets, selected_hidden, intermediate_dim, hidden_dim
        )
    else:
        if _expert_gateup_is_fused(expert0):
            gate_out, up_out = _project_fused_gateup(
                self, active_experts, intermediate_dim, selected_hidden, hidden_states.dtype, device, offsets
            )
        else:
            gate_out = _project("gate_proj", hidden_dim, intermediate_dim, selected_hidden)
            up_out = _project("up_proj", hidden_dim, intermediate_dim, selected_hidden)
        gated = _apply_expert_gate(self, gate_out, up_out)
        del gate_out, up_out

        down_out = _project("down_proj", intermediate_dim, hidden_dim, gated)
        del gated

    weighted = down_out * sample_weights_sorted.unsqueeze(-1)
    del down_out
    if sentinel_mask.any():
        sentinel_sorted = sentinel_mask[perm]
        weighted = weighted.masked_fill(sentinel_sorted.unsqueeze(-1), 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    out_per_sample = weighted[inv_perm]
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states.to(hidden_states.dtype)


def linear_loop_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Grouped GEMM fast path for `linear_loop` experts, with per-expert fallback.

    This is the function GPTQModel registers in `ALL_EXPERTS_FUNCTIONS` under
    the ``linear_loop`` key.  The grouped path is only taken for modules that have
    explicitly opted in via ``_gptqmodel_grouped_dispatch_enabled``; this keeps
    quantization/calibration on the defused per-expert loop.
    """
    if getattr(self, GROUPED_DISPATCH_FLAG, False) and _can_use_grouped_mm(self, hidden_states, top_k_index):
        try:
            return _grouped_mm_dequant_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        except Exception as exc:  # noqa: BLE001
            if DEBUG_ON:
                log.debug(
                    "grouped_mm experts forward failed for %s: %s; falling back to per-expert loop",
                    self.__class__.__name__,
                    exc,
                )
            # Latch the fallback so we do not repeat the expensive setup on every call.
            self._grouped_mm_ok = False
            self._grouped_mm_failed = True

    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as _fallback
    return _fallback(self, hidden_states, top_k_index, top_k_weights)


def enable_grouped_dispatch_for_model(model: nn.Module) -> int:
    """Enable grouped GEMM dispatch for every experts module in ``model``.

    Returns the number of modules flagged.  The flag is per-module, so loading a
    new model in the same process will not inherit the fast path.
    """
    flagged = 0
    for module in model.modules():
        if getattr(module, "num_experts", 0) > 0 and "0" in module._modules:
            setattr(module, GROUPED_DISPATCH_FLAG, True)
            flagged += 1
    return flagged


def register_linear_loop_experts() -> bool:
    """Register GPTQModel's grouped `linear_loop` experts implementation.

    This should be called after Defuser has done its unfusing work and only
    for post-quantization inference (not during per-module quantization).
    """
    if not HAS_EXPERTS_INTERFACE or ALL_EXPERTS_FUNCTIONS is None:
        return False

    if LINEAR_LOOP_IMPL not in ALL_EXPERTS_FUNCTIONS._global_mapping:
        log.warn("Defuser has not registered `linear_loop` experts implementation; cannot override.")
        return False

    ALL_EXPERTS_FUNCTIONS._global_mapping[LINEAR_LOOP_IMPL] = linear_loop_experts_forward
    log.info("Registered GPTQModel grouped GEMM `linear_loop` experts dispatch.")
    return True
