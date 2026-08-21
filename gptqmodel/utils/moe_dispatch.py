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

import os
import weakref

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

# Lazy import of the Triton fused silu*mul kernel; if Triton is unavailable or
# the kernel fails to load we fall back to the eager act_fn(gate) * up path.
_FUSED_SILU_MUL: object | None = None


def _load_fused_silu_mul() -> bool:
    """Load the fused SiLU*mul Triton kernel once; return True on success."""
    global _FUSED_SILU_MUL
    if _FUSED_SILU_MUL is None:
        try:
            from gptqmodel.nn_modules.triton_utils.kernels import fused_silu_mul

            _FUSED_SILU_MUL = fused_silu_mul
        except Exception:
            _FUSED_SILU_MUL = False
    return _FUSED_SILU_MUL is not False


def _is_silu_activation(act_fn) -> bool:
    """Return True when ``act_fn`` is a SiLU implementation."""
    if act_fn is None:
        return False
    if isinstance(act_fn, nn.Module):
        name = type(act_fn).__name__
    else:
        name = getattr(act_fn, "__name__", "")
    return name in ("SiLU", "SiLUActivation", "silu")


def _resolve_apply_gate(module: nn.Module):
    """Return a callable that applies the expert activation (gate * up) without hot-path getattr.

    Prefer the fused SiLU*mul Triton kernel for standard SwiGLU/SiLU gating; fall back
    to the module's own ``_apply_gate`` or the eager ``act_fn(gate) * up`` path when
    the activation is not SiLU or the kernel cannot be loaded.
    """
    act_fn = getattr(module, "act_fn", None)
    if _is_silu_activation(act_fn) and _load_fused_silu_mul():
        # Allocate a new contiguous output; Marlin GEMMs that follow can skip
        # a redundant contiguous() copy of the strided gate slice.
        return lambda gate, up: _FUSED_SILU_MUL(gate, up)

    try:
        apply_gate = module._apply_gate
        return lambda gate, up: apply_gate(torch.cat([gate, up], dim=-1))
    except AttributeError:
        pass

    if act_fn is None:
        raise AttributeError(f"{module.__class__.__name__} must define either `_apply_gate` or `act_fn`.")
    return lambda gate, up: act_fn(gate) * up

LINEAR_LOOP_IMPL = "linear_loop"

# Weak-key cache for per-expert dispatch metadata. Storing it on the module can
# interfere with torch.compile / serialization and is automatically cleaned up
# when the experts module is garbage-collected.
_EXPERT_DISPATCH_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

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


def _sanitize_expert_ids(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp routing ids for tensor indexing and identify entries that must contribute zero."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    sentinel_mask = (expert_ids < 0) | (expert_ids >= num_experts)
    return expert_ids.clamp(min=0, max=num_experts - 1), sentinel_mask


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

    Fully vectorized: no per-expert ``nonzero``, ``tolist`` or ``item`` calls, so the
    hot decode path does not issue device-to-host synchronizations.
    """
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_indices = torch.argsort(flat, stable=True)
    sorted_flat = flat[sorted_indices]
    active, counts = torch.unique_consecutive(sorted_flat, return_counts=True)

    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    num_active = active.numel()

    # Each active expert can contribute up to (block_size - 1) padding slots, so an
    # upper-bound allocation avoids a device-to-host size query.
    max_total_padded = numel + num_active * (block_size - 1)
    max_blocks = max_total_padded // block_size

    starts = counts.cumsum(0) - counts
    ends = starts + counts
    out_starts = padded_counts.cumsum(0) - padded_counts

    device = flat.device
    arange = torch.arange(numel, device=device, dtype=torch.int64)
    expert_of = torch.searchsorted(ends, arange, right=True)
    pos = out_starts[expert_of] + (arange - starts[expert_of])

    sorted_token_ids = torch.full(
        (max_total_padded,), numel, dtype=torch.int64, device=device
    )
    sorted_token_ids.scatter_(0, pos, sorted_indices)
    sorted_token_ids = sorted_token_ids.to(torch.int32)

    blocks_per_expert = padded_counts // block_size
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)
    if num_active > 0:
        repeated = torch.repeat_interleave(
            torch.arange(num_active, device=device, dtype=torch.int32),
            blocks_per_expert.to(torch.int64),
        )
        expert_ids[: repeated.numel()] = repeated

    num_tokens_post_padded = padded_counts.sum().to(torch.int32)
    return active, sorted_token_ids, expert_ids, num_tokens_post_padded


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
    # Planar (gptq_p) 3-bit words are incompatible with the continuous-layout
    # Triton decoder; planar 2/4/8 words are bit-identical to continuous.
    if bits == 3 and getattr(proj, "planar", False):
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
    if is_marlin:
        # The per-expert Marlin path must at least be able to run; if it cannot,
        # grouped_mm cannot save us.
        if not _probe_marlin_expert(
            self, expert0, hidden_dim, intermediate_dim, hidden_states.dtype, hidden_states.device
        ):
            self._grouped_mm_failed = True
            return False

    # For Marlin-packed experts the per-expert packed Marlin active loop is the
    # fastest known dispatch for the Laguna-S-2.1 W4G64 shape class after
    # removing D2H scalar syncs: it uses the optimized ``gptq_marlin_gemm`` kernel
    # for each active expert block and has lower launch overhead than the
    # batched/offset mega-kernel for small-to-moderate batch sizes.  Keep the
    # mega-kernel and dense grouped GEMM paths available behind explicit opt-ins.
    if is_marlin:
        backend = "marlin"
        marlin_moe_backend = os.environ.get("GPTQMODEL_MARLIN_MOE_BACKEND", "per_expert")
        supported = marlin_moe_backend == "marlin_moe" and _batched_marlin_moe_supported(
            self, expert0, hidden_dim, intermediate_dim, hidden_states.dtype
        )
        if supported:
            backend = "marlin_moe"
        elif marlin_moe_backend == "grouped_mm" and _can_extract_dense_expert(
            expert0, hidden_dim, intermediate_dim, hidden_states.dtype, hidden_states.device
        ):
            backend = "grouped_mm"
        self._grouped_mm_ok = True
        self._moe_dispatch_backend = backend
        return True

    grouped_dense_ok = _grouped_mm_available() and _can_extract_dense_expert(
        expert0, hidden_dim, intermediate_dim, hidden_states.dtype, hidden_states.device
    )

    if grouped_dense_ok:
        self._grouped_mm_ok = True
        self._moe_dispatch_backend = "grouped_mm"
        return True

    self._grouped_mm_failed = True
    return False


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
    cache = _get_expert_dispatch_cache(self)
    gateup_groups = cache["gateup_groups"]
    weights: list[torch.Tensor] = []
    for expert_idx in active_experts.tolist():
        fused, gate_group, _, _ = gateup_groups[expert_idx]
        if not fused or gate_group is None:
            raise RuntimeError(
                f"Unable to extract fused gate/up weight for expert {expert_idx} in {self.__class__.__name__}"
            )
        w = gate_group.dequantize_weight(dtype)
        if w is None or w.shape != (gate_group.in_features, gate_group.total_out_features):
            raise RuntimeError(
                f"Unable to extract fused gate/up weight for expert {expert_idx} in {self.__class__.__name__}"
            )
        weights.append(w.to(device=device, dtype=dtype).contiguous())

    stack = torch.stack(weights, dim=0)
    del weights
    gateup_out = torch.nn.functional.grouped_mm(x, stack, offs=offsets)
    del stack
    gate = gateup_out[..., :intermediate_dim]
    up = gateup_out[..., intermediate_dim:]
    return gate, up


def _get_expert_dispatch_cache(self: nn.Module) -> dict:
    """Return precomputed per-expert projection handles for `self`.

    The cache is built once per experts module and contains direct module
    references for every expert, eliminating repeated `getattr(self, str(idx))`
    and fused-group introspection in the hot forward path.
    """
    cache = _EXPERT_DISPATCH_CACHE.get(self)
    if cache is not None:
        return cache

    num_experts = int(self.num_experts)
    modules = self._modules
    experts = [modules[str(i)] for i in range(num_experts)]
    gate_projs = []
    up_projs = []
    down_projs = []
    gateup_groups = []
    for expert in experts:
        gate_proj = expert._modules["gate_proj"]
        up_proj = expert._modules["up_proj"]
        gate_group = getattr(gate_proj, "_gptqmodel_fused_group", None)
        up_group = getattr(up_proj, "_gptqmodel_fused_group", None)
        fused = gate_group is not None and gate_group is up_group
        gateup_groups.append((fused, gate_group if fused else None, gate_proj, up_proj))
        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        down_projs.append(expert._modules["down_proj"])

    cache = {
        "experts": experts,
        "gate_projs": gate_projs,
        "up_projs": up_projs,
        "down_projs": down_projs,
        "proj_by_attr": {
            "gate_proj": gate_projs,
            "up_proj": up_projs,
            "down_proj": down_projs,
        },
        "gateup_groups": gateup_groups,
        "apply_gate": _resolve_apply_gate(self),
    }
    _EXPERT_DISPATCH_CACHE[self] = cache
    return cache


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
    cache = _get_expert_dispatch_cache(self)
    gateup_groups = cache["gateup_groups"]
    down_projs = cache["down_projs"]
    apply_gate = cache["apply_gate"]

    # Pull the small routing metadata to the host once instead of issuing a
    # device->host scalar transfer inside the per-expert loop.
    active_experts_list = active_experts.tolist()
    offsets_list = offsets.tolist()
    prev = 0
    for i, expert_idx in enumerate(active_experts_list):
        end = int(offsets_list[i])
        x_block = selected_hidden[prev:end]
        prev = end
        if x_block.size(0) == 0:
            continue

        fused, gate_group, gate_proj, up_proj = gateup_groups[expert_idx]
        if fused:
            gate_up_out = gate_group._compute(x_block)
            gate_out = gate_up_out[..., gate_group.slices[0][0] : gate_group.slices[0][1]]
            up_out = gate_up_out[..., gate_group.slices[1][0] : gate_group.slices[1][1]]
        else:
            gate_out = gate_proj(x_block)
            up_out = up_proj(x_block)
        gated = apply_gate(gate_out, up_out)
        del gate_out, up_out
        down_blocks.append(down_projs[expert_idx](gated))

    return torch.cat(down_blocks, dim=0) if down_blocks else torch.empty(
        0, hidden_dim, device=selected_hidden.device, dtype=selected_hidden.dtype
    )


@torch._dynamo.disable
def _marlin_experts_project_one_token(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """M=1 decode fast path for Marlin-packed MoE experts.

    Avoids sorting, token-expert gather/scatter, and padding for a single token
    with ``top_k`` routed experts. Each active expert runs its fused gate/up
    and down Marlin kernels once; contributions are weighted-summed in FP32.
    """
    cache = _get_expert_dispatch_cache(self)
    gateup_groups = cache["gateup_groups"]
    down_projs = cache["down_projs"]
    apply_gate = cache["apply_gate"]

    hidden_dim = hidden_states.size(-1)
    device = hidden_states.device
    dtype = hidden_states.dtype
    num_experts = self.num_experts

    # top_k_index has shape (1, top_k) after the batched reshape.
    expert_ids, sentinel_mask = _sanitize_expert_ids(top_k_index[0], num_experts)
    weights = top_k_weights[0].to(dtype).masked_fill(sentinel_mask, 0.0)

    # One small device->host transfer per MoE layer instead of per pair.
    expert_ids_list = expert_ids.tolist()
    weights_list = weights.tolist()

    # Accumulate weight per expert so duplicated top-k ids only compute once.
    expert_to_weight: dict[int, float] = {}
    for e, w in zip(expert_ids_list, weights_list):
        if w == 0.0:
            continue
        expert_to_weight[e] = expert_to_weight.get(e, 0.0) + float(w)

    # Flatten to 2-D and force one contiguous copy up front so every Marlin call
    # downstream sees a row-major (M, K) input and can skip its own contiguous() copy.
    x = hidden_states.reshape(-1, hidden_dim)
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.zeros(1, hidden_dim, device=device, dtype=torch.float32)
    for expert_idx, weight in expert_to_weight.items():
        fused, gate_group, gate_proj, up_proj = gateup_groups[expert_idx]
        if fused:
            gate_up_out = gate_group._compute(x)
            gate_out = gate_up_out[..., gate_group.slices[0][0] : gate_group.slices[0][1]]
            up_out = gate_up_out[..., gate_group.slices[1][0] : gate_group.slices[1][1]]
        else:
            gate_out = gate_proj(x)
            up_out = up_proj(x)
        gated = apply_gate(gate_out, up_out)
        del gate_out, up_out
        down = down_projs[expert_idx](gated)
        out.add_(down.to(torch.float32), alpha=weight)

    return out.to(dtype)


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
    """Return True when the Marlin-packed experts can use the batched/offset MoE kernel.

    Experts may have different packed ``scales`` shapes (different ``group_size``)
    as long as their ``qweight`` shapes and runtime dimensions are uniform; such
    experts are clustered and launched as separate ``moe_wna16_marlin_gemm`` calls.
    """
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
        ref = (target.padded_in_features, target.padded_out_features, target.is_k_full, target.weight_type)
        for i in range(1, num_experts):
            other_proj = getattr(getattr(self, str(i)), attr)
            other_target = _marlin_moe_target(other_proj)
            other_ref = (other_target.padded_in_features, other_target.padded_out_features, other_target.is_k_full, other_target.weight_type)
            if tuple(other_target.qweight.shape) != ref_shape_qw or other_ref != ref:
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


def _marlin_moe_target(obj: object) -> object:
    """Return the concrete Marlin packed-weight holder for a projection object."""
    if hasattr(obj, "_gptqmodel_fused_group"):
        obj = obj._gptqmodel_fused_group
    return getattr(obj, "kernel", obj)


def _active_cluster_key(target: object) -> tuple:
    """Shape/runtime key used to group active experts into homogeneous clusters."""
    return (
        tuple(target.qweight.shape),
        tuple(target.scales.shape),
        target.padded_in_features,
        target.padded_out_features,
        target.is_k_full,
        getattr(target.weight_type, "id", target.weight_type),
    )


def _build_active_clusters_for_proj(
    active_proj_targets: list[object],
    active_global_ids: list[int],
    num_experts: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """Build one or more homogeneous Marlin MoE clusters from active experts.

    Only the experts that are actually routed for this forward are copied into a
    contiguous ``[num_active, ...]`` tensor.  This keeps the per-forward extra
    allocation small (``top_k * weight_size`` for one projection) so the
    batched/offset mega-kernel works on large checkpoints without a full prestack.
    """
    if not active_global_ids:
        return []
    if len(active_proj_targets) != len(active_global_ids):
        raise ValueError(
            "active_proj_targets and active_global_ids must contain the same number of experts"
        )

    # Callers materialize only the routed experts, so this list is compact and
    # cannot be indexed by a sparse global expert id (for example [3, 7]).
    # Preserve the upstream-known identity explicitly for clustering and copies.
    target_by_global_id = dict(zip(active_global_ids, active_proj_targets, strict=True))
    if len(target_by_global_id) != len(active_global_ids):
        raise ValueError("active_global_ids must be unique")

    cluster_map: dict[tuple, list[int]] = {}
    for global_id in active_global_ids:
        target = target_by_global_id[global_id]
        cluster_map.setdefault(_active_cluster_key(target), []).append(global_id)

    clusters: list[dict[str, object]] = []
    for global_ids in cluster_map.values():
        first = target_by_global_id[global_ids[0]]
        num = len(global_ids)

        stacked_qw = torch.empty(
            (num, *first.qweight.shape), dtype=first.qweight.dtype, device=device
        )
        stacked_sc = torch.empty(
            (num, *first.scales.shape), dtype=first.scales.dtype, device=device
        )

        # membership[global_id] = local cluster id; non-members map to 0 (a valid
        # local id used only for padding rows by moe_wna16_marlin_gemm).
        membership = torch.zeros(num_experts, dtype=torch.int64, device=device)
        in_cluster = torch.zeros(num_experts, dtype=torch.bool, device=device)
        cluster_global_ids = torch.empty(num, dtype=torch.int64, device=device)
        for local_id, global_id in enumerate(global_ids):
            target = target_by_global_id[global_id]
            stacked_qw[local_id].copy_(target.qweight)
            stacked_sc[local_id].copy_(target.scales)
            membership[global_id] = local_id
            in_cluster[global_id] = True
            cluster_global_ids[local_id] = global_id

        clusters.append({
            "qweight": stacked_qw,
            "scales": stacked_sc,
            "weight_type": first.weight_type,
            "padded_in_features": first.padded_in_features,
            "padded_out_features": first.padded_out_features,
            "is_k_full": first.is_k_full,
            "membership": membership,
            "in_cluster": in_cluster,
            "global_ids": cluster_global_ids,
        })

    return clusters


def _moe_scatter_cluster_output(
    cluster_out: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    in_cluster: torch.Tensor,
    topk_ids: torch.Tensor,
    global_out: torch.Tensor,
) -> None:
    """Scatter a cluster's kernel output back into the global pair-ordered buffer.

    Only rows whose original token-expert pair belongs to this cluster are
    written.  ``in_cluster`` is a ``[num_experts]`` bool tensor and ``topk_ids``
    is the global (batch, top_k) expert ids used to build the membership mask.
    """
    num_pairs = topk_ids.numel()
    valid = sorted_token_ids < num_pairs
    valid_global_ids = sorted_token_ids[valid]
    pair_in_cluster = in_cluster[topk_ids.flatten()]
    member_rows = pair_in_cluster[valid_global_ids]
    member_ids = valid_global_ids[member_rows]
    if member_ids.numel() > 0:
        global_out[member_ids] = cluster_out[member_ids]


def _build_single_cluster_stack(
    proj_targets: list[object],
    active_global_ids: list[int],
    device: torch.device,
) -> dict[str, object] | None:
    """Build one homogeneous cluster from active experts with a single ``torch.stack``.

    Returns ``None`` if the active experts do not share the same packed shape,
    in which case the caller should fall back to per-shape clustering.
    """
    first = proj_targets[0]
    try:
        stacked_qw = torch.stack([t.qweight for t in proj_targets], dim=0).to(device)
        stacked_sc = torch.stack([t.scales for t in proj_targets], dim=0).to(device)
    except RuntimeError:
        return None
    return {
        "qweight": stacked_qw,
        "scales": stacked_sc,
        "weight_type": first.weight_type,
        "padded_in_features": first.padded_in_features,
        "padded_out_features": first.padded_out_features,
        "is_k_full": first.is_k_full,
        "global_ids": active_global_ids,
    }


def _batched_marlin_moe_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Batched/offset Marlin MoE mega-kernel dispatch with shape clustering."""
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

    if num_tokens == 0 or num_top_k == 0:
        final_hidden_states = torch.zeros(
            num_tokens,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        if batch_size is not None:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states

    topk_ids = top_k_index.to(torch.int32)
    topk_w = top_k_weights.to(torch.float32)
    topk_ids, sentinel_mask = _sanitize_expert_ids(topk_ids, num_experts)
    topk_w = topk_w.masked_fill(sentinel_mask, 0.0)

    cache = _get_expert_dispatch_cache(self)
    expert0 = cache["experts"][0]
    gate_proj = cache["gate_projs"][0]
    up_proj = cache["up_projs"][0]
    gate_out_features = gate_proj.out_features
    up_out_features = up_proj.out_features
    fused_gateup = _expert_gateup_is_fused(expert0)

    active_global_ids_t = torch.unique(topk_ids, sorted=True)
    active_global_ids = active_global_ids_t.tolist()

    # Try to build one stacked cluster per projection; a single stacked copy is
    # much cheaper than the per-index copy loop used by the generic builder.
    stacked: dict[str, list[dict[str, object]]] = {}
    if fused_gateup:
        gateup_targets = [
            _marlin_moe_target(
                group if (fused and group is not None) else gate_p
            )
            for i in active_global_ids
            for fused, group, gate_p, up_p in (cache["gateup_groups"][i],)
        ]
        gateup_cluster = _build_single_cluster_stack(
            gateup_targets, active_global_ids, device
        )
        if gateup_cluster is None:
            stacked["gateup"] = _build_active_clusters_for_proj(
                gateup_targets, active_global_ids, num_experts, device
            )
        else:
            stacked["gateup"] = [gateup_cluster]
    else:
        gate_targets = [
            _marlin_moe_target(cache["gate_projs"][i]) for i in active_global_ids
        ]
        up_targets = [
            _marlin_moe_target(cache["up_projs"][i]) for i in active_global_ids
        ]
        gate_cluster = _build_single_cluster_stack(
            gate_targets, active_global_ids, device
        )
        up_cluster = _build_single_cluster_stack(
            up_targets, active_global_ids, device
        )
        if gate_cluster is None or up_cluster is None:
            stacked["gate"] = _build_active_clusters_for_proj(
                gate_targets, active_global_ids, num_experts, device
            )
            stacked["up"] = _build_active_clusters_for_proj(
                up_targets, active_global_ids, num_experts, device
            )
        else:
            stacked["gate"] = [gate_cluster]
            stacked["up"] = [up_cluster]

    down_targets = [
        _marlin_moe_target(cache["down_projs"][i]) for i in active_global_ids
    ]
    down_cluster = _build_single_cluster_stack(
        down_targets, active_global_ids, device
    )
    if down_cluster is None:
        stacked["down"] = _build_active_clusters_for_proj(
            down_targets, active_global_ids, num_experts, device
        )
    else:
        stacked["down"] = [down_cluster]

    workspace = getattr(self, "_marlin_moe_workspace", None)
    if workspace is None or workspace.device != device:
        from gptqmodel.utils.marlin import marlin_make_workspace_new

        workspace = marlin_make_workspace_new(device, max_blocks_per_sm=4)
        self._marlin_moe_workspace = workspace

    block_size = _moe_block_size(num_tokens, num_top_k)
    topk_weights_flat = topk_w.reshape(-1).contiguous()

    def _maybe_pad(x: torch.Tensor, padded_in: int) -> torch.Tensor:
        if x.size(-1) == padded_in:
            return x
        pad = padded_in - x.size(-1)
        return F.pad(x, (0, pad))

    def _cluster_marlin_gemm(
        x: torch.Tensor,
        cluster: dict[str, object],
        out_features: int,
        top_k: int,
        mul_topk_weights: bool,
        size_m: int,
    ) -> torch.Tensor:
        membership = cluster["membership"]
        in_cluster = cluster["in_cluster"]
        local_topk_ids = membership[topk_ids.long()].to(torch.int32)

        active, sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            local_topk_ids, block_size
        )

        x_padded = _maybe_pad(x, cluster["padded_in_features"]).contiguous()
        out = moe_wna16_marlin_gemm(
            x_padded,
            cluster["qweight"],
            cluster["scales"],
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            cluster["weight_type"],
            moe_block_size=block_size,
            top_k=top_k,
            size_m=size_m,
            size_n=cluster["padded_out_features"],
            size_k=cluster["padded_in_features"],
            is_k_full=cluster["is_k_full"],
            use_fp32_reduce=True,
            use_atomic_add=False,
            mul_topk_weights=mul_topk_weights,
        )
        if out.size(-1) > out_features:
            out = out[..., :out_features]
        return out, sorted_token_ids, in_cluster

    # Fast path: exactly one shape cluster per projection.  The same MoE
    # routing applies to gate/up/down, so alignment metadata is computed once.
    single_cluster = all(len(v) == 1 for v in stacked.values())
    if single_cluster:
        search_ids = active_global_ids_t.to(torch.int64)
        flat_topk = topk_ids.flatten().to(torch.int64)
        local_topk_ids = torch.searchsorted(search_ids, flat_topk).to(torch.int32)
        _, sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            local_topk_ids, block_size
        )

        if fused_gateup:
            group = getattr(gate_proj, "_gptqmodel_fused_group")
            gate_slice = group.slices[0]
            up_slice = group.slices[1]
            gateup_cluster = stacked["gateup"][0]
            gate_up_out = torch.empty(
                (num_tokens * num_top_k, gateup_cluster["padded_out_features"]),
                dtype=dtype,
                device=device,
            )
            moe_wna16_marlin_gemm(
                _maybe_pad(hidden_states, gateup_cluster["padded_in_features"]).contiguous(),
                gateup_cluster["qweight"],
                gateup_cluster["scales"],
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights_flat,
                gateup_cluster["weight_type"],
                moe_block_size=block_size,
                top_k=num_top_k,
                size_m=num_tokens,
                size_n=gateup_cluster["padded_out_features"],
                size_k=gateup_cluster["padded_in_features"],
                is_k_full=gateup_cluster["is_k_full"],
                mul_topk_weights=False,
                c=gate_up_out,
            )
            gate_out = gate_up_out[..., gate_slice[0] : gate_slice[1]]
            up_out = gate_up_out[..., up_slice[0] : up_slice[1]]
        else:
            gate_cluster = stacked["gate"][0]
            gate_out_buf = torch.empty(
                (num_tokens * num_top_k, gate_cluster["padded_out_features"]),
                dtype=dtype,
                device=device,
            )
            moe_wna16_marlin_gemm(
                _maybe_pad(hidden_states, gate_cluster["padded_in_features"]).contiguous(),
                gate_cluster["qweight"],
                gate_cluster["scales"],
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights_flat,
                gate_cluster["weight_type"],
                moe_block_size=block_size,
                top_k=num_top_k,
                size_m=num_tokens,
                size_n=gate_cluster["padded_out_features"],
                size_k=gate_cluster["padded_in_features"],
                is_k_full=gate_cluster["is_k_full"],
                mul_topk_weights=False,
                c=gate_out_buf,
            )
            gate_out = gate_out_buf[..., :gate_out_features]

            up_cluster = stacked["up"][0]
            up_out_buf = torch.empty(
                (num_tokens * num_top_k, up_cluster["padded_out_features"]),
                dtype=dtype,
                device=device,
            )
            moe_wna16_marlin_gemm(
                _maybe_pad(hidden_states, up_cluster["padded_in_features"]).contiguous(),
                up_cluster["qweight"],
                up_cluster["scales"],
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights_flat,
                up_cluster["weight_type"],
                moe_block_size=block_size,
                top_k=num_top_k,
                size_m=num_tokens,
                size_n=up_cluster["padded_out_features"],
                size_k=up_cluster["padded_in_features"],
                is_k_full=up_cluster["is_k_full"],
                mul_topk_weights=False,
                c=up_out_buf,
            )
            up_out = up_out_buf[..., :up_out_features]

        apply_gate = _resolve_apply_gate(self)
        act = apply_gate(gate_out, up_out)
        del gate_out, up_out

        down_cluster = stacked["down"][0]
        down_out_buf = torch.empty(
            (num_tokens * num_top_k, down_cluster["padded_out_features"]),
            dtype=dtype,
            device=device,
        )
        moe_wna16_marlin_gemm(
            _maybe_pad(act, down_cluster["padded_in_features"]).contiguous(),
            down_cluster["qweight"],
            down_cluster["scales"],
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            down_cluster["weight_type"],
            moe_block_size=block_size,
            top_k=1,
            size_m=num_tokens * num_top_k,
            size_n=down_cluster["padded_out_features"],
            size_k=down_cluster["padded_in_features"],
            is_k_full=down_cluster["is_k_full"],
            mul_topk_weights=True,
            c=down_out_buf,
        )
        down_out = down_out_buf[..., :hidden_dim]
        final_hidden_states = (
            down_out.to(torch.float32)
            .view(num_tokens, num_top_k, hidden_dim)
            .sum(dim=1)
            .to(dtype)
        )

        if batch_size is not None:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states

    # Multi-cluster fallback: process each homogeneous shape cluster separately.
    if fused_gateup:
        group = getattr(gate_proj, "_gptqmodel_fused_group")
        gate_slice = group.slices[0]
        up_slice = group.slices[1]
        total_out_features = gate_out_features + up_out_features
        gate_up_out = torch.zeros(
            (num_tokens * num_top_k, total_out_features), dtype=dtype, device=device
        )
        for cluster in stacked["gateup"]:
            fused_out, sorted_token_ids, in_cluster = _cluster_marlin_gemm(
                hidden_states,
                cluster,
                total_out_features,
                top_k=num_top_k,
                mul_topk_weights=False,
                size_m=num_tokens,
            )
            _moe_scatter_cluster_output(
                fused_out, sorted_token_ids, in_cluster, topk_ids, gate_up_out
            )
        gate_out = gate_up_out[:, gate_slice[0] : gate_slice[1]]
        up_out = gate_up_out[:, up_slice[0] : up_slice[1]]
    else:
        gate_out = torch.zeros(
            (num_tokens * num_top_k, gate_out_features), dtype=dtype, device=device
        )
        for cluster in stacked["gate"]:
            c_out, sorted_token_ids, in_cluster = _cluster_marlin_gemm(
                hidden_states,
                cluster,
                gate_out_features,
                top_k=num_top_k,
                mul_topk_weights=False,
                size_m=num_tokens,
            )
            _moe_scatter_cluster_output(
                c_out, sorted_token_ids, in_cluster, topk_ids, gate_out
            )

        up_out = torch.zeros(
            (num_tokens * num_top_k, up_out_features), dtype=dtype, device=device
        )
        for cluster in stacked["up"]:
            c_out, sorted_token_ids, in_cluster = _cluster_marlin_gemm(
                hidden_states,
                cluster,
                up_out_features,
                top_k=num_top_k,
                mul_topk_weights=False,
                size_m=num_tokens,
            )
            _moe_scatter_cluster_output(
                c_out, sorted_token_ids, in_cluster, topk_ids, up_out
            )

    apply_gate = _resolve_apply_gate(self)
    act = apply_gate(gate_out, up_out)
    del gate_out, up_out

    down_out = torch.zeros(
        (num_tokens * num_top_k, hidden_dim), dtype=torch.float32, device=device
    )
    for cluster in stacked["down"]:
        c_out, sorted_token_ids, in_cluster = _cluster_marlin_gemm(
            act,
            cluster,
            hidden_dim,
            top_k=1,
            mul_topk_weights=True,
            size_m=num_tokens * num_top_k,
        )
        _moe_scatter_cluster_output(
            c_out, sorted_token_ids, in_cluster, topk_ids, down_out
        )

    final_hidden_states = (
        down_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(dtype)
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

    if num_tokens == 0 or num_top_k == 0:
        final_hidden_states = torch.zeros(
            num_tokens,
            hidden_dim,
            device=device,
            dtype=hidden_states.dtype,
        )
        if batch_size is not None:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states

    # Single-token decode fast path for the per-expert Marlin dispatcher.
    # The general path below builds a sorted/padded token-expert index which
    # costs several kernel launches per layer; for M=1 those are all overhead.
    if num_tokens == 1 and getattr(self, "_moe_dispatch_backend", None) == "marlin":
        out = _marlin_experts_project_one_token(self, hidden_states, top_k_index, top_k_weights)
        if batch_size is not None:
            out = out.view(batch_size, seq_len, hidden_dim)
        return out

    cache = _get_expert_dispatch_cache(self)
    proj_by_attr = cache["proj_by_attr"]

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1)

    # Normalize without a device->host sync; invalid entries are masked after
    # projection so they preserve the generic per-expert fallback's zero output.
    expert_ids, sentinel_mask = _sanitize_expert_ids(expert_ids, num_experts)

    active_experts = torch.unique(expert_ids, sorted=True)
    local_ids = torch.searchsorted(active_experts, expert_ids)
    local_sorted, perm = torch.sort(local_ids)
    token_idx_sorted = token_idx[perm]
    sample_weights_sorted = sample_weights[perm]
    selected_hidden = hidden_states[token_idx_sorted]

    e_active = active_experts.numel()
    counts_local = torch.bincount(local_sorted, minlength=e_active)
    offsets = torch.cumsum(counts_local, dim=0, dtype=torch.int64).to(torch.int32)

    expert0 = cache["experts"][0]
    intermediate_dim = expert0.gate_proj.out_features

    # Defer the grouped_mm lookup until the grouped-mm branch is actually used.
    # The marlin and marlin_moe branches do not need it, so an environment without
    # torch.nn.functional.grouped_mm should still be able to take those paths.
    grouped_mm = getattr(torch.nn.functional, "grouped_mm", None)

    def _project(expert_attr: str, in_f: int, out_f: int, x: torch.Tensor) -> torch.Tensor:
        """Dequantize one projection for the active experts, stack, and grouped_mm."""
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor | None] = []
        projs = proj_by_attr[expert_attr]
        for expert_idx in active_experts.tolist():
            proj = projs[expert_idx]
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
        gated = cache["apply_gate"](gate_out, up_out)
        del gate_out, up_out

        down_out = _project("down_proj", intermediate_dim, hidden_dim, gated)
        del gated

    weighted = down_out * sample_weights_sorted.unsqueeze(-1)
    del down_out
    sentinel_sorted = sentinel_mask[perm]
    weighted = weighted.masked_fill(sentinel_sorted.unsqueeze(-1), 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    out_per_sample = weighted[inv_perm]
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states.to(hidden_states.dtype)


@torch._dynamo.disable
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
