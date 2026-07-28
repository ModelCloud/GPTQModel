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
from torch import nn

from gptqmodel.utils import log
from gptqmodel.utils.env import env_flag

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


def _can_use_grouped_mm(self: nn.Module, hidden_states: torch.Tensor) -> bool:
    """Return True when this experts module can use the grouped GEMM fast path."""
    if getattr(self, "_grouped_mm_ok", False):
        return True
    if getattr(self, "_grouped_mm_failed", False):
        return False

    if not _grouped_mm_available():
        return False
    if hidden_states.device.type != "cuda":
        return False
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
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

    for proj, in_f, out_f in (
        (expert0.gate_proj, hidden_dim, intermediate_dim),
        (expert0.up_proj, hidden_dim, intermediate_dim),
        (expert0.down_proj, intermediate_dim, hidden_dim),
    ):
        if _extract_expert_dense_weight(proj, in_f, out_f, hidden_states.dtype, hidden_states.device) is None:
            self._grouped_mm_failed = True
            return False

    self._grouped_mm_ok = True
    return True


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

    grouped_mm = torch.nn.functional.grouped_mm

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
    if getattr(self, GROUPED_DISPATCH_FLAG, False) and _can_use_grouped_mm(self, hidden_states):
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
