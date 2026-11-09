# SPDX-License-Identifier: Apache-2.0
"""Causal calibration stages for Kimi K3's dense core.

The caller installs serialized-and-reconstructed K6 weights between dependent
stages. Routed experts must already represent the immutable K2 checkpoint.
Both the running prefix and the complete attention-residual bank are required.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


@torch.inference_mode()
def attention_inputs(block, prefix_sum, block_residual):
    """Return normalized attention input and the native pre-attention state."""
    if not block.use_attn_residuals:
        raise ValueError("Kimi core stages require attention residuals")
    hidden = prefix_sum
    batch, length, width = hidden.shape
    if block_residual.shape[1]:
        hidden = block.forward.__globals__["_apply_attn_res"](
            prefix_sum.view(-1, width), block_residual,
            block.self_attention_res_proj, block.self_attention_res_norm,
        ).view(batch, length, width)
    if block.layer_idx % block.attn_res_block_size == 0:
        block_residual = torch.cat((block_residual, prefix_sum.view(-1, width).unsqueeze(1)), 1)
        prefix_sum = None
    return block.input_layernorm(hidden), prefix_sum, block_residual


@torch.inference_mode()
def attention_finish(block, prefix_sum, block_residual, attention_output):
    """Finish attention residuals, then produce the exact normalized FFN input."""
    prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
    batch, length, width = prefix_sum.shape
    hidden = block.forward.__globals__["_apply_attn_res"](
        prefix_sum.view(-1, width), block_residual,
        block.mlp_res_proj, block.mlp_res_norm,
    ).view(batch, length, width)
    return prefix_sum, block_residual, block.post_attention_layernorm(hidden)


class _CapturedLinearInput(BaseException):
    def __init__(self, owner):
        self.owner = owner


class LinearInputCapture:
    """Abort one native forward immediately before a selected linear executes.

    Use a fresh context per forward. Only this context's private sentinel is
    suppressed; model exceptions propagate after the hook is removed.
    """
    def __init__(self, target):
        self.target = target
        self.input = None
        self._handle = None
        self._used = False

    def _capture(self, module, args, kwargs):
        value = args[0] if args else kwargs.get("input")
        if not isinstance(value, torch.Tensor):
            raise TypeError("target linear did not receive a tensor input")
        self.input = value.detach()
        raise _CapturedLinearInput(self)

    def __enter__(self):
        if self._used:
            raise RuntimeError("linear input capture contexts cannot be reused")
        self._used = True
        self._handle = self.target.register_forward_pre_hook(self._capture, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._handle.remove()
        self._handle = None
        if isinstance(exc, _CapturedLinearInput) and exc.owner is self:
            return True
        if exc_type is None and self.input is None:
            raise RuntimeError("native forward did not reach the target linear")
        return False


@torch.inference_mode()
def capture_linear_input(target, forward: Callable, *args, **kwargs):
    """Run ``forward`` until ``target`` and return its actual tensor input."""
    with LinearInputCapture(target) as capture:
        forward(*args, **kwargs)
    return capture.input


@torch.inference_mode()
def shared_down_input(shared, hidden):
    """Use the checkpoint MLP's own activation and low-precision rounding."""
    if shared.config.hidden_act == "situ":
        return shared.act_fn(torch.cat((shared.gate_proj(hidden), shared.up_proj(hidden)), dim=-1))
    return shared.act_fn(shared.gate_proj(hidden)) * shared.up_proj(hidden)


@torch.inference_mode()
def routed_up_input(moe, hidden):
    """Collect the native normalized, weighted K2 aggregate for latent up."""
    if not moe.use_latent_moe:
        raise ValueError("Kimi core stages require the latent MoE path")
    # The router consumes full-width hidden states, before latent projection.
    selected, weights = moe.gate(hidden)
    latent = moe.routed_expert_down_proj(hidden.view(-1, hidden.shape[-1]))
    output = moe.moe_infer(latent, selected, weights)
    if moe.latent_moe_use_norm:
        output = moe.routed_expert_norm(output)
    return output


@torch.inference_mode()
def complete_moe(prefix_sum, block_residual, shared_input, routed_input, moe):
    """Replay final K6 down/up projections with the checkpoint's add order."""
    shared = moe.shared_experts.down_proj(shared_input)
    routed = moe.routed_expert_up_proj(routed_input).view_as(shared)
    output = routed + shared
    return prefix_sum + output, block_residual


@torch.inference_mode()
def final_head_input(prefix_sum, block_residual, *, output_attn_res_proj,
                     output_attn_res_norm, model_norm, apply_attn_res):
    """Finish the model's native output mixer and norm without computing logits."""
    batch, length, width = prefix_sum.shape
    hidden = apply_attn_res(
        prefix_sum.view(-1, width), block_residual,
        output_attn_res_proj, output_attn_res_norm,
    ).view(batch, length, width)
    return model_norm(hidden)


class DenseRawHessian:
    """Raw FP32 XTX over all dense observations, with no synthetic rows.

    Calls retain corpus order. Only one chunk's FP32 activations is temporary;
    raw matrices may be shared between equal-input projections, whose quantizer
    must independently clone/finalize them.
    """
    def __init__(self, width, *, device="cuda:0", chunk_rows=2048):
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("Hessian width must be a positive integer")
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
            raise ValueError("Hessian chunk size must be a positive integer")
        self.width = width
        self.chunk_rows = chunk_rows
        self.H = torch.zeros((width, width), dtype=torch.float32, device=device)
        self.count = 0

    @torch.inference_mode()
    def add(self, inputs):
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2
                or inputs.shape[-1] != self.width or not inputs.is_floating_point()):
            raise ValueError("invalid dense Hessian input geometry or dtype")
        rows = inputs.reshape(-1, self.width)
        if not rows.shape[0]:
            raise ValueError("dense Hessian requires nonempty observations")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError("nonfinite dense Hessian observations")
        for start in range(0, rows.shape[0], self.chunk_rows):
            value = rows[start:start + self.chunk_rows].to(device=self.H.device, dtype=torch.float32)
            self.H.addmm_(value.T, value)
        if not bool(torch.isfinite(self.H).all()):
            raise ValueError("nonfinite dense Hessian accumulation")
        self.count += rows.shape[0]

    update = add

    @torch.inference_mode()
    def raw_dict(self, *, minimum_rows=1):
        if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows <= 0:
            raise ValueError("minimum observation count must be a positive integer")
        if self.count < minimum_rows:
            raise ValueError(f"insufficient dense Hessian observations: {self.count} < {minimum_rows}")
        if (not bool(torch.isfinite(self.H).all())
                or float(self.H.diagonal().sum()) <= 0):
            raise ValueError("invalid dense Hessian covariance")
        return {"H": self.H, "count": self.count, "finalized": False}

    as_dict = raw_dict
