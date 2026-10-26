# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels for the ParoQuant symmetric quantize/dequantize STE path.

ParoQuant forces ``sym=True`` (``_require_paroquant_sym``) and the default
``fast`` quantizer implementation keeps that symmetry
(``_quantizer_sym_for_impl``), so the symmetric branch of
``pseudo_quantize_dequant`` is the branch every calibration optimizer step
evaluates. It is also the branch that never got a fused node: the composed
reference graph spends five launches on the forward (``clamp`` on the scale,
``div``, ``round``, ``clamp``, ``mul``) and ten more on the backward (the two
straight-through nodes are pass-throughs, but the multiply and the division
each contribute a value gradient plus a per-group scale reduction), and each of
those passes re-reads and re-writes the whole transformed weight. On MI350X the
stage is launch-bound -- the calibration loop runs tens of optimizer steps per
linear across every linear in the model -- so those passes cost far more in
queue latency and memory traffic than in arithmetic.

The kernel here evaluates the same arithmetic in a single pass. Every floating
point step keeps the reference semantics: ``div_rn`` for the exact
round-to-nearest division the eager graph performs, ``rint`` for the
round-half-to-even step behind ``torch.round``, and the clamp order the
reference uses, so the result is bitwise identical to the composed graph.

The forward also stores ``-scaled / bounded_scale`` in place of ``scaled``.
With ``s`` the bounded scale, ``d = w / s`` the scaled weight and ``q`` the
integer codes, the composed graph's straight-through gradients are::

    grad_w     = (grad_out * s) / s
    grad_scale = sum(grad_out * q) + sum(grad_out * s * (-d / s))

so ``-d / s`` is the only thing the backward ever needs ``d`` for. Spelling it
in the forward kernel costs the same single full-size store the old
intermediate did, and removes the ``square`` plus ``div`` launches the torch
backward used to spend rebuilding it from ``scaled`` on every optimizer step.

The backward keeps the two per-group reductions in torch. Torch's reduction
order is what makes ``grad_scale`` bitwise reproducible, and a Triton block
reduction would sum the same terms in a different order; the elementwise
partials are therefore produced in the same torch spelling the reference uses.
"""

from __future__ import annotations

from typing import Optional

import torch

from gptqmodel.utils.env import env_flag


try:  # pragma: no cover - exercised implicitly by the availability probe
    import triton
    import triton.language as tl

    if torch.version.hip is not None:
        from triton.language.extra.hip import libdevice as _libdevice
    else:
        from triton.language.extra.cuda import libdevice as _libdevice

    _SYM_STE_TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - Triton is an optional accelerator
    _SYM_STE_TRITON_AVAILABLE = False


_SYM_STE_BLOCK = 1024

# Kill switch for hosts where the Triton kernels cannot be exercised safely.
_SYM_STE_TRITON_ENABLED = env_flag("GPTQMODEL_PAROQUANT_SYM_STE_TRITON", default=True)


if _SYM_STE_TRITON_AVAILABLE:

    @triton.jit
    def _sym_qdq_forward_kernel(
        weight_ptr,
        scale_ptr,
        out_ptr,
        neg_scaled_over_scale_ptr,
        quant_ptr,
        num_elements,
        group_size,
        qmin,
        qmax,
        STORE_INTERMEDIATES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offsets < num_elements
        group = offsets // group_size
        bounded_scale = tl.load(scale_ptr + group, mask=mask, other=1.0)
        bounded_scale = tl.minimum(tl.maximum(bounded_scale, 1e-5), 1e5)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        scaled = tl.math.div_rn(weight, bounded_scale)
        quant = tl.minimum(tl.maximum(_libdevice.rint(scaled), qmin), qmax)
        # Value-only callers have no backward to feed, so the two full-size
        # intermediates stay in registers and the kernel writes one tensor.
        if STORE_INTERMEDIATES:
            tl.store(
                neg_scaled_over_scale_ptr + offsets,
                tl.math.div_rn(-scaled, bounded_scale),
                mask=mask,
            )
            tl.store(quant_ptr + offsets, quant, mask=mask)
        tl.store(out_ptr + offsets, quant * bounded_scale, mask=mask)


    def _triton_sym_qdq_forward(
        ctx,
        weight: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        group_size: int,
    ) -> torch.Tensor:
        """Run the fused forward kernel and stash the backward operands on ``ctx``."""
        view = weight.reshape(-1, group_size).contiguous()
        num_elements = view.numel()
        out = torch.empty_like(view)
        neg_scaled_over_scale = torch.empty_like(view)
        quant = torch.empty_like(view)
        grid = (triton.cdiv(num_elements, _SYM_STE_BLOCK),)
        _sym_qdq_forward_kernel[grid](
            view,
            scale,
            out,
            neg_scaled_over_scale,
            quant,
            num_elements,
            group_size,
            float(qmin),
            float(qmax),
            STORE_INTERMEDIATES=True,
            BLOCK=_SYM_STE_BLOCK,
        )
        ctx.save_for_backward(neg_scaled_over_scale, quant, scale)
        ctx.group_size = group_size
        return out.reshape_as(weight)


    def _triton_sym_qdq_forward_only(
        weight: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        group_size: int,
    ) -> torch.Tensor:
        """Run the fused forward kernel for callers that are outside the autograd graph.

        With no backward to feed, the kernel is specialized to skip the two
        full-size intermediates and the call returns the dequantized weight
        directly. ``out`` doubles as the unused intermediate pointers, which the
        compiled kernel never touches.
        """
        view = weight.reshape(-1, group_size).contiguous()
        num_elements = view.numel()
        out = torch.empty_like(view)
        grid = (triton.cdiv(num_elements, _SYM_STE_BLOCK),)
        _sym_qdq_forward_kernel[grid](
            view,
            scale,
            out,
            out,
            out,
            num_elements,
            group_size,
            float(qmin),
            float(qmax),
            STORE_INTERMEDIATES=False,
            BLOCK=_SYM_STE_BLOCK,
        )
        return out.reshape_as(weight)


    class _SymQuantDequantTritonSTE(torch.autograd.Function):
        """Fused Triton symmetric quantize/dequantize with the torch elementwise backward.

        The forward pass is where the reference graph spends its whole-tensor
        reads and writes, so it collapses into one kernel. The backward keeps the
        torch spelling of the partials: the two straight-through nodes contribute
        nothing, and the remaining multiply and division gradients are cheap
        elementwise kernels whose spelling is what makes ``grad_scale`` bitwise
        identical to the composed reference.
        """

        @staticmethod
        def forward(
            ctx,
            weight: torch.Tensor,
            scale: torch.Tensor,
            qmin: int,
            qmax: int,
            group_size: int,
        ) -> torch.Tensor:
            return _triton_sym_qdq_forward(ctx, weight, scale, qmin, qmax, group_size)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            neg_scaled_over_scale, quant, scale = ctx.saved_tensors
            bounded_scale = scale.clamp(min=1e-5, max=1e5)
            grad_view = grad_output.reshape(-1, ctx.group_size)
            grad_quant = grad_view * bounded_scale
            grad_weight = grad_quant / bounded_scale
            grad_scale = (grad_view * quant).sum(dim=1, keepdim=True) + (
                grad_quant * neg_scaled_over_scale
            ).sum(dim=1, keepdim=True)
            return grad_weight.reshape_as(grad_output), grad_scale, None, None, None


def sym_quant_dequant_ste_available() -> bool:
    """Report whether the fused Triton symmetric STE kernels can run on this host."""
    return _SYM_STE_TRITON_AVAILABLE and _SYM_STE_TRITON_ENABLED


def sym_quant_dequant_ste_forward(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Run the fused Triton symmetric forward for value-only callers, or return None.

    This is the same arithmetic as ``sym_quant_dequant_ste`` with the autograd
    node and its two saved intermediates removed, for callers that are already
    outside the autograd graph.
    """
    if not sym_quant_dequant_ste_available():
        return None
    return _triton_sym_qdq_forward_only(weight, scale, qmin, qmax, group_size)


def sym_quant_dequant_ste(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Run the fused Triton symmetric STE path, or return None when it cannot apply."""
    if not sym_quant_dequant_ste_available():
        return None
    return _SymQuantDequantTritonSTE.apply(weight, scale, qmin, qmax, group_size)
