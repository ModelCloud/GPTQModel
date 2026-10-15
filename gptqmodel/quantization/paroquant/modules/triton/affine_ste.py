# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels for the ParoQuant affine quantize/dequantize STE path.

The calibration loop evaluates the asymmetric pseudo-quantizer thousands of
times per stage. The composed reference graph and the fused autograd node both
spend ten kernel launches per call on a per-group elementwise transform: four
on the per-group quantizer parameters (``clamp``, ``neg``, ``round``,
``clamp``) and six passes over the full weight (``div``, ``round``, ``add``,
``clamp``, ``sub``, ``mul``), each of which re-reads and re-writes the whole
tensor. On MI350X/gfx950 the stage is launch-bound, so those passes cost far
more in queue latency and memory traffic than in arithmetic.

The kernels here evaluate the same arithmetic in a single pass. Every floating
point step keeps the reference semantics: ``div_rn`` for the exact
round-to-nearest division the eager graph performs, ``rint`` for the
round-half-to-even step behind ``torch.round``, and the clamp order the
reference uses, so the result is bitwise identical to the composed graph.

Both backward variants keep the two per-group reductions in torch. Torch's
reduction order is what makes ``grad_scale`` bitwise reproducible, and a Triton
block reduction would sum the same terms in a different order; the elementwise
partials are therefore produced -- in one kernel by
``_AffineQuantDequantTritonSTE``, or by the same torch spelling the reference
uses in ``_AffineQuantDequantHybridSTE`` -- and reduced by the caller.
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

    _AFFINE_STE_TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - Triton is an optional accelerator
    _AFFINE_STE_TRITON_AVAILABLE = False


_AFFINE_STE_BLOCK = 1024

# Kill switch for hosts where the Triton kernels cannot be exercised safely.
_AFFINE_STE_TRITON_ENABLED = env_flag("GPTQMODEL_PAROQUANT_AFFINE_STE_TRITON", default=True)

# The Triton backward collapses the elementwise partials into one kernel (six
# launches instead of twelve) at the cost of materialising three full-size
# partial tensors. On gfx950 that extra traffic eats the launch win: nine
# interleaved warm ParoQuant runs measured 4.19 s for the hybrid backward against
# 4.24 s for the Triton one, while both beat the torch fused node on launches and
# host time. The hybrid stays the default and the Triton backward is kept behind
# the env switch for hosts where the launch cost dominates the extra traffic.
_AFFINE_STE_TRITON_BACKWARD = env_flag("GPTQMODEL_PAROQUANT_AFFINE_STE_TRITON_BACKWARD", default=False)


if _AFFINE_STE_TRITON_AVAILABLE:

    @triton.jit
    def _affine_qdq_forward_kernel(
        weight_ptr,
        scale_ptr,
        zero_point_ptr,
        out_ptr,
        scaled_ptr,
        centered_ptr,
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
        zero_point = tl.load(zero_point_ptr + group, mask=mask, other=0.0)
        round_zero_point = tl.minimum(tl.maximum(_libdevice.rint(-zero_point), qmin), qmax)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        scaled = tl.math.div_rn(weight, bounded_scale)
        quant = tl.minimum(tl.maximum(_libdevice.rint(scaled) + round_zero_point, qmin), qmax)
        centered = quant - round_zero_point
        # Value-only callers have no backward to feed, so the two full-size
        # intermediates stay in registers and the kernel writes one tensor.
        if STORE_INTERMEDIATES:
            tl.store(scaled_ptr + offsets, scaled, mask=mask)
            tl.store(centered_ptr + offsets, centered, mask=mask)
        tl.store(out_ptr + offsets, centered * bounded_scale, mask=mask)


    @triton.jit
    def _affine_qdq_backward_kernel(
        grad_ptr,
        scale_ptr,
        scaled_ptr,
        centered_ptr,
        grad_weight_ptr,
        grad_scale_mul_ptr,
        grad_scale_div_ptr,
        num_elements,
        group_size,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offsets < num_elements
        group = offsets // group_size
        bounded_scale = tl.load(scale_ptr + group, mask=mask, other=1.0)
        bounded_scale = tl.minimum(tl.maximum(bounded_scale, 1e-5), 1e5)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        scaled = tl.load(scaled_ptr + offsets, mask=mask, other=0.0)
        centered = tl.load(centered_ptr + offsets, mask=mask, other=0.0)
        grad_scaled = grad * bounded_scale
        tl.store(grad_weight_ptr + offsets, tl.math.div_rn(grad_scaled, bounded_scale), mask=mask)
        tl.store(grad_scale_mul_ptr + offsets, grad * centered, mask=mask)
        tl.store(grad_scale_div_ptr + offsets, grad_scaled * tl.math.div_rn(-scaled, bounded_scale), mask=mask)


    def _triton_qdq_forward(
        ctx,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point_float: torch.Tensor,
        qmin: int,
        qmax: int,
        group_size: int,
    ) -> torch.Tensor:
        """Run the fused forward kernel and stash the backward operands on ``ctx``."""
        view = weight.reshape(-1, group_size).contiguous()
        num_elements = view.numel()
        out = torch.empty_like(view)
        scaled = torch.empty_like(view)
        centered = torch.empty_like(view)
        grid = (triton.cdiv(num_elements, _AFFINE_STE_BLOCK),)
        _affine_qdq_forward_kernel[grid](
            view,
            scale,
            zero_point_float,
            out,
            scaled,
            centered,
            num_elements,
            group_size,
            float(qmin),
            float(qmax),
            STORE_INTERMEDIATES=True,
            BLOCK=_AFFINE_STE_BLOCK,
        )
        ctx.save_for_backward(scaled, centered, scale)
        ctx.group_size = group_size
        ctx.zero_point_shape = zero_point_float.shape
        return out.reshape_as(weight)


    def _triton_qdq_forward_only(
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point_float: torch.Tensor,
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
        grid = (triton.cdiv(num_elements, _AFFINE_STE_BLOCK),)
        _affine_qdq_forward_kernel[grid](
            view,
            scale,
            zero_point_float,
            out,
            out,
            out,
            num_elements,
            group_size,
            float(qmin),
            float(qmax),
            STORE_INTERMEDIATES=False,
            BLOCK=_AFFINE_STE_BLOCK,
        )
        return out.reshape_as(weight)


    class _AffineQuantDequantTritonSTE(torch.autograd.Function):
        """Fused Triton affine quantize/dequantize with a Triton backward.

        The backward kernel writes the three elementwise partials in one pass and
        leaves only the two per-group reductions to torch, which keeps the
        reduction order -- and therefore ``grad_scale`` -- bitwise identical to
        the composed reference.
        """

        @staticmethod
        def forward(
            ctx,
            weight: torch.Tensor,
            scale: torch.Tensor,
            zero_point_float: torch.Tensor,
            qmin: int,
            qmax: int,
            group_size: int,
        ) -> torch.Tensor:
            return _triton_qdq_forward(ctx, weight, scale, zero_point_float, qmin, qmax, group_size)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            scaled, centered, scale = ctx.saved_tensors
            group_size = ctx.group_size
            grad_view = grad_output.reshape(-1, group_size).contiguous()
            num_elements = grad_view.numel()
            grad_weight = torch.empty_like(grad_view)
            grad_scale_mul = torch.empty_like(grad_view)
            grad_scale_div = torch.empty_like(grad_view)
            grid = (triton.cdiv(num_elements, _AFFINE_STE_BLOCK),)
            _affine_qdq_backward_kernel[grid](
                grad_view,
                scale,
                scaled,
                centered,
                grad_weight,
                grad_scale_mul,
                grad_scale_div,
                num_elements,
                group_size,
                BLOCK=_AFFINE_STE_BLOCK,
            )
            grad_scale = grad_scale_mul.sum(dim=1, keepdim=True) + grad_scale_div.sum(dim=1, keepdim=True)
            grad_zero_point = grad_scale.new_zeros(ctx.zero_point_shape)
            return grad_weight.reshape_as(grad_output), grad_scale, grad_zero_point, None, None, None


    class _AffineQuantDequantHybridSTE(torch.autograd.Function):
        """Fused Triton forward with the torch elementwise backward.

        The forward pass is where the reference graph spends its whole-tensor
        reads and writes, so it collapses into one kernel. The backward keeps the
        torch spelling of the partials: torch's elementwise kernels are cheaper
        here than the Triton backward's three full-size materialised partials,
        and reusing them keeps the bitwise equivalence argument trivial.
        """

        @staticmethod
        def forward(
            ctx,
            weight: torch.Tensor,
            scale: torch.Tensor,
            zero_point_float: torch.Tensor,
            qmin: int,
            qmax: int,
            group_size: int,
        ) -> torch.Tensor:
            return _triton_qdq_forward(ctx, weight, scale, zero_point_float, qmin, qmax, group_size)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            scaled, centered, scale = ctx.saved_tensors
            bounded_scale = scale.clamp(min=1e-5, max=1e5)
            grad_view = grad_output.reshape(-1, ctx.group_size)
            grad_scaled = grad_view * bounded_scale
            grad_weight = grad_scaled / bounded_scale
            grad_scale = (grad_view * centered).sum(dim=1, keepdim=True) + (
                grad_scaled * (-scaled / bounded_scale)
            ).sum(dim=1, keepdim=True)
            grad_zero_point = grad_scale.new_zeros(ctx.zero_point_shape)
            return grad_weight.reshape_as(grad_output), grad_scale, grad_zero_point, None, None, None


def affine_quant_dequant_ste_available() -> bool:
    """Report whether the fused Triton affine STE kernels can run on this host."""
    return _AFFINE_STE_TRITON_AVAILABLE and _AFFINE_STE_TRITON_ENABLED


def affine_quant_dequant_ste_forward(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point_float: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Run the fused Triton affine forward for value-only callers, or return None.

    This is the same arithmetic as ``affine_quant_dequant_ste`` with the autograd
    node and its two saved intermediates removed, for callers that are already
    outside the autograd graph.
    """
    if not affine_quant_dequant_ste_available():
        return None
    return _triton_qdq_forward_only(weight, scale, zero_point_float, qmin, qmax, group_size)


def affine_quant_dequant_ste(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point_float: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Run the fused Triton affine STE path, or return None when it cannot apply."""
    if not affine_quant_dequant_ste_available():
        return None
    node = _AffineQuantDequantTritonSTE if _AFFINE_STE_TRITON_BACKWARD else _AffineQuantDequantHybridSTE
    return node.apply(weight, scale, zero_point_float, qmin, qmax, group_size)
