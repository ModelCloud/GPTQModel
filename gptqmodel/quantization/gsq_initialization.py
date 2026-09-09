"""Scalar range-search primitives for the explicit GSQ paper procedure."""

import torch

from .quantizer import Quantizer
from .config import ScaleSearchConfig


@torch.no_grad()
def signed_scalar_range_search(weight, *, bits):
    """Fit one group per row using the author's signed scale search.

    Matches the symmetric W2/W3/W4 prior's FP32 operation order, exponent 2.4,
    100-point shrink grid and 0.8 maximum shrink. This is the range search only;
    GPTQ Hessian preparation and sequential error feedback remain caller-owned.
    Returns reconstructed weights, column-vector scales and integer zero points.
    Negative scales are intentional: the signed grid has unequal endpoints.
    """
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError('Signed GSQ range search supports W2/W3/W4')
    if (weight.ndim != 2 or not weight.numel() or not weight.is_floating_point()
            or not torch.isfinite(weight).all()):
        raise ValueError('Signed GSQ range search requires finite nonempty floating rows')
    value = weight.float()
    maxq = 2**bits-1
    zeros = torch.full((len(value),), (maxq+1)/2, device=value.device, dtype=value.dtype)
    lower = torch.minimum(value.min(1).values, torch.zeros_like(zeros))
    upper = torch.maximum(value.max(1).values, torch.zeros_like(zeros))
    upper = torch.maximum(lower.abs(), upper)
    lower = torch.where(lower < 0, -upper, lower)
    empty = (lower == 0) & (upper == 0)
    lower = torch.where(empty, -torch.ones_like(lower), lower)
    upper = torch.where(empty, torch.ones_like(upper), upper)
    scales = (upper-lower)/maxq
    best = torch.full_like(scales, float('inf'))

    def reconstruct(scale):
        codes = torch.clamp(torch.round(value/scale[:, None])+zeros[:, None], 0, maxq)
        return scale[:, None]*(codes-zeros[:, None])

    for index in range(80):
        fraction = 1-index/100
        candidate = (fraction*upper-fraction*lower)/maxq
        positive = reconstruct(candidate)
        negative = reconstruct(-candidate)
        positive_error = (positive-value).abs().pow(2.4).sum(1)
        negative_error = (negative-value).abs().pow(2.4).sum(1)
        use_negative = negative_error < positive_error
        error = torch.where(use_negative, negative_error, positive_error)
        improve = error < best
        best = torch.where(improve, error, best)
        scales = torch.where(improve, torch.where(use_negative, -candidate, candidate), scales)
    if not torch.isfinite(best).all() or not torch.isfinite(scales).all() or (scales == 0).any():
        raise ValueError('Signed GSQ range search exceeded finite FP32 arithmetic')
    return reconstruct(scales), scales[:, None], zeros[:, None]


class SignedGSQQuantizer(Quantizer):
    """Use the signed prior within repository GPTQ error feedback.

    This adapter selects the paper's weight-only range objective; Hessians
    supplied by GPTQ affect its error feedback, not the range-search objective.
    Full author GPTQ trajectory parity is a separate validation requirement.
    """

    def _validate(self, value, weight):
        if not weight or not self.perchannel or not self.qcfg.sym:
            raise ValueError('Signed GSQ prior requires symmetric per-row weight quantization')
        if value.device.type not in ('cpu', 'cuda'):
            raise ValueError('Signed GSQ prior supports CPU/CUDA; fused MPS parameter search is not supported')
        if self.grid != 100 or self.maxshrink != .8:
            raise ValueError('Signed GSQ prior requires the fixed author shrink schedule')
        if self.qcfg.scale_search != ScaleSearchConfig.MSE or self.qcfg.mse != 2.4:
            raise ValueError('Signed GSQ prior requires explicit MSE range search with exponent 2.4')

    def find_params(self, x, weight=False, *, hessian=None, gptq_inverse_cholesky=None):
        del hessian, gptq_inverse_cholesky
        self._validate(x, weight)
        _, self.scale, self.zero = signed_scalar_range_search(x, bits=self.qcfg.bits)

    def find_params_batched(self, x, weight=False, *, hessian=None, hessian_prepared=False,
                            gptq_inverse_cholesky=None):
        del hessian, hessian_prepared, gptq_inverse_cholesky
        self._validate(x, weight)
        if x.ndim != 3:
            raise ValueError('Batched signed GSQ prior requires [rows,groups,columns]')
        rows, groups, columns = x.shape
        _, scales, zeros = signed_scalar_range_search(x.reshape(rows*groups, columns), bits=self.qcfg.bits)
        return scales.reshape(rows, groups), zeros.reshape(rows, groups)
