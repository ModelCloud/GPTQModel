# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import os

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from .config import BaseQuantizeConfig, ScaleSearchConfig, _normalize_quant_bits, resolve_quant_format
from ._scale_search_triton import _triton_find_params_batched_activation


log = setup_logger()

HF_OPTIMUM = "hf_optimum"

# Bound temporary candidate tensors while still amortizing eager CUDA launch
# overhead for the 128-column groups used by GPTQ.
SCALE_SEARCH_TARGET_ELEMENTS = 64 * 1024 * 1024
SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK = 16
# Correlated objectives benefit more from large GEMMs than elementwise
# objectives. Give them a larger, still-bounded workspace without increasing
# activation or MSE search memory.
CORRELATED_SCALE_SEARCH_TARGET_ELEMENTS = 128 * 1024 * 1024
CORRELATED_SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK = 80


def quantize(x, scale, zero, maxq, requires_groupwise_processing: bool):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if requires_groupwise_processing:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        return scale * q
    else:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, qcfg: BaseQuantizeConfig, shape=1, name: str=None):
        super(Quantizer, self).__init__()

        self.qcfg = qcfg
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.name=name

    def requires_groupwise_processing(self) -> bool:
        return False

    # FIXME, optimum shouldn't call this directly, it should call hf_configure
    def configure(
        self,
        perchannel=False,
        grid=100,
        maxshrink=0.8,
        trits=False,
        bits: int | str | None = None, # for hf compat
        sym: bool | None = None, # for hf compat
    ):
        if self.name == HF_OPTIMUM:
            if bits is not None:
                self.qcfg.bits = _normalize_quant_bits(bits, format_value=resolve_quant_format(self.qcfg.format, self.qcfg.method))
            if sym is not None:
                self.qcfg.sym = sym

        if self.requires_groupwise_processing():
            maxq_value = 2 ** (self.qcfg.bits - 1) - 1
        else:
            maxq_value = 2 ** self.qcfg.bits - 1

        self.perchannel = perchannel
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            maxq_value = -1
        self._maxq_value = maxq_value
        self.maxq = torch.tensor(maxq_value)

    def _prepare_scale_search_hessian(
        self,
        hessian: torch.Tensor | None,
        *,
        method: ScaleSearchConfig,
        columns: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Prepare only the Hessian data required by the selected objective."""

        if hessian is None:
            return None
        if method == ScaleSearchConfig.ACTIVATION:
            # Activation search only consumes the diagonal. Avoiding a full
            # symmetric Hessian copy is especially important for 8192-wide MLPs.
            if hessian.ndim == 1 and hessian.shape == (columns,):
                importance = hessian.detach().to(device=device, dtype=torch.float32)
            elif hessian.ndim == 2 and hessian.shape == (columns, columns):
                importance = hessian.detach().diagonal().to(device=device, dtype=torch.float32)
            else:
                raise ValueError(
                    "Quantizer.find_params(): `hessian` must have shape "
                    f"({columns},) or ({columns}, {columns}) for activation scale search, "
                    f"got {tuple(hessian.shape)}."
                )
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
            diagonal_mean = importance.mean()
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            normalized = importance / safe_mean
            return torch.where(valid, normalized, torch.ones_like(normalized))

        if hessian.ndim != 2 or hessian.shape != (columns, columns):
            raise ValueError(
                "Quantizer.find_params(): `hessian` must have shape "
                f"({columns}, {columns}) for {method.value} scale search, got {tuple(hessian.shape)}."
            )

        prepared = hessian.detach().to(device=device, dtype=torch.float32)
        prepared = torch.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        prepared = (prepared + prepared.t()) * 0.5
        diagonal_mean = prepared.diagonal().clamp_min(0).mean()
        if not torch.isfinite(diagonal_mean) or diagonal_mean <= 0:
            return None
        prepared.div_(diagonal_mean)
        if method == ScaleSearchConfig.HYBRID:
            # Fold 50/50 diagonal shrinkage into the Hessian once instead of
            # allocating and reducing another candidate-sized squared-error
            # tensor for every scale-search chunk. Off-diagonal terms receive
            # half weight while diagonal terms retain their original weight.
            prepared.mul_(0.5)
            prepared.diagonal().mul_(2.0)
        return prepared

    def _scale_search_error(
        self,
        error: torch.Tensor,
        *,
        method: ScaleSearchConfig,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score one or more clipping candidates for every output row."""

        if method == ScaleSearchConfig.MSE or hessian is None:
            return error.abs_().pow_(mse).sum(dim=-1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearchConfig.ACTIVATION:
            importance = hessian if hessian.ndim == 1 else hessian.diagonal().clamp_min(0)
            return error_fp32.square_().mul_(importance).sum(dim=-1)

        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            if error_fp32.ndim not in {2, 3}:
                raise ValueError(
                    f"{method.value.capitalize()} scale search expects a two-dimensional candidate tensor or "
                    "a three-dimensional candidate batch."
                )
            # Exact within a quantization group. For an ungrouped tensor, use a
            # bounded block-diagonal approximation so scale search does not turn
            # into an O(columns^2) allocation for every candidate.
            configured_group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
            columns = error_fp32.shape[-1]
            block_size = (
                min(columns, configured_group_size)
                if configured_group_size > 0
                else min(columns, 128)
            )
            objective = torch.zeros(error_fp32.shape[:-1], dtype=torch.float32, device=error_fp32.device)
            for start in range(0, columns, block_size):
                end = min(start + block_size, columns)
                block_error = error_fp32[..., start:end]
                block_hessian = hessian[start:end, start:end]
                block_objective = (block_error.matmul(block_hessian) * block_error).sum(dim=-1)
                objective.add_(block_objective)
            return objective.clamp_min_(0)

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    def _prepare_scale_search_hessian_batched(
        self,
        hessian: torch.Tensor | None,
        *,
        method: ScaleSearchConfig,
    ) -> torch.Tensor | None:
        """Prepare Hessian/diagonal data for batched multi-group scale search."""

        if hessian is None:
            return None

        if method == ScaleSearchConfig.ACTIVATION:
            if hessian.ndim == 1:
                hessian = hessian.unsqueeze(0)
            if hessian.ndim != 2:
                raise ValueError(
                    f"Activation scale search expects a 2D group-diagonal tensor, got {tuple(hessian.shape)}."
                )
            importance = hessian.detach().to(dtype=torch.float32, device=hessian.device)
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
            # Normalize per-group to match the per-group find_params behavior.
            diagonal_mean = importance.mean(dim=-1, keepdim=True)
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            normalized = importance / safe_mean
            return torch.where(valid, normalized, torch.ones_like(normalized))

        if hessian.ndim != 3:
            raise ValueError(
                f"{method.value.capitalize()} batched scale search expects a 3D group-Hessian tensor, "
                f"got {tuple(hessian.shape)}."
            )
        prepared = hessian.detach().to(dtype=torch.float32, device=hessian.device)
        prepared = torch.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        prepared = (prepared + prepared.transpose(-2, -1)) * 0.5
        diagonal = prepared.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        # Normalize each group by its own diagonal mean to match per-group find_params.
        diagonal_mean = diagonal.mean(dim=-1, keepdim=True).unsqueeze(-1)
        valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
        safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
        prepared = prepared / safe_mean
        if method == ScaleSearchConfig.HYBRID:
            prepared.mul_(0.5)
            prepared.diagonal(dim1=-2, dim2=-1).mul_(2.0)
        return prepared

    def _scale_search_error_batched(
        self,
        error: torch.Tensor,
        *,
        method: ScaleSearchConfig,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score candidates for every (output row, group) pair in one launch."""

        if method == ScaleSearchConfig.MSE or hessian is None:
            return error.abs_().pow_(mse).sum(dim=-1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearchConfig.ACTIVATION:
            importance = hessian.unsqueeze(0).unsqueeze(0) if hessian.ndim == 2 else hessian
            return error_fp32.square_().mul_(importance).sum(dim=-1)

        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            # error: [candidates, rows, groups, group_size]
            # hessian: [groups, group_size, group_size]
            # For HYBRID the Hessian is already folded into 0.5*(full + diagonal)
            # during _prepare_scale_search_hessian_batched, so the quadratic form
            # alone matches the per-group objective.
            projected = torch.einsum("crgi,gij->crgj", error_fp32, hessian)
            objective = (error_fp32 * projected).sum(dim=-1)
            return objective.clamp_min_(0)

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    @staticmethod
    def _scale_search_candidate_chunk_size(
        x: torch.Tensor,
        candidate_count: int,
        method: ScaleSearchConfig,
    ) -> int:
        """Choose a bounded candidate batch that amortizes eager launch overhead."""

        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            target_elements = CORRELATED_SCALE_SEARCH_TARGET_ELEMENTS
            max_candidates = CORRELATED_SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK
        else:
            target_elements = SCALE_SEARCH_TARGET_ELEMENTS
            max_candidates = SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK
        elements_per_candidate = max(1, x.numel())
        return max(
            1,
            min(
                candidate_count,
                max_candidates,
                target_elements // elements_per_candidate,
            ),
        )

    def _quantize_scale_search_candidates(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        *,
        maxq_value: int,
    ) -> torch.Tensor:
        """Quantize one candidate batch without synchronizing on a CUDA scalar."""

        if maxq_value < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.div(x, scale)
        q.round_()
        if self.requires_groupwise_processing():
            return q.clamp_(-self.maxq, self.maxq).mul_(scale)
        return q.add_(zero).clamp_(0, self.maxq).sub_(zero).mul_(scale)

    def find_params(self, x, weight=False, *, hessian: torch.Tensor | None = None):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.qcfg.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            xmin = torch.where(tmp, -xmax, xmin)
        tmp = (xmin == 0) & (xmax == 0)
        xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
        xmax = torch.where(tmp, torch.ones_like(xmax), xmax)

        if maxq_value < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            if self.requires_groupwise_processing():
                self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / self.maxq
                if self.qcfg.sym:
                    self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
                else:
                    self.zero = torch.round(-xmin / self.scale)

        mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
        method = getattr(self.qcfg, "scale_search", None)
        if method is None and mse > 0:
            method = ScaleSearchConfig.MSE
        elif isinstance(method, str):
            method = ScaleSearchConfig(method)

        if method is not None and mse > 0.0:
            prepared_hessian = None
            if method != ScaleSearchConfig.MSE:
                prepared_hessian = self._prepare_scale_search_hessian(
                    hessian,
                    method=method,
                    columns=x.shape[1],
                    device=dev,
                )
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            candidate_count = int(self.maxshrink * self.grid)

            chunk_size = self._scale_search_candidate_chunk_size(x, candidate_count, method)
            # Vectorize the shrink factors on the device; materializing the
            # original Python-float list one-by-one can differ by one ULP after
            # cancellation and alter the serialized scale values.
            shrink = torch.arange(candidate_count, device=dev, dtype=torch.float32)
            shrink = 1.0 - shrink / self.grid
            # Precompute scale/zero for every candidate once so the chunk loop
            # only slices views instead of recomputing elementwise ranges.
            p = shrink.view(-1, 1)
            xmin_all = p * xmin.unsqueeze(0)
            xmax_all = p * xmax.unsqueeze(0)
            if self.requires_groupwise_processing():
                scale_all = xmax_all / self.maxq
            else:
                scale_all = (xmax_all - xmin_all) / self.maxq
            if self.qcfg.sym:
                zero_all = self.zero.unsqueeze(0).expand_as(scale_all)
            else:
                zero_all = torch.round(-xmin_all / scale_all)

            x_batch = x.unsqueeze(0)
            for start in range(0, candidate_count, chunk_size):
                end = min(start + chunk_size, candidate_count)
                scale1 = scale_all[start:end]
                zero1 = zero_all[start:end]
                candidate = self._quantize_scale_search_candidates(
                    x_batch,
                    scale1.unsqueeze(2),
                    zero1.unsqueeze(2),
                    maxq_value=maxq_value,
                )
                errors = self._scale_search_error(
                    candidate.sub_(x_batch),
                    method=method,
                    mse=mse,
                    hessian=prepared_hessian,
                )
                # torch.min returns the first index on ties. Combining one
                # winner per chunk with a strict comparison across chunks
                # exactly preserves the scalar loop's first-candidate rule.
                chunk_best, chunk_index = errors.min(dim=0)
                gather_index = chunk_index.unsqueeze(0)
                chunk_scale = scale1.gather(0, gather_index).squeeze(0)
                chunk_zero = zero1.gather(0, gather_index).squeeze(0)
                take = chunk_best < best
                best = torch.where(take, chunk_best, best)
                self.scale = torch.where(take, chunk_scale, self.scale)
                self.zero = torch.where(take, chunk_zero, self.zero)
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def find_params_batched(
        self,
        x: torch.Tensor,
        weight: bool = False,
        *,
        hessian: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scales and zeros for all groups in x at once.

        x is expected to have shape [rows, num_groups, group_size] and
        hessian is either None, a [num_groups, group_size] diagonal for
        activation search, or a [num_groups, group_size, group_size] group
        Hessian for correlated objectives.  Returns (scale, zero) of shape
        [rows, num_groups].
        """

        if x.ndim != 3:
            raise ValueError(f"find_params_batched expects a 3D tensor, got {tuple(x.shape)}.")
        if not (weight and self.perchannel):
            raise ValueError("find_params_batched is only supported for per-channel weight quantization.")

        dev = x.device
        self.maxq = self.maxq.to(dev)
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value

        rows, num_groups, group_size = x.shape

        # Min/max over the group dimension (last).
        tmp = torch.zeros((rows, num_groups), device=dev)
        xmin = torch.minimum(x.amin(dim=-1), tmp)
        xmax = torch.maximum(x.amax(dim=-1), tmp)

        if self.qcfg.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            xmin = torch.where(tmp, -xmax, xmin)
        tmp = (xmin == 0) & (xmax == 0)
        xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
        xmax = torch.where(tmp, torch.ones_like(xmax), xmax)

        if maxq_value < 0:
            scale = xmax
            zero = xmin
        else:
            if self.requires_groupwise_processing():
                scale = xmax / self.maxq
                zero = torch.zeros_like(scale)
            else:
                scale = (xmax - xmin) / self.maxq
                if self.qcfg.sym:
                    zero = torch.full_like(scale, (self.maxq + 1) / 2)
                else:
                    zero = torch.round(-xmin / scale)

        mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
        method = getattr(self.qcfg, "scale_search", None)
        if method is None and mse > 0:
            method = ScaleSearchConfig.MSE
        elif isinstance(method, str):
            method = ScaleSearchConfig(method)

        if method is not None and mse > 0.0:
            prepared_hessian = None
            if method != ScaleSearchConfig.MSE:
                prepared_hessian = self._prepare_scale_search_hessian_batched(
                    hessian,
                    method=method,
                )

            candidate_count = int(self.maxshrink * self.grid)
            if candidate_count > 0:
                shrink = torch.arange(candidate_count, device=dev, dtype=torch.float32)
                shrink = 1.0 - shrink / self.grid
                p = shrink.view(-1, 1, 1)
                xmin_all = p * xmin.unsqueeze(0)
                xmax_all = p * xmax.unsqueeze(0)
                if self.requires_groupwise_processing():
                    scale_all = xmax_all / self.maxq
                else:
                    scale_all = (xmax_all - xmin_all) / self.maxq
                if self.qcfg.sym:
                    zero_all = zero.unsqueeze(0).expand(candidate_count, -1, -1)
                else:
                    zero_all = torch.round(-xmin_all / scale_all)
            else:
                scale_all = scale.unsqueeze(0)
                zero_all = zero.unsqueeze(0)

            if (
                os.environ.get("GPTQMODEL_SCALE_SEARCH_TRITON", "1") != "0"
                and _triton_find_params_batched_activation is not None
                and method == ScaleSearchConfig.ACTIVATION
                and prepared_hessian is not None
                and not self.requires_groupwise_processing()
                and group_size <= 128
                and maxq_value >= 7
                and x.is_cuda
                and x.is_contiguous()
                and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
            ):
                try:
                    return _triton_find_params_batched_activation(
                        x,
                        xmin,
                        xmax,
                        prepared_hessian,
                        scale_all,
                        zero_all,
                        self.grid,
                        self.maxshrink,
                        float(self.maxq.item()),
                        bool(self.qcfg.sym),
                        candidate_count,
                    )
                except Exception as e:
                    log.warn(f"Triton activation scale-search failed, falling back: {e}")

            best = torch.full((rows, num_groups), float("inf"), device=dev)
            chunk_size = self._scale_search_candidate_chunk_size(x, candidate_count, method)

            x_batch = x.unsqueeze(0)
            for start in range(0, candidate_count, chunk_size):
                end = min(start + chunk_size, candidate_count)
                scale1 = scale_all[start:end]
                zero1 = zero_all[start:end]
                candidate = self._quantize_scale_search_candidates(
                    x_batch.expand(end - start, -1, -1, -1),
                    scale1.unsqueeze(-1),
                    zero1.unsqueeze(-1),
                    maxq_value=maxq_value,
                )
                errors = self._scale_search_error_batched(
                    candidate - x_batch,
                    method=method,
                    mse=mse,
                    hessian=prepared_hessian,
                )
                chunk_best, chunk_index = errors.min(dim=0)
                gather_index = chunk_index.unsqueeze(0)
                chunk_scale = scale1.gather(0, gather_index).squeeze(0)
                chunk_zero = zero1.gather(0, gather_index).squeeze(0)
                take = chunk_best < best
                best = torch.where(take, chunk_best, best)
                scale = torch.where(take, chunk_scale, scale)
                zero = torch.where(take, chunk_zero, zero)

        return scale, zero

    def quantize(self, x):
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value
        return quantize(x, self.scale, self.zero, maxq_value, self.requires_groupwise_processing())

    # def enabled(self):
    #     return self.maxq > 0

    # def ready(self):
    # return torch.all(self.scale != 0)

class QQQQuantizer(Quantizer):
    def requires_groupwise_processing(self) -> bool:
        return self.qcfg.group_size == -1 and self.qcfg.sym

__all__ = ["Quantizer"]
