# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from .config import BaseQuantizeConfig, ScaleSearchConfig, _normalize_quant_bits, resolve_quant_format


log = setup_logger()

HF_OPTIMUM = "hf_optimum"

# Bound temporary candidate tensors while still amortizing eager CUDA launch
# overhead for the 128-column groups used by GPTQ.
SCALE_SEARCH_TARGET_ELEMENTS = 16 * 1024 * 1024
SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK = 16


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
        if hessian.ndim != 2 or hessian.shape != (columns, columns):
            raise ValueError(
                "Quantizer.find_params(): `hessian` must have shape "
                f"({columns}, {columns}), got {tuple(hessian.shape)}."
            )

        if method == ScaleSearchConfig.ACTIVATION:
            # Activation search only consumes the diagonal. Avoiding a full
            # symmetric Hessian copy is especially important for 8192-wide MLPs.
            importance = hessian.detach().diagonal().to(device=device, dtype=torch.float32)
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
            diagonal_mean = importance.mean()
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            normalized = importance / safe_mean
            return torch.where(valid, normalized, torch.ones_like(normalized))

        prepared = hessian.detach().to(device=device, dtype=torch.float32)
        prepared = torch.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        prepared = (prepared + prepared.t()) * 0.5
        diagonal_mean = prepared.diagonal().clamp_min(0).mean()
        if not torch.isfinite(diagonal_mean) or diagonal_mean <= 0:
            return None
        return prepared / diagonal_mean

    def _scale_search_error(
        self,
        error: torch.Tensor,
        *,
        method: ScaleSearchConfig,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score one clipping candidate for every output row."""

        if method == ScaleSearchConfig.MSE or hessian is None:
            return error.abs().pow(mse).sum(dim=-1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearchConfig.ACTIVATION:
            importance = hessian if hessian.ndim == 1 else hessian.diagonal().clamp_min(0)
            return error_fp32.square().mul(importance).sum(dim=-1)

        if method == ScaleSearchConfig.HESSIAN:
            if error_fp32.ndim != 2:
                raise ValueError("Hessian scale search expects one two-dimensional candidate tensor.")
            # Exact within a quantization group. For an ungrouped tensor, use a
            # bounded block-diagonal approximation so scale search does not turn
            # into an O(columns^2) allocation for every candidate.
            configured_group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
            block_size = (
                min(error_fp32.shape[1], configured_group_size)
                if configured_group_size > 0
                else min(error_fp32.shape[1], 128)
            )
            objective = torch.zeros(error_fp32.shape[0], dtype=torch.float32, device=error_fp32.device)
            for start in range(0, error_fp32.shape[1], block_size):
                end = min(start + block_size, error_fp32.shape[1])
                block_error = error_fp32[:, start:end]
                block_hessian = hessian[start:end, start:end]
                objective.add_((block_error.matmul(block_hessian) * block_error).sum(dim=1))
            return objective.clamp_min_(0)

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    @staticmethod
    def _scale_search_candidate_chunk_size(x: torch.Tensor, candidate_count: int) -> int:
        """Choose a bounded candidate batch that amortizes eager launch overhead."""

        elements_per_candidate = max(1, x.numel())
        return max(
            1,
            min(
                candidate_count,
                SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK,
                SCALE_SEARCH_TARGET_ELEMENTS // elements_per_candidate,
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
        if self.requires_groupwise_processing():
            q = torch.clamp(torch.round(x / scale), -self.maxq, self.maxq)
            return scale * q
        q = torch.clamp(torch.round(x / scale) + zero, 0, self.maxq)
        return scale * (q - zero)

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

            if method == ScaleSearchConfig.HESSIAN:
                # Full Hessian scoring uses group-local matrix products and is
                # intentionally kept one candidate at a time.
                for i in range(candidate_count):
                    p = 1 - i / self.grid
                    xmin1 = p * xmin
                    xmax1 = p * xmax
                    scale1 = (
                        xmax1 / self.maxq
                        if self.requires_groupwise_processing()
                        else (xmax1 - xmin1) / self.maxq
                    )
                    zero1 = torch.round(-xmin1 / scale1) if not self.qcfg.sym else self.zero
                    candidate = self._quantize_scale_search_candidates(
                        x,
                        scale1.unsqueeze(1),
                        zero1.unsqueeze(1),
                        maxq_value=maxq_value,
                    )
                    err = self._scale_search_error(
                        candidate - x,
                        method=method,
                        mse=mse,
                        hessian=prepared_hessian,
                    )
                    take = err < best
                    best = torch.where(take, err, best)
                    self.scale = torch.where(take, scale1, self.scale)
                    self.zero = torch.where(take, zero1, self.zero)
            else:
                chunk_size = self._scale_search_candidate_chunk_size(x, candidate_count)
                # Materialize the original Python-float factors directly. A
                # device-side FP32 subtraction can differ by one ULP after
                # cancellation and alter the serialized scale values.
                shrink = torch.tensor(
                    [1 - i / self.grid for i in range(candidate_count)],
                    device=dev,
                    dtype=torch.float32,
                )
                x_batch = x.unsqueeze(0)
                for start in range(0, candidate_count, chunk_size):
                    end = min(start + chunk_size, candidate_count)
                    p = shrink[start:end].unsqueeze(1)
                    xmin1 = p * xmin.unsqueeze(0)
                    xmax1 = p * xmax.unsqueeze(0)
                    scale1 = (
                        xmax1 / self.maxq
                        if self.requires_groupwise_processing()
                        else (xmax1 - xmin1) / self.maxq
                    )
                    zero1 = torch.round(-xmin1 / scale1) if not self.qcfg.sym else self.zero.expand_as(scale1)
                    candidate = self._quantize_scale_search_candidates(
                        x_batch,
                        scale1.unsqueeze(2),
                        zero1.unsqueeze(2),
                        maxq_value=maxq_value,
                    )
                    errors = self._scale_search_error(
                        candidate - x_batch,
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

    def quantize(self, x):
        return quantize(x, self.scale, self.zero, self.maxq, self.requires_groupwise_processing())

    # def enabled(self):
    #     return self.maxq > 0

    # def ready(self):
    # return torch.all(self.scale != 0)

class QQQQuantizer(Quantizer):
    def requires_groupwise_processing(self) -> bool:
        return self.qcfg.group_size == -1 and self.qcfg.sym

__all__ = ["Quantizer"]
