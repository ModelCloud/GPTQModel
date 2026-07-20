# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from .config import BaseQuantizeConfig, ScaleSearch, _normalize_quant_bits, resolve_quant_format


log = setup_logger()

HF_OPTIMUM = "hf_optimum"

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
            self.maxq = torch.tensor(2 ** (self.qcfg.bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2 ** self.qcfg.bits - 1)

        self.perchannel = perchannel
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def _prepare_scale_search_hessian(
        self,
        hessian: torch.Tensor | None,
        *,
        columns: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a finite, symmetric, scale-normalized Hessian for range scoring."""

        if hessian is None:
            return None
        if hessian.ndim != 2 or hessian.shape != (columns, columns):
            raise ValueError(
                "Quantizer.find_params(): `hessian` must have shape "
                f"({columns}, {columns}), got {tuple(hessian.shape)}."
            )

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
        method: ScaleSearch,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score one clipping candidate for every output row."""

        if method == ScaleSearch.MSE or hessian is None:
            return error.abs().pow(mse).sum(dim=1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearch.ACTIVATION:
            importance = hessian.diagonal().clamp_min(0)
            return error_fp32.square().mul(importance.unsqueeze(0)).sum(dim=1)

        if method == ScaleSearch.HESSIAN:
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

    def find_params(self, x, weight=False, *, hessian: torch.Tensor | None = None):
        dev = x.device
        self.maxq = self.maxq.to(dev)

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
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
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
            method = ScaleSearch.MSE
        elif isinstance(method, str):
            method = ScaleSearch(method)

        if method is not None and mse > 0.0:
            prepared_hessian = None
            if method != ScaleSearch.MSE:
                prepared_hessian = self._prepare_scale_search_hessian(
                    hessian,
                    columns=x.shape[1],
                    device=dev,
                )
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (
                    xmax1 / self.maxq
                    if self.requires_groupwise_processing()
                    else (xmax1 - xmin1) / self.maxq
                )
                zero1 = torch.round(-xmin1 / scale1) if not self.qcfg.sym else self.zero
                candidate = quantize(
                    x,
                    scale1.unsqueeze(1),
                    zero1.unsqueeze(1),
                    self.maxq,
                    self.requires_groupwise_processing(),
                )
                err = self._scale_search_error(
                    candidate - x,
                    method=method,
                    mse=mse,
                    hessian=prepared_hessian,
                )
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
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
