# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import platform
import sys
import time
from functools import lru_cache

import torch
import transformers
from torch import nn
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from .config import Fallback, FallbackStrategy, RTNConfig, SmoothMSE
from .fallback_smooth import mse_optimal_quant, smooth_block
from .quantizer import HF_OPTIMUM, Quantizer

_MLX_RTN_MIN_ELEMENTS = 256 * 1024


@lru_cache(maxsize=1)
def _mlx_rtn_quantization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_array_to_torch(value, device: torch.device) -> torch.Tensor:
    return torch.from_dlpack(value).to(device=device).contiguous()


def _quantize_rtn_weight_mlx_to_torch(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import mlx.core as mx

    from .mlx_rtn import quantize_rtn_weight_mlx

    if weight.device.type == "mps":
        torch.mps.synchronize()
    result = quantize_rtn_weight_mlx(
        mx.from_dlpack(weight.contiguous()),
        bits=bits,
        group_size=group_size,
        sym=sym,
    )
    return tuple(_mlx_array_to_torch(value, weight.device) for value in result[:3])


def _should_use_mlx_rtn_quantization(weight: torch.Tensor, qcfg: RTNConfig) -> bool:
    return (
        _mlx_rtn_quantization_available()
        and weight.device.type in {"cpu", "mps"}
        and weight.ndim == 2
        and weight.numel() >= _MLX_RTN_MIN_ELEMENTS
        and qcfg.smooth is None
        and isinstance(qcfg.bits, int)
        and 2 <= qcfg.bits <= 8
        and (qcfg.group_size == -1 or qcfg.group_size > 0)
    )


def get_number_of_rows_and_cols(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        return layer.weight.shape[1], layer.weight.shape[0]

    return layer.weight.shape[0], math.prod(layer.weight.shape[1:])


class RTN:
    """Native weight-only RTN quantizer with optional smoothing.

    This path never enters GPTQ's activation/Hessian lifecycle. It quantizes the
    module weights directly, then returns tensors that can be packed into GPTQ,
    AWQ, or future export layouts by the existing packing stage.
    """

    def __init__(self, module: nn.Module, qcfg: RTNConfig):
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
            self._named_module = module
        else:
            self.module = module
            self.name = HF_OPTIMUM
            self._named_module = None

        self.validate_module(self.module)
        self.qcfg = qcfg
        self.quantizer = Quantizer(qcfg=qcfg, name=self.name)
        self.quantizer.configure(perchannel=True)
        self.nsamples = 0
        self._primary = Fallback(
            strategy=FallbackStrategy.RTN,
            threshold=True,
            smooth=qcfg.smooth,
        )

        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)

        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_cols = 0

        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

    @staticmethod
    def validate_module(module: nn.Module) -> None:
        assert isinstance(
            module,
            (nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D),
        ), f"We supports only linear and convolutional layers. actual = `{module}`"

    def clone_module(self, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=True, device=device)
        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)
        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()
        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)
        return clone.float()

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    @staticmethod
    def _collapse_group_param(param: torch.Tensor) -> torch.Tensor:
        collapsed = param if param.dim() > 1 else param.unsqueeze(1)
        if collapsed.shape[1] > 1:
            collapsed = collapsed.mean(dim=1, keepdim=True)
        return collapsed

    @torch.inference_mode()
    def quantize(self):
        maxq = 2**self.qcfg.bits - 1
        effective_group_size = (
            self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        )
        smooth_method = self.qcfg.smooth
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        start_time = time.time()
        target_device = self.module.weight.device
        weights = self.clone_module(device=target_device)
        if _should_use_mlx_rtn_quantization(weights, self.qcfg):
            quantized, scale, zero = _quantize_rtn_weight_mlx_to_torch(
                weights,
                bits=self.qcfg.bits,
                group_size=self.qcfg.group_size,
                sym=self.qcfg.sym,
            )
        else:
            quantized = torch.empty_like(weights)
            scale_chunks = []
            zero_chunks = []

            for start in range(0, self.columns, effective_group_size):
                end = min(start + effective_group_size, self.columns)
                block = weights[:, start:end]

                if isinstance(smooth_method, SmoothMSE):
                    dequant, scale, zero = mse_optimal_quant(
                        block,
                        self.qcfg,
                        maxq,
                        steps=mse_steps,
                        maxshrink=mse_maxshrink,
                    )
                else:
                    block_mod, scale_factor = smooth_block(
                        block,
                        self._primary,
                        group_size=effective_group_size,
                    )
                    self.quantizer.find_params(block_mod, weight=True)
                    dequant = self.quantizer.quantize(block_mod)
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero

                    if scale_factor is not None:
                        scale = scale * scale_factor
                        dequant = dequant * scale_factor

                quantized[:, start:end] = dequant
                scale_chunks.append(self._collapse_group_param(scale))
                zero_chunks.append(self._collapse_group_param(zero))

            scale = torch.cat(scale_chunks, dim=1)
            zero = torch.cat(zero_chunks, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            quantized = quantized[:, :valid_cols]
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        g_idx = (
            torch.arange(valid_cols, device=quantized.device, dtype=torch.int32)
            // effective_group_size
        )

        if isinstance(self.module, transformers.Conv1D):
            quantized = quantized.t()

        if quantized.shape != self.module.weight.shape:
            quantized = quantized.reshape(self.module.weight.shape).to(
                self.module.weight.dtype
            )
        else:
            quantized = quantized.to(self.module.weight.dtype)

        quantized = quantized.to(
            device=self.module.weight.data.device, non_blocking=False
        )
        mean_abs_err = (quantized - self.module.weight.data).abs().mean().item()
        duration = time.time() - start_time
        avg_loss = f"rtn: {mean_abs_err:.7f}"
        damp = 0.0

        return quantized, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples


__all__ = ["RTN", "get_number_of_rows_and_cols"]
