# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.config import (
    _normalize_fp8_fmt,
    _normalize_fp8_scale_semantics,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
)
from ...quantization.dtype import (
    available_float8_dtype_names,
    dequantize_fp8,
    device_supports_native_fp8,
)
from ...utils.backend import BACKEND
from . import WeightOnlyQuantLinear
from .gguf import _apply_optional_smoother


def _fp8_dtype_from_name(fmt: str) -> torch.dtype:
    return getattr(torch, _normalize_fp8_fmt(fmt))


def _weight_to_matrix(linear: nn.Module) -> torch.Tensor:
    weight = linear.weight.detach()
    if isinstance(linear, _ConvNd):
        weight = weight.flatten(1)
    if isinstance(linear, transformers.pytorch_utils.Conv1D):
        weight = weight.T
    return weight


def _compute_scale_inv(abs_max: torch.Tensor, fp8_max: float) -> torch.Tensor:
    abs_max = abs_max.to(torch.float32)
    eps = torch.finfo(torch.float32).tiny
    return torch.where(
        abs_max > 0,
        torch.full_like(abs_max, float(fp8_max)) / abs_max.clamp_min(eps),
        torch.ones_like(abs_max),
    )


def quantize_fp8_weight(
    weight: torch.Tensor,
    *,
    format: str = "float8_e4m3fn",
    weight_scale_method: str = "row",
    weight_block_size: Optional[Tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D weight matrix, got shape {tuple(weight.shape)}.")

    format = _normalize_fp8_fmt(format)
    if format == "float8_e8m0fnu":
        raise ValueError(
            "TorchFP8Linear does not quantize dense weights to float8_e8m0fnu; "
            "use that format only for dequantization of existing checkpoints."
        )
    block_size = _normalize_fp8_weight_block_size(weight_block_size)
    weight_scale_method = _normalize_fp8_weight_scale_method(
        weight_scale_method,
        weight_block_size=block_size,
    )
    fp8_dtype = _fp8_dtype_from_name(format)
    fp8_max = torch.finfo(fp8_dtype).max

    weight = weight.to(device="cpu", dtype=torch.float32).contiguous()

    if weight_scale_method == "tensor":
        scale_inv = _compute_scale_inv(weight.abs().amax(), fp8_max)
        quantized = torch.clamp(weight * scale_inv, min=-fp8_max, max=fp8_max).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32)

    if weight_scale_method == "row":
        scale_inv = _compute_scale_inv(weight.abs().amax(dim=1), fp8_max)
        quantized = torch.clamp(
            weight * scale_inv.unsqueeze(1),
            min=-fp8_max,
            max=fp8_max,
        ).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()

    if block_size is None:
        raise ValueError("FP8 block quantization requires `weight_block_size`.")

    block_rows, block_cols = block_size
    rows, cols = weight.shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"FP8 block quantization expects shape {tuple(weight.shape)} to be divisible by block size "
            f"{block_size}."
        )

    row_blocks = rows // block_rows
    col_blocks = cols // block_cols
    blocks = weight.reshape(row_blocks, block_rows, col_blocks, block_cols)
    scale_inv = _compute_scale_inv(blocks.abs().amax(dim=(1, 3)), fp8_max)
    scaled = blocks * scale_inv.unsqueeze(1).unsqueeze(3)
    quantized = torch.clamp(scaled, min=-fp8_max, max=fp8_max).to(fp8_dtype).reshape(rows, cols)
    return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()


class TorchFP8Linear(WeightOnlyQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.FP8_TORCH]
    SUPPORTS_METHODS = [METHOD.FP8]
    SUPPORTS_FORMATS = {FORMAT.FP8: 15}
    SUPPORTS_BITS = [8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    # torch-npu 2.9 exposes FP8 dtypes on CPU, but CANN rejects float8 tensors
    # on Ascend devices, so FP8 is not an NPU-capable torch backend yet.
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    QUANT_TYPE = "fp8"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        format: str = "float8_e4m3fn",
        weight_scale_method: str = "row",
        weight_block_size: Optional[Tuple[int, int]] = None,
        weight_scale_semantics: str = "inverse",
        **kwargs,
    ):
        self.fp8_format = _normalize_fp8_fmt(format)
        self.fp8_dtype = _fp8_dtype_from_name(self.fp8_format)
        block_size = _normalize_fp8_weight_block_size(weight_block_size)
        self.weight_scale_method = _normalize_fp8_weight_scale_method(
            weight_scale_method,
            weight_block_size=block_size,
        )
        self.weight_block_size = block_size
        self.weight_scale_semantics = _normalize_fp8_scale_semantics(weight_scale_semantics)
        self._scaled_mm_hard_disabled = False

        if self.weight_scale_method == "block" and self.weight_block_size is not None:
            block_rows, block_cols = self.weight_block_size
            if out_features % block_rows != 0 or in_features % block_cols != 0:
                raise ValueError(
                    f"TorchFP8Linear block scaling requires out_features/in_features "
                    f"to be divisible by `weight_block_size={self.weight_block_size}`."
                )

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.FP8_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            **kwargs,
        )

        if register_buffers:
            self._allocate_buffers(bias=bias)

    @classmethod
    def validate_once(cls):
        if not available_float8_dtype_names():
            return False, RuntimeError("TorchFP8Linear requires a PyTorch build with FP8 dtypes.")
        return True, None

    def smooth_block_size(self) -> int:
        if self.weight_scale_method == "block" and self.weight_block_size is not None:
            return self.weight_block_size[1]
        return -1

    def _scale_shape(self) -> tuple[int, ...]:
        if self.weight_scale_method == "tensor":
            return ()
        if self.weight_scale_method == "row":
            return (self.out_features,)
        if self.weight_block_size is None:
            raise ValueError("TorchFP8Linear block scaling requires `weight_block_size`.")
        block_rows, block_cols = self.weight_block_size
        return (
            self.out_features // block_rows,
            self.in_features // block_cols,
        )

    def _allocate_buffers(self, *, bias: bool) -> None:
        weight = torch.zeros((self.out_features, self.in_features), dtype=self.fp8_dtype)
        scale = torch.ones(self._scale_shape(), dtype=torch.float32)

        if "weight" in self._buffers:
            self.weight = weight
        else:
            self.register_buffer("weight", weight)

        if "weight_scale_inv" in self._buffers:
            self.weight_scale_inv = scale
        else:
            self.register_buffer("weight_scale_inv", scale)

        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=torch.float16)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        if hasattr(self, "weight") and self.weight is not None:
            buffers.append(self.weight)
        if hasattr(self, "weight_scale_inv") and self.weight_scale_inv is not None:
            buffers.append(self.weight_scale_inv)
        if hasattr(self, "bias") and self.bias is not None:
            buffers.append(self.bias)
        return buffers

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={self.fp8_format}, "
            f"weight_scale_method={self.weight_scale_method}"
        )

    def _weight_to_matrix(self, linear: nn.Module) -> torch.Tensor:
        return _weight_to_matrix(linear)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        block_in: int = 8192,
        workers: int = 1,
    ):
        del block_in, workers
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_gpu(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        block_in: int = 8192,
        device: torch.device | None = None,
    ):
        del block_in, device
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    @torch.inference_mode()
    def pack_original(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        smooth=None,
    ):
        del scales, zeros, g_idx

        weight = self._weight_to_matrix(linear).to(device="cpu", dtype=torch.float32)
        weight = _apply_optional_smoother(
            weight,
            smooth=smooth,
            group_size=self.smooth_block_size(),
        )
        qweight, weight_scale_inv = quantize_fp8_weight(
            weight,
            format=self.fp8_format,
            weight_scale_method=self.weight_scale_method,
            weight_block_size=self.weight_block_size,
        )

        if "weight" in self._buffers:
            self.weight = qweight
        else:
            self.register_buffer("weight", qweight)

        if "weight_scale_inv" in self._buffers:
            self.weight_scale_inv = weight_scale_inv
        else:
            self.register_buffer("weight_scale_inv", weight_scale_inv)

        if linear.bias is not None:
            bias = linear.bias.detach().to(device="cpu", dtype=torch.float16)
            if "bias" in self._buffers:
                self.bias = bias
            else:
                self.register_buffer("bias", bias)
        else:
            self.bias = None

        self._scaled_mm_hard_disabled = False

    def _resolve_target(self, device=None, dtype=None) -> tuple[torch.device, torch.dtype]:
        target_device = self.weight.device if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        return target_device, target_dtype

    def _expanded_scale_inv(self, *, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
        scale_inv = self.weight_scale_inv
        if scale_inv.device != target_device or scale_inv.dtype != target_dtype:
            scale_inv = scale_inv.to(device=target_device, dtype=target_dtype)

        if self.weight_scale_method == "tensor":
            return scale_inv
        if self.weight_scale_method == "row":
            return scale_inv.view(-1, 1)
        if self.weight_block_size is None:
            raise ValueError("TorchFP8Linear block scaling requires `weight_block_size`.")

        block_rows, block_cols = self.weight_block_size
        expanded = scale_inv.repeat_interleave(block_rows, dim=0)
        expanded = expanded.repeat_interleave(block_cols, dim=1)
        return expanded

    def dequantize_weight(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_target(device=device, dtype=dtype)

        if self.weight_scale_semantics != "inverse":
            raise NotImplementedError(
                f"Unsupported FP8 scale semantics `{self.weight_scale_semantics}`."
            )

        # Older GPUs can store the module on CUDA but still miss native FP8 math.
        # In that case, dequantize on CPU directly to fp16/bf16 and only then move.
        prefer_cpu_dequant = target_device.type == "cpu"
        # PyTorch's CUDA cast path for E8M0 currently produces NaNs when used as a
        # dense weight matrix, so force the validated CPU LUT path for that format.
        if self.fp8_format == "float8_e8m0fnu":
            prefer_cpu_dequant = True
        if target_device.type == "cuda" and not device_supports_native_fp8(target_device):
            prefer_cpu_dequant = True

        if prefer_cpu_dequant:
            weight = dequantize_fp8(
                self.weight.to(device="cpu"),
                scale_inv=self.weight_scale_inv.to(device="cpu"),
                axis=None if self.weight_scale_method == "block" else (0 if self.weight_scale_method == "row" else 0),
                target_dtype=target_dtype,
            )
            if target_device.type != "cpu":
                weight = weight.to(device=target_device)
            return weight.transpose(0, 1).contiguous()

        weight = self.weight if self.weight.device == target_device else self.weight.to(device=target_device)
        weight = weight.to(target_dtype)
        scale_inv = self._expanded_scale_inv(target_device=target_device, target_dtype=target_dtype)
        return (weight / scale_inv).transpose(0, 1).contiguous()

    def _scaled_mm_weight_scale(self, *, device: torch.device) -> torch.Tensor:
        scale_inv = self.weight_scale_inv
        if scale_inv.device != device:
            scale_inv = scale_inv.to(device=device)
        scale = torch.reciprocal(scale_inv.to(torch.float32))
        if self.weight_scale_method == "tensor":
            return scale
        if self.weight_scale_method == "row":
            return scale.view(1, -1)
        raise NotImplementedError("scaled_mm is only used for tensorwise or rowwise FP8 scales.")

    def _quantize_input_for_scaled_mm(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_max = torch.finfo(self.fp8_dtype).max
        x_work = x_flat.to(torch.float32)
        if self.weight_scale_method == "tensor":
            scale_inv = _compute_scale_inv(x_work.abs().amax(), fp8_max)
        else:
            scale_inv = _compute_scale_inv(x_work.abs().amax(dim=1, keepdim=True), fp8_max)
        x_q = torch.clamp(x_work * scale_inv, min=-fp8_max, max=fp8_max).to(self.fp8_dtype)
        return x_q, torch.reciprocal(scale_inv).to(torch.float32)

    def _can_use_scaled_mm(self, x_flat: torch.Tensor) -> bool:
        return (
            not self._scaled_mm_hard_disabled
            and hasattr(torch, "_scaled_mm")
            and x_flat.device.type == "cuda"
            and self.weight_scale_method == "tensor"
            and self.fp8_format != "float8_e8m0fnu"
            and x_flat.dtype in {torch.float16, torch.bfloat16}
            and x_flat.shape[-1] == self.in_features
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        )

    def _forward_dequant_matmul(self, x_flat: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(device=x_flat.device, dtype=x_flat.dtype)
        return torch.matmul(x_flat, weight)

    def _forward_scaled_mm(self, x_flat: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.device == x_flat.device else self.weight.to(device=x_flat.device)
        x_q, scale_a = self._quantize_input_for_scaled_mm(x_flat)
        scale_b = self._scaled_mm_weight_scale(device=x_flat.device)
        return torch._scaled_mm(
            x_q,
            weight.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=x_flat.dtype,
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if self._can_use_scaled_mm(x_flat):
            try:
                output = self._forward_scaled_mm(x_flat)
            except Exception:
                self._scaled_mm_hard_disabled = True
                output = self._forward_dequant_matmul(x_flat)
        else:
            output = self._forward_dequant_matmul(x_flat)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["TorchFP8Linear", "quantize_fp8_weight"]
