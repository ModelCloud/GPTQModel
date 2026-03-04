# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...looper.linear_mode import LinearMode
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.torch import TORCH_HAS_FUSED_OPS


def _has_int8_mm_op() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


def _requantize_to_int8(float_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric int8 re-quantization for [K, N] weights."""
    if float_weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(float_weight.shape)}")

    channel_max = float_weight.abs().amax(dim=0)
    channel_scale = (channel_max / 127.0).clamp_min(1e-10)
    weight_int8 = (
        (float_weight / channel_scale.unsqueeze(0))
        .round()
        .clamp_(-128, 127)
        .to(torch.int8)
    )
    return weight_int8, channel_scale


class Int8PackedModule(torch.nn.Module):
    """CPU fused int8 matmul wrapper around aten::_weight_int8pack_mm."""

    def __init__(self, int8_weight_nk: torch.Tensor, channel_scales: torch.Tensor):
        super().__init__()
        self.register_buffer("int8_weight_nk", int8_weight_nk, persistent=False)
        self.register_buffer("channel_scales", channel_scales, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = self.channel_scales
        if scales.dtype != x.dtype:
            scales = scales.to(dtype=x.dtype)
        return torch.ops.aten._weight_int8pack_mm(x, self.int8_weight_nk, scales)


class TorchInt8QuantLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.TORCH_INT8]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Keep auto-selection unchanged; this kernel is enabled via explicit backend selection.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "torch_int8"

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
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH_INT8),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self.linear_mode = None  # lazily initialized to inference mode
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        self.int8_op: Optional[Int8PackedModule] = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_int8_mm_op():
            return False, ImportError("aten::_weight_int8pack_mm is unavailable in this PyTorch build.")
        return True, None

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return
        super().optimize()

    def _build_ret_idx(self) -> torch.Tensor:
        existing = getattr(self, "ret_idx", None)
        total = self.g_idx.shape[0]
        if isinstance(existing, torch.Tensor) and existing.numel() == total:
            return existing

        device = self.g_idx.device
        ret_idx = torch.zeros(total, dtype=torch.int32, device=device)
        group_size = max(int(self.group_size), 1)
        groups = total // group_size
        remainder = total % group_size
        g_idx = self.g_idx.to(torch.int32)
        g_idx_2 = g_idx * group_size

        if remainder > 0:
            mask = g_idx == groups
            if mask.any():
                g_idx_2[mask] += torch.arange(remainder, device=device, dtype=torch.int32)

        if groups > 0:
            base = torch.arange(group_size, device=device, dtype=torch.int32)
            for i in range(groups):
                mask = g_idx == i
                if not mask.any():
                    continue
                count = int(mask.sum().item())
                g_idx_2[mask] += base[:count]

        ret_idx[g_idx_2] = torch.arange(total, device=device, dtype=torch.int32)
        self.ret_idx = ret_idx
        return ret_idx

    def transform_cpu(self, dtype: torch.dtype):
        # [K, N] from GPTQ int4 tensors.
        float_weight = self.dequantize_weight(num_itr=1).to(torch.float32)
        int8_weight_kn, channel_scale = _requantize_to_int8(float_weight)

        int8_weight_nk = int8_weight_kn.t().contiguous()
        channel_scale = channel_scale.to(dtype=dtype).contiguous()

        if "int8_weight_nk" not in self._buffers:
            self.register_buffer("int8_weight_nk", int8_weight_nk, persistent=False)
        else:
            self.int8_weight_nk = int8_weight_nk

        if "int8_channel_scale" not in self._buffers:
            self.register_buffer("int8_channel_scale", channel_scale, persistent=False)
        else:
            self.int8_channel_scale = channel_scale

    def transform(self, dtype: torch.dtype, device: str):
        if device != "cpu":
            raise NotImplementedError("TorchInt8QuantLinear only supports CPU.")
        self.transform_cpu(dtype)

    def forward(self, x: torch.Tensor):
        if self.training:
            raise NotImplementedError("TorchInt8QuantLinear does not support training mode.")

        # Common decode path is 2D [M, K]. Skip reshape/out-shape overhead on this hot path.
        if x.dim() == 2:
            if self.linear_mode is None:
                if not TORCH_HAS_FUSED_OPS:
                    raise RuntimeError("TorchInt8QuantLinear requires torch fused CPU int8 ops.")
                self.transform(x.dtype, x.device.type)
                self.linear_mode = LinearMode.INFERENCE
                if x.device.type == "cpu":
                    self.int8_op = Int8PackedModule(self.int8_weight_nk, self.int8_channel_scale).eval()

            if self.linear_mode != LinearMode.INFERENCE:
                raise RuntimeError("TorchInt8QuantLinear failed to initialize inference mode.")

            out = self._fused_op_forward(x)
            if self.bias is not None:
                out.add_(self.bias)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        if self.linear_mode is None:
            if not TORCH_HAS_FUSED_OPS:
                raise RuntimeError("TorchInt8QuantLinear requires torch fused CPU int8 ops.")
            self.transform(x.dtype, x.device.type)
            self.linear_mode = LinearMode.INFERENCE
            if x.device.type == "cpu":
                self.int8_op = Int8PackedModule(self.int8_weight_nk, self.int8_channel_scale).eval()

        if self.linear_mode != LinearMode.INFERENCE:
            raise RuntimeError("TorchInt8QuantLinear failed to initialize inference mode.")

        out = self._fused_op_forward(x).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise NotImplementedError("TorchInt8QuantLinear fused path is CPU-only.")
        if self.int8_op is None:
            raise RuntimeError("TorchInt8QuantLinear int8 op is not initialized.")
        return self.int8_op(x.contiguous())

    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None


def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchInt8QuantLinear):
            raise ValueError(
                "Only models loaded using TorchInt8QuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH_INT8"
            )

        if isinstance(module, TorchInt8QuantLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchInt8QuantLinear", "dequantize_model"]
