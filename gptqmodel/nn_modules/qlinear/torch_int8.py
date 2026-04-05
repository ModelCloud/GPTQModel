# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.


from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND


INT8_WEIGHT_BUFFER_NAME = "int8_weight_nk"
INT8_SCALE_BUFFER_NAME = "int8_channel_scale"


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


def _cached_int8_dequantize(module: nn.Module) -> Optional[torch.Tensor]:
    int8_weight = getattr(module, INT8_WEIGHT_BUFFER_NAME, None)
    int8_scale = getattr(module, INT8_SCALE_BUFFER_NAME, None)
    if int8_weight is None or int8_scale is None:
        return None

    weight_kn = int8_weight.t().to(torch.float32)
    scales = int8_scale.to(torch.float32)
    return weight_kn * scales.unsqueeze(0)


def _write_int8_buffers(module: nn.Module, float_weight: torch.Tensor, dtype: torch.dtype) -> None:
    int8_weight_kn, channel_scale = _requantize_to_int8(float_weight.to(torch.float32))

    int8_weight_nk = int8_weight_kn.t().contiguous()
    channel_scale = channel_scale.to(dtype=dtype).contiguous()

    if INT8_WEIGHT_BUFFER_NAME not in module._buffers:
        module.register_buffer(INT8_WEIGHT_BUFFER_NAME, int8_weight_nk, persistent=False)
    else:
        module.int8_weight_nk = int8_weight_nk

    if INT8_SCALE_BUFFER_NAME not in module._buffers:
        module.register_buffer(INT8_SCALE_BUFFER_NAME, channel_scale, persistent=False)
    else:
        module.int8_channel_scale = channel_scale


class Int8PackedModule(torch.nn.Module):
    """CPU fused int8 matmul wrapper around aten::_weight_int8pack_mm."""

    def __init__(self, int8_weight_nk: torch.Tensor, channel_scales: torch.Tensor):
        super().__init__()
        self.register_buffer(INT8_WEIGHT_BUFFER_NAME, int8_weight_nk, persistent=False)
        self.register_buffer(INT8_SCALE_BUFFER_NAME, channel_scales, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_channel_scale.dtype != x.dtype:
            self.int8_channel_scale = self.int8_channel_scale.to(dtype=x.dtype)
        return torch.ops.aten._weight_int8pack_mm(x, self.int8_weight_nk, self.int8_channel_scale)


class TorchInt8Linear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_INT8]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Keep auto-selection unchanged; this kernel is enabled via explicit backend selection.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [2, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "torch_int8"
    GPTQ_BUFFER_NAMES = ("qzeros", "qweight", "g_idx", "scales")
    UNPACK_BUFFER_NAMES = ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")

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
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH_INT8),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        self.int8_module: Optional[Int8PackedModule] = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_int8_mm_op():
            return False, ImportError("aten::_weight_int8pack_mm is unavailable in this PyTorch build.")
        return True, None

    def post_init(self):
        super().post_init()
        # One-time conversion: GPTQ packed storage (2/4/8-bit) -> int8 packed CPU kernel storage.
        # Keep only int8 tensors after conversion to reduce memory footprint.
        self.transform_cpu(dtype=torch.float32)
        self._empty_gptq_only_weights()
        self._drop_unpack_buffers()
        self.int8_module = Int8PackedModule(self.int8_weight_nk, self.int8_channel_scale).eval()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return
        super().optimize()

    def _has_all_gptq_buffers(self) -> bool:
        return all(getattr(self, name, None) is not None for name in self.GPTQ_BUFFER_NAMES)

    def _delete_attr_if_exists(self, attr_name: str):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    def _ensure_unpack_buffers(self):
        if (
            hasattr(self, "wf_unsqueeze_zero")
            and hasattr(self, "wf_unsqueeze_neg_one")
            and self.wf_unsqueeze_zero is not None
            and self.wf_unsqueeze_neg_one is not None
        ):
            return

        if self.bits not in [2, 4, 8]:
            raise NotImplementedError("TorchInt8Linear unpack only supports bits in [2, 4, 8].")

        wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=torch.int32).unsqueeze(0)
        device = self.qweight.device
        wf_zero = wf.unsqueeze(0).to(device=device)
        wf_neg_one = wf.unsqueeze(-1).to(device=device)

        if "wf_unsqueeze_zero" not in self._buffers:
            self.register_buffer("wf_unsqueeze_zero", wf_zero, persistent=False)
        else:
            self.wf_unsqueeze_zero = wf_zero

        if "wf_unsqueeze_neg_one" not in self._buffers:
            self.register_buffer("wf_unsqueeze_neg_one", wf_neg_one, persistent=False)
        else:
            self.wf_unsqueeze_neg_one = wf_neg_one

    def _drop_unpack_buffers(self):
        for name in self.UNPACK_BUFFER_NAMES:
            self._delete_attr_if_exists(name)

    def dequantize_weight(self, num_itr: int = 1):
        # Int8 fallback for dequantized export path after original GPTQ tensors are released.
        dequantized = _cached_int8_dequantize(self)
        if dequantized is not None and not self._has_all_gptq_buffers():
            return dequantized

        if num_itr != 1:
            raise NotImplementedError("TorchInt8Linear dequantize_weight only supports num_itr == 1.")

        if not self._has_all_gptq_buffers():
            raise RuntimeError("TorchInt8Linear missing GPTQ buffers for dequantization.")

        self._ensure_unpack_buffers()

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        return self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])

    @torch.inference_mode()
    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        block_in: int = 8192,
        workers: int = 1,
    ):
        raise NotImplementedError(
            "TorchInt8Linear is not packable. Load GPTQ int4 tensors and let post_init() convert to int8."
        )

    def pack(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        block_in: int = 8192,
        workers: int = 1,
    ):
        raise NotImplementedError(
            "TorchInt8Linear is not packable. Load GPTQ int4 tensors and let post_init() convert to int8."
        )

    def transform_cpu(self, dtype: torch.dtype):
        # [K, N] from GPTQ packed tensors (2/4/8-bit).
        float_weight = self.dequantize_weight(num_itr=1).to(torch.float32)
        _write_int8_buffers(self, float_weight=float_weight, dtype=dtype)

    def transform(self, dtype: torch.dtype, device: str):
        if device != "cpu":
            raise NotImplementedError("TorchInt8Linear only supports CPU.")
        self.transform_cpu(dtype)

    def forward(self, x: torch.Tensor):
        if self.training:
            raise NotImplementedError("TorchInt8Linear does not support training mode.")
        if self.int8_module is None:
            raise RuntimeError("TorchInt8Linear int8 module is not initialized. Ensure post_init() has been called.")

        # Common decode path is 2D [M, K]. Skip reshape/out-shape overhead on this hot path.
        if x.dim() == 2:
            out = self._fused_op_forward(x)
            if self.bias is not None:
                out.add_(self.bias)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        out = self._fused_op_forward(x).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise NotImplementedError("TorchInt8Linear fused path is CPU-only.")
        if self.int8_module is None:
            raise RuntimeError("TorchInt8Linear int8 module is not initialized.")
        return self.int8_module(x.contiguous())

    def _empty_gptq_only_weights(self):
        for name in self.GPTQ_BUFFER_NAMES:
            self._delete_attr_if_exists(name)


def dequantize_model(model: PreTrainedModel):
    from .torch_int8_awq import TorchInt8AwqLinear

    supported_int8_qlinears = (TorchInt8Linear, TorchInt8AwqLinear)

    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, supported_int8_qlinears):
            raise ValueError(
                "Only models loaded using TorchInt8Linear or TorchInt8AwqLinear are supported "
                "for dequantization. Please load model using backend=BACKEND.GPTQ_TORCH_INT8 or "
                "backend=BACKEND.AWQ_TORCH_INT8"
            )

        if isinstance(module, supported_int8_qlinears):
            new_module = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            if module.bias is not None:
                new_module.bias = torch.nn.Parameter(module.bias.detach().to("cpu", torch.float16))

            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = [
    "INT8_SCALE_BUFFER_NAME",
    "INT8_WEIGHT_BUFFER_NAME",
    "Int8PackedModule",
    "TorchInt8Linear",
    "_cached_int8_dequantize",
    "_has_int8_mm_op",
    "_requantize_to_int8",
    "_write_int8_buffers",
    "dequantize_model",
]
