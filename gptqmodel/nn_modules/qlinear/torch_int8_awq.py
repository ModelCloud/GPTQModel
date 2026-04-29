# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from .torch_int8 import (
    Int8PackedModule,
    _cached_int8_dequantize,
    _has_int8_mm_op,
    _write_int8_buffers,
)


class TorchInt8AwqLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_INT8]
    SUPPORTS_METHODS = [METHOD.AWQ]
    # Keep auto-selection unchanged; this kernel is enabled via explicit backend selection.
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "torch_int8_awq"
    AWQ_BUFFER_NAMES = ("qzeros", "qweight", "scales")

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
            backend=kwargs.pop("backend", BACKEND.AWQ_TORCH_INT8),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )
        self.int8_module: Optional[Int8PackedModule] = None

        if register_buffers:
            pack_cols = max(1, self.out_features // self.pack_factor)
            group_rows = max(1, math.ceil(self.in_features / max(int(self.group_size), 1)))

            self.register_buffer(
                "qweight",
                torch.zeros((self.in_features, pack_cols), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros((group_rows, pack_cols), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "scales",
                torch.zeros((group_rows, self.out_features), dtype=torch.float16),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_int8_mm_op():
            return False, ImportError("aten::_weight_int8pack_mm is unavailable in this PyTorch build.")
        return True, None

    def post_init(self):
        super().post_init()
        # One-time conversion: AWQ packed storage -> float -> int8 packed CPU kernel storage.
        # Keep only int8 tensors after conversion to reduce memory footprint.
        self.transform_cpu(dtype=torch.float32)
        self._empty_awq_only_weights()
        self.int8_module = Int8PackedModule(self.int8_weight_nk, self.int8_channel_scale).eval()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return
        super().optimize()

    def _has_all_awq_buffers(self) -> bool:
        return all(getattr(self, name, None) is not None for name in self.AWQ_BUFFER_NAMES)

    def _delete_attr_if_exists(self, attr_name: str):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    def _empty_awq_only_weights(self):
        for name in self.AWQ_BUFFER_NAMES:
            self._delete_attr_if_exists(name)

    def dequantize_weight(self):
        # Int8 fallback for dequantized export path after original AWQ tensors are released.
        dequantized = _cached_int8_dequantize(self)
        if dequantized is not None and not self._has_all_awq_buffers():
            return dequantized

        if not self._has_all_awq_buffers():
            raise RuntimeError("TorchInt8AwqLinear missing AWQ buffers for dequantization.")

        return dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        ).to(torch.float32)

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
            "TorchInt8AwqLinear is not packable. Load AWQ int4 tensors and let post_init() convert to int8."
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
            "TorchInt8AwqLinear is not packable. Load AWQ int4 tensors and let post_init() convert to int8."
        )

    def transform_cpu(self, dtype: torch.dtype):
        float_weight = self.dequantize_weight().to(torch.float32)
        _write_int8_buffers(self, float_weight=float_weight, dtype=dtype)

    def transform(self, dtype: torch.dtype, device: str):
        if device != "cpu":
            raise NotImplementedError("TorchInt8AwqLinear only supports CPU.")
        self.transform_cpu(dtype)

    def forward(self, x: torch.Tensor):
        if self.training:
            raise NotImplementedError("TorchInt8AwqLinear does not support training mode.")
        if self.int8_module is None:
            raise RuntimeError(
                "TorchInt8AwqLinear int8 module is not initialized. Ensure post_init() has been called."
            )

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
            raise NotImplementedError("TorchInt8AwqLinear fused path is CPU-only.")
        if self.int8_module is None:
            raise RuntimeError("TorchInt8AwqLinear int8 module is not initialized.")
        return self.int8_module(x.contiguous())


__all__ = ["TorchInt8AwqLinear"]
