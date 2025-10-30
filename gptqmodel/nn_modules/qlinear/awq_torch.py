# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.awq.utils.mit_repacker import (
    multiply_scale_qzero_negative as mit_multiply_scale_qzero_negative,
    qweight_unpack as mit_qweight_unpack,
)
from ...quantization.awq.utils.module import try_import
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear
from .awq_gemm import _convert_awq_v1_to_v2


log = setup_logger()
awq_ext, msg = try_import("gptqmodel_awq_kernels")


class AwqTorchQuantLinear(AWQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]


    QUANT_TYPE = "awq_torch"

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
        register_buffers: bool = False,
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
            backend=kwargs.pop("backend", BACKEND.TORCH_AWQ),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        self._legacy_qweight: torch.Tensor | None = None
        self._legacy_qzeros: torch.Tensor | None = None
        self._legacy_scales: torch.Tensor | None = None

    def post_init(self):
        super().post_init()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )

    def load_legacy_tensors(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        device = qweight.device
        for name in ("qweight", "scales", "qzeros", "bias"):
            if hasattr(self, name):
                delattr(self, name)
            if name in getattr(self, "_buffers", {}):
                del self._buffers[name]

        self._legacy_qweight = qweight.to(device)
        self._legacy_qzeros = qzeros.to(device)
        self._legacy_scales = scales.to(device)

        qweight_v2, scales_processed, zeros_processed = _convert_awq_v1_to_v2(
            qweight,
            qzeros,
            scales,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            interleave=4,
        )
        self.register_buffer("qweight", qweight_v2)
        self.register_buffer("scales", scales_processed)
        self.register_buffer("qzeros", zeros_processed)
        if bias is not None:
            self.register_buffer("bias", bias.to(device=device, dtype=scales.dtype))
        else:
            self.bias = None
        self.pack_dtype = torch.int16

    def _dequantize_weight_fallback(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._legacy_qweight is None
            or self._legacy_qzeros is None
            or self._legacy_scales is None
        ):
            raise RuntimeError("Legacy AWQ tensors unavailable for Torch fallback.")

        qweight = self._legacy_qweight.to(device=device)
        qzeros = self._legacy_qzeros.to(device=device)
        scales = self._legacy_scales.to(device=device)

        unpacked = mit_qweight_unpack(qweight)
        if unpacked.shape == (self.in_features, self.out_features):
            unpacked = unpacked.transpose(0, 1).contiguous()
        elif unpacked.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Unexpected unpacked qweight shape {tuple(unpacked.shape)}; "
                f"expected ({self.out_features}, {self.in_features})"
            )
        unpacked = unpacked.to(torch.float32)

        groups = 1 if self.group_size in (-1, 0) else self.in_features // self.group_size
        scales_groups = scales
        if scales_groups.shape == (self.out_features, groups):
            scales_groups = scales_groups.transpose(0, 1).contiguous()
        elif scales_groups.shape != (groups, self.out_features):
            raise ValueError(
                f"Unexpected legacy scales shape {tuple(scales_groups.shape)}; "
                f"expected ({groups}, {self.out_features}) or ({self.out_features}, {groups})."
            )

        pack_num = 32 // self.bits
        expected_zero_cols = self.out_features // pack_num
        qzeros_groups = qzeros
        if qzeros_groups.shape == (self.out_features, expected_zero_cols):
            qzeros_groups = qzeros_groups.transpose(0, 1).contiguous()
        elif qzeros_groups.shape == (expected_zero_cols, self.out_features):
            qzeros_groups = qzeros_groups.transpose(0, 1).contiguous()
        elif qzeros_groups.shape != (groups, expected_zero_cols):
            raise ValueError(
                f"Unexpected legacy qzeros shape {tuple(qzeros_groups.shape)}; "
                f"expected one of {{({self.out_features}, {expected_zero_cols}), ({expected_zero_cols}, {self.out_features}), ({groups}, {expected_zero_cols})}}."
            )

        scaled_zeros_groups = mit_multiply_scale_qzero_negative(scales_groups, qzeros_groups, zp_shift=0)

        weight = torch.empty((self.out_features, self.in_features), dtype=torch.float32, device=device)
        for group_idx in range(groups):
            start = group_idx * self.group_size
            end = min(start + self.group_size, self.in_features)
            weight[:, start:end] = (
                unpacked[:, start:end] * scales_groups[group_idx].to(torch.float32).unsqueeze(1)
                + scaled_zeros_groups[group_idx].to(torch.float32).unsqueeze(1)
            )

        return weight.transpose(0, 1).contiguous().to(dtype=dtype)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])

        if awq_ext is not None and hasattr(awq_ext, "dequantize_weights_cuda"):
            weight = awq_ext.dequantize_weights_cuda(
                self.qweight,
                self.scales,
                self.qzeros,
                0,
                0,
                0,
                False,
            )
            weight = weight.to(dtype=x_flat.dtype, device=device)
            output = torch.matmul(x_flat, weight)
        else:
            matmul_weight = self._dequantize_weight_fallback(device=device, dtype=torch.float32)
            matmul_input = x_flat.to(torch.float32)
            output = torch.matmul(matmul_input, matmul_weight)
            output = output.to(dtype=x_flat.dtype)

        if self.bias is not None:
            if self.bias.dtype != output.dtype:
                self.bias = self.bias.to(dtype=output.dtype)
            output = output + self.bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        output = output.reshape(original_shape)

        return output

__all__ = ["AwqTorchQuantLinear"]
