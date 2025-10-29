# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.utils.module import try_import
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width

__all__ = ["AwqGEMMQuantLinear"]

awq_ext, msg = try_import("gptqmodel_awq_kernels")

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _awq_reverse_indices(device: torch.device) -> torch.Tensor:
    return torch.tensor(AWQ_REVERSE_ORDER, dtype=torch.long, device=device)


def _qweight_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Mirror MIT TinyChat offline-weight-repacker qweight_unpack with AWQ ordering fix."""
    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)
    mask = torch.tensor(0xF, dtype=torch.int32, device=packed.device)
    offsets = torch.arange(8, device=packed.device, dtype=torch.int32)
    unpacked = (packed.unsqueeze(-1) >> (offsets * 4)) & mask
    unpacked = unpacked.index_select(-1, _awq_reverse_indices(packed.device))
    return unpacked.reshape(packed.shape[0], packed.shape[1] * 8)


def _packing_v2_from_unpacked(unpacked: torch.Tensor, interleave: int, kstride: int) -> torch.Tensor:
    """Port of TinyChat packing_v2_from_unpacked implemented with torch ops."""
    rows, cols = unpacked.shape
    kernel = unpacked.view(rows, cols // 32, 32)
    kernel = kernel.view(rows, cols // 32, 4, 4, 2).permute(0, 1, 3, 2, 4).reshape(rows, cols // 32, 32)
    kernel = kernel.view(rows, cols // 32, 4, 8)
    kernel = kernel.view(rows, cols // 32, 4, 4, 2).permute(0, 1, 2, 4, 3).reshape(rows, cols)
    kernel = kernel.view(rows // interleave, interleave, cols // kstride, kstride)
    kernel = kernel.permute(0, 2, 1, 3).reshape(rows // interleave, cols // kstride, kstride, interleave)
    kernel = (
        kernel[..., 0]
        | (kernel[..., 1] << 4)
        | (kernel[..., 2] << 8)
        | (kernel[..., 3] << 12)
    )
    return kernel.reshape(rows // interleave, cols).to(torch.int16)


def _multiply_scale_qzero_negative(scales: torch.Tensor, qzeros: torch.Tensor, zp_shift: int = 0) -> torch.Tensor:
    """Replica of TinyChat multiply_scale_qzero_negative for AWQ layouts."""
    zeros = _qweight_unpack(qzeros.to(torch.int32)).to(dtype=scales.dtype)
    scaled = scales * zeros
    if zp_shift:
        scaled = scaled + zp_shift * scales
    return -scaled


class AwqGEMMQuantLinear(AWQuantLinear):
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
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    QUANT_TYPE = "awq_gemm"
    INTERLEAVE = 4

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int16,
        adapter: Adapter = None,
        register_buffers: bool = False,
        **kwargs,
    ) -> None:
        backend = kwargs.pop("backend", BACKEND.GEMM)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=backend,
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )
        self.interleave = self.INTERLEAVE
        self.split_k_iters = 8
        self.bias = None
        if register_buffers:
            self._init_buffers()

    def _init_buffers(self) -> None:
        int16_pack = 16 // self.bits
        pack_num = 32 // self.bits
        zeros_width = calculate_zeros_width(self.in_features, self.group_size, pack_num=pack_num)
        self.register_buffer("qweight", torch.zeros((self.out_features // self.INTERLEAVE, self.in_features // int16_pack * self.INTERLEAVE), dtype=torch.int16))
        self.register_buffer("qzeros", torch.zeros((zeros_width * pack_num, self.out_features), dtype=torch.float16))
        self.register_buffer("scales", torch.zeros((zeros_width * pack_num, self.out_features), dtype=torch.float16))
        if self.bias is not None:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))

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

        unpacked = _qweight_unpack(self._legacy_qweight)
        if unpacked.shape[0] == self.in_features and unpacked.shape[1] == self.out_features:
            unpacked = unpacked.transpose(0, 1).contiguous()
        elif unpacked.shape[0] != self.out_features or unpacked.shape[1] != self.in_features:
            raise ValueError(
                f"Unexpected legacy qweight shape {tuple(unpacked.shape)}; "
                f"expected ({self.out_features}, {self.in_features}) post-unpack."
            )
        qweight_v2 = _packing_v2_from_unpacked(unpacked, self.INTERLEAVE, 64).contiguous()
        scaled_zeros = _multiply_scale_qzero_negative(scales.to(device), qzeros.to(device), zp_shift=0)
        pack_num = 32 // self.bits
        zeros_width = calculate_zeros_width(self.in_features, self.group_size, pack_num=pack_num)
        scales_processed = torch.zeros((zeros_width * pack_num, self.out_features), dtype=scales.dtype, device=device)
        scales_processed[: scales.shape[0], :] = scales.to(device)
        zeros_processed = torch.zeros_like(scales_processed)
        zeros_processed[: scaled_zeros.shape[0], :] = scaled_zeros.to(scales.dtype)
        self.register_buffer("qweight", qweight_v2)
        self.register_buffer("scales", scales_processed)
        self.register_buffer("qzeros", zeros_processed)
        if bias is not None:
            self.register_buffer("bias", bias.to(device=device, dtype=scales.dtype))
        else:
            self.bias = None
        self.pack_dtype = torch.int16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if awq_ext is None:
            raise ModuleNotFoundError("External AWQ kernels are not properly installed." + msg)
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"AWQ GEMM kernels support float16/bfloat16 inputs only. Got {x.dtype}.")
        if self.scales.dtype != x.dtype:
            self.scales = self.scales.to(dtype=x.dtype)
        if self.qzeros.dtype != x.dtype:
            self.qzeros = self.qzeros.to(dtype=x.dtype)
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias = self.bias.to(dtype=x.dtype)
        num_tokens = x.numel() // x.shape[-1]
        weight = dequantize_gemm(
            self._legacy_qweight,
            self._legacy_qzeros,
            self._legacy_scales,
            self.bits,
            self.group_size,
        ).to(device=x.device, dtype=x.dtype)
        out = torch.matmul(x, weight)
        if self.bias is not None:
            out = out + self.bias
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)
        return out
