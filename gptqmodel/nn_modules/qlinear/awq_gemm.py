# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.modules.linear.gemv_fast import pack_intweight
from ...quantization.awq.modules.linear.gemv_fast import pack_intweight
from ...quantization.awq.utils.module import try_import
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.logger import setup_logger


log = setup_logger()


def _unpack_qweight_int4(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.int32:
        raise ValueError("Legacy qweight tensor must be int32 packed.")
    n, packed_cols = qweight.shape
    mask = 0xF
    unpacked = torch.zeros((n, packed_cols * 8), dtype=torch.int32, device=qweight.device)
    for offset in range(8):
        unpacked[:, offset::8] = (qweight >> (offset * 4)) & mask
    return unpacked


def _multiply_scale_qzero_negative(scales: torch.Tensor, qzeros: torch.Tensor, zp_shift: int = 0) -> torch.Tensor:
    pack_size = 8
    groups, out_features = scales.shape
    packed_cols = qzeros.shape[1]
    scaled = torch.zeros_like(scales, dtype=scales.dtype)
    mask = 0xF
    for zero_idx in range(packed_cols):
        packed = qzeros[:, zero_idx]
        for offset in range(pack_size):
            col = zero_idx * pack_size + offset
            if col >= out_features:
                break
            zero = (packed >> (offset * 4)) & mask
            scaled[:, col] = scales[:, col] * zero.to(scales.dtype)
    if zp_shift != 0:
        scaled = scaled + zp_shift * scales
    return -scaled


awq_ext, msg = try_import("gptqmodel_awq_kernels")


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

        if register_buffers:
            self._init_buffers()

    def _init_buffers(self) -> None:
        int16_pack = 16 // self.bits
        pack_num = 32 // self.bits
        zeros_width = calculate_zeros_width(self.in_features, self.group_size, pack_num=pack_num)

        self.register_buffer(
            "qweight",
            torch.zeros(
                (self.out_features // self.INTERLEAVE, self.in_features // int16_pack * self.INTERLEAVE),
                dtype=torch.int16,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (zeros_width * pack_num, self.out_features),
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (zeros_width * pack_num, self.out_features),
                dtype=torch.float16,
            ),
        )

        if self.bias is not None:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        self._convert_legacy_layout_if_needed()
        super().post_init()

    def _convert_legacy_layout_if_needed(self) -> None:
        if not hasattr(self, "qweight"):
            return
        if self.qweight.dtype == torch.int16:
            return
        if self.qweight.numel() == 0:
            return
        if self.qweight.shape[0] == self.out_features // self.INTERLEAVE:
            return

        log.info("AWQ GEMM: converting legacy layout to v2 format.")

        device = self.qweight.device
        pack_num = 32 // self.bits
        mask = (1 << self.bits) - 1
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]

        # Unpack weight integers (legacy layout: [in_features, out_features / pack])
        num_rows, num_cols = self.qweight.shape
        full_cols = num_cols * pack_num
        intweight = torch.zeros((num_rows, full_cols), dtype=torch.int32, device=device)

        for col in range(num_cols):
            packed = self.qweight[:, col]
            for idx, order in enumerate(order_map):
                target = col * pack_num + order
                if target >= self.out_features:
                    continue
                values = (packed >> (idx * self.bits)) & mask
                intweight[:, target] = values

        intweight = intweight[:, : self.out_features]
        intweight = intweight.t().contiguous()  # [out_features, in_features]
        self.qweight = pack_intweight(intweight, interleave=self.INTERLEAVE, kstride=64)

        # Prepare padded scales
        scales_old = self.scales
        dtype = scales_old.dtype
        zeros_width = calculate_zeros_width(self.in_features, self.group_size, pack_num=pack_num)

        scales_transposed = scales_old.t().contiguous()  # [out_features, groups]
        qscales = torch.zeros(
            (scales_transposed.shape[0], zeros_width * pack_num),
            dtype=dtype,
            device=scales_transposed.device,
        )
        qscales[:, : scales_transposed.shape[1]] = scales_transposed

        zeros_rows, zeros_cols = self.qzeros.shape
        zeros_unpacked = torch.zeros((zeros_rows, zeros_cols * pack_num), dtype=torch.float32, device=self.qzeros.device)
        for col in range(zeros_cols):
            packed = self.qzeros[:, col]
            for idx, order in enumerate(order_map):
                target = col * pack_num + order
                if target >= self.out_features:
                    continue
                values = (packed >> (idx * self.bits)) & mask
                zeros_unpacked[:, target] = values.to(torch.float32)

        zeros_unpacked = zeros_unpacked[:, : scales_old.shape[1]]
        zeros_transposed = zeros_unpacked.t().contiguous()  # [out_features, groups]

        scaled_zeros = torch.zeros_like(qscales, dtype=dtype, device=qscales.device)
        scaled_zeros[:, : zeros_transposed.shape[1]] = -(
            qscales[:, : zeros_transposed.shape[1]] * zeros_transposed.to(dtype)
        )

        self.scales = qscales.transpose(1, 0).contiguous()
        self.qzeros = scaled_zeros.transpose(1, 0).contiguous()
        self.qweight = self.qweight.to(device=device)
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

        if num_tokens < 8:
            out = awq_ext.gemv_forward_cuda(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
                num_tokens,
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_ext.gemm_forward_cuda(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
            )

        if self.bias is not None:
            out = out + self.bias

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

__all__ = ["AwqGEMMQuantLinear"]
