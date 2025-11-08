# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import torch

from ...adapter.adapter import Adapter
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import TORCH_HAS_FUSED_OPS
from .torch_fused import Int4PackedOp, TorchFusedQuantLinear, pack_scales_and_zeros


log = setup_logger()


class TorchFusedAwqQuantLinear(TorchFusedQuantLinear):
    """Torch fused AWQ variant based on GPTQ fused kernels via CPU int4 packing."""

    QUANT_TYPE = "torch_fused_awq"
    SUPPORTS_BITS = TorchFusedQuantLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = TorchFusedQuantLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = TorchFusedQuantLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchFusedQuantLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchFusedQuantLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchFusedQuantLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchFusedQuantLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchFusedQuantLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchFusedQuantLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = TorchFusedQuantLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = TorchFusedQuantLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchFusedQuantLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchFusedQuantLinear.SUPPORTS_ADAPTERS

    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

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
        kwargs.setdefault("backend", BACKEND.TORCH_FUSED_AWQ)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )
        if register_buffers:
            # AWQ packs each input row into pack_factor-wide columns for int4 lanes.
            pack_cols = max(1, self.out_features // self.pack_factor)
            qweight_shape = (self.in_features, pack_cols)
            group_size = max(int(self.group_size), 1)
            # Each group holds group_size input rows; ceil ensures remaining rows are included.
            group_rows = max(1, math.ceil(self.in_features / group_size))

            self.register_buffer(
                "qweight",
                torch.zeros(qweight_shape, dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros((group_rows, pack_cols), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "scales",
                torch.zeros((group_rows, self.out_features), dtype=torch.float16),
            )
            g_idx = torch.arange(self.in_features, dtype=torch.int32) // group_size
            self.register_buffer("g_idx", g_idx)
            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    def transform_cpu_awq(self, dtype):
        src_scales = self.scales
        if src_scales.dtype != torch.float16:
            src_scales = src_scales.to(torch.float16)
        src_scales = src_scales.contiguous()

        # Unpack AWQ tensors
        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val)
        if izeros is not None:
            izeros = torch.bitwise_and(izeros, max_val)

        # Precompute the per-group zero offsets
        scale_fp16 = src_scales.clone()
        scale_fp32 = scale_fp16.to(torch.float32)
        zero_offset = 1 << (self.bits - 1)
        zeros_fp16 = (zero_offset - izeros.reshape_as(scale_fp32)).to(dtype=scale_fp32.dtype)
        zeros_fp16 = (zeros_fp16 * scale_fp32).to(torch.float16)

        # Repack AWQ-per-output rows into GPTQ-style per-input packs so the base
        # TorchFusedQuantLinear path can handle the conversion to int4pack.
        in_features, out_features = iweight.shape
        pack_factor = int(self.pack_factor)
        if in_features % pack_factor != 0:
            raise ValueError(
                f"AWQ in_features={in_features} must be divisible by pack_factor={pack_factor}."
            )

        rows = iweight.view(in_features // pack_factor, pack_factor, out_features)
        gptq_qweight = torch.zeros(
            (rows.shape[0], out_features),
            dtype=self.pack_dtype,
            device=iweight.device,
        )
        bit_shifts = list(range(0, pack_factor * self.bits, self.bits))
        for lane, shift in enumerate(bit_shifts):
            gptq_qweight |= rows[:, lane, :].to(torch.int32) << shift
        self.qweight = gptq_qweight.contiguous()

        # Reuse the GPTQ CPU transformation to convert into int4pack layout.
        super().transform_cpu(dtype)

        # Restore AWQ-specific scale/zero metadata for the fused op.
        self.scales = scale_fp16.to(dtype=dtype)
        self.qzeros = zeros_fp16.to(dtype=dtype)
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def awq_weight_dequantize(self, device, dtype):
        dense = dequantize_gemm(
            self.qweight,
            self.qzeros,
            self.scales,
            self.bits,
            self.group_size,
        ).to(device=device, dtype=torch.float32)
        return dense.to(device=device, dtype=dtype)

    def transform_cpu(self, dtype):
        self.transform_cpu_awq(dtype)

    def transform(self, dtype, device):
        if device != "cpu":
            raise NotImplementedError(
                "TorchFusedAwqQuantLinear only supports fused transforms on CPU devices."
            )
        self.transform_cpu(dtype)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        self.assert_supported_dtype(x_flat.dtype)
        if not self.training and not self.transformed and TORCH_HAS_FUSED_OPS:
            self.transform(x_flat.dtype, x_flat.device.type)
            self.transformed = True
            if x_flat.device.type == "cpu":
                self.torch_fused_op = Int4PackedOp(
                    self.qweight, self.scales_and_zeros, self.group_size
                ).eval()
                import torch._inductor.config as config
                config.freezing = True
                config.max_autotune = True

        if self.transformed:
            # log.debug("awq calling fused op")
            out = self._fused_op_forward(x_flat)
        else:
            # log.debug("awq dense path")
            weight = self.awq_weight_dequantize(device=x_flat.device, dtype=x_flat.dtype)
            out = torch.matmul(x_flat, weight)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out.reshape(out_shape)

    def assert_supported_dtype(self, dtype: torch.dtype):
        if dtype not in self.SUPPORTS_DTYPES:
            supported = ", ".join(str(d) for d in self.SUPPORTS_DTYPES)
            raise TypeError(
                f"{self.__class__.__name__} only supports input dtypes [{supported}], but received {dtype}."
            )


__all__ = ["TorchFusedAwqQuantLinear"]
