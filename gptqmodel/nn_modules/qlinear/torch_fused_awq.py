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

    # inherit from torch fused
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
    REQUIRES_FORMAT_V2 = TorchFusedQuantLinear.REQUIRES_FORMAT_V2

    # AWQ kernels are only accuracy validate for float16 for now
    SUPPORTS_DTYPES = [torch.float16]

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
            # Skip base buffer init, we need to manually init buffers for awq
            register_buffers=False,
            **kwargs,
        )

        # Create awq buffers
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

            self.register_buffer("g_idx", torch.arange(self.in_features, dtype=torch.int32) // group_size)

            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    def prepare_awq_fused_tensors(self, need_zeros: bool = True):
        self.scales.to(torch.float16).contiguous()

        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val)
        if izeros is None:
            raise RuntimeError("AWQ fused kernel requires zero points.")
        izeros = torch.bitwise_and(izeros, max_val)

        if need_zeros:
            zero_offset = 1 << (self.bits - 1)
            zeros = (zero_offset - izeros.reshape_as(self.scales)) * self.scales

        gptq_qweight = self.pack_awq_qweight(iweight)
        gptq_qzeros = self.pack_awq_qzeros(izeros)
        return gptq_qweight, gptq_qzeros, self.scales, zeros if need_zeros else None

    def pack_awq_qweight(self, iweight: torch.Tensor) -> torch.Tensor:
        in_features, out_features = iweight.shape
        pack_factor = int(self.pack_factor)
        if in_features % pack_factor != 0:
            raise ValueError(
                f"AWQ in_features={in_features} must be divisible by pack_factor={pack_factor}."
            )
        rows = iweight.view(in_features // pack_factor, pack_factor, out_features)
        packed = torch.zeros(
            (rows.shape[0], out_features),
            dtype=self.pack_dtype,
            device=iweight.device,
        )
        shifts = range(0, pack_factor * self.bits, self.bits)
        for lane, shift in enumerate(shifts):
            packed |= rows[:, lane, :].to(torch.int32) << shift
        return packed.contiguous()

    def pack_awq_qzeros(self, izeros: torch.Tensor) -> torch.Tensor:
        pack_factor = int(self.pack_factor)
        if izeros.shape[1] % pack_factor != 0:
            raise ValueError(
                f"AWQ qzeros dimension {izeros.shape[1]} must be divisible by pack_factor={pack_factor}."
            )
        cols = izeros.view(izeros.shape[0], izeros.shape[1] // pack_factor, pack_factor)
        packed = torch.zeros(
            (cols.shape[0], cols.shape[1]),
            dtype=self.pack_dtype,
            device=izeros.device,
        )
        shifts = range(0, pack_factor * self.bits, self.bits)
        for lane, shift in enumerate(shifts):
            packed |= cols[:, :, lane].to(torch.int32) << shift
        return packed.contiguous()

    def transform_cpu_awq(self, dtype):
        self.qweight, self.qzeros, scales, zeros = self.prepare_awq_fused_tensors()

        super().transform_cpu(dtype, do_scales_and_zeros=False)

        self.scales = scales.to(device=self.qweight.device, dtype=dtype).contiguous()
        self.qzeros = zeros.to(device=self.qweight.device, dtype=dtype).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def transform_xpu_awq(self, dtype):
        self.qweight, self.qzeros, scales, _ = self.prepare_awq_fused_tensors(need_zeros=False)

        super().transform_xpu(dtype)

        self.scales = scales.to(device=self.qweight.device, dtype=dtype).contiguous()

    def transform_cpu(self, dtype):
        self.transform_cpu_awq(dtype)

    def awq_weight_dequantize(self, device, dtype):
        return dequantize_gemm(
            self.qweight,
            self.qzeros,
            self.scales,
            self.bits,
            self.group_size,
        ).to(device=device, dtype=dtype)

    def transform(self, dtype, device):
        if device == "cpu":
            self.transform_cpu(dtype)
        elif device == "xpu":
            self.transform_xpu_awq(dtype)
        else:
            raise NotImplementedError(
                "TorchFusedAwqQuantLinear only supports fused transforms on CPU or XPU devices."
            )

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
