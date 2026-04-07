# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq
from ...utils.backend import BACKEND
from .bitblas import (
    BITBLAS_OPTIMIZE_FEATURES,
    BITBLAS_PROPAGATE_WEIGHTS,
    BitblasBaseQuantLinear,
    BitblasQuantizationConfig,
)


class AWQBitBlasKernel(BitblasBaseQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_BITBLAS]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0, FORMAT.BITBLAS: 30}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False, True]
    SUPPORTS_SYM = [False, True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [16]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    QUANT_TYPE = "awq_bitblas"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        dtype: torch.dtype = torch.float16,
        adapter: Adapter = None,
        enable_tuning: bool = False,
        fast_decoding: bool = True,
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, list[int]] = BITBLAS_OPTIMIZE_FEATURES,
        layout: str = "nt",
        register_buffers: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            dtype=dtype,
            adapter=adapter,
            enable_tuning=enable_tuning,
            fast_decoding=fast_decoding,
            propagate_b=propagate_b,
            opt_features=opt_features,
            layout=layout,
            register_buffers=register_buffers,
            backend=kwargs.pop("backend", BACKEND.AWQ_BITBLAS),
            **kwargs,
        )

    def _build_quant_config(
        self,
        *,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        dtype: torch.dtype,
    ) -> BitblasQuantizationConfig:
        del sym

        # AWQ stores unsigned weight codes plus zero-points, even for symmetric checkpoints.
        return BitblasQuantizationConfig(
            weight_bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            is_sym=False,
            torch_dtype=dtype,
            quant_method="awq",
        )

    @torch.inference_mode()
    def pack(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        del g_idx

        if scales is None or zeros is None:
            raise ValueError("AWQBitBlasKernel.pack requires both scales and zeros.")

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()

        weight = linear.weight.detach().contiguous()
        group_idx = torch.arange(self.in_features, device=weight.device, dtype=torch.int64) // self.group_size
        maxq = (1 << self.bits) - 1

        # Broadcast per-group affine parameters across K so we can recover the stored AWQ integer codes.
        scale_zeros = (zeros * scales).index_select(0, group_idx).t()
        scales_by_input = scales.index_select(0, group_idx).t()
        intweight = torch.round((weight + scale_zeros) / scales_by_input).clamp_(0, maxq).to(torch.int8)

        self._load_bitblas_quant_state(
            intweight_out_in=intweight,
            scales_out_group=scales.t().contiguous(),
            intzeros_group_out=torch.round(zeros).to(torch.int8).contiguous(),
            bias=linear.bias.detach() if linear.bias is not None else None,
        )

    @torch.inference_mode()
    def repack_from_awq(self, awq_module) -> None:
        qzeros = getattr(awq_module, "qzeros", None)
        if qzeros is None:
            raise ValueError("AWQBitBlasKernel requires qzeros to repack AWQ checkpoints.")

        intweight, intzeros = unpack_awq(
            awq_module.qweight.detach(),
            qzeros.detach(),
            self.bits,
        )
        intweight, intzeros = reverse_awq_order(intweight, intzeros, self.bits)

        maxq = (1 << self.bits) - 1
        intweight = torch.bitwise_and(intweight, maxq).to(torch.int8).t().contiguous()
        intzeros = torch.bitwise_and(intzeros, maxq).to(torch.int8).contiguous()

        self._load_bitblas_quant_state(
            intweight_out_in=intweight,
            scales_out_group=awq_module.scales.detach().t().contiguous(),
            intzeros_group_out=intzeros,
            bias=awq_module.bias.detach() if getattr(awq_module, "bias", None) is not None else None,
        )


AwqBitBLASLinear = AWQBitBlasKernel

__all__ = ["AWQBitBlasKernel", "AwqBitBLASLinear"]
