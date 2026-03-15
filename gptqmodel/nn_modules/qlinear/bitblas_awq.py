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
from ...nn_modules.qlinear import BaseQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq
from ...utils.backend import BACKEND
from .bitblas import (
    BITBLAS_AVAILABLE,
    BITBLAS_INSTALL_HINT,
    BITBLAS_OPTIMIZE_FEATURES,
    BITBLAS_PROPAGATE_WEIGHTS,
    BitblasQuantLinear,
    BitblasQuantizationConfig,
    import_bitblas,
)


class AWQBitBlasKernel(BitblasQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.BITBLAS_AWQ]
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
        adapter: Adapter = None,
        enable_tuning: bool = False,
        fast_decoding: bool = True,
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, list[int]] = BITBLAS_OPTIMIZE_FEATURES,
        layout: str = "nt",
        register_buffers: bool = False,
        **kwargs,
    ) -> None:
        del fast_decoding
        del register_buffers

        BaseQuantLinear.__init__(
            self,
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.BITBLAS_AWQ),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        if not BITBLAS_AVAILABLE:
            raise ImportError(BITBLAS_INSTALL_HINT)

        # AWQ stores unsigned weight codes plus zero-points, even for symmetric checkpoints.
        self.quant_config = BitblasQuantizationConfig(
            weight_bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            is_sym=False,
            quant_method="awq",
        )
        self.enable_tuning = enable_tuning
        self.layout = layout
        self.opt_features = list(opt_features) if isinstance(opt_features, list) else [opt_features]
        self.propagate_b = propagate_b

        import_bitblas()

        self._validate_parameters(in_features, out_features)
        self._configure_bitblas_matmul(
            in_features,
            out_features,
            self.TORCH_DTYPE,
            enable_tuning,
            False,
            layout,
            bits,
        )
        self._initialize_buffers(in_features, out_features, bias)

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

        # Broadcast per-group affine parameters across K so we can recover the stored AWQ integer codes.
        scale_zeros = (zeros * scales).index_select(0, group_idx).t()
        scales_by_input = scales.index_select(0, group_idx).t()
        intweight = torch.round((weight + scale_zeros) / scales_by_input).clamp_(0, self.maxq).to(torch.int8)

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


AwqBitBLASQuantLinear = AWQBitBlasKernel

__all__ = ["AWQBitBlasKernel", "AwqBitBLASQuantLinear"]
