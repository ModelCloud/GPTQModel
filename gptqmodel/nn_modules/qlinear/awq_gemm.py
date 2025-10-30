# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.utils.mit_repacker import (
    multiply_scale_qzero_negative as mit_multiply_scale_qzero_negative,
    packing_v2_from_unpacked as mit_packing_v2_from_unpacked,
    qweight_unpack as mit_qweight_unpack,
)
from ...quantization.awq.utils.module import try_import
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width

__all__ = ["AwqGEMMQuantLinear"]

awq_ext, msg = try_import("gptqmodel_awq_kernels")

def _convert_awq_v1_to_v2(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = qweight.device
    unpacked = mit_qweight_unpack(qweight.to(device))
    if unpacked.shape == (in_features, out_features):
        unpacked = unpacked.transpose(0, 1).contiguous()
    elif unpacked.shape != (out_features, in_features):
        raise ValueError(
            f"Unexpected legacy qweight shape {tuple(unpacked.shape)}; expected "
            f"({out_features}, {in_features}) or ({in_features}, {out_features})."
        )
    qweight_v2 = mit_packing_v2_from_unpacked(unpacked, interleave, 64).contiguous()

    pack_num = 32 // bits
    zeros_width = calculate_zeros_width(in_features, group_size, pack_num=pack_num)

    groups = 1 if group_size in (-1, 0) else in_features // group_size
    scales_legacy = scales.to(device=device).contiguous()
    if scales_legacy.shape == (out_features, groups):
        scales_groups_first = scales_legacy.transpose(0, 1).contiguous()
    elif scales_legacy.shape == (groups, out_features):
        scales_groups_first = scales_legacy
    else:
        raise ValueError(
            f"Unexpected legacy scales shape {tuple(scales_legacy.shape)}; "
            f"expected ({out_features}, {groups}) or ({groups}, {out_features})."
        )

    qzeros_legacy = qzeros.to(device=device).contiguous()
    expected_zero_cols = out_features // pack_num
    if qzeros_legacy.shape == (out_features, expected_zero_cols):
        qzeros_groups_first = qzeros_legacy.transpose(0, 1).contiguous()
    elif qzeros_legacy.shape == (expected_zero_cols, out_features):
        qzeros_groups_first = qzeros_legacy.transpose(0, 1).contiguous()
    elif qzeros_legacy.shape == (groups, expected_zero_cols):
        qzeros_groups_first = qzeros_legacy
    else:
        raise ValueError(
            f"Unexpected legacy qzeros shape {tuple(qzeros_legacy.shape)}; "
            f"expected one of {{({out_features}, {expected_zero_cols}), ({expected_zero_cols}, {out_features}), ({groups}, {expected_zero_cols})}}."
        )

    scaled_zeros_groups_first = mit_multiply_scale_qzero_negative(
        scales_groups_first, qzeros_groups_first, zp_shift=0
    )

    padded_rows = zeros_width * pack_num
    scales_processed = torch.zeros(
        (padded_rows, out_features),
        dtype=scales_groups_first.dtype,
        device=device,
    )
    zeros_processed = torch.zeros_like(scales_processed)
    rows = min(padded_rows, scales_groups_first.shape[0])
    scales_processed[:rows, :] = scales_groups_first[:rows, :]
    zeros_processed[:rows, :] = scaled_zeros_groups_first[:rows, :]
    return qweight_v2, scales_processed, zeros_processed


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
        self.register_buffer(
            "qweight",
            torch.zeros(
                (self.out_features // self.INTERLEAVE, self.in_features // int16_pack * self.INTERLEAVE),
                dtype=torch.int16,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((self.in_features // self.group_size, self.out_features), dtype=torch.float16),
        )
        self.register_buffer(
            "scales",
            torch.zeros((self.in_features // self.group_size, self.out_features), dtype=torch.float16),
        )
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
        qweight_v2, scales_processed, zeros_processed = _convert_awq_v1_to_v2(
            qweight,
            qzeros,
            scales,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            interleave=self.INTERLEAVE,
        )
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
        use_fp32_accum = True
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
                use_fp32_accum,
            )
        if self.bias is not None:
            out = out + self.bias
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)
        return out
