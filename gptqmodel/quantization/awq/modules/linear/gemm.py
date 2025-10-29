# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn

from gptqmodel.quantization.awq.utils.module import try_import
from .gemv_fast import calculate_zeros_width, pack_intweight


awq_ext, msg = try_import("gptqmodel_awq_kernels")


class WQLinear_GEMM(nn.Module):
    """AWQ GEMM kernel wrapper for the v2 layout."""

    INTERLEAVE = 4

    def __init__(
        self,
        w_bit: int,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool,
        dev: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()

        if w_bit != 4:
            raise NotImplementedError("Only 4-bit AWQ kernels are supported.")

        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0
        assert self.out_features % self.INTERLEAVE == 0

        int16_pack = 16 // self.w_bit
        pack_num = 32 // self.w_bit
        zeros_width = calculate_zeros_width(self.in_features, self.group_size, pack_num=pack_num)

        self.register_buffer(
            "qweight",
            torch.zeros(
                (self.out_features // self.INTERLEAVE, self.in_features // int16_pack * self.INTERLEAVE),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (zeros_width * pack_num, self.out_features),
                dtype=dtype,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (zeros_width * pack_num, self.out_features),
                dtype=dtype,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=dtype, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        w_bit: int,
        group_size: int,
        init_only: bool = False,
        scales: torch.Tensor | None = None,
        zeros: torch.Tensor | None = None,
    ) -> "WQLinear_GEMM":
        awq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dev=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if init_only:
            return awq_linear

        if scales is None or zeros is None:
            raise ValueError("`scales` and `zeros` tensors are required to build AWQ kernels.")

        dtype = scales.dtype
        pack_num = 32 // w_bit
        zeros_width = calculate_zeros_width(linear.in_features, group_size, pack_num=pack_num)

        qscales = torch.zeros(
            (scales.shape[0], zeros_width * pack_num),
            dtype=dtype,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        awq_linear.scales = qscales.transpose(1, 0).contiguous()

        if linear.bias is not None:
            awq_linear.bias = linear.bias.detach().clone().to(dtype=dtype)

        scale_zeros = zeros * scales

        columns = []
        for idx in range(awq_linear.in_features):
            group_idx = idx // awq_linear.group_size
            col = torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, group_idx]) / qscales[:, group_idx]
            ).to(torch.int32)
            columns.append(col.unsqueeze(1))
        intweight = torch.cat(columns, dim=1)
        awq_linear.qweight = pack_intweight(intweight.contiguous(), interleave=cls.INTERLEAVE, kstride=64)

        zeros_float = zeros.to(dtype=torch.float32)
        scaled_zeros = torch.zeros_like(qscales, dtype=dtype, device=qscales.device)
        scaled_zeros[:, : scales.shape[1]] = -(qscales[:, : scales.shape[1]] * zeros_float[:, : scales.shape[1]])
        awq_linear.qzeros = scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if awq_ext is None:
            raise ModuleNotFoundError("External AWQ kernels are not properly installed." + msg)

        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"AWQ kernels require float16/bfloat16 inputs; received {x.dtype}.")

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
            bias = self.bias.to(dtype=out.dtype, device=out.device)
            out = out + bias

        return out
