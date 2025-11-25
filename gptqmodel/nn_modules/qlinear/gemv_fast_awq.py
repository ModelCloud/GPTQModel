# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.utils.module import try_import
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.logger import setup_logger


log = setup_logger()

awq_v2_ext, msg = try_import("gptqmodel_awq_v2_kernels")


def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class AwqGEMVFastQuantLinear(AWQuantLinear):
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

    SUPPORTS_DTYPES = [torch.float16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemv_fast"

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
        backend = kwargs.pop("backend", BACKEND.GEMV_FAST)
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
            **kwargs)

        self.split_k_iters = 8
        self.interleave = 4

        int32_pack_factor = 32 // self.bits

        self.bias = None

        if register_buffers:
            self.register_buffer(
                "qweight",
                torch.zeros((out_features // self.interleave, in_features // self.pack_factor * self.interleave), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    calculate_zeros_width(in_features, self.group_size) * int32_pack_factor,
                    out_features,
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    calculate_zeros_width(in_features, self.group_size) * int32_pack_factor,
                    out_features,
                    dtype=torch.float16,
                ),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        super().post_init()

    def forward(self, x: torch.Tensor):
        if awq_v2_ext is None:
            raise ModuleNotFoundError("External AWQ V2 kernels are not properly installed." + msg)

        inputs = x
        batch_size, n_tokens, _ = inputs.shape

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()

        if batch_size < 8 and n_tokens == 1:
            out = awq_v2_ext.gemv_forward_cuda_decode(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_v2_ext.gemm_forward_cuda_prefill(
                inputs, self.qweight, self.scales, self.qzeros
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // self.bits
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, self.group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        self.register_buffer("scales", qscales.transpose(1, 0).contiguous())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.clone().half())
        else:
            self.bias = None

        intweight = []
        for idx in range(self.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // self.group_size])
                    / qscales[:, idx // self.group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        self.register_buffer("qweight", pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        ))

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros_like(qscales)

        qzeros[:, : scales.shape[1]] = -(
                qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(torch.float16)
        self.register_buffer("qzeros", qzeros.transpose(1, 0).contiguous())

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, bits={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.bits,
                self.group_size,
            )
        )

__all__ = ["AwqGEMVFastQuantLinear"]
