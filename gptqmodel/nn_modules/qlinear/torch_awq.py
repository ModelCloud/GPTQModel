# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear


log = setup_logger()


class AwqTorchLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 10}
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

    REQUIRES_FORMAT_V2 = False

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
        self.compute_dtype = kwargs.get("dtype") or torch.float16
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        if register_buffers:
            if self.scales is not None and self.scales.dtype != self.compute_dtype:
                self.scales = self.scales.to(dtype=self.compute_dtype)
            if self.bias is not None and self.bias.dtype != self.compute_dtype:
                self.bias = self.bias.to(dtype=self.compute_dtype)

    def post_init(self):
        if self.scales is not None and self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(dtype=torch.float16)
        if self.bias is not None and self.bias.dtype not in (torch.float16, torch.bfloat16):
            self.bias = self.bias.to(dtype=torch.float16)
        super().post_init()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        del g_idx
        assert scales is not None and zeros is not None

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales

        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        self.register_buffer("scales", scales.clone().to(scale_dtype))
        if linear.bias is not None:
            bias_dtype = linear.bias.dtype if linear.bias.dtype in (torch.float16, torch.bfloat16) else scale_dtype
            self.register_buffer("bias", linear.bias.clone().to(bias_dtype))
        else:
            self.bias = None

        pack_num = 32 // self.bits

        intweight = []
        for idx in range(self.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // self.group_size])
                    / self.scales[idx // self.group_size]
                ).to(torch.int32)[:, None]
            )
        intweight = torch.cat(intweight, dim=1).t().contiguous()

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * self.bits),
            dtype=torch.int32,
            device=intweight.device,
        )
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits),
            dtype=torch.int32,
            device=zeros.device,
        )

        if self.bits != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]

        for col in range(intweight.shape[1] // pack_num):
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * self.bits)

        for col in range(zeros.shape[1] // pack_num):
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]].to(torch.int32)
                qzeros[:, col] |= qzero_col << (i * self.bits)

        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)

    def _ensure_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.scales.device != device or self.scales.dtype != dtype or not self.scales.is_contiguous():
            self.scales = self.scales.to(device=device, dtype=dtype).contiguous()
        if self.bias is not None and (
            self.bias.device != device or self.bias.dtype != dtype or not self.bias.is_contiguous()
        ):
            self.bias = self.bias.to(device=device, dtype=dtype).contiguous()

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or x_flat.device != device:
            x_flat = x_flat.to(device=device, dtype=compute_dtype)
        elif not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        self._ensure_runtime_dtype(device=device, dtype=compute_dtype)

        weight = dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        )
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise AssertionError(f"weight {weight.dtype} is not float16 or bfloat16")
        if weight.dtype != compute_dtype or weight.device != device or not weight.is_contiguous():
            weight = weight.to(device=device, dtype=compute_dtype).contiguous()

        output = torch.matmul(x_flat, weight)

        if self.bias is not None:
            output = output + self.bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        if output.dtype != input_dtype:
            output = output.to(dtype=input_dtype)

        output = output.reshape(original_shape)

        return output

__all__ = ["AwqTorchLinear"]
