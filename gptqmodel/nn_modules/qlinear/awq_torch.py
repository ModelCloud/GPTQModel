# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.awq.utils.module import try_import
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear
from .awq_gemm import _convert_awq_v1_to_v2


log = setup_logger()
awq_ext, msg = try_import("gptqmodel_awq_kernels")


class AwqTorchQuantLinear(AWQuantLinear):
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
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH_AWQ),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        self._legacy_qweight: torch.Tensor | None = None
        self._legacy_qzeros: torch.Tensor | None = None
        self._legacy_scales: torch.Tensor | None = None

    def post_init(self):
        super().post_init()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )

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

        qweight_v2, scales_processed, zeros_processed = _convert_awq_v1_to_v2(
            qweight,
            qzeros,
            scales,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            interleave=4,
        )
        self.register_buffer("qweight", qweight_v2)
        self.register_buffer("scales", scales_processed)
        self.register_buffer("qzeros", zeros_processed)
        if bias is not None:
            self.register_buffer("bias", bias.to(device=device, dtype=scales.dtype))
        else:
            self.bias = None
        self.pack_dtype = torch.int16

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])

        if awq_ext is not None:
            weight = awq_ext.dequantize_weights_cuda(
                self.qweight,
                self.scales,
                self.qzeros,
                0,
                0,
                0,
                False,
            )
            weight = weight.to(dtype=x_flat.dtype, device=device)
        else:
            if (
                self._legacy_qweight is None
                or self._legacy_qzeros is None
                or self._legacy_scales is None
            ):
                raise RuntimeError("Legacy AWQ tensors unavailable for Torch fallback.")
            weight = dequantize_gemm(
                self._legacy_qweight,
                self._legacy_qzeros,
                self._legacy_scales,
                self.bits,
                self.group_size,
            )
            scales_expanded = self._legacy_scales.repeat_interleave(self.group_size, dim=0).to(weight.dtype)
            weight = weight - 8 * scales_expanded
            weight = weight.to(device=device, dtype=x_flat.dtype)

        output = torch.matmul(x_flat, weight)

        if self.bias is not None:
            if self.bias.dtype != output.dtype:
                self.bias = self.bias.to(dtype=output.dtype)
            output = output + self.bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        output = output.reshape(original_shape)

        return output

__all__ = ["AwqTorchQuantLinear"]
