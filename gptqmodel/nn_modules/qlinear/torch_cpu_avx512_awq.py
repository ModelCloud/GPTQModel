# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...extension import op as extension_op
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from ...utils.logger import setup_logger
from . import AWQuantLinear
from .torch_aten_kernel import TorchAtenLinear
from .torch_cpu_avx512 import unpack_hf_int4


log = setup_logger()


class TorchCpuAvx512AwqLinear(AWQuantLinear):
    """AWQ W4A16 adapter for the shared AVX512-BF16 int4 GEMM."""

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_CPU_AVX512]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 120}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128, 256, 512]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [2]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS
    # Shared with the BF16-only AVX512-BF16 GPTQ kernel. FP16 AWQ checkpoints
    # use the native ATen FP16 int4pack backend.
    SUPPORTS_DTYPES = [torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "torch_cpu_avx512_awq"

    gemm_int4_qzero_group_scaled_cpu_kernel = None
    pack_awq_int4_kernel = None

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
        adapter=None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH_CPU_AVX512)
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
            register_buffers=register_buffers,
            **kwargs,
        )
        self._cpu_transformed = False
        self.linear_mode = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not has_avx512_bf16():
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_awq_int4_kernel = None
            err = RuntimeError(avx512_bf16_error())
            log.warning(str(err))
            return False, err

        try:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "gemm_int4_qzero_group_scaled_cpu_forward"
            )
            cls.pack_awq_int4_kernel = extension_op("int4_qzero_group_scaled_cpu", "pack_awq_int4")
        except Exception as exc:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_awq_int4_kernel = None
            return False, exc
        return True, None

    def _cpu_fused_available(self) -> bool:
        if self.gemm_int4_qzero_group_scaled_cpu_kernel is not None:
            return True
        ok, _ = type(self).cached_validate_once()
        return ok

    def _fused_input_dtype(self, x: torch.Tensor) -> bool:
        return x.dtype == torch.bfloat16

    def train(self, mode: bool = True):
        if mode:
            raise NotImplementedError(
                f"{self.__class__.__name__} is an inference-only CPU kernel. "
                "Use BACKEND.AWQ_TORCH_ATEN for trainable AWQ layers."
            )
        return super().train(False)

    def transform_cpu(self):
        if self._cpu_transformed:
            return

        self._dequant_scales_dtype = self.scales.dtype
        if self.pack_awq_int4_kernel is None:
            ok, err = type(self).cached_validate_once()
            if not ok:
                raise RuntimeError(str(err))

        # Keep the checkpoint scales in FP32. The shared AVX512 kernel uses
        # BF16 activations and weights with FP32 accumulation; rounding scales
        # to BF16 first would consume most of the inference error budget.
        self.scales = self.scales.to(torch.float32).contiguous()
        if self.bias is not None:
            self.bias = self.bias.to(torch.float32).contiguous()
        self.qweight, self.qzeros = self.pack_awq_int4_kernel(
            self.qweight,
            self.qzeros,
            self.out_features,
            self.in_features,
        )
        self._cpu_transformed = True

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
        else:
            raise NotImplementedError(
                "TorchCpuAvx512AwqLinear only supports fused transforms on CPU devices."
            )

    def dequantize_weight(self, num_itr: int = 1):
        if not self._cpu_transformed:
            return dequantize_gemm(
                qweight=self.qweight,
                qzeros=self.qzeros,
                scales=self.scales,
                bits=self.bits,
                group_size=self.group_size,
            )
        if num_itr != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__}.dequantize_weight only supports num_itr=1"
            )

        codes = unpack_hf_int4(
            self.qweight,
            out_features=self.out_features,
            in_features=self.in_features,
        )
        group_size = max(int(self.group_size), 1)
        groups = torch.arange(self.in_features, device=codes.device) // group_size
        zeros = self.qzeros.index_select(0, groups)
        scales = self.scales.index_select(0, groups)
        weight = (codes.t().float() - zeros.float()) * scales.float()
        return weight.to(self._dequant_scales_dtype)

    def awq_weight_dequantize(self, device, dtype):
        return self.dequantize_weight().to(device=device, dtype=dtype)

    @torch.no_grad()
    def _fused_op_forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        if not self._fused_input_dtype(x):
            raise RuntimeError(
                f"{self.__class__.__name__} requires bfloat16 activations, got {x.dtype}."
            )
        out = self.gemm_int4_qzero_group_scaled_cpu_kernel(
            x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.group_size,
            self.bias,
        )
        return out

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if (
            not self.training
            and not x_flat.requires_grad
            and self.linear_mode is None
            and self._cpu_fused_available()
            and x_flat.device.type == "cpu"
            and self._fused_input_dtype(x_flat)
        ):
            self.transform("cpu")
            self.linear_mode = "inference"
        elif self.linear_mode is None:
            self.linear_mode = "train"

        if self.linear_mode == "inference":
            if not self._fused_input_dtype(x_flat):
                raise RuntimeError(
                    f"{self.__class__.__name__} was transformed for bfloat16 "
                    f"but received {x_flat.dtype} activations."
                )
            out = self._fused_op_forward(x_flat).reshape(out_shape)
        else:
            weight = self.dequantize_weight().to(x_flat.dtype)
            out = torch.matmul(x_flat, weight).reshape(out_shape)
            if self.bias is not None:
                out = out + self.bias
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["TorchCpuAvx512AwqLinear"]
