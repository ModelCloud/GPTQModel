# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ...adapter.adapter import Adapter, Lora
from ...models._const import CPU, DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.logger import setup_logger
from ...utils.mxfp4_cpu import dequantize_mxfp4, load_mxfp4_cpu_kernel, quantize_mxfp4
from . import FormatSupport, WeightOnlyQuantLinear


log = setup_logger()


def _weight_to_matrix(linear: nn.Module) -> torch.Tensor:
    weight = linear.weight.detach()
    if isinstance(linear, _ConvNd):
        weight = weight.flatten(1)
    if isinstance(linear, transformers.pytorch_utils.Conv1D):
        weight = weight.T
    return weight


class Mxfp4CpuLinear(WeightOnlyQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.MXFP4_CPU]
    SUPPORTS_METHODS = [METHOD.MXFP4]
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.MXFP4: FormatSupport(priority=15, bits=(4,)),
    }
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float32]
    SUPPORTS_GROUP_SIZE = [-1]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]

    QUANT_TYPE = "mxfp4_cpu"

    @classmethod
    def validate_once(cls) -> tuple[bool, Optional[Exception]]:
        src = Path(__file__).resolve().parents[2].parent / "gptqmodel_ext" / "mxfp4_cpu_kernel.cpp"
        if not src.exists():
            return False, FileNotFoundError(src)
        return True, None

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
        backend: BACKEND = BACKEND.MXFP4_CPU,
        name: str = None,
        register_buffers: bool = True,
        dtype: Optional[torch.dtype] = None,
        variant: int = 0,
        fp16_flush: int = 0,
        use_vnni: Optional[bool] = None,
        **kwargs,
    ):
        del kwargs
        self.mxfp4_variant = int(variant)
        self.mxfp4_fp16_flush = int(fp16_flush)
        if use_vnni is None:
            use_vnni = env_flag("GPTQMODEL_MXFP4_USE_VNNI", True)
        self.use_vnni = bool(use_vnni)

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=backend,
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            name=name,
            dtype=dtype,
        )

        if register_buffers:
            self._allocate_buffers(bias=bias)

    def _allocate_buffers(self, *, bias: bool) -> None:
        qweight = torch.zeros((self.out_features, self.in_features // 2), dtype=torch.uint8)
        scales = torch.zeros((self.out_features, self.in_features // 32), dtype=torch.uint8)

        if "qweight" in self._buffers:
            self.qweight = qweight
        else:
            self.register_buffer("qweight", qweight)

        if "scales" in self._buffers:
            self.scales = scales
        else:
            self.register_buffer("scales", scales)

        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=torch.float32)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def _weight_to_matrix(self, linear: nn.Module) -> torch.Tensor:
        return _weight_to_matrix(linear)

    def _maybe_prepack_vnni(self) -> None:
        if not self.use_vnni:
            return
        if hasattr(self, "qpack") and self.qpack is not None:
            return
        ext = load_mxfp4_cpu_kernel()
        qpack, spack = ext.mxfp4_prepack_vnni(self.qweight, self.scales)
        self.register_buffer("qpack", qpack)
        self.register_buffer("spack", spack)

    def post_init(self) -> None:
        self._maybe_prepack_vnni()
        super().post_init()

    def list_buffers(self):
        buffers = []
        for name in ("qweight", "scales", "qpack", "spack", "bias"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                buffers.append(tensor)
        return buffers

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, use_vnni={self.use_vnni}"
        )

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        block_in: int = 8192,
        workers: int = 1,
    ):
        del block_in, workers
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_gpu(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        block_in: int = 8192,
        device: torch.device | None = None,
    ):
        del block_in, device
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    @torch.inference_mode()
    def pack_original(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        smooth=None,
    ):
        del scales, zeros, g_idx, smooth

        weight = self._weight_to_matrix(linear).to(device=CPU, dtype=torch.float32)
        qweight, scales_u8 = quantize_mxfp4(weight)

        if "qweight" in self._buffers:
            self.qweight = qweight
        else:
            self.register_buffer("qweight", qweight)

        if "scales" in self._buffers:
            self.scales = scales_u8
        else:
            self.register_buffer("scales", scales_u8)

        if linear.bias is not None:
            bias = linear.bias.detach().to(device=CPU, dtype=torch.float32)
            if "bias" in self._buffers:
                self.bias = bias
            else:
                self.register_buffer("bias", bias)
        else:
            self.bias = None

        # Prepack for VNNI immediately if enabled so weight-only quantization
        # already has the optimized layout in memory.
        if self.use_vnni:
            self._maybe_prepack_vnni()

    def dequantize_weight(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device = CPU if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        weight = dequantize_mxfp4(self.qweight, self.scales).to(device=target_device, dtype=target_dtype)
        return weight.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        target_dtype = x_flat.dtype

        if x_flat.device.type != "cpu":
            x_flat = x_flat.to(CPU)

        ext = load_mxfp4_cpu_kernel()

        if self.use_vnni:
            self._maybe_prepack_vnni()
            x_fp8 = x_flat.to(torch.float8_e4m3fn)
            out = ext.mxfp4_linear_cpu_vnni(
                x_fp8,
                self.qpack,
                self.spack,
                self.out_features,
                0,
            )
            out = out.to(target_dtype)
        else:
            if target_dtype == torch.float32:
                x_compute = x_flat.to(torch.bfloat16)
            elif target_dtype == torch.float8_e4m3fn:
                x_compute = x_flat
            else:
                x_compute = x_flat
            out = ext.mxfp4_linear_cpu(
                x_compute,
                self.qweight,
                self.scales,
                threads=0,
                variant=self.mxfp4_variant,
                fp16_flush=self.mxfp4_fp16_flush,
            )
            if target_dtype == torch.float32:
                out = out.to(torch.float32)

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)

        if self.adapter is not None:
            out = self.adapter.apply(x=x_flat, out=out)

        if input_shape[:-1] != out.shape[:-1]:
            out = out.reshape(*input_shape[:-1], self.out_features)

        return out


__all__ = ["Mxfp4CpuLinear"]
