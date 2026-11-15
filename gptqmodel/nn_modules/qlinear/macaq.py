# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-FileCopyrightText: 2026 the MACAQ authors
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# MACaQ MXFP4 inference kernels.
#
# Portions of this file are adapted from Huawei CSL's MACaQ reference
# implementation (https://github.com/huawei-csl/MACaQ, Apache License 2.0):
#   `quantize/macaq_mxfp4.py` (OCP MXFP4 packing / dequantization)
#   `vllm/macaq_weight_only.py` (W4A16 forward)
# Algorithm: "MACaQ: Fast, Model Agnostic Calibrated Quantization for
# Low-Precision Weights and Activations", Muller et al., arXiv:2611.10649.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization import macaq as macaq_core
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.macaq_triton import (
    TRITON_AVAILABLE,
    mxfp4_w4a4_linear,
    mxfp4_w4a16_linear,
    triton_mxfp4_available,
)
from ...utils.rocm import IS_ROCM
from . import WeightOnlyQuantLinear


log = setup_logger()


class MacaqLinear(WeightOnlyQuantLinear):
    """Weight-only-safe MACaQ Linear with a portable Torch fallback.

    The packed weight stays in its MXFP4 representation between calls; the
    dense rotated-frame weight is reconstructed only for the current forward
    and released immediately.
    """

    SUPPORTS_BACKENDS = [BACKEND.MACAQ_TORCH]
    SUPPORTS_METHODS = [METHOD.MACAQ]
    SUPPORTS_FORMATS = {FORMAT.MACAQ: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "macaq"

    def __init__(
        self,
        bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        block_size: int = macaq_core.MXFP4_BLOCK,
        had_block: int = macaq_core.MXFP4_BLOCK,
        activation_precision: str = "w4a4",
        pre_mode: str = "balance",
        pre_alpha: float = 0.5,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            bits=int(bits),
            in_features=int(in_features),
            out_features=int(out_features),
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.MACAQ_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            dtype=dtype,
            **kwargs,
        )

        self.block_size = int(block_size)
        self.had_block = int(had_block)
        self.activation_precision = str(activation_precision).strip().lower()
        self.pre_mode = str(pre_mode)
        self.pre_alpha = float(pre_alpha)
        if self.block_size != macaq_core.MXFP4_BLOCK:
            raise ValueError("MacaqLinear: OCP MXFP4 requires `block_size=32`.")
        if self.had_block < 32 or self.had_block & (self.had_block - 1):
            raise ValueError("MacaqLinear: `had_block` must be a power of two of at least 32.")
        if self.in_features % self.block_size or self.in_features % self.had_block:
            raise ValueError(
                f"MacaqLinear: in_features {self.in_features} must be divisible by "
                f"block_size={self.block_size} and had_block={self.had_block}."
            )
        if self.activation_precision not in ("w4a4", "w4a16"):
            raise ValueError("MacaqLinear: `activation_precision` must be `w4a4` or `w4a16`.")

        self.compute_dtype = (
            dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
        )
        self._triton_disabled = False

        # `BH`/`pre_scale` are small and always materialized so the module has
        # a real device to allocate the packed storage on later (the looper
        # moves the module to the quantization device before calling `pack`).
        device = next(iter(self._parameters.values())).device if self._parameters else torch.device("cpu")
        self.register_buffer(
            "BH", macaq_core.make_hadamard(self.had_block, device=device, dtype=self.compute_dtype)
        )
        self.register_buffer(
            "pre_scale", torch.ones(self.in_features, dtype=self.compute_dtype, device=device)
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(self.out_features, dtype=self.compute_dtype, device=device)
            )
        else:
            self.bias = None

        if register_buffers:
            self._allocate_buffers(bias=bias)

    # ------------------------------------------------------------------
    # buffers
    # ------------------------------------------------------------------
    def _allocate_buffers(self, *, bias: bool) -> None:
        """Allocate packed storage outside inference mode."""

        device = self.BH.device
        groups = self.in_features // self.block_size
        with torch.inference_mode(False):
            if "W_packed" not in self._buffers:
                self.register_buffer(
                    "W_packed",
                    torch.zeros((self.out_features, self.in_features // 2), dtype=torch.uint8, device=device),
                )
            if "W_scales" not in self._buffers:
                # Store E8M0 as raw bytes so every safetensors writer can
                # serialize the checkpoint; kernels view them as E8M0.
                self.register_buffer(
                    "W_scales",
                    torch.zeros((self.out_features, groups), dtype=torch.uint8, device=device),
                )
            if bias and self.bias is None:
                self.register_buffer(
                    "bias", torch.zeros(self.out_features, dtype=self.compute_dtype, device=device)
                )

    def _validate_loaded_state(self) -> None:
        groups = self.in_features // self.block_size
        if tuple(self.W_packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(
                f"MacaqLinear `{self.name}`: unexpected W_packed shape {tuple(self.W_packed.shape)}; "
                f"expected {(self.out_features, self.in_features // 2)}."
            )
        if tuple(self.W_scales.shape) != (self.out_features, groups):
            raise ValueError(
                f"MacaqLinear `{self.name}`: unexpected W_scales shape {tuple(self.W_scales.shape)}; "
                f"expected {(self.out_features, groups)}."
            )
        if tuple(self.pre_scale.shape) != (self.in_features,):
            raise ValueError(f"MacaqLinear `{self.name}`: unexpected pre_scale shape.")
        if tuple(self.BH.shape) != (self.had_block, self.had_block):
            raise ValueError(f"MacaqLinear `{self.name}`: unexpected BH shape.")

    def post_init(self):
        self._validate_loaded_state()
        super().post_init()

    # ------------------------------------------------------------------
    # packing
    # ------------------------------------------------------------------
    def pack(self, linear: nn.Module = None, *, macaq_result=None, **kwargs):
        """Fill the packed buffers from a :class:`MacaqQuantResult`."""

        del kwargs
        if macaq_result is None:
            raise ValueError(
                "MacaqLinear.pack requires a `macaq_result` produced by the MACaQ quantizer."
            )
        bias = None if linear is None else getattr(linear, "bias", None)
        if "W_packed" not in self._buffers:
            self._allocate_buffers(bias=bias is not None)
        self.load_quant_result(macaq_result, bias=bias)

    @torch.no_grad()
    def load_quant_result(self, result, bias: Optional[torch.Tensor] = None) -> None:
        device = self.W_packed.device
        self.W_packed.data.copy_(result.W_packed.to(device=device, dtype=torch.uint8))
        self.W_scales.data.copy_(result.W_scales.to(device=device).view(torch.uint8))
        if result.pre_scale is not None:
            self.pre_scale.data.copy_(result.pre_scale.to(device=device, dtype=self.compute_dtype))
        else:
            self.pre_scale.data.fill_(1.0)
        self.BH.data.copy_(result.BH.to(device=device, dtype=self.compute_dtype))
        if bias is not None and self.bias is not None:
            self.bias.data.copy_(bias.to(device=device, dtype=self.compute_dtype))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def dequantize(self) -> torch.Tensor:
        """Decode the packed MXFP4 weight in the rotated frame (current call only)."""

        return macaq_core.dequantize_mxfp4(
            self.W_packed,
            self.W_scales,
            self.in_features,
            dtype=self.compute_dtype,
        )

    def _forward_torch(self, x_flat: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize()
        if self.activation_precision == "w4a4":
            xb = macaq_core.macaq_quantize_activations(
                x_flat, self.pre_scale, self.BH, self.block_size
            ).to(self.compute_dtype)
        else:
            xb = x_flat.to(self.compute_dtype)
            if self.pre_scale is not None:
                xb = xb * self.pre_scale.to(xb.dtype)
            if self.BH is not None:
                xb = macaq_core.rotate_blocks(xb, self.BH)
        return torch.nn.functional.linear(xb, weight, self.bias)

    def _forward_triton(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self.activation_precision == "w4a4":
            return mxfp4_w4a4_linear(
                x_flat,
                self.W_packed,
                self.W_scales,
                pre_scale=self.pre_scale,
                bh=self.BH,
                bias=self.bias,
            )
        return mxfp4_w4a16_linear(
            x_flat,
            self.W_packed,
            self.W_scales,
            pre_scale=self.pre_scale,
            bh=self.BH,
            bias=self.bias,
            compute_dtype=self.compute_dtype,
        )

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            not self._triton_disabled
            and self.backend == BACKEND.MACAQ_TRITON
            and TRITON_AVAILABLE
            and x.is_cuda
            and not IS_ROCM
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if x_flat.shape[0] == 0:
            output = self.empty_linear_output(x_flat)
        elif self._can_use_triton(x_flat):
            try:
                output = self._forward_triton(x_flat)
            except Exception as exc:
                self._triton_disabled = True
                log.warn.once(
                    f"MACaQ Triton kernel unavailable on {x_flat.device} "
                    f"({type(exc).__name__}: {exc}); using the portable Torch path."
                )
                output = self._forward_torch(x_flat)
        else:
            output = self._forward_torch(x_flat)

        output = output.to(x.dtype)
        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)
        return output.reshape(original_shape)


class MacaqTritonLinear(MacaqLinear):
    """CUDA Triton MXFP4 Linear (native ``tl.dot_scaled`` W4A4 GEMM)."""

    SUPPORTS_BACKENDS = [BACKEND.MACAQ_TRITON]
    SUPPORTS_METHODS = [METHOD.MACAQ]
    SUPPORTS_FORMATS = {FORMAT.MACAQ: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_once(cls):
        if not TRITON_AVAILABLE:
            return False, RuntimeError("MacaqTritonLinear requires Triton.")
        if not torch.cuda.is_available() or IS_ROCM:
            return False, RuntimeError("MacaqTritonLinear requires an available CUDA device.")
        return True, None

    @classmethod
    def validate_device(cls, device):
        if not triton_mxfp4_available():
            raise NotImplementedError(
                "MacaqTritonLinear requires a CUDA device and Triton; use BACKEND.MACAQ_TORCH otherwise."
            )
        return super().validate_device(device)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.MACAQ_TRITON)
        super().__init__(*args, **kwargs)
        self.backend = BACKEND.MACAQ_TRITON


__all__ = ["MacaqLinear", "MacaqTritonLinear"]
