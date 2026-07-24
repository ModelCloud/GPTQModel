# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from .gemm_awq_triton import AwqGEMMTritonLinear
from .tritonv2 import TritonV2Linear


class _TrilinLinearMixin:
    """Own the native CUDA Trilin path while retaining an explicit fallback backend."""

    SUPPORTS_BITS = [3]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_TRAINING = False
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    _TRILIN_RUNTIME_QWEIGHT = "qweight"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        valid, error = super().validate_once()
        if not valid:
            return valid, error

        try:
            from ...utils.trilin import prewarm_trilin_extension

            if not prewarm_trilin_extension():
                return False, RuntimeError("Trilin native 3-bit CUDA extension is unavailable.")
        except Exception as exc:
            return False, exc
        return True, None

    def _trilin_runtime_qweight(self) -> torch.Tensor:
        return getattr(self, self._TRILIN_RUNTIME_QWEIGHT)

    def post_init(self):
        super().post_init()

        from ..triton_utils.three_bit import (
            prepare_marlin_3bit,
            prepare_trilin_3bit,
            prepare_trilin_lora_3bit,
        )

        # Accelerate applies the requested model dtype to floating-point buffers
        # while loading. Trilin accepts FP16 or BF16 activations, but its native
        # CUDA ABI deliberately stores scales as FP16 for both paths.
        self.scales = self.scales.to(dtype=torch.float16).contiguous()
        runtime_qweight = self._trilin_runtime_qweight()
        self._trilin_native_3bit = prepare_trilin_3bit(
            runtime_qweight,
            self.scales,
            self.requested_group_size,
        )
        lora_workspace = (
            prepare_trilin_lora_3bit(
                self.adapter,
                device=runtime_qweight.device,
                in_features=self.in_features,
                out_features=self.out_features,
                group_size=self.requested_group_size,
            )
            if self.adapter is not None and self._trilin_native_3bit
            else None
        )
        if lora_workspace is not None:
            self.register_buffer("_trilin_lora_workspace", lora_workspace, persistent=False)

        marlin_state = prepare_marlin_3bit(runtime_qweight, self.scales, self.requested_group_size)
        if marlin_state is not None:
            self.register_buffer("_trilin_marlin_qweight", marlin_state.qweight, persistent=False)
            self.register_buffer("_trilin_marlin_scales", marlin_state.scales, persistent=False)
            self.register_buffer("_trilin_marlin_workspace", marlin_state.workspace, persistent=False)
            self.register_buffer("_trilin_marlin_empty", marlin_state.empty, persistent=False)

    def forward(self, x: torch.Tensor):
        if self.training:
            return super().forward(x)

        from ..triton_utils.three_bit import (
            matmul_marlin_3bit,
            matmul_trilin_3bit,
            matmul_trilin_lora_3bit,
        )

        capability = torch.cuda.get_device_capability(self.qweight.device)
        if capability < (8, 0):
            return super().forward(x)

        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.stride(-1) != 1:
            x_flat = x_flat.contiguous()

        runtime_qweight = self._trilin_runtime_qweight()
        native_qweight = getattr(self, "_trilin_marlin_qweight", None)
        adapter_applied = False
        if (
            x_flat.dtype in (torch.float16, torch.bfloat16)
            and 0 < x_flat.shape[0] <= 16
            and getattr(self, "_trilin_native_3bit", False)
        ):
            fused_out = None
            if self.adapter is not None:
                fused_out = matmul_trilin_lora_3bit(
                    self.adapter,
                    x_flat,
                    runtime_qweight,
                    self.scales,
                    getattr(self, "_trilin_lora_workspace", None),
                    bias=self.bias,
                    group_size=self.requested_group_size,
                )
            if fused_out is not None:
                out = fused_out
                adapter_applied = True
            else:
                out = matmul_trilin_3bit(
                    x_flat,
                    runtime_qweight,
                    self.scales,
                    bias=self.bias,
                    group_size=self.requested_group_size,
                )
        elif x_flat.dtype == torch.float16 and native_qweight is not None:
            out = matmul_marlin_3bit(
                x_flat,
                native_qweight,
                self._trilin_marlin_scales,
                self._trilin_marlin_workspace,
                self._trilin_marlin_empty,
                k=self.in_features,
                n=self.out_features,
                bias=self.bias,
            )
        else:
            return super().forward(x)

        out = out.reshape(out_shape)
        if self.adapter and not adapter_applied:
            out = self.adapter.apply(x=x, out=out)
        return out.to(dtype=x.dtype)


class TrilinLinear(_TrilinLinearMixin, TritonV2Linear):
    SUPPORTS_BACKENDS = [BACKEND.TRILIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 60, FORMAT.GPTQ_V2: 60}
    SUPPORTS_BITS = [3]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = TritonV2Linear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "trilin"

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = BACKEND.TRILIN
        super().__init__(*args, **kwargs)


class AwqTrilinLinear(_TrilinLinearMixin, AwqGEMMTritonLinear):
    SUPPORTS_BACKENDS = [BACKEND.TRILIN]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 60}
    SUPPORTS_BITS = [3]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 96, 128, 192, 256, 384, 512]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = AwqGEMMTritonLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "awq_trilin"

    _TRILIN_RUNTIME_QWEIGHT = "_triton_3bit_qweight"

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = BACKEND.TRILIN
        super().__init__(*args, **kwargs)


__all__ = ["AwqTrilinLinear", "TrilinLinear"]
