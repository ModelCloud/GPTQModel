# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.gptq_gemm import (
    _validate_gptq_gemm_device_support,
    apply_gptq_gemm_linear,
    gptq_gemm_qweight_to_b_packed,
    gptq_gemm_runtime_available,
    gptq_gemm_runtime_error,
)
from ...utils.rocm import IS_ROCM


def _effective_group_size(group_size: int, in_features: int) -> int:
    return in_features if group_size == -1 else group_size


class GPTQGemmLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_GEMM]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Explicit backend only until benchmarked on the local target GPUs.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]

    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "gptq-gemm"

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
        register_buffers: bool = True,
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
            backend=kwargs.pop("backend", BACKEND.GPTQ_GEMM),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=False,
            **kwargs,
        )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not gptq_gemm_runtime_available():
            return False, ImportError(gptq_gemm_runtime_error())
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("GPTQ-GEMM kernel is not supported on ROCm.")
            if not _validate_gptq_gemm_device_support():
                raise NotImplementedError("GPTQ-GEMM kernel requires compute capability >= 8.0.")

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = True,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err

        effective_group_size = in_features if (group_size == -1 and in_features is not None) else group_size
        if effective_group_size is not None and effective_group_size > 0 and (effective_group_size % 16) != 0:
            return False, NotImplementedError(
                f"{cls} requires group_size to be a positive multiple of 16: actual group_size = `{effective_group_size}`"
            )
        return True, None

    def post_init(self):
        if self.qweight.device.type != "cuda":
            raise ValueError("GPTQ-GEMM backend requires CUDA-resident packed weights before post_init().")

        group_size = _effective_group_size(self.group_size, self.in_features)
        expected_g_idx = torch.arange(
            self.in_features,
            device=self.g_idx.device,
            dtype=self.g_idx.dtype,
        ) // group_size
        if not torch.equal(self.g_idx, expected_g_idx) and not bool(torch.all(self.g_idx == 0).item()):
            raise ValueError("GPTQ-GEMM backend only supports non-act-order GPTQ checkpoints.")

        b_packed = gptq_gemm_qweight_to_b_packed(self.qweight)
        if "b_packed" not in self._buffers:
            self.register_buffer("b_packed", b_packed, persistent=False)
        else:
            self.b_packed = b_packed

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "b_packed") and self.b_packed is not None and not any(
            tensor is self.b_packed for tensor in buf
        ):
            buf.append(self.b_packed)
        return buf

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.empty(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"GPTQ-GEMM backend expected input dim {self.in_features}, got {x.shape[-1]}."
            )

        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        out = apply_gptq_gemm_linear(
            input=x.contiguous(),
            b_packed=self.b_packed,
            scales=self.scales.contiguous(),
            group_size=_effective_group_size(self.group_size, self.in_features),
        )

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)


__all__ = ["GPTQGemmLinear"]
