# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...extension import op as extension_op
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from ...utils.logger import setup_logger
from .torch_aten_kernel import TorchAtenLinear


log = setup_logger()


def pack_hf_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[out_features, in_features]`` uint4 codes for the HF CPU ABI."""

    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape={tuple(codes.shape)}")
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")

    out_features, in_features = codes.shape
    if out_features % 32 != 0:
        raise ValueError("out_features must be divisible by 32")
    if in_features % 2 != 0:
        raise ValueError("in_features must be even")

    groups = out_features // 32
    grouped = codes.reshape(groups, 32, in_features // 2, 2)
    low_rows = grouped[:, :16]
    high_rows = grouped[:, 16:]
    packed = (low_rows | (high_rows << 4)).permute(0, 2, 1, 3)
    return packed.reshape(out_features, in_features // 2).contiguous()


def unpack_hf_int4(
    packed: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_hf_int4`.

    The kernel-facing tensor is shaped ``[out_features, in_features // 2]``,
    but its flat layout is grouped as ``[out_features // 32, in_features // 2,
    32 bytes]``. Reshape through the same grouping before splitting nibbles.
    """

    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape={tuple(packed.shape)}")
    if out_features % 32 != 0:
        raise ValueError("out_features must be divisible by 32")
    if in_features % 2 != 0:
        raise ValueError("in_features must be even")
    if packed.shape != (out_features, in_features // 2):
        raise ValueError(
            "packed shape does not match the requested layer: "
            f"expected {(out_features, in_features // 2)}, got {tuple(packed.shape)}"
        )

    groups = out_features // 32
    half_in = in_features // 2
    grouped = packed.reshape(groups, half_in, 16, 2).permute(0, 2, 1, 3)
    low_rows = torch.bitwise_and(grouped, 0x0F)
    high_rows = torch.bitwise_and(torch.bitwise_right_shift(grouped, 4), 0x0F)
    return torch.cat((low_rows, high_rows), dim=1).reshape(out_features, in_features)


class TorchCpuAvx512Linear(TorchAtenLinear):
    """CPU GPTQ W4A16 kernel using AVX512-BF16."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_CPU_AVX512]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 120, FORMAT.GPTQ_V2: 120}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    # The fused kernel has no backward path. Keeping this backend inference-only
    # lets AUTO_TRAINABLE fall back to TorchAtenLinear instead of silently
    # training through a slow dequantized matmul.
    SUPPORTS_TRAINING = False
    # The AVX512 kernel walks full groups and requires in_features % group_size == 0.
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [2]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "torch_cpu_avx512"

    gemm_int4_qzero_group_scaled_cpu_kernel = None
    pack_gptq_int4_kernel = None

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.GPTQ_TORCH_CPU_AVX512)
        super().__init__(*args, **kwargs)
        self._cpu_transformed = False

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not has_avx512_bf16():
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            err = RuntimeError(avx512_bf16_error())
            log.warning(str(err))
            return False, err

        try:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "gemm_int4_qzero_group_scaled_cpu_forward"
            )
            cls.pack_gptq_int4_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "pack_gptq_int4"
            )
        except Exception as exc:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_gptq_int4_kernel = None
            return False, exc
        return True, None

    def _cpu_fused_available(self) -> bool:
        if self.gemm_int4_qzero_group_scaled_cpu_kernel is not None:
            return True
        ok, _ = type(self).cached_validate_once()
        return ok

    def transform_cpu(self):
        if self._cpu_transformed:
            return

        self._dequant_scales_dtype = self.scales.dtype
        self.scales = self.scales.to(torch.bfloat16).contiguous()

        if self.pack_gptq_int4_kernel is None:
            ok, err = type(self).cached_validate_once()
            if not ok:
                raise RuntimeError(str(err))

        ret_idx = (
            torch.empty(0, dtype=torch.int32)
            if self._ret_idx_is_identity()
            else self._build_ret_idx()
        )
        self.qweight, self.qzeros = self.pack_gptq_int4_kernel(
            self.qweight,
            self.qzeros,
            ret_idx,
            self.out_features,
            self.in_features,
        )
        self._cpu_transformed = True

    def train(self, mode: bool = True):
        if mode:
            raise NotImplementedError(
                f"{self.__class__.__name__} is an inference-only CPU kernel. "
                "Use BACKEND.GPTQ_TORCH_ATEN for trainable GPTQ layers."
            )
        return super().train(False)

    def dequantize_weight(self, num_itr: int = 1):
        if not self._cpu_transformed:
            return super().dequantize_weight(num_itr=num_itr)
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
        zeros = self.qzeros.index_select(0, groups).t()
        scales = self.scales.index_select(0, groups).t()
        weight = (codes.to(torch.float32) - zeros.to(torch.float32)) * scales.to(torch.float32)

        # ``transform_cpu`` sorts input channels by GPTQ group before packing.
        # Restore the checkpoint's original channel order for nn.Linear users.
        if not self._ret_idx_is_identity():
            ret_idx = self._build_ret_idx().to(device=weight.device, dtype=torch.long)
            restored = torch.empty_like(weight)
            restored[:, ret_idx] = weight
            weight = restored
        return weight.t().contiguous().to(self._dequant_scales_dtype)

    @torch.no_grad
    def _fused_op_forward(self, x):
        if not self._ret_idx_is_identity():
            x = x[:, self.ret_idx].contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        original_dtype = x.dtype
        if original_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        out = self.gemm_int4_qzero_group_scaled_cpu_kernel(
            x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.group_size,
        )
        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out


__all__ = ["TorchCpuAvx512Linear", "pack_hf_int4", "unpack_hf_int4"]
