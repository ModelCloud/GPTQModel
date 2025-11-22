# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.machete import (
    _validate_machete_device_support,
    machete_import_exception,
    machete_mm,
    machete_prepack_B,
    pack_quantized_values_into_int32,
)
from ...utils.marlin import replace_parameter, unpack_cols
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class AwqMacheteQuantLinear(AWQuantLinear):
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False]  # AWQ kernels do not reorder activations
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_machete"

    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

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
            register_buffers: bool = False,
            **kwargs):
        if machete_import_exception is not None:
            raise ValueError(
                "Trying to use the machete backend, but could not import the "
                f"C++/CUDA dependencies with the following error: {machete_import_exception}"
            )

        if bits not in self.TYPE_MAP:
            raise ValueError(f"Unsupported num_bits = {bits}. Supported: {list(self.TYPE_MAP.keys())}")

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.MACHETE),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.weight_type = self.TYPE_MAP[self.bits]
        self.has_zero_points = True

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if machete_import_exception is not None:
            return False, ImportError(machete_import_exception)
        return cls._validate(**args)

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Machete kernel is not supported on ROCm.")
            if not _validate_machete_device_support():
                raise NotImplementedError("Machete kernel requires compute capability >= 9.0.")

    def post_init(self):
        device = self.qweight.device

        # Reconstruct integer weights from packed AWQ representation
        qweight_int = unpack_cols(
            self.qweight,
            self.bits,
            self.in_features,
            self.out_features,
        ).to(device=device)

        packed = pack_quantized_values_into_int32(
            qweight_int,
            self.weight_type,
            packed_dim=0,
        )
        packed = packed.t().contiguous().t()
        prepacked = machete_prepack_B(
            packed,
            a_type=self.scales.dtype,
            b_type=self.weight_type,
            group_scales_type=self.scales.dtype,
        )
        replace_parameter(
            self,
            "qweight",
            torch.nn.Parameter(prepacked.contiguous(), requires_grad=False),
        )

        # Ensure scales are contiguous and resident on the correct device.
        replace_parameter(
            self,
            "scales",
            torch.nn.Parameter(self.scales.contiguous(), requires_grad=False),
        )

        # Convert zero-points: unpack columns, then pre-apply scales as expected by machete_mm
        effective_group_size = self.in_features if self.group_size == -1 else self.group_size
        num_groups = self.in_features // effective_group_size

        qzeros_unpacked = unpack_cols(
            self.qzeros,
            self.bits,
            num_groups,
            self.out_features,
        ).to(device=device)

        scales = self.scales
        qzeros_fp = (-1.0 * scales.to(dtype=scales.dtype) * qzeros_unpacked.to(scales.dtype)).contiguous()
        replace_parameter(
            self,
            "qzeros",
            torch.nn.Parameter(qzeros_fp, requires_grad=False),
        )

        if self.bias is not None:
            self.bias = self.bias.to(device=device)

        super().post_init()

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        input_2d = x.reshape(-1, x.shape[-1])
        group_scales = self.scales.to(dtype=input_2d.dtype)
        group_zeros = self.qzeros.to(dtype=input_2d.dtype)

        output = machete_mm(
            a=input_2d,
            b_q=self.qweight,
            b_type=self.weight_type,
            b_group_scales=group_scales,
            b_group_zeros=group_zeros,
            b_group_size=self.group_size,
        )

        if self.bias is not None:
            output.add_(self.bias)

        result = output.reshape(x.shape[:-1] + (self.out_features,))

        if self.adapter:
            result = self.adapter.apply(x=x, out=result)

        return result


__all__ = ["AwqMacheteQuantLinear"]
