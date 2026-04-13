# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.machete import (
    _validate_machete_device_support,
    machete_mm,
    machete_prepack_B,
    machete_runtime_available,
    machete_runtime_error,
    pack_quantized_values_into_int32,
)
from ...utils.marlin import replace_parameter, replace_tensor, unpack_cols
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


def _undo_awq_interleave(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    if num_bits == 4:
        undo_interleave = [0, 4, 1, 5, 2, 6, 3, 7]
    elif num_bits == 8:
        undo_interleave = [0, 2, 1, 3]
    else:
        raise ValueError(f"Unsupported AWQ num_bits={num_bits}")

    return (
        values.reshape(-1, len(undo_interleave))[:, undo_interleave]
        .reshape(values.shape)
        .contiguous()
    )


def _replace_registered_tensor(
    module: torch.nn.Module,
    name: str,
    new_tensor: torch.Tensor,
) -> None:
    if name in module._parameters:
        replace_parameter(
            module,
            name,
            torch.nn.Parameter(new_tensor, requires_grad=False),
        )
        return

    if name in module._buffers:
        current = getattr(module, name)
        if (
            current.dtype == new_tensor.dtype
            and current.untyped_storage().nbytes() == new_tensor.untyped_storage().nbytes()
        ):
            replace_tensor(module, name, new_tensor)
        else:
            module._buffers[name] = new_tensor
        return

    raise KeyError(f"{module.__class__.__name__}.{name} is not a registered tensor")


class AwqMacheteLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_MACHETE]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 100, FORMAT.MARLIN: 100}
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
            backend=kwargs.pop("backend", BACKEND.AWQ_MACHETE),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.weight_type = self.TYPE_MAP[self.bits]
        self.has_zero_points = True

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not machete_runtime_available():
            return False, ImportError(machete_runtime_error())
        return True, None

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
        qweight_int = _undo_awq_interleave(qweight_int, self.bits)

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
        _replace_registered_tensor(self, "qweight", prepacked.contiguous())

        # Ensure scales are contiguous and resident on the correct device.
        _replace_registered_tensor(self, "scales", self.scales.contiguous())

        # Convert zero-points: unpack columns, then pre-apply scales as expected by machete_mm
        effective_group_size = self.in_features if self.group_size == -1 else self.group_size
        num_groups = self.in_features // effective_group_size

        qzeros_unpacked = unpack_cols(
            self.qzeros,
            self.bits,
            num_groups,
            self.out_features,
        ).to(device=device)
        qzeros_unpacked = _undo_awq_interleave(qzeros_unpacked, self.bits)

        scales = self.scales
        qzeros_fp = (-1.0 * scales.to(dtype=scales.dtype) * qzeros_unpacked.to(scales.dtype)).contiguous()
        _replace_registered_tensor(self, "qzeros", qzeros_fp)

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


__all__ = ["AwqMacheteLinear"]
