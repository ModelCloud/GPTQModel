# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.machete import (
    _validate_machete_device_support,
    check_machete_supports_shape,
    machete_import_exception,
    machete_mm,
    machete_prepack_B,
)
from ...utils.marlin_scalar_type import ScalarType, scalar_types


log = setup_logger()


class MacheteQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]

    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "machete"

    TYPE_MAP: Dict[Tuple[int, bool], ScalarType] = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
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
        register_buffers: bool = False,
        adapter: Adapter | None = None,
        **kwargs,
    ):
        if machete_import_exception is not None:
            raise ValueError(
                "Trying to use the machete backend, but its CUDA extension could "
                f"not be imported: {machete_import_exception}"
            )

        ok_shape, msg = check_machete_supports_shape(in_features, out_features)
        if not ok_shape:
            raise ValueError(msg)

        if (bits, sym) not in self.TYPE_MAP:
            raise ValueError(f"Unsupported quantization config bits={bits}, sym={sym}")

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.MACHETE),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        self.sym = sym
        self.weight_type = self.TYPE_MAP[(self.bits, sym)]

        rows = self.in_features // self.pack_factor
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(rows, self.out_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        groups = max(1, math.ceil(self.in_features / self.group_size))
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(groups, self.out_features, dtype=torch.float16),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(groups, self.out_features // self.pack_factor, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                torch.empty(self.in_features, dtype=torch.int32), requires_grad=False
            ),
        )

        if bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.zeros(self.out_features, dtype=torch.float16), requires_grad=False
                ),
            )
        else:
            self.bias = None

        self._prepacked_cache: Dict[torch.dtype, torch.Tensor] = {}

    @classmethod
    def validate(cls, **args):
        if machete_import_exception is not None:
            return False, ImportError(machete_import_exception)
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features is not None and out_features is not None:
            ok, msg = check_machete_supports_shape(in_features, out_features)
            if not ok:
                return False, ValueError(msg)
        group_size = args.get("group_size")
        desc_act = args.get("desc_act")
        dtype = args.get("pack_dtype", torch.int32)
        if desc_act and desc_act not in cls.SUPPORTS_DESC_ACT:
            return False, NotImplementedError("Machete does not support desc_act=True.")
        if dtype not in cls.SUPPORTS_PACK_DTYPES:
            return False, NotImplementedError(
                f"Machete only supports pack_dtype=torch.int32, got {dtype}."
            )
        return cls._validate(**args)

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA and not _validate_machete_device_support():
            raise NotImplementedError(
                "Machete kernels require an NVIDIA Hopper (SM90+) GPU."
            )

    def post_init(self):
        if not _validate_machete_device_support():
            raise RuntimeError(
                "Machete kernel currently supports Hopper GPUs (SM90+) and CUDA."
            )

        self._prepacked_cache.clear()

        super().post_init()

    def _ensure_prepacked(self, act_dtype: torch.dtype) -> torch.Tensor:
        cached = self._prepacked_cache.get(act_dtype)
        if cached is not None and cached.device == self.qweight.device:
            return cached

        group_scales_type = self.scales.dtype if self.scales is not None else None
        weight = self.qweight.data
        if weight.stride(0) != 1:
            weight = weight.t().contiguous().t()

        prepacked = machete_prepack_B(
            weight,
            a_type=act_dtype,
            b_type=self.weight_type,
            group_scales_type=group_scales_type,
        ).detach()

        self._prepacked_cache[act_dtype] = prepacked
        return prepacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        if machete_import_exception is not None:
            raise RuntimeError(machete_import_exception)

        act_dtype = x.dtype
        if act_dtype not in self.SUPPORTS_DTYPES:
            raise ValueError(f"Machete kernel does not support dtype {act_dtype}.")

        x_2d = x.reshape(-1, x.shape[-1])

        prepacked = self._ensure_prepacked(act_dtype)
        group_scales = self.scales.to(dtype=act_dtype, device=x_2d.device)

        output_2d = machete_mm(
            a=x_2d.contiguous(),
            b_q=prepacked,
            b_type=self.weight_type,
            b_group_scales=group_scales,
            b_group_zeros=None,
            b_group_size=self.group_size,
            out_dtype=act_dtype,
        )

        if self.bias is not None:
            output_2d = output_2d + self.bias.to(dtype=output_2d.dtype, device=output_2d.device)

        output = output_2d.reshape(*x.shape[:-1], self.out_features)

        if self.adapter is not None:
            output = self.adapter.apply(x=x, out=output)

        return output


__all__ = ["MacheteQuantLinear"]
