# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.machete import (
    _validate_machete_device_support,
    check_machete_supports_shape,
    machete_mm,
    machete_prepack_B,
    machete_runtime_available,
    machete_runtime_error,
    pack_quantized_values_into_int32,
    query_machete_supported_group_sizes,
    query_machete_supported_quant_types,
    unpack_quantized_values_into_int32,
)
from ...utils.marlin import replace_parameter
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class MacheteLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_MACHETE]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 100}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
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

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "machete"

    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
        (4, False): scalar_types.uint4,
        (8, False): scalar_types.uint8,
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
            adapter: Adapter = None,
            **kwargs):
        if (bits, sym) not in self.TYPE_MAP:
            raise ValueError(f"Unsupported quantization config: bits={bits}, sym={sym}")

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_MACHETE),
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        # Quantized weights (packed)
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    self.in_features // self.pack_factor,
                    self.out_features,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )

        # Activation order indices
        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                torch.empty(self.in_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        # Scales
        grouped_rows = math.ceil(self.in_features / self.group_size)
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(
                    grouped_rows,
                    self.out_features,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            ),
        )

        # Register GPTQ checkpoint-compatible qzero storage even for symmetric
        # configs so Accelerate can bind tensors before post_init().
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    grouped_rows,
                    self.out_features // self.pack_factor,
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
        else:
            self.bias = None

        self.weight_type = self.TYPE_MAP[(self.bits, self.sym)]
        self.has_zero_points = False

        # Buffer storing permutation applied to activations (empty when unused)
        self.register_buffer("input_perm", torch.empty(0, dtype=torch.int32))

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not machete_runtime_available():
            return False, ImportError(machete_runtime_error())
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = cls._validate(**args)
        if not ok:
            return ok, err

        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features is not None and out_features is not None:
            supported, reason = check_machete_supports_shape(in_features, out_features)
            if not supported:
                return False, ValueError(reason)

        bits = args.get("bits")
        sym = args.get("sym", True)
        quant_type = cls.TYPE_MAP.get((bits, sym))
        if quant_type is None:
            return False, ValueError(f"Machete does not support bits={bits}, sym={sym}")
        if quant_type not in query_machete_supported_quant_types(zero_points=not sym):
            return False, ValueError(f"Machete does not support bits={bits}, sym={sym}")

        group_size = args.get("group_size")
        dtype = args.get("dtype") or torch.float16
        if group_size not in query_machete_supported_group_sizes(dtype):
            return False, ValueError(
                f"Machete does not support group_size={group_size} for dtype={dtype}"
            )

        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Machete kernel is not supported on ROCm.")
            if not _validate_machete_device_support():
                raise NotImplementedError(machete_runtime_error())

    def post_init(self):
        device = self.qweight.device

        perm = None
        if self.desc_act:
            perm = torch.argsort(self.g_idx).to(torch.int32)
            sorted_g_idx = self.g_idx[perm]
            replace_parameter(
                self,
                "g_idx",
                torch.nn.Parameter(sorted_g_idx.to(device=device), requires_grad=False),
            )
            self.input_perm = perm.to(device=device)
        else:
            self.input_perm = torch.empty(0, dtype=torch.int32, device=device)

        qweight_unpacked = unpack_quantized_values_into_int32(
            self.qweight.data, self.weight_type, packed_dim=0)
        if perm is not None:
            qweight_unpacked = qweight_unpacked[perm, :]

        qweight_packed = pack_quantized_values_into_int32(
            qweight_unpacked, self.weight_type, packed_dim=0)
        qweight_packed = qweight_packed.t().contiguous().t()
        prepacked = machete_prepack_B(
            qweight_packed,
            a_type=self.scales.dtype,
            b_type=self.weight_type,
            group_scales_type=self.scales.dtype,
        )
        replace_parameter(
            self,
            "qweight",
            torch.nn.Parameter(prepacked.contiguous(), requires_grad=False),
        )

        scales = self.scales.data.contiguous()
        replace_parameter(
            self,
            "scales",
            torch.nn.Parameter(scales, requires_grad=False),
        )

        if self.sym:
            replace_parameter(
                self,
                "qzeros",
                torch.nn.Parameter(
                    torch.empty(0, dtype=self.pack_dtype, device=device),
                    requires_grad=False,
                ),
            )
            self.has_zero_points = False
        else:
            qzeros_unpacked = unpack_quantized_values_into_int32(
                self.qzeros.data,
                self.weight_type,
                packed_dim=1,
            )
            qzeros_fp = (-1.0 * scales * qzeros_unpacked.to(scales.dtype)).contiguous()
            replace_parameter(
                self,
                "qzeros",
                torch.nn.Parameter(qzeros_fp, requires_grad=False),
            )
            self.has_zero_points = True

        if self.bias is not None:
            self.bias = self.bias.to(device=device)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "input_perm") and self.input_perm is not None:
            buf.append(self.input_perm)
        return buf

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        input_2d = x.reshape(-1, x.shape[-1])

        if self.input_perm.numel() > 0:
            perm = self.input_perm
            if perm.device != input_2d.device:
                perm = perm.to(device=input_2d.device)
            input_2d = input_2d[:, perm]

        group_scales = self.scales
        if group_scales.dtype != input_2d.dtype:
            group_scales = group_scales.to(dtype=input_2d.dtype)

        if self.has_zero_points:
            assert self.qzeros is not None and self.qzeros.numel() > 0, (
                "Asymmetric MacheteLinear requires non-empty qzeros after post_init()."
            )
            group_zeros = self.qzeros
        else:
            group_zeros = None

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


__all__ = ["MacheteLinear"]
