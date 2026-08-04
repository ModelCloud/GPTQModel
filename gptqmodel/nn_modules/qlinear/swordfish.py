# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Swordfish (Blackwell sm100/sm110) weight-quantized linear layer."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.marlin import replace_parameter
from ...utils.marlin_scalar_type import scalar_types

from ...utils.rocm import IS_ROCM
from ...utils.swordfish import (
    _validate_swordfish_device_support,
    check_swordfish_supports_shape,
    query_swordfish_supported_group_sizes,
    query_swordfish_supported_quant_types,
    swordfish_mm,
    swordfish_prepack_B,
    swordfish_runtime_available,
    swordfish_runtime_error,
)

log = setup_logger()


class SwordfishLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_SWORDFISH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 101}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "swordfish"

    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
        (4, False): scalar_types.uint4,
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
        format=None,
        **kwargs,
    ):
        if (bits, sym) not in self.TYPE_MAP:
            raise ValueError(f"Unsupported quantization config: bits={bits}, sym={sym}")

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_SWORDFISH),
            adapter=adapter,
            register_buffers=False,
            format=format,
            **kwargs,
        )

        # Pre-allocate checkpoint-compatible GPTQ buffer shapes so that
        # `from_quantized` loading and manual `pack` both write to the same
        # fields.  post_init() repacks qweight into the Swordfish ABI.
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    (self.in_features // self.pack_factor, self.out_features),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                torch.empty(self.in_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        grouped_rows = math.ceil(self.in_features / self.group_size)
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features),
                    dtype=self.compute_dtype,
                ),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features // self.pack_factor),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((self.out_features,), dtype=self.compute_dtype),
            )
        else:
            self.bias = None

        self.weight_type = self.TYPE_MAP[(self.bits, self.sym)]
        self.has_zero_points = False

        self.register_buffer("input_perm", torch.empty(0, dtype=torch.int32))

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not swordfish_runtime_available():
            return False, ImportError(swordfish_runtime_error())
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super().validate(**args)
        if not ok:
            return ok, err

        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features is not None and out_features is not None:
            supported, reason = check_swordfish_supports_shape(in_features, out_features)
            if not supported:
                return False, ValueError(reason)

        bits = args.get("bits")
        sym = args.get("sym", True)
        desc_act = args.get("desc_act", False)
        if bits == 8 and desc_act:
            return False, ValueError("Swordfish 8-bit weights do not support activation reordering (desc_act).")
        if bits == 8 and not sym:
            return False, ValueError("Swordfish 8-bit weights do not support zero points (sym=False).")

        quant_type = cls.TYPE_MAP.get((bits, sym))
        if quant_type is None:
            return False, ValueError(f"Swordfish does not support bits={bits}, sym={sym}")
        if quant_type not in query_swordfish_supported_quant_types(zero_points=not sym):
            return False, ValueError(f"Swordfish does not support bits={bits}, sym={sym}")

        group_size = args.get("group_size")
        dtype = args.get("dtype") or torch.float16
        if group_size not in query_swordfish_supported_group_sizes(dtype):
            return False, ValueError(
                f"Swordfish does not support group_size={group_size} for dtype={dtype}"
            )

        if in_features is not None and group_size is not None and group_size != -1:
            if in_features % group_size != 0 or group_size % 32 != 0:
                return False, ValueError(
                    f"Swordfish requires in_features % group_size == 0 and group_size % 32 == 0, "
                    f"got in_features={in_features}, group_size={group_size}"
                )

        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Swordfish kernel is not supported on ROCm.")
            if not _validate_swordfish_device_support():
                raise NotImplementedError(swordfish_runtime_error())

    def post_init(self):
        device = self.qweight.device

        perm = None
        if self.desc_act:
            perm = torch.argsort(self.g_idx).to(torch.int32)
            # g_idx is not needed at runtime once the permutation is folded
            # into the packed weight.
            replace_parameter(
                self,
                "g_idx",
                torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int32, device=device),
                    requires_grad=False,
                ),
            )

        prepacked = swordfish_prepack_B(
            self.qweight.data,
            self.in_features,
            self.out_features,
            num_bits=self.bits,
            perm=perm,
        )
        replace_parameter(
            self,
            "qweight",
            torch.nn.Parameter(prepacked.contiguous(), requires_grad=False),
        )

        scales = self.scales.data.contiguous()
        if scales.dtype != self.compute_dtype:
            scales = scales.to(self.compute_dtype)
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
            # AWQ-style zero points: fold (half_range - zp) * scale into fp group tensor.
            from ...utils.machete import unpack_quantized_values_into_int32

            half_range = float(1 << (self.bits - 1))
            qzeros_unpacked = unpack_quantized_values_into_int32(
                self.qzeros.data,
                self.weight_type,
                packed_dim=1,
            )
            qzeros_fp = ((half_range - qzeros_unpacked.to(scales.dtype)) * scales).contiguous()
            replace_parameter(
                self,
                "qzeros",
                torch.nn.Parameter(qzeros_fp, requires_grad=False),
            )
            self.has_zero_points = True

        # Store the permutation for activation reordering in forward().
        if perm is not None:
            self.input_perm = perm.to(device=device)
        else:
            self.input_perm = torch.empty(0, dtype=torch.int32, device=device)

        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=self.compute_dtype)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "input_perm") and self.input_perm is not None:
            buf.append(self.input_perm)
        return buf

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            result = torch.empty(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)
            if self.adapter is not None:
                result = self.adapter.apply(x=x, out=result)
            return result

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
                "Asymmetric SwordfishLinear requires non-empty qzeros after post_init()."
            )
            group_zeros = self.qzeros
            if group_zeros.dtype != input_2d.dtype:
                group_zeros = group_zeros.to(dtype=input_2d.dtype)
        else:
            group_zeros = None

        # Swordfish expects -1 for channelwise/per-output-channel; the base
        # class stores that as in_features for its own shape bookkeeping.
        kernel_group_size = -1 if self.requested_group_size == -1 else self.group_size
        output = swordfish_mm(
            a=input_2d,
            b_packed=self.qweight,
            group_scales=group_scales,
            group_size=kernel_group_size,
            size_k=self.in_features,
            size_n=self.out_features,
            group_zps=group_zeros,
            num_bits=self.bits,
        )

        if self.bias is not None:
            output.add_(self.bias.to(dtype=output.dtype))

        result = output.reshape(x.shape[:-1] + (self.out_features,))
        if self.adapter is not None:
            result = self.adapter.apply(x=x, out=result)

        return result


def _unpack_cols_torch(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """GPU-friendly unpacking of int32-packed quantized weights into columns."""
    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0, f"size_n={size_n} not divisible by pack_factor={pack_factor}"
    assert packed_q_w.shape == (
        size_k,
        size_n // pack_factor,
    ), f"packed_q_w.shape={packed_q_w.shape} != ({size_k}, {size_n // pack_factor})"

    mask = (1 << num_bits) - 1
    q_res = torch.zeros((size_k, size_n), dtype=torch.int32, device=packed_q_w.device)
    for i in range(pack_factor):
        vals = (packed_q_w & mask).to(torch.int32)
        q_res[:, i::pack_factor] = vals
        packed_q_w = packed_q_w >> num_bits

    return q_res.contiguous()


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


class AwqSwordfishLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_SWORDFISH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 101}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_swordfish"

    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (4, False): scalar_types.uint4,
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
        **kwargs,
    ):
        if (bits, sym) not in self.TYPE_MAP:
            raise ValueError(f"Unsupported quantization config: bits={bits}, sym={sym}")

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_SWORDFISH),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        pack_factor = self.pack_dtype_bits // self.bits

        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    (self.in_features, self.out_features // pack_factor),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        effective_group_size = self.in_features if self.requested_group_size == -1 else self.group_size
        grouped_rows = math.ceil(self.in_features / effective_group_size)
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features),
                    dtype=self.compute_dtype,
                ),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features // pack_factor),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((self.out_features,), dtype=self.compute_dtype),
            )
        else:
            self.bias = None

        self.weight_type = self.TYPE_MAP[(self.bits, self.sym)]
        self.has_zero_points = not self.sym

        self.register_buffer("input_perm", torch.empty(0, dtype=torch.int32))

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not swordfish_runtime_available():
            return False, ImportError(swordfish_runtime_error())
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super().validate(**args)
        if not ok:
            return ok, err

        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features is not None and out_features is not None:
            supported, reason = check_swordfish_supports_shape(in_features, out_features)
            if not supported:
                return False, ValueError(reason)

        bits = args.get("bits")
        sym = args.get("sym", True)
        quant_type = cls.TYPE_MAP.get((bits, sym))
        if quant_type is None:
            return False, ValueError(f"AwqSwordfishLinear does not support bits={bits}, sym={sym}")
        if quant_type not in query_swordfish_supported_quant_types(zero_points=not sym):
            return False, ValueError(
                f"Swordfish does not support AWQ {bits}-bit weights with sym={sym} (zero_points={not sym})"
            )

        group_size = args.get("group_size")
        dtype = args.get("dtype") or torch.float16
        if group_size not in query_swordfish_supported_group_sizes(dtype):
            return False, ValueError(
                f"Swordfish does not support group_size={group_size} for dtype={dtype}"
            )

        if in_features is not None and group_size is not None and group_size != -1:
            if in_features % group_size != 0 or group_size % 32 != 0:
                return False, ValueError(
                    f"AwqSwordfishLinear requires in_features % group_size == 0 and group_size % 32 == 0, "
                    f"got in_features={in_features}, group_size={group_size}"
                )

        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Swordfish kernel is not supported on ROCm.")
            if not _validate_swordfish_device_support():
                raise NotImplementedError(swordfish_runtime_error())

    def post_init(self):
        device = self.qweight.device

        # Convert AWQ-interleaved packed weights to GPTQ/Marlin layout.
        qweight_int = _unpack_cols_torch(
            self.qweight,
            self.bits,
            self.in_features,
            self.out_features,
        )
        qweight_int = _undo_awq_interleave(qweight_int, self.bits)

        # Repack the AWQ-unpacked integer weights into the GPTQ row-major
        # layout expected by gptq_marlin_repack inside swordfish_prepack_B.
        pack_factor = 32 // self.bits
        qweight_int = qweight_int.view(
            self.in_features // 32, 32 // pack_factor, pack_factor, self.out_features
        )
        shifts = torch.arange(
            0, 32, self.bits, dtype=torch.int32, device=qweight_int.device
        )
        packed = (qweight_int << shifts.view(1, 1, pack_factor, 1)).sum(dim=2, dtype=torch.int32)
        packed = packed.view(self.in_features // pack_factor, self.out_features).contiguous()

        prepacked = swordfish_prepack_B(
            packed,
            self.in_features,
            self.out_features,
            num_bits=self.bits,
            perm=None,
        )
        replace_parameter(
            self,
            "qweight",
            torch.nn.Parameter(prepacked.contiguous(), requires_grad=False),
        )

        scales = self.scales.data.contiguous()
        if scales.dtype != self.compute_dtype:
            scales = scales.to(self.compute_dtype)
        replace_parameter(
            self,
            "scales",
            torch.nn.Parameter(scales, requires_grad=False),
        )

        effective_group_size = self.in_features if self.requested_group_size == -1 else self.group_size
        num_groups = self.in_features // effective_group_size

        if self.has_zero_points:
            qzeros_unpacked = _unpack_cols_torch(
                self.qzeros,
                self.bits,
                num_groups,
                self.out_features,
            )
            qzeros_unpacked = _undo_awq_interleave(qzeros_unpacked, self.bits)

            half_range = float(1 << (self.bits - 1))
            qzeros_fp = ((half_range - qzeros_unpacked.to(scales.dtype)) * scales).contiguous()
            replace_parameter(
                self,
                "qzeros",
                torch.nn.Parameter(qzeros_fp, requires_grad=False),
            )
        else:
            replace_parameter(
                self,
                "qzeros",
                torch.nn.Parameter(
                    torch.empty(0, dtype=scales.dtype, device=device), requires_grad=False
                ),
            )

        self.input_perm = torch.empty(0, dtype=torch.int32, device=device)

        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=self.compute_dtype)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "input_perm") and self.input_perm is not None:
            buf.append(self.input_perm)
        return buf

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            result = torch.empty(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)
            if self.adapter is not None:
                result = self.adapter.apply(x=x, out=result)
            return result

        input_2d = x.reshape(-1, x.shape[-1])

        group_scales = self.scales
        if group_scales.dtype != input_2d.dtype:
            group_scales = group_scales.to(dtype=input_2d.dtype)

        group_zeros = self.qzeros if self.has_zero_points else None
        if group_zeros is not None and group_zeros.dtype != input_2d.dtype:
            group_zeros = group_zeros.to(dtype=input_2d.dtype)

        kernel_group_size = -1 if self.requested_group_size == -1 else self.group_size
        output = swordfish_mm(
            a=input_2d,
            b_packed=self.qweight,
            group_scales=group_scales,
            group_size=kernel_group_size,
            size_k=self.in_features,
            size_n=self.out_features,
            group_zps=group_zeros,
            num_bits=self.bits,
        )

        if self.bias is not None:
            output.add_(self.bias.to(dtype=output.dtype))

        result = output.reshape(x.shape[:-1] + (self.out_features,))
        if self.adapter is not None:
            result = self.adapter.apply(x=x, out=result)

        return result


__all__ = ["SwordfishLinear", "AwqSwordfishLinear"]
