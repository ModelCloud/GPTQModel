# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import List, Optional

import torch

from ._extension_loader import load_extension_module
from .logger import setup_logger
from .marlin_scalar_type import ScalarType, scalar_types


log = setup_logger()

machete_import_exception: Optional[str] = None
try:
    gptqmodel_machete_kernels = load_extension_module("gptqmodel_machete_kernels")
except ImportError as e:  # pragma: no cover - surfaced at runtime
    machete_import_exception = str(e)
    gptqmodel_machete_kernels = None

MACHETE_PREPACKED_BLOCK_SHAPE = (64, 128)


def _validate_machete_device_support() -> bool:
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9)


def query_machete_supported_quant_types(zero_points: bool) -> List[ScalarType]:
    if zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_machete_supported_act_types(_zero_points: bool) -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def query_machete_supported_group_sizes(act_type: torch.dtype) -> List[int]:
    if act_type in (torch.float16, torch.bfloat16):
        return [-1, 64, 128]
    return [-1, 128]


def check_machete_supports_shape(in_features: int,
                                 out_features: int) -> tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return (False,
                f"Input features size must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[0]}")
    if out_features % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return (False,
                f"Output features size must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[1]}")
    return (True, None)


def _ensure_machete_loaded():
    if machete_import_exception is not None:
        raise ImportError(
            f"Trying to use the machete backend, but could not import the C++/CUDA dependencies: {machete_import_exception}"
        )


def _maybe_scalar_type(t: Optional[torch.Tensor]) -> Optional[torch.dtype]:
    return t.dtype if t is not None else None


def machete_prepack_B(
        weight: torch.Tensor,
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    _ensure_machete_loaded()
    return gptqmodel_machete_kernels.machete_prepack_B(
        weight,
        a_type,
        b_type.id,
        group_scales_type,
    )


def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype] = None,
        group_zeros_type: Optional[torch.dtype] = None,
        channel_scales_type: Optional[torch.dtype] = None,
        token_scales_type: Optional[torch.dtype] = None,
        out_type: Optional[torch.dtype] = None) -> List[str]:
    _ensure_machete_loaded()
    return gptqmodel_machete_kernels.machete_supported_schedules(
        a_type,
        b_type.id,
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
        *,
        a: torch.Tensor,
        b_q: torch.Tensor,
        b_type: ScalarType,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        out_type: Optional[torch.dtype] = None,
        schedule: Optional[str] = None) -> torch.Tensor:
    _ensure_machete_loaded()
    return gptqmodel_machete_kernels.machete_mm(
        a,
        b_q,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


def pack_quantized_values_into_int32(
        tensor: torch.Tensor,
        qtype: ScalarType,
        packed_dim: int = 0) -> torch.Tensor:
    perm = tuple(i for i in range(tensor.ndim) if i != packed_dim) + (packed_dim,)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    temp = tensor.permute(perm)

    pack_factor = 32 // qtype.size_bits
    mask = (1 << qtype.size_bits) - 1

    assert temp.shape[-1] % pack_factor == 0
    new_shape = list(temp.shape)
    new_shape[-1] //= pack_factor

    result = torch.zeros(new_shape, dtype=torch.int32, device=tensor.device)
    for i in range(pack_factor):
        result |= ((temp[..., i::pack_factor] & mask) << (qtype.size_bits * i))

    return result.permute(inv_perm)


def unpack_quantized_values_into_int32(
        tensor: torch.Tensor,
        qtype: ScalarType,
        packed_dim: int = 0) -> torch.Tensor:
    perm = tuple(i for i in range(tensor.ndim) if i != packed_dim) + (packed_dim,)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    temp = tensor.permute(perm)

    pack_factor = 32 // qtype.size_bits
    mask = (1 << qtype.size_bits) - 1

    new_shape = list(temp.shape)
    new_shape[-1] *= pack_factor

    result = torch.zeros(new_shape, dtype=torch.int32, device=tensor.device)
    for i in range(pack_factor):
        result[..., i::pack_factor] = (temp >> (qtype.size_bits * i)) & mask

    return result.permute(inv_perm)


__all__ = [
    "_validate_machete_device_support",
    "check_machete_supports_shape",
    "machete_import_exception",
    "machete_mm",
    "machete_prepack_B",
    "machete_supported_schedules",
    "pack_quantized_values_into_int32",
    "query_machete_supported_act_types",
    "query_machete_supported_group_sizes",
    "query_machete_supported_quant_types",
    "unpack_quantized_values_into_int32",
]
