# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

from typing import List, Optional

import torch

from ._extension_loader import load_extension_module
from .logger import setup_logger
from .marlin_scalar_type import ScalarType, scalar_types
from .rocm import IS_ROCM


log = setup_logger()

machete_import_exception: Optional[str] = None
try:
    gptqmodel_machete_kernels = load_extension_module("gptqmodel_machete_kernels")
except ImportError as exc:  # pragma: no cover - runtime guard
    machete_import_exception = str(exc)


MACHETE_PREPACKED_BLOCK_SHAPE = (64, 128)


def _validate_machete_device_support() -> bool:
    """
    Returns ``True`` when the active CUDA device can execute the Machete kernel.
    """
    if not torch.cuda.is_available() or IS_ROCM:
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


def machete_prepack_B(
    B: torch.Tensor,
    *,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: Optional[torch.dtype],
) -> torch.Tensor:
    if machete_import_exception is not None:
        raise ImportError(machete_import_exception)
    return torch.ops.gptqmodel_machete_kernels.machete_prepack_B(
        B,
        a_type,
        b_type.id,
        group_scales_type,
    )


def machete_mm(
    *,
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_type: ScalarType,
    b_group_scales: Optional[torch.Tensor],
    b_group_zeros: Optional[torch.Tensor],
    b_group_size: Optional[int],
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if machete_import_exception is not None:
        raise ImportError(machete_import_exception)
    return torch.ops.gptqmodel_machete_kernels.machete_mm(
        a,
        b_q,
        b_type.id,
        out_dtype,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        None,  # channel scales currently unused
        None,  # token scales currently unused
        None,  # schedule hint
    )


def machete_supported_schedules(
    *,
    a_type: torch.dtype,
    b_type: ScalarType,
    b_group_scales_type: Optional[torch.dtype],
    b_group_zeros_type: Optional[torch.dtype],
    out_type: Optional[torch.dtype] = None,
) -> List[str]:
    if machete_import_exception is not None:
        raise ImportError(machete_import_exception)
    return torch.ops.gptqmodel_machete_kernels.machete_supported_schedules(
        a_type,
        b_type.id,
        b_group_scales_type,
        b_group_zeros_type,
        None,
        None,
        out_type,
    )


def query_machete_supported_quant_types(include_zero_points: bool) -> List[ScalarType]:
    if include_zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_machete_supported_act_types(_with_zero_points: bool) -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def query_machete_supported_group_sizes(act_type: torch.dtype) -> List[int]:
    if act_type in (torch.float16, torch.bfloat16):
        return [-1, 64, 128]
    return [-1, 128]


def check_machete_supports_shape(in_features: int, out_features: int) -> tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return (
            False,
            f"in_features must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[0]} (got {in_features})",
        )
    if out_features % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return (
            False,
            f"out_features must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[1]} (got {out_features})",
        )
    return True, None


__all__ = [
    "_validate_machete_device_support",
    "check_machete_supports_shape",
    "machete_import_exception",
    "machete_mm",
    "machete_prepack_B",
    "machete_supported_schedules",
    "query_machete_supported_act_types",
    "query_machete_supported_group_sizes",
    "query_machete_supported_quant_types",
]
