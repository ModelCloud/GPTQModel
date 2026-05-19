# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GrassHopper GPTQ grouped GEMV/GEMM helpers.

This module is the user-facing alias for the existing VecQuant3 extension
namespace while callers migrate to the broader 3/4/8-bit kernel name.
"""

from __future__ import annotations

from .vecquant3 import (
    SUPPORTED_BITS,
    gemm,
    gemm_lora,
    gemm_lora_int8,
    gemv,
    gemv_lora,
    gemv_lora_int8,
    grasshopper_runtime_available,
    grasshopper_runtime_error,
    grasshopper_supported,
)

runtime_available = grasshopper_runtime_available
runtime_error = grasshopper_runtime_error
supported = grasshopper_supported

__all__ = [
    "SUPPORTED_BITS",
    "gemm",
    "gemm_lora",
    "gemm_lora_int8",
    "gemv",
    "gemv_lora",
    "gemv_lora_int8",
    "grasshopper_runtime_available",
    "grasshopper_runtime_error",
    "grasshopper_supported",
    "runtime_available",
    "runtime_error",
    "supported",
]
