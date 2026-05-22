# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)


_GRASSHOPPER_OPS_NAME = "gptqmodel_grasshopper_ops"
_GRASSHOPPER_NAMESPACE = "gptqmodel_grasshopper"
_GRASSHOPPER_REQUIRED_CUDA_HEADERS = ("cuda_runtime_api.h",)
_GRASSHOPPER_ACCUMULATION_FLOAT32 = 0
_GRASSHOPPER_ACCUMULATION_INPUT = 1
SUPPORTED_BITS = (3, 4, 8)


def _grasshopper_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "grasshopper"


def _grasshopper_sources() -> list[str]:
    root = _grasshopper_root()
    return [
        str(root / "grasshopper.cpp"),
        str(root / "grasshopper_kernel.cu"),
    ]


def _grasshopper_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_grasshopper_root())],
        required_header_names=_GRASSHOPPER_REQUIRED_CUDA_HEADERS,
    )


def _grasshopper_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_GRASSHOPPER_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_GRASSHOPPER_OPS_NAME,
    namespace=_GRASSHOPPER_NAMESPACE,
    required_ops=(
        "gemv",
        "gemv_lora",
        "gemv_lora_int8",
        "gemm",
        "gemm_lora",
        "gemm_lora_int8",
    ),
    sources=_grasshopper_sources,
    build_root_env="GPTQMODEL_GRASSHOPPER_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("grasshopper"),
    display_name="GrassHopper GPTQ grouped GEMV/GEMM",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_grasshopper_extra_cuda_cflags,
    extra_include_paths=_grasshopper_include_paths,
    force_rebuild_env="GPTQMODEL_GRASSHOPPER_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def grasshopper_supported() -> bool:
    return torch.cuda.is_available()


def grasshopper_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "GrassHopper GPTQ grouped GEMV/GEMM requires CUDA."
    return _GRASSHOPPER_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def grasshopper_runtime_available() -> bool:
    if not grasshopper_supported():
        return False
    return _extension_api().is_available("grasshopper")


def _normalize_accumulation_dtype(
    accumulation_dtype: str | torch.dtype | None,
    input_dtype: torch.dtype,
) -> int:
    if input_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "GrassHopper GPTQ grouped GEMV supports only torch.float16 or torch.bfloat16 input tensors."
        )

    if accumulation_dtype is None:
        return _GRASSHOPPER_ACCUMULATION_FLOAT32

    if isinstance(accumulation_dtype, torch.dtype):
        if accumulation_dtype == torch.float32:
            return _GRASSHOPPER_ACCUMULATION_FLOAT32
        if accumulation_dtype == input_dtype:
            return _GRASSHOPPER_ACCUMULATION_INPUT
        raise ValueError(
            "`accumulation_dtype` must be torch.float32, 'input', or the same low-precision dtype as the input tensor."
        )

    normalized = str(accumulation_dtype).strip().lower().replace("torch.", "")
    normalized = normalized.replace("-", "_")
    if normalized in {"float32", "fp32", "f32", "float"}:
        return _GRASSHOPPER_ACCUMULATION_FLOAT32
    if normalized in {"input", "input_dtype", "native", "low", "low_precision"}:
        return _GRASSHOPPER_ACCUMULATION_INPUT
    if normalized in {"float16", "fp16", "f16", "half"} and input_dtype == torch.float16:
        return _GRASSHOPPER_ACCUMULATION_INPUT
    if normalized in {"bfloat16", "bf16"} and input_dtype == torch.bfloat16:
        return _GRASSHOPPER_ACCUMULATION_INPUT
    raise ValueError(
        "`accumulation_dtype` must be one of float32/fp32, input/native, "
        "float16/fp16 for fp16 inputs, or bfloat16/bf16 for bf16 inputs."
    )


def _normalize_bits(bits: int) -> int:
    normalized = int(bits)
    if normalized not in SUPPORTED_BITS:
        raise ValueError("GrassHopper GPTQ grouped GEMV/GEMM supports only 3, 4, or 8-bit base weights.")
    return normalized


def gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemv(x, qweight, scales, qzeros, int(group_size), accumulation_type, _normalize_bits(bits))


def gemv_lora(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemv_lora(
        x, qweight, scales, qzeros, down, up, int(group_size), accumulation_type, _normalize_bits(bits)
    )


def gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemm(x, qweight, scales, qzeros, int(group_size), accumulation_type, _normalize_bits(bits))


def gemm_lora(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemm_lora(
        x, qweight, scales, qzeros, down, up, int(group_size), accumulation_type, _normalize_bits(bits)
    )


def _normalize_lora_int8_up_shape(up_shape: torch.Tensor | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(up_shape, torch.Tensor):
        shape_values = up_shape.detach().cpu().reshape(-1).tolist()
    else:
        shape_values = list(up_shape)
    if len(shape_values) != 2:
        raise ValueError("GrassHopper int8 LoRA-B shape must contain exactly [rank, out_features].")
    rank, out_features = (int(shape_values[0]), int(shape_values[1]))
    if rank <= 0 or out_features <= 0:
        raise ValueError("GrassHopper int8 LoRA-B shape values must be positive.")
    return rank, out_features


def gemv_lora_int8(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    down: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_shape: torch.Tensor | tuple[int, int] | list[int],
    group_size: int,
    lora_group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    rank, out_features = _normalize_lora_int8_up_shape(up_shape)
    if int(down.reshape(-1).numel()) != rank:
        raise ValueError("GrassHopper int8 LoRA-B rank must match the down projection length.")
    if int(qweight.size(1)) != out_features:
        raise ValueError("GrassHopper int8 LoRA-B out_features must match the quantized base output width.")
    if lora_group_size <= 0:
        raise ValueError("GrassHopper int8 LoRA-B group size must be positive.")
    expected_values = rank * out_features
    if int(up_qweight.numel()) < expected_values:
        raise ValueError("GrassHopper int8 LoRA-B qweight is too small for the provided shape.")
    expected_scale_groups = (expected_values + int(lora_group_size) - 1) // int(lora_group_size)
    if int(up_scales.numel()) < expected_scale_groups:
        raise ValueError("GrassHopper int8 LoRA-B scales are too small for the provided shape/group size.")

    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemv_lora_int8(
        x,
        qweight,
        scales,
        qzeros,
        down,
        up_qweight,
        up_scales,
        int(group_size),
        int(lora_group_size),
        accumulation_type,
        _normalize_bits(bits),
    )


def gemm_lora_int8(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    down: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_shape: torch.Tensor | tuple[int, int] | list[int],
    group_size: int,
    lora_group_size: int,
    accumulation_dtype: str | torch.dtype | None = torch.float32,
    bits: int = 3,
) -> torch.Tensor:
    rank, out_features = _normalize_lora_int8_up_shape(up_shape)
    if int(down.size(-1)) != rank:
        raise ValueError("GrassHopper int8 LoRA-B rank must match the down projection width.")
    if int(qweight.size(1)) != out_features:
        raise ValueError("GrassHopper int8 LoRA-B out_features must match the quantized base output width.")
    if lora_group_size <= 0:
        raise ValueError("GrassHopper int8 LoRA-B group size must be positive.")
    expected_values = rank * out_features
    if int(up_qweight.numel()) < expected_values:
        raise ValueError("GrassHopper int8 LoRA-B qweight is too small for the provided shape.")
    expected_scale_groups = (expected_values + int(lora_group_size) - 1) // int(lora_group_size)
    if int(up_scales.numel()) < expected_scale_groups:
        raise ValueError("GrassHopper int8 LoRA-B scales are too small for the provided shape/group size.")

    ops = _extension_api().namespace(name="grasshopper")
    accumulation_type = _normalize_accumulation_dtype(accumulation_dtype, x.dtype)
    return ops.gemm_lora_int8(
        x,
        qweight,
        scales,
        qzeros,
        down,
        up_qweight,
        up_scales,
        int(group_size),
        int(lora_group_size),
        accumulation_type,
        _normalize_bits(bits),
    )


runtime_available = grasshopper_runtime_available
runtime_error = grasshopper_runtime_error
supported = grasshopper_supported


__all__ = [
    "SUPPORTED_BITS",
    "_GRASSHOPPER_TORCH_OPS_EXTENSION",
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
