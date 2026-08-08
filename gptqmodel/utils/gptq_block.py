# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""JIT wrapper for the quantization-time CUDA GPTQ block kernel."""

from __future__ import annotations

import threading
from collections.abc import Callable
from operator import index
from pathlib import Path

import torch
from torch import Tensor

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_GPTQ_BLOCK_OPS_NAME = "gptqmodel_gptq_block_ops"
_GPTQ_BLOCK_NAMESPACE = "gptqmodel_gptq_block"
_MAX_BLOCK_COLUMNS = 128
_MAX_BLOCK_ROWS = 2**31 - 1
_MAX_QUANTIZED_CODE = 2**8 - 1
_GPTQ_BLOCK_OP: Callable | None = None
_GPTQ_BLOCK_OP_INIT_LOCK = threading.Lock()


def _gptq_block_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "gptq_block"


def _gptq_block_sources() -> list[str]:
    return [str(_gptq_block_root() / "gptq_block_cuda.cu")]


def _gptq_block_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_diag_suppress=True,
    )


_GPTQ_BLOCK_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_GPTQ_BLOCK_OPS_NAME,
    namespace=_GPTQ_BLOCK_NAMESPACE,
    required_ops=("quantize",),
    sources=_gptq_block_sources,
    build_root_env="GPTQMODEL_GPTQ_BLOCK_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("gptq_block"),
    display_name="GPTQ CUDA block quantization",
    extra_cflags=default_jit_cflags,
    extra_cuda_cflags=_gptq_block_cuda_cflags,
    force_rebuild_env="GPTQMODEL_GPTQ_BLOCK_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _gptq_block_op() -> Callable:
    """Resolve the torch op once without serializing hot multi-GPU launches."""
    global _GPTQ_BLOCK_OP
    if _GPTQ_BLOCK_OP is None:
        with _GPTQ_BLOCK_OP_INIT_LOCK:
            if _GPTQ_BLOCK_OP is None:
                _GPTQ_BLOCK_OP = _extension_api().op("gptq_block", "quantize")
    return _GPTQ_BLOCK_OP


def gptq_block_cuda_supported() -> bool:
    return torch.cuda.is_available() and torch.version.hip is None


def gptq_block_cuda_available() -> bool:
    return gptq_block_cuda_supported() and _extension_api().is_available("gptq_block")


def gptq_block_cuda_error() -> str:
    if not torch.cuda.is_available():
        return "GPTQ CUDA block quantization requires CUDA."
    if torch.version.hip is not None:
        return "GPTQ CUDA block quantization requires NVIDIA CUDA; ROCm is not supported."
    return _extension_api().error("gptq_block")


def prewarm_gptq_block_cuda() -> bool:
    return _extension_api().load(name="gptq_block")["gptq_block"]


def _integer_argument(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc


def gptq_block_cuda(
    weights: Tensor,
    hessian_inverse: Tensor,
    scale: Tensor,
    zero: Tensor,
    maxq: int,
    group_size: int,
    *,
    groupwise: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize one serial GPTQ column block on the current CUDA stream."""

    maxq = _integer_argument("maxq", maxq)
    group_size = _integer_argument("group_size", group_size)
    if not isinstance(groupwise, bool):
        raise TypeError(f"groupwise must be a bool, got {type(groupwise).__name__}")

    if weights.ndim != 2:
        raise ValueError(
            f"weights must be two-dimensional, got shape {tuple(weights.shape)}"
        )
    rows, count = weights.shape
    if rows <= 0 or count <= 0:
        raise ValueError(
            f"weights dimensions must be positive, got shape {tuple(weights.shape)}"
        )
    if rows > _MAX_BLOCK_ROWS:
        raise ValueError(
            f"CUDA block kernel supports rows <= {_MAX_BLOCK_ROWS}, got {rows}"
        )
    if count > _MAX_BLOCK_COLUMNS:
        raise ValueError(
            f"CUDA block kernel supports count <= {_MAX_BLOCK_COLUMNS}, got {count}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if count % group_size != 0:
        raise ValueError(f"group_size {group_size} must divide count {count}")
    if maxq <= 0:
        raise ValueError(f"maxq must be positive, got {maxq}")
    if maxq > _MAX_QUANTIZED_CODE:
        raise ValueError(
            f"CUDA block kernel supports maxq <= {_MAX_QUANTIZED_CODE}, got {maxq}"
        )

    operands = (weights, hessian_inverse, scale, zero)
    if any(tensor.dtype != torch.float32 for tensor in operands):
        raise TypeError(
            "CUDA GPTQ block kernel expects float32 weights/Hinv/scale/zero"
        )
    devices = {tensor.device for tensor in operands}
    if len(devices) != 1:
        raise ValueError(
            f"CUDA GPTQ block tensors must share one device, got {sorted(map(str, devices))}"
        )
    if not weights.is_cuda:
        raise ValueError(
            f"CUDA GPTQ block tensors must be CUDA tensors, got {weights.device}"
        )
    if hessian_inverse.shape != (count, count):
        raise ValueError(
            f"hessian_inverse must have shape {(count, count)}, got {tuple(hessian_inverse.shape)}"
        )
    expected_scale_shape = (rows, count // group_size)
    if scale.shape != expected_scale_shape or zero.shape != expected_scale_shape:
        raise ValueError(
            f"scale/zero must have shape {expected_scale_shape}, got {tuple(scale.shape)}/{tuple(zero.shape)}"
        )

    if out is None:
        quantized = torch.empty_like(weights, memory_format=torch.contiguous_format)
        errors = torch.empty_like(weights, memory_format=torch.contiguous_format)
    else:
        quantized, errors = out
        if quantized.shape != weights.shape or errors.shape != weights.shape:
            raise ValueError(
                f"out tensors must match weights shape {tuple(weights.shape)}, got "
                f"{tuple(quantized.shape)}/{tuple(errors.shape)}"
            )
        if quantized.dtype != torch.float32 or errors.dtype != torch.float32:
            raise TypeError("CUDA GPTQ block out tensors must have dtype float32")
        if quantized.device != weights.device or errors.device != weights.device:
            raise ValueError(
                "CUDA GPTQ block out tensors must share the weights device"
            )
        if not quantized.is_contiguous() or not errors.is_contiguous():
            raise ValueError("CUDA GPTQ block out tensors must be contiguous")
        output_storage = {
            quantized.untyped_storage().data_ptr(),
            errors.untyped_storage().data_ptr(),
        }
        input_storage = {tensor.untyped_storage().data_ptr() for tensor in operands}
        if len(output_storage) != 2 or output_storage.intersection(input_storage):
            raise ValueError(
                "CUDA GPTQ block out tensors must not alias each other or any input"
            )

    prepared_operands = tuple(tensor.contiguous() for tensor in operands)
    # Cold op resolution is serialized once; cached torch-dispatch/CUDA launches
    # carry no mutable Python JIT state and need no process-wide launch lock.
    op = _gptq_block_op()

    # The custom operator will launch asynchronously on the current stream.
    # Record every caller-owned tensor and contiguous temporary before launch so
    # the CUDA allocator cannot recycle storage from another free-threaded worker
    # while it is in flight. Deduplicate the normal path to keep this to six calls.
    stream = torch.cuda.current_stream(weights.device)
    recorded_storage: set[int] = set()
    for tensor in (*operands, *prepared_operands, quantized, errors):
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr not in recorded_storage:
            tensor.record_stream(stream)
            recorded_storage.add(storage_ptr)
    op(
        *prepared_operands,
        int(maxq),
        int(group_size),
        bool(groupwise),
        quantized,
        errors,
    )
    return quantized, errors


__all__ = [
    "gptq_block_cuda",
    "gptq_block_cuda_available",
    "gptq_block_cuda_error",
    "gptq_block_cuda_supported",
    "prewarm_gptq_block_cuda",
]
