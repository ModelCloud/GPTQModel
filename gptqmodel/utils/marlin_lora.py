# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)
from .env import env_flag
from .logger import setup_logger


log = setup_logger()

_MARLIN_LORA_OPS_NAME = "gptqmodel_marlin_lora_ops"
_MARLIN_LORA_NAMESPACE = "gptqmodel_marlin_lora"
_MARLIN_LORA_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
)
_FUSED_ENV = "GPTQMODEL_MARLIN_LORA_FUSED"
_FUSED_COOPERATIVE_ENV = "GPTQMODEL_MARLIN_LORA_COOPERATIVE"
_FUSED_CUDA_UP_ADD_ENV = "GPTQMODEL_MARLIN_LORA_CUDA_UP_ADD"
_FUSED_MAX_M_ENV = "GPTQMODEL_MARLIN_LORA_FUSED_MAX_M"
_FUSED_MAX_RANK_ENV = "GPTQMODEL_MARLIN_LORA_FUSED_MAX_RANK"
_FUSED_DISABLED_REASON: Optional[str] = None


def _marlin_lora_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "marlin_lora"


def _marlin_lora_sources() -> list[str]:
    root = _marlin_lora_root()
    return [
        str(root / "marlin_lora.cpp"),
        str(root / "marlin_lora_kernel.cu"),
    ]


def _marlin_lora_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_marlin_lora_root())],
        required_header_names=_MARLIN_LORA_REQUIRED_CUDA_HEADERS,
    )


def _marlin_lora_extra_cuda_cflags() -> list[str]:
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


_MARLIN_LORA_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MARLIN_LORA_OPS_NAME,
    namespace=_MARLIN_LORA_NAMESPACE,
    required_ops=("lora_fused_add", "lora_fused_add_prepared", "lora_up_add"),
    sources=_marlin_lora_sources,
    build_root_env="GPTQMODEL_MARLIN_LORA_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("marlin_lora"),
    display_name="Marlin fused LoRA",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_marlin_lora_extra_cuda_cflags,
    extra_include_paths=_marlin_lora_include_paths,
    force_rebuild_env="GPTQMODEL_MARLIN_LORA_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def marlin_lora_supported() -> bool:
    return torch.cuda.is_available()


def marlin_lora_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "Marlin fused LoRA requires CUDA."
    return _MARLIN_LORA_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def marlin_lora_runtime_available() -> bool:
    if not marlin_lora_supported():
        return False
    return _extension_api().is_available("marlin_lora")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        log.warn.once(f"Invalid {name}={raw!r}; using default {default}.")
        return default


def _can_try_fused_lora(x: torch.Tensor, out: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor) -> bool:
    if not env_flag(_FUSED_ENV, default=True):
        return False
    if not torch.cuda.is_available() or x.device.type != "cuda" or out.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or out.dtype != x.dtype:
        return False
    if lora_a is None or lora_b is None:
        return False
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        return False
    if x.shape[-1] != lora_a.shape[0] or lora_a.shape[1] != lora_b.shape[0] or out.shape[-1] != lora_b.shape[1]:
        return False
    if out.dim() != x.dim() or not out.is_contiguous():
        return False

    rows = x.numel() // x.shape[-1]
    max_m = _env_int(_FUSED_MAX_M_ENV, 16)
    max_rank = _env_int(_FUSED_MAX_RANK_ENV, 512)
    return rows <= max_m and lora_a.shape[1] <= max_rank


def _ensure_lora_tensors_for(
    adapter,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lora_a = adapter.lora_A
    lora_b = adapter.lora_B
    if lora_a.dtype != dtype or lora_a.device != device or lora_b.dtype != dtype or lora_b.device != device:
        log.info.once(
            f"Adapter: Lora A/B auto changed from `{lora_a.dtype}` on `{lora_a.device}` "
            f"to `{dtype}` on `{device}` to match forward input."
        )
        adapter.lora_A = lora_a.to(device=device, dtype=dtype).contiguous()
        adapter.lora_B = lora_b.to(device=device, dtype=dtype).contiguous()
    elif not lora_a.is_contiguous() or not lora_b.is_contiguous():
        adapter.lora_A = lora_a.contiguous()
        adapter.lora_B = lora_b.contiguous()
    return adapter.lora_A, adapter.lora_B


def _ensure_lora_tensors(adapter, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _ensure_lora_tensors_for(adapter, device=x.device, dtype=x.dtype)


def marlin_lora_cuda_up_add_enabled() -> bool:
    return env_flag(_FUSED_ENV, default=True) and env_flag(_FUSED_CUDA_UP_ADD_ENV, default=False)


def prepare_marlin_fused_lora(
    adapter,
    *,
    device: torch.device,
    dtype: torch.dtype,
    in_features: int,
    out_features: int,
    use_prepared_marlin: bool = False,
):
    """Prepare the Ampere cooperative LoRA state, or return None for the portable fallback."""

    global _FUSED_DISABLED_REASON

    if not env_flag(_FUSED_ENV, default=True) or not env_flag(_FUSED_COOPERATIVE_ENV, default=True):
        return None
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return None
    if torch.cuda.get_device_capability(device) != (8, 0):
        return None
    has_compressed_lora = getattr(adapter, "_has_compressed_lora", None)
    if callable(has_compressed_lora) and has_compressed_lora():
        return None

    lora_a = getattr(adapter, "lora_A", None)
    lora_b = getattr(adapter, "lora_B", None)
    if lora_a is None or lora_b is None or lora_a.dim() != 2 or lora_b.dim() != 2:
        return None
    rank = lora_a.shape[1]
    max_rows = _env_int(_FUSED_MAX_M_ENV, 16)
    max_rank = _env_int(_FUSED_MAX_RANK_ENV, 512)
    if (
        max_rows < 1
        or rank < 1
        or rank > max_rank
        or lora_a.shape != (in_features, rank)
        or lora_b.shape != (rank, out_features)
    ):
        return None
    lora_a, lora_b = _ensure_lora_tensors_for(adapter, device=device, dtype=dtype)
    marlin_extension = "marlin_bf16" if dtype == torch.bfloat16 else "marlin_fp16"
    dtype_tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    prepared_suffix = "_prepared" if use_prepared_marlin else ""
    op_name = f"gptq_marlin_gemm_lora{prepared_suffix}_{dtype_tag}"
    try:
        op = _extension_api().op(marlin_extension, op_name)
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        if _FUSED_DISABLED_REASON != reason:
            log.warn(f"Marlin cooperative LoRA unavailable; using standard adapter path: {reason}")
            _FUSED_DISABLED_REASON = reason
        return None

    workspace = None if use_prepared_marlin else torch.empty((1, rank), dtype=torch.float32, device=device)
    return op, lora_a, lora_b, workspace, max_rows, use_prepared_marlin


def apply_marlin_fused_lora(
    adapter,
    *,
    x: torch.Tensor,
    out: torch.Tensor,
    cooperative_buffer: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Try the optional Marlin+LoRA fused tail and return None on fallback."""

    global _FUSED_DISABLED_REASON

    lora_a = getattr(adapter, "lora_A", None)
    lora_b = getattr(adapter, "lora_B", None)
    if not _can_try_fused_lora(x, out, lora_a, lora_b):
        return None

    lora_a, lora_b = _ensure_lora_tensors(adapter, x)
    x_2d = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    out_2d = out if out.dim() == 2 else out.reshape(-1, out.shape[-1])

    if env_flag(_FUSED_COOPERATIVE_ENV, default=True) and cooperative_buffer is not None:
        if torch.cuda.get_device_capability(x.device) != (8, 0):
            return None
        try:
            op = _extension_api().op("marlin_lora", "lora_fused_add")
            op(x_2d, lora_a, lora_b, out_2d, cooperative_buffer)
            return out
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            if _FUSED_DISABLED_REASON != reason:
                log.warn(f"Marlin cooperative LoRA disabled; falling back to standard adapter path: {reason}")
                _FUSED_DISABLED_REASON = reason
            return None

    down = torch.matmul(x_2d, lora_a).contiguous()

    if not env_flag(_FUSED_CUDA_UP_ADD_ENV, default=False):
        torch.addmm(out_2d, down, lora_b, beta=1.0, alpha=1.0, out=out_2d)
        return out

    try:
        op = _extension_api().op("marlin_lora", "lora_up_add")
        op(down, lora_b, out_2d)
        return out
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        if _FUSED_DISABLED_REASON != reason:
            log.warn(f"Marlin fused LoRA disabled; falling back to standard adapter path: {reason}")
            _FUSED_DISABLED_REASON = reason
        return None
