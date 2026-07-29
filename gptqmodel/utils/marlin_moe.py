# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""JIT extension and Python helper for the batched/offset Marlin MoE mega-kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)
from .marlin_scalar_type import ScalarType


_MARLIN_MOE_OPS_NAME = "gptqmodel_marlin_moe_ops"
_MARLIN_MOE_NAMESPACE = "gptqmodel_marlin_moe"


def _marlin_moe_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "marlin_moe"


def _marlin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "marlin"


def _marlin_moe_environment_error() -> str:
    if not torch.cuda.is_available():
        return "Marlin MoE kernel requires CUDA."
    major, minor = torch.cuda.get_device_capability()
    if major < 8 and not (major == 7 and minor >= 5):
        return f"Marlin MoE kernel requires compute capability >= 7.5, got {major}.{minor}."
    return ""


marlin_moe_import_exception = _marlin_moe_environment_error() or None


def _ensure_generated_marlin_moe_kernels() -> Path:
    root = _marlin_moe_root()
    generated = sorted(root.glob("sm*_kernel_*.cu"))
    selector = root / "kernel_selector.h"
    if generated and selector.exists():
        return root

    generator = root / "generate_kernels.py"
    archs = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        archs.add(f"{major}.{minor}")
    arch_arg = ",".join(sorted(archs)) if archs else "8.0"

    result = subprocess.run(
        [sys.executable, str(generator), arch_arg],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Marlin MoE kernel generation failed"
            + (f": {details}" if details else ".")
        )
    return root


def _marlin_moe_sources() -> List[str]:
    root = _ensure_generated_marlin_moe_kernels()
    sources = [
        str(root / "marlin_moe.cpp"),
        str(root / "ops.cu"),
    ]
    sources.extend(str(path) for path in sorted(root.glob("sm*_kernel_*.cu")))
    if len(sources) <= 2:
        raise RuntimeError(f"Marlin MoE kernel sources are incomplete under `{root}`.")
    return sources


def _marlin_moe_include_paths() -> List[str]:
    return cuda_include_paths_with_fallback(
        [str(_marlin_moe_root()), str(_marlin_root())],
    )


def _marlin_moe_extra_cflags() -> List[str]:
    return default_jit_cflags()


def _marlin_moe_extra_cuda_cflags() -> List[str]:
    flags = default_jit_cuda_cflags(
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


_MARLIN_MOE_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MARLIN_MOE_OPS_NAME,
    namespace=_MARLIN_MOE_NAMESPACE,
    required_ops=("moe_wna16_marlin_gemm",),
    sources=_marlin_moe_sources,
    build_root_env="GPTQMODEL_MARLIN_MOE_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("marlin_moe"),
    display_name="Marlin MoE",
    extra_cflags=_marlin_moe_extra_cflags,
    extra_cuda_cflags=_marlin_moe_extra_cuda_cflags,
    extra_include_paths=_marlin_moe_include_paths,
    force_rebuild_env="GPTQMODEL_MARLIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def marlin_moe_runtime_available() -> bool:
    if marlin_moe_import_exception is not None:
        return False
    from gptqmodel import extension as extension_api

    return extension_api.is_available("marlin_moe")


def marlin_moe_runtime_error() -> str:
    if marlin_moe_import_exception is not None:
        return marlin_moe_import_exception
    from gptqmodel import extension as extension_api

    if extension_api.is_available("marlin_moe"):
        return ""
    return extension_api.error("marlin_moe") or "Marlin MoE runtime unavailable."


def _resolve_marlin_moe_op():
    from gptqmodel import extension as extension_api

    return extension_api.op("marlin_moe", "moe_wna16_marlin_gemm")


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    b_q_type: ScalarType,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_fp32_reduce: bool = True,
    use_atomic_add: bool = False,
    mul_topk_weights: bool = False,
    b_bias: Optional[torch.Tensor] = None,
    a_scales: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    b_zeros: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    perm: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
    is_zp_float: bool = False,
) -> torch.Tensor:
    op = _resolve_marlin_moe_op()
    return op(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        thread_k,
        thread_n,
        blocks_per_sm,
    )


__all__ = [
    "marlin_moe_import_exception",
    "marlin_moe_runtime_available",
    "marlin_moe_runtime_error",
    "moe_wna16_marlin_gemm",
]
