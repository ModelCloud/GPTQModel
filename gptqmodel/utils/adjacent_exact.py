# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""JIT wrapper for the CUDA adjacent-rounding solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import Tensor

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_ADJACENT_EXACT_OPS_NAME = "gptqmodel_adjacent_exact_ops"
_ADJACENT_EXACT_NAMESPACE = "gptqmodel_adjacent_exact"
_MAX_WORKER_WARPS = 1 << 20
_MAX_BRANCH_BOUND_DECISIONS = 128
_MAX_BRANCH_BOUND_SPLIT_DEPTH = 20


@dataclass(frozen=True)
class AdjacentBranchBoundCandidates:
    """Native branch-and-bound candidates and certification metadata."""

    states: Tensor
    costs: Tensor
    root_lower_bounds: Tensor
    nodes_visited: Tensor
    completed: Tensor
    global_best: Tensor


def _adjacent_exact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "adjacent_exact"


def _adjacent_exact_sources() -> list[str]:
    return [
        str(_adjacent_exact_root() / "adjacent_exact_cuda.cu"),
        str(_adjacent_exact_root() / "adjacent_branch_bound_cuda.cu"),
    ]


def _adjacent_exact_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_diag_suppress=True,
    )


_ADJACENT_EXACT_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_ADJACENT_EXACT_OPS_NAME,
    namespace=_ADJACENT_EXACT_NAMESPACE,
    required_ops=("exact_candidates", "branch_bound"),
    sources=_adjacent_exact_sources,
    build_root_env="GPTQMODEL_ADJACENT_EXACT_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("adjacent_exact"),
    display_name="AdjacentExact CUDA solvers",
    extra_cflags=default_jit_cflags,
    extra_cuda_cflags=_adjacent_exact_cuda_cflags,
    force_rebuild_env="GPTQMODEL_ADJACENT_EXACT_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def adjacent_exact_cuda_supported() -> bool:
    return torch.cuda.is_available()


def adjacent_exact_cuda_available() -> bool:
    return adjacent_exact_cuda_supported() and _extension_api().is_available(
        "adjacent_exact"
    )


def adjacent_exact_cuda_error() -> str:
    if not torch.cuda.is_available():
        return "AdjacentExact CUDA requires CUDA."
    return _extension_api().error("adjacent_exact")


def prewarm_adjacent_exact_cuda() -> bool:
    return _extension_api().load(name="adjacent_exact")["adjacent_exact"]


def adjacent_exact_candidates(
    constant: Tensor,
    linear: Tensor,
    interaction: Tensor,
    *,
    warps: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return one exhaustive-search winner and energy per CUDA worker warp."""

    if constant.ndim != 0 or constant.dtype != torch.float64 or not constant.is_cuda:
        raise ValueError("constant must be a scalar CUDA float64 tensor.")
    if linear.ndim != 1 or linear.dtype != torch.float64 or not linear.is_cuda:
        raise ValueError("linear must be a rank-one CUDA float64 tensor.")
    if not 1 <= linear.numel() <= 32:
        raise ValueError("linear must contain between 1 and 32 binary decisions.")
    if (
        interaction.shape != (linear.numel(), linear.numel())
        or interaction.dtype != torch.float64
        or not interaction.is_cuda
    ):
        raise ValueError(
            "interaction must be a square CUDA float64 tensor matching linear."
        )
    if constant.device != linear.device or interaction.device != linear.device:
        raise ValueError(
            "constant, linear, and interaction must be on the same CUDA device."
        )
    if not bool(torch.isfinite(constant)) or not bool(torch.isfinite(linear).all()):
        raise ValueError("constant and linear must contain only finite values.")
    if not bool(torch.isfinite(interaction).all()):
        raise ValueError("interaction must contain only finite values.")
    if not torch.equal(interaction, interaction.mT):
        raise ValueError("interaction must be exactly symmetric.")
    if bool(torch.count_nonzero(interaction.diagonal())):
        raise ValueError("interaction diagonal must be exactly zero.")
    if not 0 <= warps <= _MAX_WORKER_WARPS:
        raise ValueError(
            f"warps must be zero (automatic) or in [1, {_MAX_WORKER_WARPS}]."
        )

    op = _extension_api().op("adjacent_exact", "exact_candidates")
    return op(
        constant.contiguous(), linear.contiguous(), interaction.contiguous(), warps
    )


def adjacent_branch_bound_candidates(
    constant: Tensor,
    linear: Tensor,
    interaction: Tensor,
    *,
    split_depth: int,
    max_nodes_per_worker: int = 0,
    certificate_tolerance: float = 1e-12,
) -> AdjacentBranchBoundCandidates:
    """Search one dense 1–128-variable QUBO with native CUDA branch-and-bound."""

    if constant.ndim != 0 or constant.dtype != torch.float64 or not constant.is_cuda:
        raise ValueError("constant must be a scalar CUDA float64 tensor.")
    if linear.ndim != 1 or linear.dtype != torch.float64 or not linear.is_cuda:
        raise ValueError("linear must be a rank-one CUDA float64 tensor.")
    if not 1 <= linear.numel() <= _MAX_BRANCH_BOUND_DECISIONS:
        raise ValueError(
            f"linear must contain between 1 and {_MAX_BRANCH_BOUND_DECISIONS} binary decisions."
        )
    if (
        interaction.shape != (linear.numel(), linear.numel())
        or interaction.dtype != torch.float64
        or not interaction.is_cuda
    ):
        raise ValueError(
            "interaction must be a square CUDA float64 tensor matching linear."
        )
    if constant.device != linear.device or interaction.device != linear.device:
        raise ValueError(
            "constant, linear, and interaction must be on the same CUDA device."
        )
    if not bool(torch.isfinite(constant)) or not bool(torch.isfinite(linear).all()):
        raise ValueError("constant and linear must contain only finite values.")
    if not bool(torch.isfinite(interaction).all()):
        raise ValueError("interaction must contain only finite values.")
    if not torch.equal(interaction, interaction.mT):
        raise ValueError("interaction must be exactly symmetric.")
    if bool(torch.count_nonzero(interaction.diagonal())):
        raise ValueError("interaction diagonal must be exactly zero.")
    if not 0 <= split_depth <= min(
        int(linear.numel()), _MAX_BRANCH_BOUND_SPLIT_DEPTH
    ):
        raise ValueError(
            "split_depth must be in "
            f"[0, min(linear.numel(), {_MAX_BRANCH_BOUND_SPLIT_DEPTH})]."
        )
    if max_nodes_per_worker < 0:
        raise ValueError(
            "max_nodes_per_worker must be zero (unlimited) or positive."
        )
    if not math.isfinite(certificate_tolerance) or certificate_tolerance < 0:
        raise ValueError("certificate_tolerance must be finite and non-negative.")

    op = _extension_api().op("adjacent_exact", "branch_bound")
    states, costs, lower_bounds, nodes, completed, global_best = op(
        constant.contiguous(),
        linear.contiguous(),
        interaction.contiguous(),
        split_depth,
        max_nodes_per_worker,
        certificate_tolerance,
    )
    return AdjacentBranchBoundCandidates(
        states=states,
        costs=costs,
        root_lower_bounds=lower_bounds,
        nodes_visited=nodes,
        completed=completed,
        global_best=global_best,
    )


__all__ = [
    "AdjacentBranchBoundCandidates",
    "adjacent_branch_bound_candidates",
    "adjacent_exact_candidates",
    "adjacent_exact_cuda_available",
    "adjacent_exact_cuda_error",
    "adjacent_exact_cuda_supported",
    "prewarm_adjacent_exact_cuda",
]
