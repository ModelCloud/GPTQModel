# SPDX-FileCopyrightText: 2024-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 ModelCloud.ai (qubitium@modelcloud.ai)
# SPDX-License-Identifier: Apache-2.0
# EoRA arXiv https://arxiv.org/abs/2410.21271
# EoRA Official Repo: https://github.com/NVlabs/EoRA
# This file has been modified by ModelCloud.AI team and qubitium@modelcloud.ai for adoption into GPT-QModel

# EoRA
# @article{liu2024eora,
#   title={EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation},
#   author={Liu, Shih-Yang and Yang, Huck and Wang, Chien-Yi and Fung, Nai Chit and Yin, Hongxu and Sakr, Charbel and Muralidharan, Saurav and Cheng, Kwang-Ting and Kautz, Jan and Wang, Yu-Chiang Frank and others},
#   journal={arXiv preprint arXiv:2410.21271},
#   year={2024}
# }

import os
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..utils.env import env_flag
from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM
from ..utils.torch import TORCH_GTE_210

log = setup_logger()

_EORA_CHOLESKY_ENV = "GPTQMODEL_EORA_CHOLESKY"


def _eora_covariance_rtol(size: int) -> float:
    """Return the numerical-rank tolerance for float32 EoRA covariance data."""

    return min(1.0, max(1, int(size)) * torch.finfo(torch.float32).eps)


def eora_process_input(
        input: Tensor,
        name: str,
        sample_size: int,
        device: torch.device,
) -> Tuple[int, torch.Tensor, float]:
    """Prepare the per-batch covariance contribution required for EoRA.

    The contribution remains on the originating device so multi-GPU execution
    can accumulate locally before a single merge step.
    """

    inp = input[0].to(device=device, dtype=torch.float32)
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)

    batch = inp.shape[0]
    adds = torch.matmul(inp.transpose(1, 2), inp)
    adds_sum = torch.sum(adds, dim=0).detach()

    contribution = adds_sum.to(dtype=torch.float32)
    contribution /= float(sample_size)

    # Adding batch to denominator is only for mathematical stability
    scale = float(sample_size) / (float(sample_size) + float(batch))

    del inp, adds, adds_sum

    return batch, contribution, scale


def merge_eora_segments(segments: Sequence[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Combine pre-aggregated EoRA segments using their scale products.

    Each segment entry is a tuple ``(total, scale_product)`` where ``total`` is
    the sequential accumulation result for that segment starting from zero, and
    ``scale_product`` is the product of per-batch scale factors encountered in
    the same segment.  The function mutates the first segment tensor in place
    and returns it as the merged result.
    """
    if not segments:
        raise ValueError("EoRA merge requires at least one segment.")

    result: torch.Tensor | None = None
    for total, scale_product in segments:
        if result is None:
            result = total
        else:
            result.mul_(float(scale_product))
            result.add_(total)

    assert result is not None
    return result


def _eora_compute_lora_eigh(
        w_wq_delta: Tensor,
        name: str,
        raw_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)

    if not torch.isfinite(L).all():
        raise FloatingPointError(f"EoRA covariance eigensolve produced non-finite eigenvalues for `{name}`.")

    # EoRA covariance contributions are accumulated in float32 before this
    # eigensolve. Eigenvalues below the corresponding numerical-rank cutoff
    # cannot be inverted reliably, even though the decomposition uses float64.
    relative_tolerance = _eora_covariance_rtol(L.numel())
    maximum = L[-1].clamp_min(0)
    cutoff = maximum * relative_tolerance
    retained = L > cutoff
    discarded_count = int((~retained).sum().item())

    if discarded_count:
        negative_count = int((L < 0).sum().item())
        log.warning(
            f"EoRA: covariance for `{name}` is numerically rank deficient; using a truncated pseudoinverse "
            f"and discarding {discarded_count}/{L.numel()} eigenvalues at or below {cutoff.item():.3e} "
            f"(negative={negative_count}, min={L[0].item():.3e}, max={L[-1].item():.3e}, "
            f"rtol={relative_tolerance:.3e})."
        )

    sqrt_eigenvalues = torch.zeros_like(L)
    inverse_sqrt_eigenvalues = torch.zeros_like(L)
    sqrt_eigenvalues[retained] = torch.sqrt(L[retained])
    inverse_sqrt_eigenvalues[retained] = torch.rsqrt(L[retained])

    # Q @ diag(sqrt_eigenvalues) and diag(inverse_sqrt_eigenvalues) @ Q.T,
    # expressed without materializing either dense diagonal matrix.
    scaling_diag_matrix = Q * sqrt_eigenvalues.unsqueeze(0)
    scaling_matrix_inv = inverse_sqrt_eigenvalues.unsqueeze(1) * Q.T

    scaling_diag_matrix = scaling_diag_matrix.to(dtype=torch.float32)
    scaling_matrix_inv = scaling_matrix_inv.to(dtype=torch.float32)

    delta_scale = torch.matmul(w_wq_delta, scaling_diag_matrix)

    U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
    lowrank_r = rank
    sqrt_s = torch.sqrt(S[:lowrank_r])
    B = (U[:, :lowrank_r] * sqrt_s.unsqueeze(0)).to(dtype=dtype)
    truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
    A = (sqrt_s.unsqueeze(1) * truc_v).to(dtype=dtype)

    del L, Q, U, S, V,
    del sqrt_eigenvalues, inverse_sqrt_eigenvalues, scaling_diag_matrix, scaling_matrix_inv, delta_scale
    del retained, maximum, cutoff, sqrt_s, truc_v

    return A.contiguous(), B.contiguous()


def _eora_compute_lora_cholesky(
        w_wq_delta: Tensor,
        name: str,
        raw_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
) -> Tuple[Tensor, Tensor] | None:
    if not hasattr(torch.linalg, "cholesky_ex"):
        log.warn.once("EoRA: Cholesky fast path requires torch.linalg.cholesky_ex; falling back to eigensolve.")
        return None

    scaling_diag_matrix, info = torch.linalg.cholesky_ex(raw_scaling_diag_matrix, check_errors=False)
    info_value = int(info.item())
    if info_value != 0:
        log.warn.once(
            f"EoRA: Cholesky fast path skipped for `{name}` because the covariance matrix is not positive definite "
            f"(cholesky_ex info={info_value}); falling back to eigensolve."
        )
        return None

    scaling_diag_matrix = scaling_diag_matrix.to(dtype=torch.float32)
    delta_scale = torch.matmul(w_wq_delta, scaling_diag_matrix)

    U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
    lowrank_r = rank
    sqrt_s = torch.sqrt(S[:lowrank_r])
    B = (U[:, :lowrank_r] * sqrt_s.unsqueeze(0)).to(dtype=dtype)

    # Equivalent to V[:rank, :] @ inv(scaling_diag_matrix), without materializing
    # the dense inverse. `left=False` solves X @ scaling_diag_matrix = V[:rank, :].
    truc_v = torch.linalg.solve_triangular(
        scaling_diag_matrix,
        V[:lowrank_r, :],
        upper=False,
        left=False,
    )
    A = (sqrt_s.unsqueeze(1) * truc_v).to(dtype=dtype)

    del U, S, V
    del scaling_diag_matrix, delta_scale, sqrt_s, truc_v

    return A.contiguous(), B.contiguous()


def eora_compute_lora(
        w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cholesky: Optional[bool] = True,
) -> Tuple[Tensor, Tensor]:

    assert w_wq_delta.dtype == torch.float32

    # save this later for SVD
    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)

    if IS_ROCM and not TORCH_GTE_210:
        # hip cannot resolve linalg ops
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")

    if os.getenv(_EORA_CHOLESKY_ENV) is not None:
        use_cholesky = env_flag(_EORA_CHOLESKY_ENV, default=use_cholesky)
    else:
        use_cholesky = bool(use_cholesky)
    result = None
    if use_cholesky and not (IS_ROCM and not TORCH_GTE_210):
        result = _eora_compute_lora_cholesky(
            w_wq_delta=w_wq_delta,
            name=name,
            raw_scaling_diag_matrix=raw_scaling_diag_matrix,
            rank=rank,
            dtype=dtype,
        )
    if result is None:
        result = _eora_compute_lora_eigh(
            w_wq_delta=w_wq_delta,
            name=name,
            raw_scaling_diag_matrix=raw_scaling_diag_matrix,
            rank=rank,
            dtype=dtype,
        )

    A, B = result

    del w_wq_delta, raw_scaling_diag_matrix

    # revert linalg backend
    if IS_ROCM and not TORCH_GTE_210:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B
