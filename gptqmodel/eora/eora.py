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

from typing import Sequence, Tuple

import torch
from torch import Tensor

from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM

log = setup_logger()

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

def eora_compute_lora(
        w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
) -> Tuple[Tensor, Tensor]:

    assert w_wq_delta.dtype == torch.float32

    # save this later for SVD
    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)

    if IS_ROCM:
        # hip cannot resolve linalg ops
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")

    L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)

    if (L < 0).any():
        ## When expanding the calibration data size for EoRA, I suggest maintaining the balance by allocating 50% to general input (C4) and the remaining 50% to downstream task data.
        log.warn(f"Found negative eigenvalues in `{name}`. Please increase your calibration data set for EoRA.")
        minimum = torch.min(L[L > 0])
        L[L < 0] = minimum

    sqrtEigenvalues = torch.sqrt(L)
    scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)

    scaling_matrix_inv = torch.diag(1/sqrtEigenvalues) @ Q.T

    scaling_diag_matrix = scaling_diag_matrix.to(dtype=torch.float32)
    scaling_matrix_inv = scaling_matrix_inv.to(dtype=torch.float32)

    delta_scale = torch.matmul(w_wq_delta, scaling_diag_matrix)

    U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
    lowrank_r = rank
    truc_s = S[:lowrank_r]
    truc_u = U[:, :lowrank_r]
    truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
    truc_sigma = torch.diag(truc_s)

    sqrtS = torch.sqrt(truc_sigma)
    B = torch.matmul(truc_u, sqrtS).to(dtype=dtype) # default to float16, check if we should save to float32
    A = torch.matmul(sqrtS, truc_v).to(dtype=dtype) # default to float16, check if we should save to float32


    del L, Q, U, S, V,
    del w_wq_delta, raw_scaling_diag_matrix, sqrtEigenvalues, scaling_diag_matrix, scaling_matrix_inv, delta_scale
    del truc_s, truc_u, truc_v, truc_sigma, sqrtS

    # revert linalg backend
    if IS_ROCM:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B
