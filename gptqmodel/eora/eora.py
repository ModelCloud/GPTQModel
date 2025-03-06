# Copyright 2024-2025 NVIDIA CORPORATION
# EoRA arXiv: https://arxiv.org/abs/2410.21271
# EoRA Official Repo: https://github.com/NVlabs/EoRA

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..looper.named_module import NamedModule
from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM

log = setup_logger()

def eora_process_input(input: Tensor, name: str, eigen_scaling_diag_matrix: Dict[str, torch.dtype], sample_size: int):
    inp = input[0].to(dtype=torch.float32)
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)

    tmp = inp.shape[0]
    adds = torch.matmul(inp.transpose(1, 2), inp)
    adds_sum = torch.sum(adds, dim=0)

    ## Adding tmp to denominator is only for mathmatical stability
    eigen_scaling_diag_matrix[name] *= sample_size / (sample_size + tmp)
    eigen_scaling_diag_matrix[name] += adds_sum / sample_size

    del inp, tmp, adds, adds_sum

def eora_compute_lora(
        device: torch.device,
        w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
        module: NamedModule,
        eigen_scaling_diag_matrix: torch.dtype,
        rank: int) -> Tuple[Tensor, Tensor]:

    assert w_wq_delta.dtype == torch.float32

    # save this later for SVD
    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(dtype=torch.float64, device=device)

    if IS_ROCM:
        # hip cannot resolve linalg ops
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")

    L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)

    if (L < 0).any():
        ## When expanding the calibration data size for EoRA, I suggest maintaining the balance by allocating 50% to general input (C4) and the remaining 50% to downstream task data.
        log.warn(f"Found negative eigenvalues in `{module.name}`. Please increase your calibration data set for EoRA.")
        minimum = torch.min(L[L > 0])
        L[L < 0] = minimum

    sqrtEigenvalues = torch.sqrt(L)
    scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
    
    try:
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    except Exception:
        scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(device)
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

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
    B = torch.matmul(truc_u, sqrtS).to(dtype=torch.float16)
    A = torch.matmul(sqrtS, truc_v).to(dtype=torch.float16)


    del L, Q, U, S, V,
    del w_wq_delta, raw_scaling_diag_matrix, sqrtEigenvalues, scaling_diag_matrix, scaling_matrix_inv, delta_scale
    del truc_s, truc_u, truc_v, truc_sigma, sqrtS

    # revert linalg backend
    if IS_ROCM:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B
