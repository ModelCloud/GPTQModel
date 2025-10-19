# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Based on Group Aware Reordering (GAR)
# @article{gar,
#   title={Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference, CVPRW 2025.},
#   author={T. Gafni, A. Karnieli, Y. Hanani},
#   journal={arXiv preprint arXiv:2505.14638},
#   year={2025}
# }
# https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Gafni_Dual_Precision_Quantization_for_Efficient_and_Accurate_Deep_Neural_Networks_CVPRW_2025_paper.html

import torch


def compute_local_perms_original(diag_H, groupsize):
    """
    For each group, compute a permutation that orders the indices in descending order
    based on the corresponding diagonal values of H.

    Args:
        diag_H (Tensor): 1D tensor representing the diagonal of the Hessian.
        groupsize (int): Number of columns/weights per group.

    Returns:
        local_perms (list of Tensors): Each element is a permutation (indices) for that group.
    """
    n = diag_H.numel()
    num_groups = n // groupsize
    local_perms = []
    for g in range(num_groups):
        start = g * groupsize
        end = start + groupsize
        sub_diag = diag_H[start:end]
        # Get local permutation: indices that would sort sub_diag in descending order.
        local_perm = torch.argsort(sub_diag, descending=True)
        local_perms.append(local_perm)
    return local_perms


def compute_global_perm_original(diag_H, groupsize):
    """
    Compute a permutation for the groups themselves. Here we choose the maximum diagonal value
    within each group as the group metric and sort the groups in descending order.

    Args:
        diag_H (Tensor): 1D tensor representing the diagonal of the Hessian.
        groupsize (int): Number of columns/weights per group.

    Returns:
        global_perm (Tensor): 1D tensor of length num_groups with the new order of groups.
    """
    n = diag_H.numel()
    num_groups = n // groupsize
    group_metric = []
    for g in range(num_groups):
        start = g * groupsize
        end = start + groupsize
        group_metric.append(diag_H[start:end].max().item())
    # Create a tensor on the same device as diag_H.
    group_metric = torch.tensor(group_metric, device=diag_H.device)
    global_perm = torch.argsort(group_metric, descending=True)
    return global_perm


def compose_final_perm_original(local_perms, global_perm, groupsize):
    """
    Compose the final overall permutation from the local and global permutations.

    Args:
        local_perms (list of Tensors): Local permutation for each group.
        global_perm (Tensor): Global group permutation.
        groupsize (int): Number of indices per group.

    Returns:
        final_perm (Tensor): 1D tensor that maps original indices to new positions.
    """
    num_groups = len(local_perms)
    final_perm = []
    # Process groups in the order specified by global_perm.
    for new_group in range(num_groups):
        # Get the original group index.
        orig_group = global_perm[new_group].item()
        offset = orig_group * groupsize
        local_perm = local_perms[orig_group]
        # Adjust local indices to the full index space.
        for idx in local_perm:
            final_perm.append(idx.item() + offset)
    return torch.tensor(final_perm, dtype=torch.long)
