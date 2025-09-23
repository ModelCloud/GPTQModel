# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

# optimized
def compute_local_perms(diag_H: torch.Tensor, groupsize: int) -> torch.Tensor:
    """
    Returns a (num_groups, groupsize) LongTensor of within-group permutations.
    """
    n = diag_H.numel()
    num_groups = n // groupsize
    if num_groups == 0:
        return torch.empty(0, groupsize, dtype=torch.long, device=diag_H.device)

    H = diag_H[: num_groups * groupsize].view(num_groups, groupsize)
    # Argsort along the group dimension -> indices that sort each row descending
    local_perms = torch.argsort(H, dim=1, descending=True)
    return local_perms

# optimized
def compute_global_perm(diag_H: torch.Tensor, groupsize: int, metric: str = "max") -> torch.Tensor:
    """
    Returns a (num_groups,) LongTensor with the order of groups by a chosen metric.
    metric ∈ {"max", "mean", "sum", "median"}.
    """
    n = diag_H.numel()
    num_groups = n // groupsize
    if num_groups == 0:
        return torch.empty(0, dtype=torch.long, device=diag_H.device)

    H = diag_H[: num_groups * groupsize].view(num_groups, groupsize)

    if metric == "max":
        scores = H.max(dim=1).values
    elif metric == "mean":
        scores = H.mean(dim=1)
    elif metric == "sum":
        scores = H.sum(dim=1)
    elif metric == "median":
        scores = H.median(dim=1).values
    else:
        raise ValueError(f"Unknown metric: {metric}")

    global_perm = torch.argsort(scores, descending=True)
    return global_perm

# optimized
def compose_final_perm(local_perms, global_perm, groupsize: int) -> torch.Tensor:
    """
    Vectorized:
      - Stack local_perms once (if it's a list) -> (G, S)
      - Add per-group base offsets
      - Reorder groups by global_perm
      - Flatten
    Returns a permutation p with the SAME semantics as your original:
      p[new_pos] = old_index
    """
    # Make (G, S) LongTensor on a single device
    if isinstance(local_perms, list):
        # Ensure same device & dtype across elements
        device = local_perms[0].device
        local = torch.stack([lp.to(device=device, dtype=torch.long) for lp in local_perms], dim=0)
    else:
        local = local_perms.to(dtype=torch.long)

    G, S = local.shape
    assert S == groupsize, f"groupsize mismatch: got {groupsize}, local has {S}"

    # Base offsets for each group: [0, S, 2S, ...]
    base = (torch.arange(G, device=local.device) * groupsize).view(G, 1)  # (G,1)

    # Adjust local indices into the flat space, then reorder groups by global_perm
    # NOTE: we index rows (groups) by global_perm, then flatten
    perm2d = (local + base)[global_perm.to(device=local.device, dtype=torch.long)]  # (G,S)
    return perm2d.reshape(-1)  # (G*S,)


# original algo
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

# original algo
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

def invert_perm(perm):
    """
    Compute the inverse of a permutation vector.

    Args:
        perm (Tensor): A 1D tensor containing a permutation of indices.

    Returns:
        inv (Tensor): The inverse permutation such that inv[perm] == torch.arange(len(perm)).
    """
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv
