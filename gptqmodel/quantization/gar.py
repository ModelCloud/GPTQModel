import torch


def compute_local_perms(diag_H, groupsize):
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

def compute_global_perm(diag_H, groupsize):
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

def compose_final_perm(local_perms, global_perm, groupsize):
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
