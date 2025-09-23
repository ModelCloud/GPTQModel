# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from typing import Optional, List, Tuple, Callable, Union

import torch

from .marlin_scalar_type import ScalarType
from ..utils.logger import setup_logger
from .rocm import IS_ROCM

log = setup_logger()

marlin_import_exception = None
try:
    import gptqmodel_marlin_kernels
except ImportError as e:
    marlin_import_exception = str(e)

# Validate marlin support
def _validate_marlin_device_support() -> bool:
    """
    Validates if the current device is compatible for Marlin.
    ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

    Returns:
        bool: indicates if CUDA device is compatible for Marlin
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and not IS_ROCM


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(mod: torch.nn.Module, name: str,
                      new: Union[torch.Tensor, torch.nn.Parameter]) -> None:
    old = getattr(mod, name)
    if type(old) is type(new) and old.dtype == new.dtype and \
            old.untyped_storage().nbytes() == new.untyped_storage().nbytes():
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name,
                               torch.nn.Parameter(new, requires_grad=False))


def _transform_param(layer: torch.nn.Module, name: Optional[str],
                     fn: Callable) -> None:
    if name is not None and getattr(layer, name, None) is not None:
        old_param = getattr(layer, name)
        new_param = fn(old_param)
        # replace the parameter with torch.nn.Parameter for TorchDynamo
        # compatibility
        replace_parameter(
            layer, name,
            torch.nn.Parameter(new_param.data, requires_grad=False))


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str,
                   new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)
    del new_t


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


# Whether to use atomicAdd reduce in gptq/awq marlin kernel. experimental
GPTQMODEL_MARLIN_USE_ATOMIC_ADD = True


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    if GPTQMODEL_MARLIN_USE_ATOMIC_ADD:
        return
    log.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "GPTQMODEL_MARLIN_USE_ATOMIC_ADD to 1 if possible.")


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        log.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible.")


def should_use_atomic_add_reduce(m: int, n: int, k: int, device: torch.device,
                                 dtype: torch.dtype) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with GPTQMODEL_MARLIN_USE_ATOMIC_ADD=1
    if not GPTQMODEL_MARLIN_USE_ATOMIC_ADD:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = True) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = gptq_marlin_gemm(reshaped_x,
                              None,
                              weight,
                              bias,
                              weight_scale,
                              None,
                              weight_zp,
                              g_idx,
                              g_idx_sort_indices,
                              workspace,
                              wtype,
                              size_m=reshaped_x.shape[0],
                              size_n=output_size_per_partition,
                              size_k=input_size_per_partition,
                              is_k_full=is_k_full,
                              use_atomic_add=use_atomic_add,
                              use_fp32_reduce=use_fp32_reduce,
                              is_zp_float=False)

    return output.reshape(out_shape)


def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
                     b_bias: Optional[torch.Tensor],
                     b_scales: torch.Tensor,
                     global_scale: Optional[torch.Tensor],
                     b_zeros: Optional[torch.Tensor],
                     g_idx: Optional[torch.Tensor],
                     perm: Optional[torch.Tensor],
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool = True,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return gptqmodel_marlin_kernels.gptq_marlin_gemm(a, c, b_q_weight, b_bias, b_scales,
                                                     global_scale, b_zeros, g_idx, perm,
                                                     workspace, b_q_type.id, size_m,
                                                     size_n, size_k, is_k_full,
                                                     use_atomic_add, use_fp32_reduce,
                                                     is_zp_float)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return gptqmodel_marlin_kernels.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                                       num_bits)