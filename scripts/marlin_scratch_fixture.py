# SPDX-License-Identifier: Apache-2.0
"""Deterministic packed fixtures and independent dense references for Marlin."""
import torch


def make_layer(method, dtype, k=256, n=128, *, act_order=False, bits=4,
               device="cuda:0", seed=17, bias=True):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
    from gptqmodel.utils.backend import BACKEND

    if method == "awq" and (act_order or bits not in (4, 8)):
        raise ValueError("This fixture covers AWQ 4/8-bit without act-order")
    group = 32
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (k, n), generator=gen, dtype=torch.int32)
    scales = (torch.rand(k // group, n, generator=gen) * .01 + .002).to(dtype)
    indices = torch.arange(k, dtype=torch.int32) // group
    if act_order:
        indices = indices[torch.randperm(k, generator=gen)]
    zero = (torch.full((k // group, n), 1 << (bits - 1), dtype=torch.int32)
            if method == "gptq" else
            torch.randint(0, 1 << bits, (k // group, n), generator=gen, dtype=torch.int32))
    factor = 32 // bits

    def pack_cols(values, awq=False):
        packed = torch.zeros(values.shape[0], values.shape[1] // factor, dtype=torch.int32)
        order = ((0, 2, 4, 6, 1, 3, 5, 7) if bits == 4 else (0, 2, 1, 3)) if awq else range(factor)
        for lane, source in enumerate(order):
            packed.bitwise_or_(values[:, source::factor] << (bits * lane))
        return packed

    if method == "gptq":
        weight = torch.zeros(k // factor, n, dtype=torch.int32)
        for lane in range(factor):
            weight.bitwise_or_(codes[lane::factor] << (bits * lane))
        cls, backend = MarlinLinear, BACKEND.GPTQ_MARLIN
    else:
        weight = pack_cols(codes, True)
        cls, backend = AwqMarlinLinear, BACKEND.AWQ_MARLIN
    layer = cls(bits=bits, group_size=group, desc_act=act_order,
                sym=method == "gptq", in_features=k, out_features=n,
                bias=bias, dtype=dtype, backend=backend).to(device)
    bias_value = (torch.randn(n, generator=gen) * .01).to(dtype) if bias else None
    with torch.no_grad():
        layer.qweight.copy_(weight.to(device))
        layer.qzeros.copy_(pack_cols(zero, method == "awq").to(device))
        layer.scales.copy_(scales.to(device))
        if method == "gptq":
            layer.g_idx.copy_(indices.to(device))
        if bias:
            layer.bias.copy_(bias_value.to(device))
    # Independent reference uses logical codes and group IDs, before repacking.
    dense = ((codes.float() - zero[indices.long()].float()) *
             scales[indices.long()].float()).to(device=device, dtype=dtype)
    layer.eval()
    layer.post_init()
    return layer, dense, None if bias_value is None else bias_value.to(device)


def gemm_arguments(layer, x, *, fp32=True):
    from gptqmodel.utils.marlin import marlin_pad_dim, should_use_atomic_add_reduce
    n, k = layer._marlin_tile_padding or (layer.out_features, layer.in_features)
    a = marlin_pad_dim(x.reshape(-1, x.shape[-1]), layer.in_features, k)
    atomic = False
    if type(layer).__name__ == "AwqMarlinLinear":
        atomic = should_use_atomic_add_reduce(a.shape[0], n, k, a.device, a.dtype)
    return dict(a=a, c=None, b_q_weight=layer.qweight, b_bias=layer.bias,
                b_scales=layer.scales, global_scale=None, b_zeros=layer.qzeros,
                g_idx=layer.g_idx, perm=layer.g_idx_sort_indices,
                workspace=layer.workspace, b_q_type=layer.weight_type,
                size_m=a.shape[0], size_n=n, size_k=k,
                is_k_full=getattr(layer, "is_k_full", True),
                use_atomic_add=atomic, use_fp32_reduce=fp32, is_zp_float=False)
