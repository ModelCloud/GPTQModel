# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def pack_exllama(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=iweights.device)

    # packing rowwise
    iweights = iweights.view(iweights.shape[0] // (32 // bits), 32 // bits, -1)
    qweight = (
        torch.bitwise_left_shift(iweights, shifts[None, :, None])
        .sum(dim=1)
        .to(torch.int32)
    )

    # packing columnwise
    izeros = izeros.view(-1, izeros.shape[1] // (32 // bits), 32 // bits)
    qzeros = (
        torch.bitwise_left_shift(izeros, shifts[None, None, :])
        .sum(dim=-1)
        .to(torch.int32)
    )

    return qweight, qzeros


def unpack_reorder_pack(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # TODO: Why is exlalma v2 kernel doing +1 inference causing us to do -1 here in packing?
    # Subtract 1 from the izeros tensor (exllama adds 1 during inference)
    # We can remove it if we remove the +1 in the exllama code
    
    # TODO: Exllama v2 kernel may still mishandle true zero-points at 0 due to clamping below.
    # Exllama adds +1 during inference, so we offset by -1 here.
    # If izeros is 0, subtracting 1 underflows to -1; as uint8 this is 255,
    # and 4-bit packing keeps the low nibble (0xF/15), corrupting zero-points.
    # Clamp to 0 after the subtraction to avoid the underflow.
    izeros = (izeros - 1).clamp(min=0)
    
    # Pack the qweight and qzeros tensors
    qweight, qzeros = pack_exllama(iweight, izeros, bits)

    return qweight, qzeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales

    return iweight
