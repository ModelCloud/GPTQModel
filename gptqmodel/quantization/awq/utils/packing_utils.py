# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq

import math

import torch


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _pack_contiguous(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the final dimension as a continuous little-endian bit stream."""
    if bits not in range(2, 9):
        raise ValueError(f"AWQ packing requires 2 through 8 bits, got {bits}")
    count = values.shape[-1]
    blocks = math.ceil(count / 32)
    padded = torch.zeros((*values.shape[:-1], blocks * 32), dtype=torch.int64, device=values.device)
    padded[..., :count] = values.to(torch.int64) & ((1 << bits) - 1)
    codes = padded.view(*values.shape[:-1], blocks, 32)
    words = torch.zeros((*values.shape[:-1], blocks, bits), dtype=torch.int64, device=values.device)
    for index in range(32):
        bit = index * bits
        word, shift = divmod(bit, 32)
        words[..., word] |= codes[..., index] << shift
        if shift + bits > 32:
            words[..., word + 1] |= codes[..., index] >> (32 - shift)
    packed = words.reshape(*values.shape[:-1], blocks * bits)
    return packed[..., :math.ceil(count * bits / 32)].to(torch.int32)


def pack_awq(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    """Pack natural AWQ codes while retaining the established INT4 layout."""
    if bits == 4:
        iweights = iweights.view(*iweights.shape[:-1], -1, 8)[..., AWQ_ORDER].flatten(-2)
        izeros = izeros.view(*izeros.shape[:-1], -1, 8)[..., AWQ_ORDER].flatten(-2)
    return _pack_contiguous(iweights, bits), _pack_contiguous(izeros, bits)


def _unpack_columnwise(qtensor: torch.Tensor, bits: int, count: int = None) -> torch.Tensor:
    """Unpack continuous words, including codes that cross word boundaries."""
    if bits not in range(2, 9):
        raise ValueError(f"AWQ unpacking requires 2 through 8 bits, got {bits}")
    if bits == 4:
        if qtensor.device.type == "npu":
            shifts = torch.arange(0, 32, bits, dtype=torch.int64, device=qtensor.device)
            divisors = torch.pow(torch.full_like(shifts, 2), shifts)
            unpacked = torch.floor_divide(qtensor[:, :, None].to(torch.int64), divisors[None, None, :])
        else:
            shifts = torch.arange(0, 32, bits, device=qtensor.device)
            unpacked = torch.bitwise_right_shift(qtensor[:, :, None], shifts[None, None, :])
        unpacked = unpacked.view(qtensor.shape[0], -1) & 15
        return unpacked[..., :count] if count is not None else unpacked
    count = count or (qtensor.shape[-1] * 32 // bits)
    blocks = math.ceil(count / 32)
    words = torch.zeros((*qtensor.shape[:-1], blocks * bits), dtype=torch.int64, device=qtensor.device)
    words[..., :qtensor.shape[-1]] = qtensor.to(torch.int64) & 0xFFFFFFFF
    words = words.view(*qtensor.shape[:-1], blocks, bits)
    codes = torch.empty((*qtensor.shape[:-1], blocks, 32), dtype=torch.int64, device=qtensor.device)
    mask = (1 << bits) - 1
    for index in range(32):
        bit = index * bits
        word, shift = divmod(bit, 32)
        value = words[..., word] >> shift
        if shift + bits > 32:
            value |= words[..., word + 1] << (32 - shift)
        codes[..., index] = value & mask
    return codes.flatten(-2)[..., :count]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int, count: int = None):
    # unpacking columnwise
    iweights = _unpack_columnwise(qweight, bits, count)

    # unpacking columnwise
    if qzeros is not None:
        izeros = _unpack_columnwise(qzeros, bits, count)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    if bits != 4:
        return iweights, izeros
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=iweights.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def pack_exllama(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    if bits == 4:
        shifts = torch.arange(0, 32, bits, device=iweights.device)
        iweights = iweights.view(iweights.shape[0] // 8, 8, -1)
        if iweights.device.type == "npu":
            powers = torch.pow(torch.full_like(shifts, 2, dtype=torch.int64), shifts.to(torch.int64))
            qweight = (iweights.to(torch.int64) * powers[None, :, None]).sum(dim=1).to(torch.int32)
        else:
            qweight = torch.bitwise_left_shift(iweights, shifts[None, :, None]).sum(dim=1).to(torch.int32)
        izeros = izeros.view(-1, izeros.shape[1] // 8, 8)
        if izeros.device.type == "npu":
            powers = torch.pow(torch.full_like(shifts, 2, dtype=torch.int64), shifts.to(torch.int64))
            qzeros = (izeros.to(torch.int64) * powers[None, None, :]).sum(dim=-1).to(torch.int32)
        else:
            qzeros = torch.bitwise_left_shift(izeros, shifts[None, None, :]).sum(dim=-1).to(torch.int32)
        return qweight, qzeros
    # GPTQ/ExLlama packs the input axis of qweight and the output axis of
    # qzeros. Transposing lets the same continuous stream helper cover both.
    return _pack_contiguous(iweights.T, bits).T.contiguous(), _pack_contiguous(izeros, bits)


def unpack_reorder_pack(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # Pack the qweight and qzeros tensors
    qweight, qzeros = pack_exllama(iweight, izeros, bits)

    return qweight, qzeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits, scales.shape[-1])
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
