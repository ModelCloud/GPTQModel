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
        # The classic layout is one int32 per eight codes. Pack it directly
        # instead of materializing the padded int64 bit-stream used by the
        # generic 2..8-bit path.
        if iweights.shape[-1] % 8 != 0 or izeros.shape[-1] % 8 != 0:
            raise ValueError("INT4 AWQ packing requires the packed dimension to be divisible by 8.")
        mask = (1 << bits) - 1
        iweights = torch.bitwise_and(iweights, mask).to(torch.int32)
        izeros = torch.bitwise_and(izeros, mask).to(torch.int32)
        return (
            pack_awq_columns(iweights, bits, order=AWQ_ORDER),
            pack_awq_columns(izeros, bits, order=AWQ_ORDER),
        )
    return _pack_contiguous(iweights, bits), _pack_contiguous(izeros, bits)


def pack_awq_columns(codes: torch.Tensor, bits: int, order=None) -> torch.Tensor:
    """Pack code columns into 32-bit words with the classic shift/OR order.

    ``codes`` is ``[rows, columns]``; every ``32 // bits`` consecutive columns
    collapse into one output word.  ``order`` optionally permutes the codes
    inside each group (the AWQ ``[0, 2, 4, 6, 1, 3, 5, 7]`` interleave).
    Codes are widened and masked before shifting so no field spills into its
    neighbors. Any trailing columns that do not fill a whole group are
    ignored, mirroring the loop that iterates ``range(columns // pack_num)``.
    """

    if codes.dim() < 2:
        raise ValueError(f"codes must have at least two dimensions, got shape={tuple(codes.shape)}")
    pack_num = 32 // bits
    leading_shape = codes.shape[:-1]
    columns = codes.shape[-1] // pack_num
    grouped = codes[..., : columns * pack_num].to(torch.int32) & ((1 << bits) - 1)
    grouped = grouped.reshape(*leading_shape, columns, pack_num)
    if order is not None:
        grouped = grouped[..., order]
    packed = torch.zeros((*leading_shape, columns), dtype=torch.int32, device=codes.device)
    for index in range(pack_num):
        packed |= grouped[..., index] << (index * bits)
    return packed


def _unpack_columnwise(
    qtensor: torch.Tensor,
    bits: int,
    count: int = None,
    dtype: torch.dtype = None,
    shifts: torch.Tensor = None,
) -> torch.Tensor:
    """Unpack continuous words, including codes that cross word boundaries.

    ``dtype`` selects the integer width of the returned codes and defaults to
    ``torch.int64``, which the packing paths keep for their widened bit-stream
    arithmetic.  ``dequantize_gemm`` requests ``torch.int32``: a code is at most
    eight bits wide, so the wider default only doubled the traffic of the
    largest transient allocation in the AWQ forward.

    ``shifts`` overrides the four-bit extraction offsets, in output-column
    order. Passing the AWQ reverse order as the offsets emits the codes already
    in natural column order, which removes the separate column gather.
    """
    if bits not in range(2, 9):
        raise ValueError(f"AWQ unpacking requires 2 through 8 bits, got {bits}")
    if bits == 4:
        if shifts is None:
            shifts = torch.arange(0, 32, bits, device=qtensor.device)
        if qtensor.device.type == "npu":
            shifts = shifts.to(torch.int64)
            divisors = torch.pow(torch.full_like(shifts, 2), shifts)
            unpacked = torch.floor_divide(qtensor[:, :, None].to(torch.int64), divisors[None, None, :])
        else:
            if dtype is not None:
                shifts = shifts.to(dtype)
            unpacked = torch.bitwise_right_shift(qtensor[:, :, None], shifts[None, None, :])
        # Masking in place avoids a second tile-sized allocation.
        unpacked = unpacked.view(qtensor.shape[0], -1)
        unpacked.bitwise_and_(15)
        unpacked = unpacked[..., :count] if count is not None else unpacked
        return unpacked if dtype is None else unpacked.to(dtype)
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
    codes = codes.flatten(-2)[..., :count]
    return codes if dtype is None else codes.to(dtype)


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


# Upper bound on one tile of INT32 codes inside `dequantize_gemm`. The tile is
# what keeps the transient footprint of a wide layer proportional to the
# reconstructed weight instead of a multiple of it; 8 MiB keeps the extra
# allocations far below the matmul operand while leaving enough rows per tile
# that the launch overhead stays negligible.
_AWQ_DEQUANT_TILE_BYTES = 8 * 1024 * 1024


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    """Reconstruct the dense AWQ weight ``[in_features, out_features]``.

    The reconstruction is tiled over input rows, and each group broadcasts its
    single scale and zero-point row instead of expanding both to
    ``[in_features, out_features]`` with ``repeat_interleave``. That keeps the
    per-call scratch bounded by one tile: the previous implementation allocated
    several full-size INT64/INT32 copies per forward, which is why one 4096x4096
    four-bit layer needed 448 MiB of transient memory for a 32 MiB weight.

    The returned tensor is the matmul operand for the current call only. Nothing
    is cached, so no dequantized weight outlives the operation that consumes it.
    """
    if qzeros is None:
        raise ValueError("dequantize_gemm requires packed AWQ zero points")

    in_features = qweight.shape[0]
    out_features = scales.shape[-1]
    device = qweight.device
    # `group_size <= 0` is the per-channel layout: one group spans every row.
    group = in_features if group_size is None or int(group_size) <= 0 else int(group_size)
    num_groups = scales.shape[0]
    if num_groups == 0:
        raise ValueError("dequantize_gemm requires at least one scale group")

    # Folding the AWQ reverse order into the extraction offsets returns both the
    # codes and the zero points in natural column order, so neither needs the
    # full-size column gather that `reverse_awq_order` performs.
    shifts = None
    if bits == 4:
        shifts = torch.tensor(
            [lane * bits for lane in AWQ_REVERSE_ORDER], dtype=torch.int64, device=device
        )

    # Zero points hold one row per group, so they stay small when unpacked once.
    izeros = _unpack_columnwise(qzeros, bits, out_features, dtype=torch.int32, shifts=shifts)

    weight = torch.empty((in_features, out_features), dtype=scales.dtype, device=device)
    # Align every tile to a group boundary so one broadcast scale/zero slice
    # serves the whole tile. Both slices are views, so no expansion is written.
    rows_per_tile = max(1, _AWQ_DEQUANT_TILE_BYTES // (out_features * 4))
    rows_per_tile = max(1, rows_per_tile // group) * group
    group_covered_rows = num_groups * group
    # `out=` is rejected while an input tracks gradients; the explicit
    # expression keeps a trainable scale usable.
    fused_write = not scales.requires_grad

    for start in range(0, in_features, rows_per_tile):
        stop = min(start + rows_per_tile, in_features)
        # `_unpack_columnwise` already masks every code to `maxq`.
        codes = _unpack_columnwise(
            qweight[start:stop], bits, out_features, dtype=torch.int32, shifts=shifts
        )

        if (stop - start) % group == 0 and stop <= group_covered_rows:
            tiles = (stop - start) // group
            first_group = start // group
            # `reshape` also accepts the sliced view that a packed word count
            # above `out_features` produces, where `view` would refuse.
            block = codes.reshape(tiles, group, out_features)
            block.sub_(izeros[first_group:first_group + tiles].unsqueeze(1))
            target = weight[start:stop].reshape(tiles, group, out_features)
            if fused_write:
                torch.mul(
                    block, scales[first_group:first_group + tiles].unsqueeze(1), out=target
                )
            else:
                target.copy_(
                    block * scales[first_group:first_group + tiles].unsqueeze(1)
                )
            continue

        # A tail that does not fill a whole group, or rows past the last group,
        # is rare and small: broadcast one group row at a time.
        first_group = min(start // group, num_groups - 1)
        last_group = min((stop - 1) // group, num_groups - 1)
        for group_index in range(first_group, last_group + 1):
            group_start = group_index * group
            # The last group also covers a tail of rows that does not fill it.
            group_stop = (
                in_features
                if group_index == num_groups - 1
                else min(group_start + group, in_features)
            )
            lo = max(start, group_start)
            hi = min(stop, group_stop)
            if lo >= hi:
                continue
            block = codes[lo - start:hi - start]
            block.sub_(izeros[group_index])
            if fused_write:
                torch.mul(block, scales[group_index], out=weight[lo:hi])
            else:
                weight[lo:hi] = block * scales[group_index]

    return weight
