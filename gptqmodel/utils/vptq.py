# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared VPTQ helpers: index packing, group layout, and dequantization.

The packed-index layout matches the VPTQ reference implementation
(https://github.com/microsoft/VPTQ, MIT licensed): a row of `n` indices with
`index_bits` bits each (optionally merged with a `res_bits` residual index) is
serialized as one contiguous little-endian bitstream and stored as 32-bit
words. The reference packs `(residual << index_bits) | index` per element.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def vptq_index_bits(num_centroids: int) -> int:
    """Return the number of bits needed to address `num_centroids` entries."""

    if num_centroids < 1:
        raise ValueError(f"VPTQ: `num_centroids` must be positive, got `{num_centroids}`.")
    return int(math.ceil(math.log2(num_centroids)))


def vptq_packed_words(num_elements: int, total_bits: int) -> int:
    """Return the number of 32-bit words needed to pack `num_elements` indices."""

    if num_elements < 0:
        raise ValueError(f"VPTQ: `num_elements` must be non-negative, got `{num_elements}`.")
    if total_bits <= 0:
        raise ValueError(f"VPTQ: `total_bits` must be positive, got `{total_bits}`.")
    return int(math.ceil(num_elements * total_bits / 32))


def vptq_group_setting(
    in_features: int,
    *,
    npercent: float,
    group_size: int,
    group_num: int,
    enable_outlier: bool = True,
) -> Tuple[int, int, int]:
    """Resolve `(outlier_size, group_size, group_num)` for the VPTQ group layout.

    The non-outlier input channels are partitioned into `group_num` contiguous
    groups of `group_size` channels; each group owns one codebook. Exactly one
    of `group_size`/`group_num` may be set; `-1` lets the other one decide.

    With `group_num` the channels left over by the division are added to the
    outlier set so every group keeps exactly `group_size` channels. That is
    only possible while the outlier codebook is part of the layout, so
    `enable_outlier=False` keeps `outlier_size` at zero and lets the last group
    carry the remainder instead (the runtime drops its padding on dequantize).
    """

    if in_features < 1:
        raise ValueError(f"VPTQ: `in_features` must be positive, got `{in_features}`.")
    if group_size != -1 and group_num != -1:
        raise ValueError("VPTQ: `group_size` and `group_num` cannot both be set.")
    if group_size != -1 and group_size < 1:
        raise ValueError(f"VPTQ: `group_size` must be `-1` or positive, got `{group_size}`.")
    if group_num != -1 and group_num < 1:
        raise ValueError(f"VPTQ: `group_num` must be `-1` or positive, got `{group_num}`.")

    initial_outlier_size = int(math.ceil((npercent / 100.0) * in_features))
    initial_outlier_size = max(0, min(in_features, initial_outlier_size))
    if not enable_outlier:
        initial_outlier_size = 0
    remaining_columns = in_features - initial_outlier_size

    if group_num != -1:
        if group_num > remaining_columns:
            raise ValueError(
                f"VPTQ: `group_num` ({group_num}) exceeds the {remaining_columns} input channels "
                "available for grouped quantization, which would leave empty codebook groups."
            )
        if enable_outlier:
            pad = remaining_columns % group_num
            outlier_size = initial_outlier_size + pad
            resolved_group_size = (in_features - outlier_size) // group_num
        else:
            outlier_size = 0
            resolved_group_size = int(math.ceil(remaining_columns / group_num))
        resolved_group_num = group_num
    elif group_size != -1:
        outlier_size = initial_outlier_size
        resolved_group_num = int(math.ceil((in_features - outlier_size) / group_size))
        resolved_group_size = group_size
    else:
        outlier_size = initial_outlier_size
        resolved_group_num = 1
        resolved_group_size = in_features - outlier_size

    if resolved_group_num < 1 or resolved_group_size < 1:
        raise ValueError(
            "VPTQ: group configuration leaves no channels to quantize "
            f"(outlier_size={outlier_size}, group_size={resolved_group_size}, group_num={resolved_group_num})."
        )
    return outlier_size, resolved_group_size, resolved_group_num


def vptq_outlier_enabled(
    *,
    npercent: float,
    outlier_vector_len: int,
    outlier_num_centroids: int,
) -> bool:
    """Report whether the configuration carries a usable outlier codebook."""

    return outlier_vector_len > 0 and outlier_num_centroids > 1 and npercent > 0


def vptq_group_for_column(column: int, *, outlier_size: int, group_size: int) -> int:
    """Return the codebook index for an input channel (`0` is the outlier codebook)."""

    if column < 0:
        raise ValueError(f"VPTQ: `column` must be non-negative, got `{column}`.")
    if column < outlier_size:
        return 0
    return 1 + (column - outlier_size) // group_size


def pack_vptq_indices(
    indices: torch.Tensor,
    index_bits: int,
    *,
    res_indices: Optional[torch.Tensor] = None,
    res_bits: int = 0,
) -> torch.Tensor:
    """Pack `indices` (and optional residual indices) into int32 words.

    Both tensors must share a shape `[..., num_elements]`. The returned tensor
    has shape `[..., ceil(num_elements * (index_bits + res_bits) / 32)]`.
    """

    if index_bits <= 0:
        raise ValueError(f"pack_vptq_indices: `index_bits` must be positive, got `{index_bits}`.")
    if res_indices is not None and res_bits <= 0:
        raise ValueError("pack_vptq_indices: `res_bits` must be positive when residual indices are given.")
    if res_indices is None:
        if res_bits:
            raise ValueError("pack_vptq_indices: `res_bits` must be zero when `res_indices` is None.")
        res_bits = 0

    total_bits = index_bits + res_bits
    if total_bits > 32:
        raise ValueError(
            f"pack_vptq_indices: `index_bits` + `res_bits` must be at most 32, got `{total_bits}`."
        )
    num_elements = indices.shape[-1]
    if num_elements == 0:
        return torch.zeros(*indices.shape[:-1], 0, dtype=torch.int32, device=indices.device)

    merged = indices.to(torch.int64)
    merged_min, merged_max = (
        int(value) for value in torch.stack(torch.aminmax(merged)).tolist()
    )
    if merged_min < 0:
        raise ValueError("pack_vptq_indices: `indices` must be non-negative.")
    if merged_max >= (1 << index_bits):
        raise ValueError(
            f"pack_vptq_indices: `indices` values exceed the {index_bits}-bit packing range."
        )
    if res_indices is not None:
        if res_indices.shape != indices.shape:
            raise ValueError(
                f"pack_vptq_indices: residual index shape {tuple(res_indices.shape)} "
                f"must match index shape {tuple(indices.shape)}."
            )
        res_merged = res_indices.to(torch.int64)
        res_min, res_max = (
            int(value) for value in torch.stack(torch.aminmax(res_merged)).tolist()
        )
        if res_min < 0:
            raise ValueError("pack_vptq_indices: `res_indices` must be non-negative.")
        if res_max >= (1 << res_bits):
            raise ValueError(
                f"pack_vptq_indices: `res_indices` values exceed the "
                f"{res_bits}-bit packing range."
            )
        merged = merged | (res_merged << index_bits)

    bit_offsets = torch.arange(total_bits, device=indices.device, dtype=torch.int64)
    bits = (merged.unsqueeze(-1) >> bit_offsets) & 1
    bits = bits.reshape(*indices.shape[:-1], num_elements * total_bits)

    words = vptq_packed_words(num_elements, total_bits)
    pad = words * 32 - bits.shape[-1]
    if pad > 0:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(*indices.shape[:-1], words, 32)

    word_offsets = torch.arange(32, device=indices.device, dtype=torch.int64)
    packed = (bits << word_offsets).sum(dim=-1)
    return packed.to(torch.int32)


def unpack_vptq_indices(
    packed: torch.Tensor,
    index_bits: int,
    num_elements: int,
    *,
    res_bits: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Inverse of :func:`pack_vptq_indices`.

    Returns `(indices, res_indices)`; `res_indices` is `None` when `res_bits`
    is zero.
    """

    if index_bits <= 0:
        raise ValueError(f"unpack_vptq_indices: `index_bits` must be positive, got `{index_bits}`.")
    if num_elements < 0:
        raise ValueError(f"unpack_vptq_indices: `num_elements` must be non-negative, got `{num_elements}`.")

    total_bits = index_bits + res_bits
    if total_bits > 32:
        raise ValueError(
            f"unpack_vptq_indices: `index_bits` + `res_bits` must be at most 32, got `{total_bits}`."
        )
    if num_elements == 0:
        empty = torch.zeros(*packed.shape[:-1], 0, dtype=torch.int64, device=packed.device)
        return empty, (empty.clone() if res_bits > 0 else None)

    word_offsets = torch.arange(32, device=packed.device, dtype=torch.int64)
    bits = (packed.to(torch.int64).unsqueeze(-1) >> word_offsets) & 1
    bits = bits.reshape(*packed.shape[:-1], -1)

    needed = num_elements * total_bits
    if bits.shape[-1] < needed:
        raise ValueError(
            f"unpack_vptq_indices: packed tensor holds {bits.shape[-1]} bits, "
            f"need at least {needed}."
        )
    bits = bits[..., :needed].reshape(*packed.shape[:-1], num_elements, total_bits)

    element_offsets = torch.arange(total_bits, device=packed.device, dtype=torch.int64)
    merged = (bits << element_offsets).sum(dim=-1)

    index_mask = (1 << index_bits) - 1
    indices = merged & index_mask
    if res_bits > 0:
        res_mask = (1 << res_bits) - 1
        res_indices = (merged >> index_bits) & res_mask
    else:
        res_indices = None
    return indices, res_indices
