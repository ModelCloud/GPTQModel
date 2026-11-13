# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Canonical packed GPTQ layouts for Torch/XLA and Tenstorrent kernels.

The checkpoint layout favours byte-addressable row-major unpacking on CPUs and
GPUs: ``qweight`` is ``[in_features / pack_factor, out_features]`` and each
word packs consecutive input features for one output column.  A TT kernel wants
output-major packed weight tiles, so every output row can be streamed into a
tile without gathering one value per column from each source row.

This module owns that conversion.  It performs the transpose while the words
are still packed and uploads exact uint8 byte planes. It never creates dense
weights or expanded per-code storage.

Canonical 4-bit layout:

- ``weight_bytes[o, k, b]`` contains byte ``b`` of the packed word for output
  ``o`` and inputs ``k*8 .. k*8+7``. Code ``j`` is at bit offset ``j*4``.
- ``zero_bytes[g, o/8, b]`` contains the byte planes for output zero points
  ``o .. o+7`` in group ``g``. Code ``j`` is at bit offset ``j*4``.

Words are converted from signed checkpoint ``int32`` to exact little-endian
uint8 planes on CPU before upload. TT's PJRT 32-bit shift path can mislegalize
or drop the high bit, while uint8 mask/shift is bit-exact.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


TT_GPTQ_PACK_BITS = 4
TT_GPTQ_PACK_FACTOR = 32 // TT_GPTQ_PACK_BITS
_UINT32_MASK = (1 << 32) - 1


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class TTPackedGPTQLayout:
    """Persistent, packed-only tensors consumed by a TT GPTQ kernel."""

    weight_bytes: torch.Tensor
    zero_bytes: torch.Tensor
    zero_neg_one: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int
    group_size: int
    bits: int = TT_GPTQ_PACK_BITS

    @property
    def pack_factor(self) -> int:
        return 32 // self.bits

    @property
    def weight_word_shape(self) -> tuple[int, int]:
        return (self.out_features, _ceil_div(self.in_features, self.pack_factor))

    @property
    def zero_word_shape(self) -> tuple[int, int]:
        return (self.scales.shape[0], _ceil_div(self.out_features, self.pack_factor))

    @property
    def weight_byte_shape(self) -> tuple[int, int, int]:
        return (self.out_features, _ceil_div(self.in_features, self.pack_factor), 4)


def _require_4bit_gptq(
    bits: int,
    pack_dtype: torch.dtype,
    in_features: int,
    out_features: int,
    sym: bool,
    planar: bool,
) -> None:
    if bits != TT_GPTQ_PACK_BITS:
        raise NotImplementedError(f"TT packed GPTQ currently requires 4-bit weights, got bits={bits}.")
    if pack_dtype != torch.int32:
        raise NotImplementedError(f"TT packed GPTQ requires int32 checkpoint words, got pack_dtype={pack_dtype}.")
    if planar:
        raise NotImplementedError("TT packed GPTQ does not yet support planar GPTQ formats.")
    if not sym:
        raise NotImplementedError("TT packed GPTQ currently requires symmetric zero points.")
    if in_features % 8:
        raise ValueError(f"TT packed GPTQ requires in_features divisible by 8, got {in_features}.")
    if out_features % 8:
        raise ValueError(f"TT packed GPTQ requires out_features divisible by 8, got {out_features}.")


def _as_uint32_words(words: torch.Tensor) -> torch.Tensor:
    """Return a CPU uint32 tensor with the exact checkpoint bit pattern."""
    if words.dtype != torch.int32:
        raise TypeError(f"expected int32 packed words, got {words.dtype}")
    return words.detach().to(device="cpu", copy=True).view(torch.uint32).contiguous()


def _strip_padding(words: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    if words.shape[0] < rows or words.shape[1] < cols:
        raise ValueError(
            f"packed tensor {tuple(words.shape)} is smaller than logical shape {(rows, cols)}"
        )
    return words[:rows, :cols].contiguous()


def _repack_weight_words(qweight: torch.Tensor, in_features: int, out_features: int) -> torch.Tensor:
    """Transpose packed GPTQ weight words into ``[out, in/8]`` uint32."""
    words_in = _ceil_div(in_features, TT_GPTQ_PACK_FACTOR)
    if qweight.shape[0] < words_in or qweight.shape[1] < out_features:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} cannot provide "
            f"[{words_in}, {out_features}] packed words"
        )
    source = _as_uint32_words(qweight)[:words_in, :out_features]
    return source.transpose(0, 1).contiguous()


def _packed_words_to_byte_planes(words: torch.Tensor) -> torch.Tensor:
    """Return little-endian byte planes for packed uint32 words.

    TT's StableHLO lowerer cannot reliably legalize ``shift_right_arithmetic``
    for 32-bit integers and its signed identity view can lose the high bit.
    Four byte planes preserve the exact word bits in the same 0.5-byte-per-
    weight footprint, and uint8 mask/shift is exact on TT.  The final singleton
    dimension is selected so every plane travels as a legal uint8 tile.
    """
    if words.dtype != torch.uint32:
        raise TypeError(f"expected uint32 packed words, got {words.dtype}")
    if words.device.type != "cpu":
        raise ValueError("byte-plane packing must be built from CPU words")
    # numpy's byte view is explicit little-endian on the supported x86 hosts;
    # reverse on a hypothetical big-endian process rather than misdecoding.
    raw = words.numpy().view(np.uint8).reshape(*words.shape, 4)
    if sys.byteorder != "little":
        raw = raw[..., ::-1]
    return torch.from_numpy(raw.copy()).contiguous()


def _repack_zero_words(
    qzeros: torch.Tensor,
    *,
    groups: int,
    out_features: int,
    qzero_format: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert GPTQ zero words from output-per-column to packed output rows."""
    words_out = _ceil_div(out_features, TT_GPTQ_PACK_FACTOR)
    if qzeros.shape[0] < groups or qzeros.shape[1] < words_out:
        raise ValueError(
            f"qzeros shape {tuple(qzeros.shape)} cannot provide "
            f"[{groups}, {words_out}] packed words"
        )
    source = _as_uint32_words(qzeros)[:groups, :words_out]
    if qzero_format == 1:
        # GPTQ v1 serializes ``zero_point - 1``.  TorchLinear converts v1 to
        # v2 with ``(stored + 1) & 0xF`` and v2 decoders use that nibble as the
        # logical zero point directly.  Decode and repack in logical space so a
        # packed-word carry cannot cross an adjacent 4-bit field.
        shifts = torch.arange(0, 32, TT_GPTQ_PACK_BITS, dtype=torch.int64).view(1, 1, -1)
        stored = (source.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
        nibbles = (stored + 1) & 0xF
        source = (nibbles << shifts).sum(dim=-1, dtype=torch.int64).to(torch.uint32).reshape(source.shape)
    elif qzero_format != 2:
        raise ValueError(f"unsupported GPTQ qzero format {qzero_format!r}; expected 1 or 2")

    # GPTQ v2 stores the logical zero point directly.  Keep a stable mask in the
    # packed ABI for compatibility, but no current GPTQ qzero format decodes to
    # -1 as TorchLinear's zero-point unpack does not subtract an additional one.
    neg_one = torch.zeros((groups, words_out, 8), dtype=torch.bool)
    return source.contiguous(), neg_one.reshape(groups, -1).contiguous()


def repack_gptq_for_tt(
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bias: Optional[torch.Tensor],
    in_features: int,
    out_features: int,
    group_size: int,
    bits: int,
    pack_dtype: torch.dtype,
    sym: bool,
    planar: bool,
    qzero_format: int,
    device: Optional[torch.device] = None,
) -> TTPackedGPTQLayout:
    """Build the persistent packed TT view without expanding weight codes.

    Repacking runs entirely on CPU.  When ``device`` is supplied, the final
    words, scales and indices are uploaded after conversion; no device-side
    ``view(uint32)`` is used on TT.
    """
    _require_4bit_gptq(bits, pack_dtype, in_features, out_features, sym, planar)
    groups = scales.shape[0]
    weight_words = _repack_weight_words(qweight, in_features, out_features)
    weight_bytes = _packed_words_to_byte_planes(weight_words).reshape(
        out_features, _ceil_div(in_features, TT_GPTQ_PACK_FACTOR), 4
    )
    zero_words, zero_neg_one = _repack_zero_words(
        qzeros,
        groups=groups,
        out_features=out_features,
        qzero_format=qzero_format,
    )
    zero_bytes = _packed_words_to_byte_planes(zero_words).reshape(
        groups, _ceil_div(out_features, TT_GPTQ_PACK_FACTOR), 4
    )
    packed_scales = scales.detach().to(device="cpu", dtype=torch.bfloat16, copy=True).contiguous()
    packed_g_idx = g_idx.detach().to(device="cpu", dtype=torch.int32, copy=True).contiguous()
    groups = packed_scales.shape[0]
    if int(packed_g_idx.min()) < 0:
        packed_g_idx = (packed_g_idx.to(torch.int64) % groups).to(torch.int32)
    packed_bias = (
        None
        if bias is None
        else bias.detach().to(device="cpu", dtype=torch.bfloat16, copy=True).contiguous()
    )

    # The byte-plane ABI is fixed-width. Validate every uploaded index/scale
    # dimension before placement so a malformed checkpoint fails here rather
    # than during the first device forward.
    if packed_scales.shape != (groups, out_features):
        raise ValueError(
            f"scales shape must be [{groups}, {out_features}], got {tuple(packed_scales.shape)}"
        )
    if packed_g_idx.numel() != in_features:
        raise ValueError(
            f"g_idx length must equal in_features ({in_features}), got {packed_g_idx.numel()}"
        )
    if int(packed_g_idx.min()) < 0 or int(packed_g_idx.max()) >= groups:
        raise ValueError(
            f"g_idx range [{int(packed_g_idx.min())}, {int(packed_g_idx.max())}] "
            f"is outside [0, {groups - 1}]"
        )

    if device is not None and torch.device(device).type != "cpu":
        weight_bytes = weight_bytes.to(device=device)
        zero_bytes = zero_bytes.to(device=device)
        zero_neg_one = zero_neg_one.to(device=device, dtype=torch.uint8)
        packed_scales = packed_scales.to(device=device)
        packed_g_idx = packed_g_idx.to(device=device)
        if packed_bias is not None:
            packed_bias = packed_bias.to(device=device)

    return TTPackedGPTQLayout(
        weight_bytes=weight_bytes,
        zero_bytes=zero_bytes,
        zero_neg_one=zero_neg_one,
        scales=packed_scales,
        g_idx=packed_g_idx,
        bias=packed_bias,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        bits=bits,
    )


def unpack_tt_word_4bit(words: torch.Tensor) -> torch.Tensor:
    """CPU oracle for the canonical per-output packed 4-bit words."""
    shifts = torch.arange(0, 32, TT_GPTQ_PACK_BITS, dtype=torch.int64)
    if words.dtype == torch.uint32:
        # CPU torch has no uint32 right shift in every supported version.
        words = words.to(torch.int64) & _UINT32_MASK
    else:
        words = words.to(torch.int64) & _UINT32_MASK
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(words.shape[0], -1)


def unpack_tt_byte_planes_4bit(byte_planes: torch.Tensor) -> torch.Tensor:
    """CPU oracle for the byte-plane ABI used by the Torch/XLA kernel."""
    if byte_planes.ndim != 3 or byte_planes.shape[-1] != 4:
        raise ValueError(f"expected [..., 4] byte planes, got {tuple(byte_planes.shape)}")
    # Assemble with four pairwise stacks to match TT's stable lowering shape;
    # this function is also the independent oracle for the device sequence.
    low = byte_planes & 0xF
    high = (byte_planes >> 4) & 0xF
    nibbles = torch.stack([low, high], dim=-1)
    return nibbles.reshape(byte_planes.shape[0], -1)


def unpack_tt_byte_planes_bf16(byte_planes: torch.Tensor) -> torch.Tensor:
    """Expand byte planes to BF16 codes with TT-supported operations.

    Keep the mask/shift in uint8, cast each nibble plane before stacking, and
    avoid device-side uint8 cat/tilize. The result is transient expanded codes,
    not a persistent dequantized-weight cache.
    """
    if byte_planes.ndim != 3 or byte_planes.shape[-1] != 4:
        raise ValueError(f"expected [out, words, 4] byte planes, got {tuple(byte_planes.shape)}")
    low = (byte_planes & 0xF).to(torch.bfloat16)
    high = ((byte_planes >> 4) & 0xF).to(torch.bfloat16)
    return torch.stack([low, high], dim=-1).reshape(byte_planes.shape[0], -1)


def dequantize_tt_packed_4bit(layout: TTPackedGPTQLayout) -> torch.Tensor:
    """CPU oracle returning ``[in_features, out_features]`` float weights."""
    if layout.weight_bytes.device.type != "cpu":
        raise ValueError("dequantize_tt_packed_4bit is a CPU oracle; move packed words to CPU first")
    codes = unpack_tt_byte_planes_4bit(layout.weight_bytes)[:, :layout.in_features].to(torch.float32)
    zero_codes = unpack_tt_byte_planes_4bit(layout.zero_bytes)[:, :layout.out_features].to(torch.float32)
    neg_one = layout.zero_neg_one.bool()[:, :layout.out_features]
    zero_codes = torch.where(neg_one, torch.full_like(zero_codes, -1.0), zero_codes)
    g_idx = layout.g_idx.to(device="cpu", dtype=torch.long)
    scales = layout.scales.to(device="cpu", dtype=torch.float32)
    return (scales[g_idx] * (codes.transpose(0, 1) - zero_codes[g_idx])).contiguous()


__all__ = [
    "TT_GPTQ_PACK_BITS",
    "TT_GPTQ_PACK_FACTOR",
    "TTPackedGPTQLayout",
    "dequantize_tt_packed_4bit",
    "repack_gptq_for_tt",
    "unpack_tt_byte_planes_bf16",
    "unpack_tt_byte_planes_4bit",
    "unpack_tt_word_4bit",
]
