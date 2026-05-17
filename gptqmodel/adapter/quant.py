# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Tuple

import torch

LORA_INT4_FORMAT = "int4_grouped"
LORA_INT6_FORMAT = "int6_grouped"
LORA_INT8_FORMAT = "int8_grouped"
LORA_GROUPED_FORMATS = {
    4: LORA_INT4_FORMAT,
    6: LORA_INT6_FORMAT,
    8: LORA_INT8_FORMAT,
}
LORA_INT8_QWEIGHT_SUFFIX = ".qweight"
LORA_INT8_SCALES_SUFFIX = ".scales"
LORA_INT8_SHAPE_SUFFIX = ".shape"
SUPPORTED_LORA_GROUPED_BITS = tuple(LORA_GROUPED_FORMATS)


def dtype_from_name(name: str | None, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Maps adapter config dtype names to torch dtypes for scale storage."""

    if name is None:
        return default
    normalized = str(name).lower().removeprefix("torch.")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported LoRA int8 scale dtype: `{name}`.")


def compressed_weight_keys(weight_key: str) -> Tuple[str, str, str]:
    """Returns qweight, scales, and shape keys for one serialized LoRA tensor."""

    return (
        f"{weight_key}{LORA_INT8_QWEIGHT_SUFFIX}",
        f"{weight_key}{LORA_INT8_SCALES_SUFFIX}",
        f"{weight_key}{LORA_INT8_SHAPE_SUFFIX}",
    )


def bits_from_lora_weight_format(weight_format: str | None) -> int | None:
    """Extracts the bit width encoded by a GPTQModel grouped-LoRA format string."""

    if weight_format is None:
        return None
    normalized = str(weight_format).strip().lower()
    for bits, fmt in LORA_GROUPED_FORMATS.items():
        if normalized == fmt:
            return bits
    return None


def is_grouped_lora_weight_format(weight_format: str | None) -> bool:
    """Reports whether a LoRA format is one of GPTQModel's grouped low-bit formats."""

    return bits_from_lora_weight_format(weight_format) is not None


def lora_grouped_format_from_bits(bits: int) -> str:
    """Returns the serialized GPTQModel grouped-LoRA format for a bit width."""

    bits = int(bits)
    if bits not in LORA_GROUPED_FORMATS:
        raise ValueError(f"LoRA grouped quantization bits must be one of {SUPPORTED_LORA_GROUPED_BITS}: actual = `{bits}`.")
    return LORA_GROUPED_FORMATS[bits]


def normalize_lora_grouped_bits(bits: int | None, weight_format: str | None = None) -> int:
    """Resolves grouped-LoRA bit width from explicit config and/or the format name."""

    format_bits = bits_from_lora_weight_format(weight_format)
    if bits is None:
        bits = format_bits if format_bits is not None else 8
    bits = int(bits)
    if bits not in LORA_GROUPED_FORMATS:
        raise ValueError(f"LoRA grouped quantization bits must be one of {SUPPORTED_LORA_GROUPED_BITS}: actual = `{bits}`.")
    if format_bits is not None and bits != format_bits:
        raise ValueError(
            f"LoRA grouped quantization bits `{bits}` do not match format `{weight_format}` "
            f"which implies `{format_bits}` bits."
        )
    return bits


def _pack_signed_groupwise_values(q_signed: torch.Tensor, bits: int) -> torch.Tensor:
    """Packs signed low-bit LoRA qvalues into bytes for adapter storage."""

    if bits == 8:
        return q_signed.to(torch.int8).view(-1)

    maxq = (1 << (bits - 1)) - 1
    q_unsigned = (q_signed + maxq).to(torch.uint8).view(-1)
    if bits == 4:
        if q_unsigned.numel() % 2:
            q_unsigned = torch.nn.functional.pad(q_unsigned, (0, 1))
        pairs = q_unsigned.to(torch.int16).view(-1, 2)
        packed = (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)
        return packed.contiguous()

    if bits == 6:
        pad = (-q_unsigned.numel()) % 4
        if pad:
            q_unsigned = torch.nn.functional.pad(q_unsigned, (0, pad))
        quads = q_unsigned.to(torch.int16).view(-1, 4)
        byte0 = ((quads[:, 0] & 0x3F) | ((quads[:, 1] & 0x03) << 6)).to(torch.uint8)
        byte1 = (((quads[:, 1] >> 2) & 0x0F) | ((quads[:, 2] & 0x0F) << 4)).to(torch.uint8)
        byte2 = (((quads[:, 2] >> 4) & 0x03) | ((quads[:, 3] & 0x3F) << 2)).to(torch.uint8)
        return torch.stack((byte0, byte1, byte2), dim=1).reshape(-1).contiguous()

    raise ValueError(f"LoRA grouped quantization bits must be one of {SUPPORTED_LORA_GROUPED_BITS}: actual = `{bits}`.")


def _unpack_signed_groupwise_values(qweight: torch.Tensor, *, bits: int, padded_numel: int, device: torch.device) -> torch.Tensor:
    """Unpacks serialized grouped LoRA qvalues into signed int16 values."""

    if bits == 8:
        if qweight.numel() < padded_numel:
            raise ValueError(
                f"Compressed LoRA int8 qweight stores `{qweight.numel()}` values, "
                f"but `{padded_numel}` padded values are required."
            )
        return qweight.to(device=device, non_blocking=True).view(-1)[:padded_numel].to(torch.int16)

    packed = qweight.to(device=device, non_blocking=True).view(-1).to(torch.int16)
    maxq = (1 << (bits - 1)) - 1
    if bits == 4:
        required_bytes = (padded_numel + 1) // 2
        if packed.numel() < required_bytes:
            raise ValueError(
                f"Compressed LoRA int4 qweight stores `{packed.numel()}` bytes, "
                f"but `{required_bytes}` bytes are required."
            )
        packed = packed[:required_bytes]
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unpacked = torch.stack((low, high), dim=1).reshape(-1)[:padded_numel]
        return unpacked - maxq

    if bits == 6:
        required_bytes = ((padded_numel + 3) // 4) * 3
        if packed.numel() < required_bytes:
            raise ValueError(
                f"Compressed LoRA int6 qweight stores `{packed.numel()}` bytes, "
                f"but `{required_bytes}` bytes are required."
            )
        packed = packed[:required_bytes].view(-1, 3)
        byte0 = packed[:, 0]
        byte1 = packed[:, 1]
        byte2 = packed[:, 2]
        q0 = byte0 & 0x3F
        q1 = ((byte0 >> 6) & 0x03) | ((byte1 & 0x0F) << 2)
        q2 = ((byte1 >> 4) & 0x0F) | ((byte2 & 0x03) << 4)
        q3 = (byte2 >> 2) & 0x3F
        unpacked = torch.stack((q0, q1, q2, q3), dim=1).reshape(-1)[:padded_numel]
        return unpacked - maxq

    raise ValueError(f"LoRA grouped quantization bits must be one of {SUPPORTED_LORA_GROUPED_BITS}: actual = `{bits}`.")


def quantize_tensor_groupwise_int(
    tensor: torch.Tensor,
    *,
    bits: int = 8,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetrically quantizes a tensor into packed grouped signed integer values."""

    bits = normalize_lora_grouped_bits(bits)
    if group_size <= 0:
        raise ValueError(f"LoRA grouped group_size must be positive, actual = `{group_size}`.")

    source = tensor.detach().contiguous()
    shape = torch.tensor(source.shape, dtype=torch.int64)
    flat = source.to(dtype=torch.float32).view(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    groups = flat.view(-1, group_size)
    max_abs = groups.abs().amax(dim=1)
    maxq = float((1 << (bits - 1)) - 1)
    scales = torch.where(max_abs > 0, max_abs / maxq, torch.ones_like(max_abs))
    q_signed = torch.round(groups / scales[:, None]).clamp_(-maxq, maxq).to(torch.int16).view(-1)
    qweight = _pack_signed_groupwise_values(q_signed, bits)
    return qweight.cpu(), scales.to(dtype=scale_dtype).cpu(), shape.cpu()


def quantize_tensor_groupwise_int8(
    tensor: torch.Tensor,
    *,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetrically quantizes a tensor into int8 values plus per-group scales."""

    return quantize_tensor_groupwise_int(
        tensor,
        bits=8,
        group_size=group_size,
        scale_dtype=scale_dtype,
    )


def dequantize_tensor_groupwise_int(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    shape: torch.Tensor,
    bits: int = 8,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materializes a grouped low-bit tensor into the requested dense dtype."""

    bits = normalize_lora_grouped_bits(bits)
    if group_size <= 0:
        raise ValueError(f"LoRA grouped group_size must be positive, actual = `{group_size}`.")

    shape_tuple = tuple(int(v) for v in shape.detach().cpu().tolist())
    expected_numel = math.prod(shape_tuple)
    padded_numel = math.ceil(expected_numel / group_size) * group_size
    expected_groups = padded_numel // group_size
    if scales.numel() < expected_groups:
        raise ValueError(
            f"Compressed LoRA shape `{shape_tuple}` requires `{expected_groups}` scale values, "
            f"but scales only stores `{scales.numel()}` values."
        )

    q = _unpack_signed_groupwise_values(
        qweight,
        bits=bits,
        padded_numel=padded_numel,
        device=device,
    ).view(-1, group_size).to(torch.float32)
    s = scales.to(device=device, dtype=torch.float32, non_blocking=True).view(-1, 1)
    dense = (q * s).view(-1)[:expected_numel].view(shape_tuple)
    return dense.to(dtype=dtype)


def dequantize_tensor_groupwise_int8(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    shape: torch.Tensor,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materializes a grouped-int8 tensor into the requested dense dtype."""

    return dequantize_tensor_groupwise_int(
        qweight=qweight,
        scales=scales,
        shape=shape,
        bits=8,
        group_size=group_size,
        device=device,
        dtype=dtype,
    )
