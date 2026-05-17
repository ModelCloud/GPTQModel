# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Tuple

import torch

LORA_INT8_FORMAT = "int8_grouped"
LORA_INT8_QWEIGHT_SUFFIX = ".qweight"
LORA_INT8_SCALES_SUFFIX = ".scales"
LORA_INT8_SHAPE_SUFFIX = ".shape"


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


def quantize_tensor_groupwise_int8(
    tensor: torch.Tensor,
    *,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetrically quantizes a tensor into int8 values plus per-group scales."""

    if group_size <= 0:
        raise ValueError(f"LoRA int8 group_size must be positive, actual = `{group_size}`.")

    source = tensor.detach().contiguous()
    shape = torch.tensor(source.shape, dtype=torch.int64)
    flat = source.to(dtype=torch.float32).view(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    groups = flat.view(-1, group_size)
    max_abs = groups.abs().amax(dim=1)
    scales = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    qweight = torch.round(groups / scales[:, None]).clamp_(-127, 127).to(torch.int8).view(-1)
    return qweight.cpu(), scales.to(dtype=scale_dtype).cpu(), shape.cpu()


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

    if qweight.numel() % group_size != 0:
        raise ValueError(
            f"Compressed LoRA qweight length `{qweight.numel()}` is not divisible by group_size `{group_size}`."
        )

    shape_tuple = tuple(int(v) for v in shape.detach().cpu().tolist())
    expected_numel = math.prod(shape_tuple)
    if expected_numel > qweight.numel():
        raise ValueError(
            f"Compressed LoRA shape `{shape_tuple}` requires `{expected_numel}` values, "
            f"but qweight only stores `{qweight.numel()}` values."
        )

    q = qweight.to(device=device, non_blocking=True).view(-1, group_size).to(torch.float32)
    s = scales.to(device=device, dtype=torch.float32, non_blocking=True).view(-1, 1)
    dense = (q * s).view(-1)[:expected_numel].view(shape_tuple)
    return dense.to(dtype=dtype)
