# SPDX-License-Identifier: Apache-2.0
"""Decode MiMo mixed MXFP4/FP8 sources into canonical dense weights."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..quantization.dtype import dequantize_f4_e2m1


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def is_mimo_mixed_source(config: Any) -> bool:
    quant = _field(config, "quantization_config", {})
    return (
        _field(config, "model_type") == "mimo_v2"
        and _field(quant, "quant_method") == "fp8"
        and _field(quant, "store_dtype") == "mxfp4"
    )


def is_mimo_encoded_weight(config: Any, name: str, weight: torch.Tensor) -> bool:
    if not is_mimo_mixed_source(config) or not name.endswith(".weight"):
        return False
    if name.startswith("model.mtp."):
        return False  # Auxiliary predictors are preserved in their source encoding.
    expert = re.fullmatch(
        r"model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)\.weight",
        name,
    )
    encoded = weight.dtype == torch.float8_e4m3fn or (
        expert is not None and weight.dtype == torch.uint8
    )
    if not encoded and weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"Unsupported MiMo source storage: {name}: {weight.dtype}")
    return encoded


def decode_mxfp4(
    weight: torch.Tensor, scale: torch.Tensor, *, target_dtype: torch.dtype
) -> torch.Tensor:
    """Decode adjacent low/high E2M1 nibbles and U8 E8M0 scale bytes."""
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError("MiMo MXFP4 weights and scales must use uint8 storage")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("MiMo MXFP4 weights and scales must be matrices")
    rows, packed_cols = weight.shape
    cols = packed_cols * 2
    if not rows or not cols or cols % 32 or scale.shape != (rows, cols // 32):
        raise ValueError("MiMo MXFP4 scale shape must be [out, logical_in / 32]")
    if (scale == 255).any():
        raise ValueError("MiMo MXFP4 contains reserved E8M0 scale byte 255")
    result = torch.empty((rows, cols), dtype=target_dtype, device=weight.device)
    for start in range(0, rows, 256):
        packed = weight[start : start + 256].contiguous()
        factors = torch.exp2(scale[start : start + 256].float() - 127)
        decoded = dequantize_f4_e2m1(packed, target_dtype=torch.float32).unflatten(
            1, (-1, 32)
        )
        decoded.mul_(factors.unsqueeze(-1))
        result[start : start + 256] = decoded.flatten(1)
    if not torch.isfinite(result).all():
        raise ValueError("MiMo MXFP4 decoded weights overflow the target dtype")
    return result


def _decode_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    rows, cols = weight.shape
    br, bc = block
    expected = ((rows + br - 1) // br, (cols + bc - 1) // bc)
    if tuple(scale.shape) != expected:
        raise ValueError(f"MiMo FP8 scale shape {tuple(scale.shape)} != {expected}")
    result = torch.empty_like(weight, dtype=target_dtype)
    for start in range(0, rows, br):
        factors = scale[start // br].float().repeat_interleave(bc)[:cols]
        result[start : start + br] = (weight[start : start + br].float() * factors).to(
            target_dtype
        )
    if not torch.isfinite(result).all():
        raise ValueError("MiMo FP8 decoded weights are nonfinite")
    return result


def decode_mimo_weight(
    config: Any,
    name: str,
    weight: torch.Tensor,
    lookup: Callable[[str], torch.Tensor | None],
    *,
    target_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Decode a recognized source tensor, fetching its exact scale companion."""
    if not is_mimo_encoded_weight(config, name, weight):
        return None
    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("MiMo decoding requires a floating-point target dtype")
    quant = _field(config, "quantization_config")
    if weight.dtype == torch.uint8:
        if _field(quant, "mxfp4_block_size") != 32:
            raise ValueError("MiMo MXFP4 requires mxfp4_block_size=32")
        scale_name = name + "_scale"
    else:
        scale_name = name + "_scale_inv"
    scale = lookup(scale_name)
    if scale is None:
        raise ValueError(f"Missing MiMo scale tensor: {scale_name}")
    scale = scale.to(device=weight.device)
    if weight.dtype == torch.uint8:
        return decode_mxfp4(weight, scale, target_dtype=target_dtype)
    if weight.ndim != 2 or scale.dtype != torch.float32 or scale.ndim != 2:
        raise ValueError("MiMo FP8 requires matrix weights and FP32 scale grids")
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("MiMo FP8 scales must be finite and positive")
    block = _field(quant, "weight_block_size")
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(type(x) is not int or x <= 0 for x in block)
    ):
        raise ValueError("MiMo FP8 requires two positive weight_block_size values")
    block = tuple(block)
    match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight", name)
    if match is None:
        return _decode_fp8_blocks(weight, scale, block, target_dtype)
    if _field(config, "attention_projection_layout") != "fused_qkv":
        raise ValueError("MiMo qkv_proj requires fused_qkv source layout")
    layer = int(match[1])
    pattern = _field(config, "hybrid_layer_pattern", [])
    if layer >= len(pattern) or pattern[layer] not in (0, 1):
        raise ValueError(f"Missing MiMo attention geometry for layer {layer}")
    prefix = "swa_" if pattern[layer] else ""
    tp = _field(config, "num_key_value_heads")
    heads = _field(config, prefix + "num_attention_heads")
    kv_heads = _field(config, prefix + "num_key_value_heads")
    hd = _field(config, prefix + "head_dim")
    vd = _field(config, prefix + "v_head_dim")
    if any(type(x) is not int or x <= 0 for x in (tp, heads, kv_heads, hd, vd)):
        raise ValueError("MiMo fused QKV requires explicit positive head geometry")
    if heads % tp or kv_heads % tp:
        raise ValueError("MiMo fused QKV heads must divide checkpoint TP")
    sizes = (heads // tp * hd, kv_heads // tp * hd, kv_heads // tp * vd)
    shard_rows = sum(sizes)
    scale_rows = (shard_rows + block[0] - 1) // block[0]
    if weight.shape[0] != shard_rows * tp or scale.shape[0] != scale_rows * tp:
        raise ValueError("MiMo fused QKV weight/scale rows do not match checkpoint TP")
    output = torch.empty_like(weight, dtype=target_dtype)
    offsets = (0, sizes[0] * tp, (sizes[0] + sizes[1]) * tp)
    for rank in range(tp):
        shard = _decode_fp8_blocks(
            weight[rank * shard_rows : (rank + 1) * shard_rows],
            scale[rank * scale_rows : (rank + 1) * scale_rows],
            block,
            target_dtype,
        )
        for part, size, offset in zip(shard.split(sizes), sizes, offsets):
            output[offset + rank * size : offset + (rank + 1) * size] = part
    return output
