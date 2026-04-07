# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.modules.conv import _ConvNd

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import (
    FORMAT,
    METHOD,
    Fallback,
    FallbackStrategy,
    GGUFBits,
    SmoothMethod,
    _normalize_quant_bits,
)
from ...quantization.fallback_smooth import smooth_block
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import WeightOnlyQuantLinear


try:
    import gguf as gguf_lib

    _GGUF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gguf_lib = None
    _GGUF_AVAILABLE = False


setup_logger()

_GGUF_TYPE_INFO = {
    "Q1_0": {"bits": 1, "block_size": 32, "type_size": 6},
    "Q1_0_g128": {"bits": 1, "block_size": 128, "type_size": 18},
    "Q4_0": {"bits": 4, "block_size": 32, "type_size": 18},
    "Q8_0": {"bits": 8, "block_size": 32, "type_size": 34},
    "Q4_K": {"bits": 4, "block_size": 256, "type_size": 144},
    "Q5_K": {"bits": 5, "block_size": 256, "type_size": 176},
    "Q6_K": {"bits": 6, "block_size": 256, "type_size": 210},
}
_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE = {
    "q1_0": "Q1_0",
    "q1_0_g128": "Q1_0_g128",
    "q4_0": "Q4_0",
    "q8_0": "Q8_0",
    "q4_k": "Q4_K",
    "q4_k_s": "Q4_K",
    "q4_k_m": "Q4_K",
    "q5_k": "Q5_K",
    "q5_k_s": "Q5_K",
    "q5_k_m": "Q5_K",
    "q6_k": "Q6_K",
}
_GGUF_SCALE_QUANT_MAX = 63
_GGUF_Q6_SCALE_QUANT_MAX = 127
_GGUF_K_QTYPES = {"Q4_K", "Q5_K", "Q6_K"}
PRISM_Q1_0_G128_NAME = "Q1_0_g128"
PRISM_Q1_0_G128_VALUE = 41
PRISM_Q1_0_G128_BLOCK_SIZE = 128
PRISM_Q1_0_G128_TYPE_SIZE = 18
_GGUF_SIGN_ONLY_TYPE_INFO = {
    "Q1_0": {"block_size": 32, "type_size": 6},
    PRISM_Q1_0_G128_NAME: {
        "block_size": PRISM_Q1_0_G128_BLOCK_SIZE,
        "type_size": PRISM_Q1_0_G128_TYPE_SIZE,
    },
}
_GGUF_TENSOR_QTYPE_BY_VALUE = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    30: "BF16",
    40: "Q1_0",
    PRISM_Q1_0_G128_VALUE: PRISM_Q1_0_G128_NAME,
}
_GGUF_SIGN_ONLY_LUT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little").astype(np.int8) * 2 - 1
)
_GGUF_SIGN_ONLY_TORCH_LUT: dict[str, torch.Tensor] = {}


def _normalize_gguf_bits(bits) -> tuple[GGUFBits, str]:
    bits_spec = _normalize_quant_bits(bits, format_value=FORMAT.GGUF)
    tensor_qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE.get(bits_spec.name)
    if tensor_qtype is None:
        supported = ", ".join(sorted(_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE))
        raise ValueError(f"Unsupported GGUF bits `{bits}`. Supported values: {supported}.")

    qtype_info = _GGUF_TYPE_INFO[tensor_qtype]
    if qtype_info["bits"] != bits_spec.width:
        raise ValueError(
            f"GGUF bits `{bits_spec.name}` require {qtype_info['bits']}-bit GGUF packing, but got bits={bits_spec.width}."
        )

    return bits_spec, tensor_qtype


def _apply_optional_smoother(
    weight: torch.Tensor,
    *,
    smooth: SmoothMethod | None,
    group_size: int,
) -> torch.Tensor:
    if smooth is None:
        return weight

    effective_group_size = weight.shape[1] if group_size == -1 else group_size
    if effective_group_size <= 0:
        effective_group_size = weight.shape[1]

    fallback = Fallback(
        strategy=FallbackStrategy.RTN,
        threshold=True,
        smooth=smooth,
    )
    smoothed = weight.clone()

    for start in range(0, weight.shape[1], effective_group_size):
        end = min(start + effective_group_size, weight.shape[1])
        block, scale_factor = smooth_block(
            smoothed[:, start:end],
            fallback,
            group_size=effective_group_size,
        )
        if scale_factor is not None:
            raise ValueError(
                "GGUF direct packing does not support smoothers that require post-quant rescaling."
            )
        smoothed[:, start:end] = block

    return smoothed


def _gguf_quantize_q4_0(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    block_size = _GGUF_TYPE_INFO["Q4_0"]["block_size"]

    imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
    max_vals = np.take_along_axis(blocks, imax, axis=-1)

    d = max_vals / -8.0
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1.0 / d)

    # Match ggml's q4_0 reference path by truncating after the +8.5 offset.
    qs = np.trunc((blocks.astype(np.float64) * inv_d.astype(np.float64)) + 8.5).astype(np.uint8)
    qs = np.clip(qs, 0, 15)
    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[:, 0, :] | (qs[:, 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1)


def _gguf_quantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    d = np.abs(blocks).max(axis=-1, keepdims=True) / 127.0
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1.0 / d)
    qs = np.round(blocks * inv_d).astype(np.int8).view(np.uint8)
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1)


def _gguf_quantize_sign_only(blocks: np.ndarray, *, block_size: int) -> np.ndarray:
    scales = np.mean(np.abs(blocks), axis=-1).astype(np.float16, copy=False)
    sign_bits = np.packbits((blocks >= 0).astype(np.uint8, copy=False), axis=-1, bitorder="little")

    packed = np.empty((blocks.shape[0], 2 + (block_size // 8)), dtype=np.uint8)
    packed[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = sign_bits
    return packed


def _pack_q4_k_scale_min(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    scales = scales.astype(np.uint8, copy=False)
    mins = mins.astype(np.uint8, copy=False)

    d = (scales[:, :4] & np.uint8(0x3F)) | ((scales[:, 4:] & np.uint8(0x30)) << np.uint8(2))
    m = (mins[:, :4] & np.uint8(0x3F)) | ((mins[:, 4:] & np.uint8(0x30)) << np.uint8(2))
    md = (scales[:, 4:] & np.uint8(0x0F)) | ((mins[:, 4:] & np.uint8(0x0F)) << np.uint8(4))

    return np.concatenate([d, m, md], axis=-1)


def _unpack_q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = scales.astype(np.uint8, copy=False).reshape((-1, 3, 4))
    d, m, md = np.split(packed, 3, axis=-2)

    sc = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mins = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((-1, 8)), mins.reshape((-1, 8))


def _unpack_q4_k_scale_min_torch(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = scales.reshape(*scales.shape[:-1], 3, 4)
    d = packed[..., 0, :]
    m = packed[..., 1, :]
    md = packed[..., 2, :]

    sc = torch.cat((d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)), dim=-1)
    mins = torch.cat((m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)), dim=-1)
    return sc, mins


def _quantize_k_subblocks(
    subblocks: np.ndarray,
    *,
    maxq: int,
    scale_quant_max: int,
    signed: bool,
) -> tuple[np.ndarray, ...]:
    if signed:
        scale = np.abs(subblocks).max(axis=-1) / maxq
        base = scale.max(axis=-1, keepdims=True) / scale_quant_max
        with np.errstate(divide="ignore", invalid="ignore"):
            quant_scales = np.where(base > 0, np.rint(scale / base), 0.0)
        quant_scales = np.clip(quant_scales, 0, scale_quant_max).astype(np.int32)
        eff_scale = base * quant_scales.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(eff_scale[..., None] > 0, np.rint(subblocks / eff_scale[..., None]), 0.0)
        q = np.clip(q, -32, 31).astype(np.int8)
        return base.astype(np.float16), quant_scales.astype(np.int8), q

    mins = np.maximum(-subblocks.min(axis=-1), 0.0)
    scale = (subblocks.max(axis=-1) + mins) / maxq

    base = scale.max(axis=-1, keepdims=True) / scale_quant_max
    min_base = mins.max(axis=-1, keepdims=True) / scale_quant_max

    with np.errstate(divide="ignore", invalid="ignore"):
        quant_scales = np.where(base > 0, np.rint(scale / base), 0.0)
        quant_mins = np.where(min_base > 0, np.rint(mins / min_base), 0.0)

    quant_scales = np.clip(quant_scales, 0, scale_quant_max).astype(np.uint8)
    quant_mins = np.clip(quant_mins, 0, scale_quant_max).astype(np.uint8)

    eff_scale = base * quant_scales.astype(np.float32)
    eff_min = min_base * quant_mins.astype(np.float32)
    shifted = subblocks + eff_min[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(eff_scale[..., None] > 0, np.rint(shifted / eff_scale[..., None]), 0.0)
    q = np.clip(q, 0, maxq).astype(np.uint8)

    return base.astype(np.float16), min_base.astype(np.float16), quant_scales, quant_mins, q


def _gguf_quantize_q4_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    subblocks = blocks.reshape(n_blocks, 8, 32)
    d, dmin, sc, mins, q = _quantize_k_subblocks(
        subblocks,
        maxq=15,
        scale_quant_max=_GGUF_SCALE_QUANT_MAX,
        signed=False,
    )
    scales = _pack_q4_k_scale_min(sc, mins)
    q_pairs = q.reshape(n_blocks, 4, 2, 32)
    qs = q_pairs[:, :, 0, :] | (q_pairs[:, :, 1, :] << np.uint8(4))
    return np.concatenate(
        [
            d.view(np.uint8),
            dmin.view(np.uint8),
            scales,
            qs.reshape(n_blocks, 128),
        ],
        axis=-1,
    )


def _gguf_quantize_q5_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    subblocks = blocks.reshape(n_blocks, 8, 32)
    d, dmin, sc, mins, q = _quantize_k_subblocks(
        subblocks,
        maxq=31,
        scale_quant_max=_GGUF_SCALE_QUANT_MAX,
        signed=False,
    )
    scales = _pack_q4_k_scale_min(sc, mins)
    q_pairs = q.reshape(n_blocks, 4, 2, 32)
    qs = (q_pairs[:, :, 0, :] & np.uint8(0x0F)) | ((q_pairs[:, :, 1, :] & np.uint8(0x0F)) << np.uint8(4))
    qh = np.sum(
        (((q >> np.uint8(4)) & np.uint8(0x01)).astype(np.uint16) << np.arange(8, dtype=np.uint16).reshape(1, 8, 1)),
        axis=1,
        dtype=np.uint16,
    ).astype(np.uint8)
    return np.concatenate(
        [
            d.view(np.uint8),
            dmin.view(np.uint8),
            scales,
            qh,
            qs.reshape(n_blocks, 128),
        ],
        axis=-1,
    )


def _gguf_quantize_q6_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    subblocks = blocks.reshape(n_blocks, 16, 16)
    d, scales, q = _quantize_k_subblocks(
        subblocks,
        maxq=31,
        scale_quant_max=_GGUF_Q6_SCALE_QUANT_MAX,
        signed=True,
    )
    q_raw = (q.astype(np.int16) + 32).astype(np.uint8).reshape(n_blocks, 8, 32)

    ql = np.empty((n_blocks, 128), dtype=np.uint8)
    qh = np.empty((n_blocks, 64), dtype=np.uint8)

    for group in range(2):
        base_q = group * 4
        base_ql = group * 64
        base_qh = group * 32

        ql[:, base_ql : base_ql + 32] = (
            (q_raw[:, base_q + 0, :] & np.uint8(0x0F))
            | ((q_raw[:, base_q + 2, :] & np.uint8(0x0F)) << np.uint8(4))
        )
        ql[:, base_ql + 32 : base_ql + 64] = (
            (q_raw[:, base_q + 1, :] & np.uint8(0x0F))
            | ((q_raw[:, base_q + 3, :] & np.uint8(0x0F)) << np.uint8(4))
        )

        qh[:, base_qh : base_qh + 32] = (
            ((q_raw[:, base_q + 0, :] >> np.uint8(4)) & np.uint8(0x03))
            | (((q_raw[:, base_q + 1, :] >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(2))
            | (((q_raw[:, base_q + 2, :] >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(4))
            | (((q_raw[:, base_q + 3, :] >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(6))
        )

    return np.concatenate(
        [
            ql,
            qh,
            scales.view(np.uint8),
            d.view(np.uint8),
        ],
        axis=-1,
    )


def _fallback_gguf_quantize(weight: np.ndarray, tensor_qtype: str) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"GGUF quantization expects a 2D weight matrix, got shape {weight.shape}.")
    qtype_info = _GGUF_TYPE_INFO[tensor_qtype]
    block_size = qtype_info["block_size"]
    if weight.shape[1] % block_size != 0:
        raise ValueError(
            f"GGUF quantization expects the input dimension to be divisible by {block_size}, got {weight.shape[1]}."
        )

    blocks = weight.reshape(-1, block_size)
    if tensor_qtype in _GGUF_SIGN_ONLY_TYPE_INFO:
        quantized_blocks = _gguf_quantize_sign_only(blocks, block_size=block_size)
    elif tensor_qtype == "Q4_0":
        quantized_blocks = _gguf_quantize_q4_0(blocks)
    elif tensor_qtype == "Q8_0":
        quantized_blocks = _gguf_quantize_q8_0(blocks)
    elif tensor_qtype == "Q4_K":
        quantized_blocks = _gguf_quantize_q4_k(blocks)
    elif tensor_qtype == "Q5_K":
        quantized_blocks = _gguf_quantize_q5_k(blocks)
    elif tensor_qtype == "Q6_K":
        quantized_blocks = _gguf_quantize_q6_k(blocks)
    else:  # pragma: no cover - guarded by class SUPPORTS_BITS
        raise NotImplementedError(f"Unsupported GGUF qtype: {tensor_qtype}")

    bytes_per_block = qtype_info["type_size"]
    rows = weight.shape[0]
    return quantized_blocks.reshape(rows, (weight.shape[1] // block_size) * bytes_per_block)


def _gguf_quantize(weight: np.ndarray, tensor_qtype: str) -> np.ndarray:
    return _quantize_gguf_tensor_numpy(weight, tensor_qtype)


def _resolve_gguf_tensor_qtype(tensor_type) -> str:
    if isinstance(tensor_type, str):
        normalized = tensor_type.strip()
        if normalized in _GGUF_TYPE_INFO or normalized in _GGUF_SIGN_ONLY_TYPE_INFO:
            return normalized
        raise NotImplementedError(f"Unsupported GGUF qtype: {tensor_type}")

    tensor_name = getattr(tensor_type, "name", None)
    if tensor_name in _GGUF_TYPE_INFO or tensor_name in _GGUF_SIGN_ONLY_TYPE_INFO:
        return tensor_name

    try:
        tensor_value = int(tensor_type)
    except (TypeError, ValueError):
        tensor_value = None

    if tensor_value is None:
        raise NotImplementedError(f"Unsupported GGUF qtype: {tensor_type}")

    resolved = _GGUF_TENSOR_QTYPE_BY_VALUE.get(tensor_value)
    if resolved is None:
        raise NotImplementedError(f"Unsupported GGUF qtype value: {tensor_value}")
    return resolved


def _is_prism_q1_0_g128(tensor_type) -> bool:
    return _resolve_gguf_tensor_qtype(tensor_type) == PRISM_Q1_0_G128_NAME


def _dequantize_sign_only_numpy(
    data: np.ndarray,
    *,
    block_size: int,
    type_size: int,
) -> np.ndarray:
    rows = np.asarray(data, dtype=np.uint8)
    if rows.shape[-1] % type_size != 0:
        raise ValueError(
            f"GGUF sign-only row byte width must be divisible by {type_size}, got "
            f"{rows.shape[-1]} for shape {rows.shape}."
        )

    n_blocks = rows.shape[-1] // type_size
    blocks = rows.reshape(*rows.shape[:-1], n_blocks, type_size)
    scales = np.ascontiguousarray(blocks[..., :2]).view(np.float16).astype(np.float32)[..., 0]
    sign_bits = np.unpackbits(blocks[..., 2:], axis=-1, bitorder="little")
    weights = np.where(sign_bits == 1, scales[..., None], -scales[..., None]).astype(np.float32, copy=False)
    return weights.reshape(*rows.shape[:-1], n_blocks * block_size)


def _get_sign_only_torch_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    lut = _GGUF_SIGN_ONLY_TORCH_LUT.get(key)
    if lut is None or lut.device != device:
        lut = torch.from_numpy(_GGUF_SIGN_ONLY_LUT).to(device=device, dtype=torch.int8)
        _GGUF_SIGN_ONLY_TORCH_LUT[key] = lut
    return lut


def _dequantize_sign_only_torch(
    data: np.ndarray | torch.Tensor,
    *,
    block_size: int,
    type_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if torch.is_tensor(data):
        rows = data
        if rows.dtype != torch.uint8:
            rows = rows.to(dtype=torch.uint8)
    else:
        rows = torch.from_numpy(np.array(data, dtype=np.uint8, copy=True, order="C"))

    target_device = rows.device if device is None else torch.device(device)
    if rows.device != target_device:
        rows = rows.to(device=target_device, non_blocking=rows.device.type == "cpu" and target_device.type == "cuda")
    if not rows.is_contiguous():
        rows = rows.contiguous()

    if rows.shape[-1] % type_size != 0:
        raise ValueError(
            f"GGUF sign-only row byte width must be divisible by {type_size}, got "
            f"{rows.shape[-1]} for shape {tuple(rows.shape)}."
        )

    n_blocks = rows.shape[-1] // type_size
    blocks = rows.reshape(*rows.shape[:-1], n_blocks, type_size)
    scales = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
    if scales.dtype != dtype:
        scales = scales.to(dtype)

    sign_bytes = blocks[..., 2:].to(dtype=torch.long)
    sign_lut = _get_sign_only_torch_lut(target_device)
    signs = sign_lut[sign_bytes].reshape(*rows.shape[:-1], n_blocks, block_size)
    if signs.dtype != dtype:
        signs = signs.to(dtype)

    weights = scales.unsqueeze(-1) * signs
    return weights.reshape(*rows.shape[:-1], n_blocks * block_size)


def _dequantize_prism_q1_0_g128(data: np.ndarray) -> np.ndarray:
    return _dequantize_sign_only_numpy(
        data,
        block_size=PRISM_Q1_0_G128_BLOCK_SIZE,
        type_size=PRISM_Q1_0_G128_TYPE_SIZE,
    )


def _dequantize_prism_q1_0_g128_torch(
    data: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return _dequantize_sign_only_torch(
        data,
        block_size=PRISM_Q1_0_G128_BLOCK_SIZE,
        type_size=PRISM_Q1_0_G128_TYPE_SIZE,
        device=device,
        dtype=dtype,
    )


def _quantize_gguf_tensor_numpy(weight: np.ndarray, tensor_qtype) -> np.ndarray:
    resolved_qtype = _resolve_gguf_tensor_qtype(tensor_qtype)
    if _GGUF_AVAILABLE:
        qtype = getattr(gguf_lib.GGMLQuantizationType, resolved_qtype, None)
        try:
            if qtype is not None:
                return gguf_lib.quantize(weight, qtype)
        except NotImplementedError:
            pass
    return _fallback_gguf_quantize(weight, resolved_qtype)


def _dequantize_q4_k_numpy(qweight: np.ndarray) -> np.ndarray:
    rows = qweight.shape[0]
    type_size = _GGUF_TYPE_INFO["Q4_K"]["type_size"]
    blocks = qweight.reshape(-1, type_size)

    d = blocks[:, :2].view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:]

    sc, mins = _unpack_q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((-1, 8, 1))
    dm = (dmin * mins.astype(np.float32)).reshape((-1, 8, 1))

    q = qs.reshape((-1, 4, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    q = (q & np.uint8(0x0F)).reshape((-1, 8, 32)).astype(np.float32)

    return (d * q - dm).reshape(rows, -1)


def _dequantize_q5_k_numpy(qweight: np.ndarray) -> np.ndarray:
    rows = qweight.shape[0]
    type_size = _GGUF_TYPE_INFO["Q5_K"]["type_size"]
    blocks = qweight.reshape(-1, type_size)

    d = blocks[:, :2].view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:]

    sc, mins = _unpack_q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((-1, 8, 1))
    dm = (dmin * mins.astype(np.float32)).reshape((-1, 8, 1))

    ql = qs.reshape((-1, 4, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((-1, 1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))

    ql = (ql & np.uint8(0x0F)).reshape((-1, 8, 32))
    qh = (qh & np.uint8(0x01)).reshape((-1, 8, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)

    return (d * q - dm).reshape(rows, -1)


def _dequantize_q6_k_numpy(qweight: np.ndarray) -> np.ndarray:
    rows = qweight.shape[0]
    type_size = _GGUF_TYPE_INFO["Q6_K"]["type_size"]
    blocks = qweight.reshape(-1, type_size)

    ql = blocks[:, :128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].view(np.float16).astype(np.float32)
    d = (d * scales).reshape((-1, 16, 1))

    ql = ql.reshape((-1, 2, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((-1, 8, 32))
    qh = qh.reshape((-1, 2, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((-1, 8, 32))

    q = (ql | (qh << np.uint8(4))).astype(np.int16) - 32
    q = q.reshape((-1, 16, 16)).astype(np.float32)

    return (d * q).reshape(rows, -1)


def _dequantize_gguf_tensor_numpy(data: np.ndarray, tensor_type) -> np.ndarray:
    resolved_qtype = _resolve_gguf_tensor_qtype(tensor_type)

    if resolved_qtype == "F32":
        return np.asarray(data, dtype=np.float32)
    if resolved_qtype == "F16":
        return np.asarray(data, dtype=np.float16).astype(np.float32)
    if resolved_qtype == "BF16":
        rows = np.asarray(data, dtype=np.uint16).astype(np.uint32)
        return np.left_shift(rows, np.uint32(16)).view(np.float32)
    if resolved_qtype == "Q4_0":
        rows = np.asarray(data, dtype=np.uint8)
        type_size = _GGUF_TYPE_INFO["Q4_0"]["type_size"]
        blocks = rows.reshape(-1, type_size)
        d = blocks[:, :2].view(np.float16).astype(np.float32)
        qs = blocks[:, 2:].reshape((-1, 1, 16))
        low = qs & np.uint8(0x0F)
        high = qs >> np.uint8(4)
        q = np.concatenate([low, high], axis=1).reshape((-1, 32)).astype(np.int16) - 8
        return (d * q.astype(np.float32)).reshape(rows.shape[0], -1)
    if resolved_qtype == "Q8_0":
        rows = np.asarray(data, dtype=np.uint8)
        type_size = _GGUF_TYPE_INFO["Q8_0"]["type_size"]
        blocks = rows.reshape(-1, type_size)
        d = blocks[:, :2].view(np.float16).astype(np.float32)
        q = blocks[:, 2:].view(np.int8).astype(np.float32)
        return (d * q).reshape(rows.shape[0], -1)
    if resolved_qtype == "Q4_K":
        return _dequantize_q4_k_numpy(np.asarray(data, dtype=np.uint8))
    if resolved_qtype == "Q5_K":
        return _dequantize_q5_k_numpy(np.asarray(data, dtype=np.uint8))
    if resolved_qtype == "Q6_K":
        return _dequantize_q6_k_numpy(np.asarray(data, dtype=np.uint8))
    if resolved_qtype == "Q1_0":
        return _dequantize_sign_only_numpy(
            data,
            block_size=_GGUF_SIGN_ONLY_TYPE_INFO["Q1_0"]["block_size"],
            type_size=_GGUF_SIGN_ONLY_TYPE_INFO["Q1_0"]["type_size"],
        )
    if resolved_qtype == PRISM_Q1_0_G128_NAME:
        return _dequantize_prism_q1_0_g128(np.asarray(data, dtype=np.uint8))

    raise NotImplementedError(f"Unsupported GGUF qtype: {resolved_qtype}")


class GGUFTorchLinear(WeightOnlyQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_TORCH]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 15}
    SUPPORTS_BITS = [1, 4, 5, 6, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    AUTOTUNE = True

    QUANT_TYPE = "gguf"
    GGUF_FUSED_CUDA_MAX_ROWS = max(0, int(os.environ.get("GPTQMODEL_GGUF_FUSED_CUDA_MAX_ROWS", "32")))
    GGUF_FUSED_CUDA_MIN_MATRIX_ELEMENTS = max(
        0,
        int(os.environ.get("GPTQMODEL_GGUF_FUSED_CUDA_MIN_MATRIX_ELEMENTS", "8388608")),
    )
    GGUF_FUSED_CPU_MAX_ROWS = max(0, int(os.environ.get("GPTQMODEL_GGUF_FUSED_CPU_MAX_ROWS", "64")))
    GGUF_FUSED_CPU_MIN_MATRIX_ELEMENTS = max(
        0,
        int(os.environ.get("GPTQMODEL_GGUF_FUSED_CPU_MIN_MATRIX_ELEMENTS", "0")),
    )
    GGUF_FUSED_CHUNK_BLOCKS = max(1, int(os.environ.get("GPTQMODEL_GGUF_FUSED_CHUNK_BLOCKS", "8")))
    GGUF_FUSED_AUTOTUNE_WARMUP = max(0, int(os.environ.get("GPTQMODEL_GGUF_FUSED_AUTOTUNE_WARMUP", "1")))
    GGUF_FUSED_AUTOTUNE_ITERS = max(1, int(os.environ.get("GPTQMODEL_GGUF_FUSED_AUTOTUNE_ITERS", "2")))
    GGUF_FUSED_AUTOTUNE_MARGIN = max(0.0, float(os.environ.get("GPTQMODEL_GGUF_FUSED_AUTOTUNE_MARGIN", "0.05")))

    def __init__(
        self,
        bits,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        bits_spec, self.gguf_tensor_qtype = _normalize_gguf_bits(bits)
        qtype_info = _GGUF_TYPE_INFO[self.gguf_tensor_qtype]
        self.gguf_block_size = qtype_info["block_size"]
        self.gguf_type_size = qtype_info["type_size"]
        self.padded_in_features = math.ceil(in_features / self.gguf_block_size) * self.gguf_block_size
        self.gguf_fused_cuda_max_rows = self.GGUF_FUSED_CUDA_MAX_ROWS
        self.gguf_fused_cuda_min_matrix_elements = self.GGUF_FUSED_CUDA_MIN_MATRIX_ELEMENTS
        self.gguf_fused_cpu_max_rows = self.GGUF_FUSED_CPU_MAX_ROWS
        self.gguf_fused_cpu_min_matrix_elements = self.GGUF_FUSED_CPU_MIN_MATRIX_ELEMENTS
        self.gguf_fused_chunk_blocks = self.GGUF_FUSED_CHUNK_BLOCKS
        self.gguf_fused_autotune_warmup = self.GGUF_FUSED_AUTOTUNE_WARMUP
        self.gguf_fused_autotune_iters = self.GGUF_FUSED_AUTOTUNE_ITERS
        self.gguf_fused_autotune_margin = self.GGUF_FUSED_AUTOTUNE_MARGIN

        super().__init__(
            bits=int(bits_spec),
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.GGUF_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            **kwargs,
        )

        self.bits = bits_spec

        if register_buffers:
            self._allocate_buffers(bias=bias)

    def _bytes_per_row(self) -> int:
        return (self.padded_in_features // self.gguf_block_size) * self.gguf_type_size

    def smooth_block_size(self) -> int:
        return self.gguf_block_size

    def _allocate_buffers(self, *, bias: bool) -> None:
        bytes_per_row = self._bytes_per_row()
        qweight = torch.zeros((self.out_features, bytes_per_row), dtype=torch.uint8)
        if "qweight" in self._buffers:
            self.qweight = qweight
        else:
            self.register_buffer("qweight", qweight)

        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=torch.float16)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def clear_weight_cache(self) -> None:
        return None

    def post_init(self):
        self.clear_weight_cache()
        super().post_init()

    def train(self, mode: bool = True):
        self.clear_weight_cache()
        return super().train(mode=mode)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}"
        )

    def _weight_to_matrix(self, linear: nn.Module) -> torch.Tensor:
        weight = linear.weight.detach()
        if isinstance(linear, _ConvNd):
            weight = weight.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            weight = weight.T
        return weight

    def _pack_weight_tensor(
        self,
        linear: nn.Module,
        *,
        smooth: SmoothMethod | None = None,
    ) -> torch.Tensor:
        weight = self._weight_to_matrix(linear).to(device="cpu", dtype=torch.float32)
        weight = _apply_optional_smoother(
            weight,
            smooth=smooth,
            group_size=self.smooth_block_size(),
        )
        if weight.shape[1] != self.padded_in_features:
            weight = torch.nn.functional.pad(weight, (0, self.padded_in_features - weight.shape[1]))

        quantized = _gguf_quantize(weight.contiguous().numpy(), self.gguf_tensor_qtype)
        return torch.from_numpy(np.ascontiguousarray(quantized)).to(torch.uint8)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        block_in: int = 8192,
        workers: int = 1,
    ):
        del block_in, workers
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_gpu(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        block_in: int = 8192,
        device: torch.device | None = None,
    ):
        del block_in, device
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    @torch.inference_mode()
    def pack_original(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        *,
        smooth: SmoothMethod | None = None,
    ):
        del scales, zeros, g_idx

        qweight = self._pack_weight_tensor(linear, smooth=smooth)
        expected_shape = (self.out_features, self._bytes_per_row())
        if tuple(qweight.shape) != expected_shape:
            raise RuntimeError(
                f"{self.__class__.__name__} produced an invalid GGUF packed shape {tuple(qweight.shape)}; "
                f"expected {expected_shape} for padded_in_features={self.padded_in_features}."
            )
        if "qweight" in self._buffers:
            self.qweight = qweight
        else:
            self.register_buffer("qweight", qweight)

        if linear.bias is not None:
            bias = linear.bias.detach().to(device="cpu", dtype=torch.float16)
            if "bias" in self._buffers:
                self.bias = bias
            else:
                self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.clear_autotune()
        self.clear_weight_cache()

    def _resolve_dequant_target(
        self,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        target_device = self.qweight.device if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        if target_dtype not in self.SUPPORTS_DTYPES:
            supported = ", ".join(str(dt).removeprefix("torch.") for dt in self.SUPPORTS_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports GGUF dequantization dtypes {{{supported}}}, got `{target_dtype}`."
            )
        return target_device, target_dtype

    def _reshape_blocks(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        target_device = self.qweight.device if device is None else torch.device(device)
        qweight = self.qweight if self.qweight.device == target_device else self.qweight.to(device=target_device)
        rows = qweight.shape[0]
        num_blocks = qweight.shape[1] // self.gguf_type_size
        blocks = qweight.contiguous().view(rows, num_blocks, self.gguf_type_size)
        return blocks, rows, num_blocks

    @staticmethod
    def _u8_shift(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.uint8, device=device)

    def _dequantize_q4_0(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
        blocks, rows, _ = self._reshape_blocks(device=target_device)

        d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if d.dtype != target_dtype:
            d = d.to(target_dtype)

        qs = blocks[..., 2:]
        low = torch.bitwise_and(qs, 0x0F)
        high = torch.bitwise_right_shift(qs, 4)
        values = torch.cat((low, high), dim=-1).to(torch.int16) - 8

        weight = d.unsqueeze(-1) * values.to(target_dtype)
        return weight.reshape(rows, self.padded_in_features)

    def _dequantize_q8_0(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
        blocks, rows, _ = self._reshape_blocks(device=target_device)

        d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if d.dtype != target_dtype:
            d = d.to(target_dtype)

        x = blocks[..., 2:].contiguous().view(torch.int8).to(target_dtype)

        weight = d.unsqueeze(-1) * x
        return weight.reshape(rows, self.padded_in_features)

    def _dequantize_sign_only(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
        weight = _dequantize_sign_only_torch(
            self.qweight,
            block_size=self.gguf_block_size,
            type_size=self.gguf_type_size,
            device=target_device,
            dtype=target_dtype,
        )
        return weight.reshape(self.out_features, self.padded_in_features)

    def _dequantize_numpy(
        self,
        fn,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
        qweight = self.qweight.detach().cpu().numpy()
        weight = fn(qweight)
        tensor = torch.from_numpy(np.ascontiguousarray(weight))
        if tensor.device != target_device or tensor.dtype != target_dtype:
            tensor = tensor.to(device=target_device, dtype=target_dtype)
        return tensor

    def _dequantize_q4_k_blocks(self, blocks: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        rows, num_blocks = blocks.shape[0], blocks.shape[1]

        d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).to(target_dtype)
        dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1).to(target_dtype)
        scales = blocks[..., 4:16]
        qs = blocks[..., 16:]

        sc, mins = _unpack_q4_k_scale_min_torch(scales)
        d = d.unsqueeze(-1) * sc.to(target_dtype)
        dm = dmin.unsqueeze(-1) * mins.to(target_dtype)

        q = qs.reshape(rows, num_blocks, 4, 1, 32)
        q = torch.bitwise_right_shift(
            q,
            self._u8_shift((0, 4), device=blocks.device).view(1, 1, 1, 2, 1),
        )
        q = torch.bitwise_and(q, 0x0F).reshape(rows, num_blocks, 8, 32)

        return (d.unsqueeze(-1) * q.to(target_dtype) - dm.unsqueeze(-1)).reshape(rows, num_blocks * 256)

    def _dequantize_q5_k_blocks(self, blocks: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        rows, num_blocks = blocks.shape[0], blocks.shape[1]

        d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).to(target_dtype)
        dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1).to(target_dtype)
        scales = blocks[..., 4:16]
        qh = blocks[..., 16:48]
        qs = blocks[..., 48:]

        sc, mins = _unpack_q4_k_scale_min_torch(scales)
        d = d.unsqueeze(-1) * sc.to(target_dtype)
        dm = dmin.unsqueeze(-1) * mins.to(target_dtype)

        ql = qs.reshape(rows, num_blocks, 4, 1, 32)
        ql = torch.bitwise_right_shift(
            ql,
            self._u8_shift((0, 4), device=blocks.device).view(1, 1, 1, 2, 1),
        )
        ql = torch.bitwise_and(ql, 0x0F).reshape(rows, num_blocks, 8, 32)

        qh = torch.bitwise_right_shift(
            qh.unsqueeze(-2),
            self._u8_shift(tuple(range(8)), device=blocks.device).view(1, 1, 8, 1),
        )
        qh = torch.bitwise_and(qh, 0x01).reshape(rows, num_blocks, 8, 32)
        q = torch.bitwise_or(ql, torch.bitwise_left_shift(qh, 4))

        return (d.unsqueeze(-1) * q.to(target_dtype) - dm.unsqueeze(-1)).reshape(rows, num_blocks * 256)

    def _dequantize_q6_k_blocks(self, blocks: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        rows, num_blocks = blocks.shape[0], blocks.shape[1]

        ql = blocks[..., :128]
        qh = blocks[..., 128:192]
        scales = blocks[..., 192:208].contiguous().view(torch.int8).to(target_dtype)
        d = blocks[..., 208:210].contiguous().view(torch.float16).squeeze(-1).to(target_dtype)
        d = (d.unsqueeze(-1) * scales).reshape(rows, num_blocks, 16, 1)

        ql = ql.reshape(rows, num_blocks, 2, 1, 64)
        ql = torch.bitwise_right_shift(
            ql,
            self._u8_shift((0, 4), device=blocks.device).view(1, 1, 1, 2, 1),
        )
        ql = torch.bitwise_and(ql, 0x0F).reshape(rows, num_blocks, 8, 32)

        qh = qh.reshape(rows, num_blocks, 2, 1, 32)
        qh = torch.bitwise_right_shift(
            qh,
            self._u8_shift((0, 2, 4, 6), device=blocks.device).view(1, 1, 1, 4, 1),
        )
        qh = torch.bitwise_and(qh, 0x03).reshape(rows, num_blocks, 8, 32)

        q = torch.bitwise_or(ql, torch.bitwise_left_shift(qh, 4)).to(torch.int16) - 32
        q = q.reshape(rows, num_blocks, 16, 16).to(target_dtype)

        return (d * q).reshape(rows, num_blocks * 256)

    def dequantize_weight(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.gguf_tensor_qtype == "Q4_0":
            weight = self._dequantize_q4_0(device=device, dtype=dtype)
        elif self.gguf_tensor_qtype == "Q8_0":
            weight = self._dequantize_q8_0(device=device, dtype=dtype)
        elif self.gguf_tensor_qtype in _GGUF_SIGN_ONLY_TYPE_INFO:
            weight = self._dequantize_sign_only(device=device, dtype=dtype)
        elif self.gguf_tensor_qtype == "Q4_K":
            target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
            blocks, _, _ = self._reshape_blocks(device=target_device)
            weight = self._dequantize_q4_k_blocks(blocks, target_dtype)
        elif self.gguf_tensor_qtype == "Q5_K":
            target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
            blocks, _, _ = self._reshape_blocks(device=target_device)
            weight = self._dequantize_q5_k_blocks(blocks, target_dtype)
        elif self.gguf_tensor_qtype == "Q6_K":
            target_device, target_dtype = self._resolve_dequant_target(device=device, dtype=dtype)
            blocks, _, _ = self._reshape_blocks(device=target_device)
            weight = self._dequantize_q6_k_blocks(blocks, target_dtype)
        else:  # pragma: no cover - guarded by class SUPPORTS_BITS
            raise NotImplementedError(f"Unsupported GGUF qtype: {self.gguf_tensor_qtype}")

        return weight[:, : self.in_features].transpose(0, 1).contiguous()

    def _is_fused_k_forward_candidate(self, x_flat: torch.Tensor) -> bool:
        if x_flat.device.type == "cuda":
            max_rows = self.gguf_fused_cuda_max_rows
            min_matrix_elements = self.gguf_fused_cuda_min_matrix_elements
        elif x_flat.device.type == "cpu":
            max_rows = self.gguf_fused_cpu_max_rows
            min_matrix_elements = self.gguf_fused_cpu_min_matrix_elements
        else:
            return False

        return (
            self.gguf_tensor_qtype in (_GGUF_K_QTYPES | set(_GGUF_SIGN_ONLY_TYPE_INFO))
            and self.adapter is None
            and not self.training
            and max_rows > 0
            and (self.in_features * self.out_features) >= min_matrix_elements
            and x_flat.shape[0] <= max_rows
        )

    @staticmethod
    def _sync_benchmark_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def _benchmark_forward_runner(self, fn, *, device: torch.device) -> float:
        with torch.inference_mode():
            for _ in range(self.gguf_fused_autotune_warmup):
                fn()
            self._sync_benchmark_device(device)

            start = time.perf_counter()
            for _ in range(self.gguf_fused_autotune_iters):
                fn()
            self._sync_benchmark_device(device)

        return (time.perf_counter() - start) / self.gguf_fused_autotune_iters

    def _benchmark_dense_forward(self, x_flat: torch.Tensor) -> float:
        return self._benchmark_forward_runner(
            lambda: self._forward_dequant_matmul(x_flat),
            device=x_flat.device,
        )

    def _benchmark_fused_forward(self, x_flat: torch.Tensor) -> float:
        return self._benchmark_forward_runner(
            lambda: self._forward_fused_k(x_flat),
            device=x_flat.device,
        )

    def _autotune(self, x_flat: torch.Tensor) -> bool:
        try:
            fused_time = self._benchmark_fused_forward(x_flat)
            dense_time = self._benchmark_dense_forward(x_flat)
            return fused_time <= dense_time * (1.0 - self.gguf_fused_autotune_margin)
        except Exception:
            return False

    def _should_use_fused_k_forward(self, x_flat: torch.Tensor) -> bool:
        if not self._is_fused_k_forward_candidate(x_flat):
            return False

        if not self.autotune_enabled:
            return True

        return bool(self.maybe_autotune(x_flat))

    def _forward_dequant_matmul(self, x_flat: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(device=x_flat.device, dtype=x_flat.dtype)
        return torch.matmul(x_flat, weight)

    def _forward_fused_k(self, x_flat: torch.Tensor) -> torch.Tensor:
        target_dtype = x_flat.dtype
        blocks, _, num_blocks = self._reshape_blocks(device=x_flat.device)

        if x_flat.shape[-1] != self.padded_in_features:
            x_work = F.pad(x_flat, (0, self.padded_in_features - x_flat.shape[-1]))
        else:
            x_work = x_flat

        output = torch.zeros((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=target_dtype)

        for start in range(0, num_blocks, self.gguf_fused_chunk_blocks):
            end = min(start + self.gguf_fused_chunk_blocks, num_blocks)
            block_chunk = blocks[:, start:end, :]

            if self.gguf_tensor_qtype == "Q4_K":
                weight_chunk = self._dequantize_q4_k_blocks(block_chunk, target_dtype)
            elif self.gguf_tensor_qtype == "Q5_K":
                weight_chunk = self._dequantize_q5_k_blocks(block_chunk, target_dtype)
            elif self.gguf_tensor_qtype == "Q6_K":
                weight_chunk = self._dequantize_q6_k_blocks(block_chunk, target_dtype)
            elif self.gguf_tensor_qtype in _GGUF_SIGN_ONLY_TYPE_INFO:
                weight_chunk = _dequantize_sign_only_torch(
                    block_chunk.reshape(block_chunk.shape[0], -1),
                    block_size=self.gguf_block_size,
                    type_size=self.gguf_type_size,
                    device=x_flat.device,
                    dtype=target_dtype,
                )
            else:  # pragma: no cover - guarded by _should_use_fused_k_forward
                raise NotImplementedError(f"Unsupported GGUF fused qtype: {self.gguf_tensor_qtype}")

            x_chunk = x_work[:, start * self.gguf_block_size : end * self.gguf_block_size]
            output = output + torch.matmul(x_chunk, weight_chunk.transpose(0, 1))

        return output

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if self._should_use_fused_k_forward(x_flat):
            output = self._forward_fused_k(x_flat)
        else:
            output = self._forward_dequant_matmul(x_flat)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["GGUFTorchLinear"]
