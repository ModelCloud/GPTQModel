# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.modules.conv import _ConvNd

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FailSafe, FailSafeStrategy, FORMAT, GGUFBits, METHOD, SmoothMethod, _normalize_quant_bits
from ...quantization.failsafe_smooth import smooth_block
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import BaseQuantLinear


try:
    import gguf as gguf_lib

    _GGUF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gguf_lib = None
    _GGUF_AVAILABLE = False


log = setup_logger()

_GGUF_TYPE_INFO = {
    "Q4_0": {"bits": 4, "block_size": 32, "type_size": 18},
    "Q8_0": {"bits": 8, "block_size": 32, "type_size": 34},
    "Q4_K": {"bits": 4, "block_size": 256, "type_size": 144},
    "Q5_K": {"bits": 5, "block_size": 256, "type_size": 176},
    "Q6_K": {"bits": 6, "block_size": 256, "type_size": 210},
}
_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE = {
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


def _normalize_gguf_bits(bits) -> tuple[GGUFBits, str]:
    bits_spec = _normalize_quant_bits(bits, format_value=FORMAT.GGUF)
    tensor_qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE.get(bits_spec.name)
    if tensor_qtype is None:
        supported = ", ".join(sorted(_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE))
        raise ValueError(f"Unsupported GGUF bits `{bits}`. Supported values: {supported}.")

    qtype_info = _GGUF_TYPE_INFO[tensor_qtype]
    if qtype_info["bits"] != bits_spec.width:
        raise ValueError(
            f"GGUF bits `{bits_spec.name}` require {qtype_info['bits']}-bit RTN export, but got bits={bits_spec.width}."
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

    failsafe = FailSafe(
        strategy=FailSafeStrategy.RTN,
        threshold=True,
        smooth=smooth,
    )
    smoothed = weight.clone()

    for start in range(0, weight.shape[1], effective_group_size):
        end = min(start + effective_group_size, weight.shape[1])
        block, scale_factor = smooth_block(
            smoothed[:, start:end],
            failsafe,
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
    if tensor_qtype == "Q4_0":
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
    if _GGUF_AVAILABLE:
        qtype = getattr(gguf_lib.GGMLQuantizationType, tensor_qtype)
        try:
            return gguf_lib.quantize(weight, qtype)
        except NotImplementedError:
            pass
    return _fallback_gguf_quantize(weight, tensor_qtype)


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


class GGUFTorchQuantLinear(BaseQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 15}
    SUPPORTS_BITS = [4, 5, 6, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
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
        self._cached_weights: Dict[Tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self.gguf_fused_cuda_max_rows = self.GGUF_FUSED_CUDA_MAX_ROWS
        self.gguf_fused_cuda_min_matrix_elements = self.GGUF_FUSED_CUDA_MIN_MATRIX_ELEMENTS
        self.gguf_fused_cpu_max_rows = self.GGUF_FUSED_CPU_MAX_ROWS
        self.gguf_fused_cpu_min_matrix_elements = self.GGUF_FUSED_CPU_MIN_MATRIX_ELEMENTS
        self.gguf_fused_chunk_blocks = self.GGUF_FUSED_CHUNK_BLOCKS

        super().__init__(
            bits=int(bits_spec),
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        self.bits = bits_spec

        if register_buffers:
            self._allocate_buffers(bias=bias)

    def _allocate_buffers(self, *, bias: bool) -> None:
        bytes_per_row = (self.padded_in_features // self.gguf_block_size) * self.gguf_type_size
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
        self._cached_weights.clear()

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

    def _cache_key(self, device: torch.device, dtype: torch.dtype) -> Tuple[str, int | None, torch.dtype]:
        return device.type, device.index, dtype

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
            group_size=self.group_size,
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

    def _should_use_fused_k_forward(self, x_flat: torch.Tensor) -> bool:
        if x_flat.device.type == "cuda":
            max_rows = self.gguf_fused_cuda_max_rows
            min_matrix_elements = self.gguf_fused_cuda_min_matrix_elements
        elif x_flat.device.type == "cpu":
            max_rows = self.gguf_fused_cpu_max_rows
            min_matrix_elements = self.gguf_fused_cpu_min_matrix_elements
        else:
            return False

        return (
            self.gguf_tensor_qtype in _GGUF_K_QTYPES
            and self.adapter is None
            and not self.training
            and max_rows > 0
            and (self.in_features * self.out_features) >= min_matrix_elements
            and x_flat.shape[0] <= max_rows
        )

    def _forward_dequant_matmul(self, x_flat: torch.Tensor, *, cache: bool = True) -> torch.Tensor:
        weight = self._get_cached_weight(x_flat) if cache else None
        if weight is None:
            weight = self.dequantize_weight(device=x_flat.device, dtype=x_flat.dtype)
            if cache:
                self._set_cached_weight(x_flat, weight)
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
            else:  # pragma: no cover - guarded by _should_use_fused_k_forward
                raise NotImplementedError(f"Unsupported GGUF fused qtype: {self.gguf_tensor_qtype}")

            x_chunk = x_work[:, start * self.gguf_block_size : end * self.gguf_block_size]
            output = output + torch.matmul(x_chunk, weight_chunk.transpose(0, 1))

        return output

    def _get_cached_weight(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.training:
            return None
        return self._cached_weights.get(self._cache_key(x.device, x.dtype))

    def _set_cached_weight(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        if self.training:
            return
        self._cached_weights[self._cache_key(x.device, x.dtype)] = weight

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if self._should_use_fused_k_forward(x_flat) and self._get_cached_weight(x_flat) is None:
            output = self._forward_fused_k(x_flat)
        else:
            output = self._forward_dequant_matmul(x_flat, cache=True)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["GGUFTorchQuantLinear"]
