# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Clean-room EXL3 torch reference kernel derived from the public EXL3 tensor
# format and documented runtime layout.

from __future__ import annotations

import math
import struct
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .exllamav3 import _EXL3_BUFFER_NAMES, _torch_dtype


_EXL3_3INST_MULT = 89226354
_EXL3_3INST_ADD = 64248484
_EXL3_MCG_MULT = 0xCBAC1FED
_EXL3_MUL1_MULT = 0x83DCD12D
_EXL3_MUL1_ACC = 0x6400
_EXL3_LOP3_MASK = 0x8FFF8FFF
_EXL3_LOP3_BIAS = 0x3B603B60


def _tensor_core_perm_values() -> list[int]:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return perm


def _inverse_perm_values(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for index, value in enumerate(perm):
        inv[value] = index
    return inv


def _half_scalar_from_bits(bits: int) -> float:
    # Convert a uint16 bit pattern to its IEEE-754 binary16 (float16) value
    # without allocating a torch tensor. The previous implementation used
    # ``torch.tensor([bits], dtype=torch.uint16).view(torch.float16).item()``
    # at module import, which fails under a meta-device preload context with
    # ``RuntimeError: Tensor.item() cannot be called on meta tensors``. The
    # transformers AWQ pathway (``replace_with_awq_linear``) imports
    # ``gptqmodel.quantization`` inside ``with torch.device('meta'):``, which
    # triggered the failure for any GPTQ load on transformers >= 5.6.
    #
    # ``struct`` format ``<e`` is IEEE-754 binary16 little-endian and matches
    # ``torch.float16``'s byte layout on all supported platforms (x86_64 /
    # aarch64). The conversion is bit-equivalent to the tensor-based path.
    packed = struct.pack("<H", int(bits) & 0xFFFF)
    return float(struct.unpack("<e", packed)[0])


_EXL3_MUL1_INV = _half_scalar_from_bits(0x1EEE)
_EXL3_MUL1_BIAS = _half_scalar_from_bits(0xC931)


@lru_cache(maxsize=None)
def _tensor_core_perm(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(_tensor_core_perm_values(), dtype=torch.long, device=device)


@lru_cache(maxsize=None)
def _tensor_core_perm_i(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(_inverse_perm_values(_tensor_core_perm_values()), dtype=torch.long, device=device)


@lru_cache(maxsize=None)
def _hadamard_128(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    had = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while had.shape[0] < 128:
        had = torch.cat(
            (
                torch.cat((had, had), dim=1),
                torch.cat((had, -had), dim=1),
            ),
            dim=0,
        )
    had *= 1.0 / math.sqrt(128.0)
    return had.contiguous()


@lru_cache(maxsize=None)
def _codebook_lut(
    codebook: str,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int64, device=device)

    if codebook == "3inst":
        raw = (values * _EXL3_3INST_MULT + _EXL3_3INST_ADD) & 0xFFFFFFFF
        raw = _EXL3_LOP3_BIAS ^ (raw & _EXL3_LOP3_MASK)
        halves = torch.stack(
            (
                (raw & 0xFFFF).to(torch.uint16),
                ((raw >> 16) & 0xFFFF).to(torch.uint16),
            ),
            dim=-1,
        ).contiguous()
        floats = halves.view(torch.float16).to(torch.float32)
        return (floats[..., 0] + floats[..., 1]).contiguous()

    if codebook == "mcg":
        raw = (values * _EXL3_MCG_MULT) & 0xFFFFFFFF
        raw = _EXL3_LOP3_BIAS ^ (raw & _EXL3_LOP3_MASK)
        halves = torch.stack(
            (
                (raw & 0xFFFF).to(torch.uint16),
                ((raw >> 16) & 0xFFFF).to(torch.uint16),
            ),
            dim=-1,
        ).contiguous()
        floats = halves.view(torch.float16).to(torch.float32)
        return (floats[..., 0] + floats[..., 1]).contiguous()

    if codebook == "mul1":
        raw = (values * _EXL3_MUL1_MULT) & 0xFFFFFFFF
        byte_sum = (
            (raw & 0xFF)
            + ((raw >> 8) & 0xFF)
            + ((raw >> 16) & 0xFF)
            + ((raw >> 24) & 0xFF)
        )
        accum = (byte_sum + _EXL3_MUL1_ACC).to(torch.uint16).contiguous()
        floats = accum.view(torch.float16).to(torch.float32)
        return (floats * _EXL3_MUL1_INV + _EXL3_MUL1_BIAS).contiguous()

    raise ValueError(f"Unsupported EXL3 codebook: {codebook}")


def _apply_hadamard_left(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] % 128 != 0:
        raise ValueError(f"EXL3 expects in_features to be divisible by 128, got {x.shape[0]}.")
    had = _hadamard_128(x.device.type, x.device.index)
    return (had @ x.view(-1, 128, x.shape[1])).view_as(x)


def _apply_hadamard_right(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] % 128 != 0:
        raise ValueError(f"EXL3 expects out_features to be divisible by 128, got {x.shape[1]}.")
    had = _hadamard_128(x.device.type, x.device.index)
    return (x.view(x.shape[0], -1, 128) @ had).view_as(x)


class ExllamaV3TorchLinear(nn.Module):
    QUANT_TYPE = "exl3"
    SUPPORTS_SHARDS = True

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensor_storage: Optional[Dict[str, Any]] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        out_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.out_dtype = out_dtype
        self.tensor_storage = tensor_storage or {}

        self.weight = torch.zeros((1,), dtype=torch.float16, device="meta")
        self._cache_signature: Optional[tuple[Any, ...]] = None
        self._inner_weight_fp32: Optional[torch.Tensor] = None
        self._weight_fp32: Optional[torch.Tensor] = None

        if tensors is not None:
            for buffer_name in _EXL3_BUFFER_NAMES:
                tensor = tensors.get(buffer_name)
                if tensor is None:
                    setattr(self, buffer_name, None)
                else:
                    self.register_buffer(buffer_name, tensor)
            return

        stored_tensors = (self.tensor_storage or {}).get("stored_tensors", {})
        for buffer_name in _EXL3_BUFFER_NAMES:
            metadata = stored_tensors.get(f"{name}.{buffer_name}")
            if metadata is None:
                setattr(self, buffer_name, None)
                continue

            shape = tuple(metadata["shape"])
            dtype = _torch_dtype(metadata["torch_dtype"])
            self.register_buffer(buffer_name, torch.empty(shape, dtype=dtype, device="meta"))

    @classmethod
    def from_tensors(
        cls,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensors: Dict[str, torch.Tensor],
    ) -> "ExllamaV3TorchLinear":
        return cls(
            in_features=in_features,
            out_features=out_features,
            name=name,
            tensors=tensors,
        )

    def _current_signature(self) -> tuple[Any, ...]:
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type == "meta":
            return ("meta",)

        signature: list[Any] = [str(trellis.device)]
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                signature.append(None)
                continue
            signature.append((tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype)))
        return tuple(signature)

    def _drop_cache(self) -> None:
        self._cache_signature = None
        self._inner_weight_fp32 = None
        self._weight_fp32 = None

    def _apply(self, fn):
        self._drop_cache()
        return super()._apply(fn)

    def post_init(self) -> None:
        self._drop_cache()

    def _codebook_name(self) -> str:
        if getattr(self, "mcg", None) is not None:
            return "mcg"
        if getattr(self, "mul1", None) is not None:
            return "mul1"
        return "3inst"

    def _bits_per_weight(self) -> int:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        return int(trellis.shape[-1] // 16)

    def _runtime_weight_dtype(self) -> torch.dtype:
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type == "cpu":
            return torch.float32
        return torch.float16

    def _unpack_indices(self) -> torch.Tensor:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        if trellis.device.type == "meta":
            raise RuntimeError(f"EXL3 module `{self.name}` has not been materialized from checkpoint tensors yet.")

        bits = self._bits_per_weight()
        mask = (1 << bits) - 1
        words = (trellis.to(torch.int32) & 0xFFFF).contiguous()
        words = words.view(*words.shape[:-1], -1, 2).flip(-1).reshape(*words.shape)
        words = words.view(*words.shape[:-1], 16, bits)

        symbols = torch.empty(
            (*words.shape[:-2], 256),
            dtype=torch.long,
            device=words.device,
        )
        for pos in range(16):
            bit_offset = pos * bits
            word_idx = bit_offset // 16
            bit_in_word = bit_offset % 16
            if bit_in_word + bits <= 16:
                shift = 16 - bit_in_word - bits
                value = (words[..., word_idx] >> shift) & mask
            else:
                bits_first = 16 - bit_in_word
                bits_second = bits - bits_first
                high = (words[..., word_idx] & ((1 << bits_first) - 1)) << bits_second
                low = words[..., word_idx + 1] >> (16 - bits_second)
                value = (high | low) & mask
            symbols[..., pos::16] = value.to(torch.long)

        warmup = (16 + bits - 1) // bits - 1
        state = torch.zeros_like(symbols[..., 0], dtype=torch.long)
        for idx in range(256 - warmup, 256):
            state = ((state << bits) | symbols[..., idx]) & 0xFFFF

        encoded = torch.empty_like(symbols)
        for idx in range(256):
            state = ((state << bits) | symbols[..., idx]) & 0xFFFF
            encoded[..., idx] = state

        return encoded

    def _ensure_inner_weight_fp32(self) -> torch.Tensor:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        if trellis.device.type == "meta":
            raise RuntimeError(f"EXL3 module `{self.name}` has not been materialized from checkpoint tensors yet.")

        signature = self._current_signature()
        if self._inner_weight_fp32 is not None and self._cache_signature == signature:
            return self._inner_weight_fp32

        encoded = self._unpack_indices()
        lut = _codebook_lut(self._codebook_name(), trellis.device.type, trellis.device.index)
        decoded = lut[encoded]

        perm_i = _tensor_core_perm_i(trellis.device.type, trellis.device.index)
        decoded = decoded[..., perm_i]
        tiles_k, tiles_n = decoded.shape[:2]
        inner = decoded.view(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
            tiles_k * 16, tiles_n * 16
        )

        self._cache_signature = signature
        self._inner_weight_fp32 = inner.contiguous().to(torch.float32)
        self._weight_fp32 = None
        return self._inner_weight_fp32

    def get_inner_weight_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        inner = self._ensure_inner_weight_fp32()
        target_dtype = dtype or self._runtime_weight_dtype()
        if inner.dtype == target_dtype:
            return inner
        return inner.to(dtype=target_dtype)

    def _ensure_weight_fp32(self) -> torch.Tensor:
        signature = self._current_signature()
        if self._weight_fp32 is not None and self._cache_signature == signature:
            return self._weight_fp32

        inner = self._ensure_inner_weight_fp32().clone()
        inner = _apply_hadamard_left(inner)
        inner *= getattr(self, "suh").to(dtype=torch.float32).unsqueeze(1)
        inner = _apply_hadamard_right(inner)
        inner *= getattr(self, "svh").to(dtype=torch.float32).unsqueeze(0)

        self._weight_fp32 = inner.contiguous()
        return self._weight_fp32

    def get_weight_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        weight = self._ensure_weight_fp32()
        target_dtype = dtype or self._runtime_weight_dtype()
        if weight.dtype == target_dtype:
            return weight
        return weight.to(dtype=target_dtype)

    def get_bias_tensor(self) -> torch.Tensor | None:
        return getattr(self, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        compute_dtype = torch.float32 if x.device.type == "cpu" else torch.float16
        x_2d = x.view(-1, self.in_features).to(compute_dtype)
        weight = self.get_weight_tensor(dtype=compute_dtype)
        y = x_2d @ weight
        bias = getattr(self, "bias", None)
        if bias is not None:
            y = y + bias.to(dtype=compute_dtype)
        y = y.view(*x.shape[:-1], self.out_features)
        return y.to(input_dtype)

    def _multiplier_value(self, name: str) -> Optional[int]:
        tensor = getattr(self, name, None)
        if tensor is None:
            return None
        return int(tensor.view(torch.uint32).item())

    def tensor_storage_entry(self) -> Dict[str, Any]:
        stored_tensors: Dict[str, Dict[str, Any]] = {}
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                continue
            stored_tensors[f"{self.name}.{buffer_name}"] = {
                "shape": list(tensor.shape),
                "torch_dtype": str(tensor.dtype).split(".")[-1],
            }

        entry: Dict[str, Any] = {
            "stored_tensors": stored_tensors,
            "quant_format": "exl3",
            "bits_per_weight": self._bits_per_weight(),
        }

        mcg_multiplier = self._multiplier_value("mcg")
        if mcg_multiplier is not None:
            entry["mcg_multiplier"] = mcg_multiplier

        mul1_multiplier = self._multiplier_value("mul1")
        if mul1_multiplier is not None:
            entry["mul1_multiplier"] = mul1_multiplier

        return entry


__all__ = ["ExllamaV3TorchLinear"]
