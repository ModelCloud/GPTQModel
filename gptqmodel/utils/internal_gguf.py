# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from enum import IntEnum
from typing import Any, Literal, NamedTuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pcre
import torch

from ..nn_modules.qlinear.gguf import (
    _dequantize_gguf_tensor_numpy,
    _dequantize_sign_only_torch,
    _quantize_gguf_tensor_numpy,
)


__version__ = "0.10.0"
_INTERNAL_GGUF_DEQUANT_DEVICE_ENV = "GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE"
_INTERNAL_GGUF_DEQUANT_MAX_BYTES_ENV = "GPTQMODEL_INTERNAL_GGUF_DEQUANT_MAX_BYTES"
_INTERNAL_GGUF_DEQUANT_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_INTERNAL_GGUF_QUANTIZED_LOADER_ENV = "GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER"

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
QK_K = 256
READER_SUPPORTED_VERSIONS = (2, GGUF_VERSION)


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    Q1_0 = 40
    Q1_0_g128 = 41


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.Q4_0: (32, 2 + 16),
    GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0: (32, 2 + 32),
    GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K: (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K: (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K: (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS: (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S: (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL: (32, 2 + 16),
    GGMLQuantizationType.IQ3_S: (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S: (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8: (1, 1),
    GGMLQuantizationType.I16: (1, 2),
    GGMLQuantizationType.I32: (1, 4),
    GGMLQuantizationType.I64: (1, 8),
    GGMLQuantizationType.F64: (1, 8),
    GGMLQuantizationType.IQ1_M: (256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    GGMLQuantizationType.BF16: (1, 2),
    GGMLQuantizationType.TQ1_0: (256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0: (256, 2 + 64),
    GGMLQuantizationType.MXFP4: (32, 1 + 16),
    GGMLQuantizationType.Q1_0: (32, 2 + 4),
    GGMLQuantizationType.Q1_0_g128: (128, 2 + 16),
}
_TORCH_SIGN_ONLY_QTYPES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.Q1_0: GGML_QUANT_SIZES[GGMLQuantizationType.Q1_0],
    GGMLQuantizationType.Q1_0_g128: GGML_QUANT_SIZES[GGMLQuantizationType.Q1_0_g128],
}

MODEL_ARCH_QWEN3 = "qwen3"
MODEL_ARCH_NAMES = {
    MODEL_ARCH_QWEN3: "qwen3",
}

_GGUF_SCALAR_TO_NP: dict[GGUFValueType, type[np.generic]] = {
    GGUFValueType.UINT8: np.uint8,
    GGUFValueType.INT8: np.int8,
    GGUFValueType.UINT16: np.uint16,
    GGUFValueType.INT16: np.int16,
    GGUFValueType.UINT32: np.uint32,
    GGUFValueType.INT32: np.int32,
    GGUFValueType.FLOAT32: np.float32,
    GGUFValueType.UINT64: np.uint64,
    GGUFValueType.INT64: np.int64,
    GGUFValueType.FLOAT64: np.float64,
    GGUFValueType.BOOL: np.bool_,
}

_QWEN3_DIRECT_NAME_MAP = {
    "model.embed_tokens": "token_embd",
    "model.norm": "output_norm",
}
_QWEN3_BLOCK_PATTERNS: tuple[tuple[pcre.Pattern, str], ...] = (
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj$"), "blk.{bid}.attn_q"),
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj$"), "blk.{bid}.attn_k"),
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj$"), "blk.{bid}.attn_v"),
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj$"), "blk.{bid}.attn_output"),
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm$"), "blk.{bid}.attn_q_norm"),
    (pcre.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm$"), "blk.{bid}.attn_k_norm"),
    (pcre.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj$"), "blk.{bid}.ffn_gate"),
    (pcre.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj$"), "blk.{bid}.ffn_up"),
    (pcre.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$"), "blk.{bid}.ffn_down"),
    (pcre.compile(r"^model\.layers\.(\d+)\.input_layernorm$"), "blk.{bid}.attn_norm"),
    (pcre.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm$"), "blk.{bid}.ffn_norm"),
)
_QWEN3_LINEAR_TENSOR_RE = pcre.compile(
    r"^blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"
)
_GGUF_BITS_ALIAS_BY_QTYPE: dict[GGMLQuantizationType, str] = {
    GGMLQuantizationType.Q1_0: "q1_0",
    GGMLQuantizationType.Q1_0_g128: "q1_0_g128",
    GGMLQuantizationType.Q4_0: "q4_0",
    GGMLQuantizationType.Q8_0: "q8_0",
    GGMLQuantizationType.Q4_K: "q4_k",
    GGMLQuantizationType.Q5_K: "q5_k",
    GGMLQuantizationType.Q6_K: "q6_k",
}


class GGUFQuantizedCheckpointSpec(NamedTuple):
    model_type: str
    bits_alias: str
    tensor_qtype: GGMLQuantizationType
    lm_head_quantized: bool


def quant_shape_to_byte_shape(shape: tuple[int, ...] | list[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(int(quant_type))]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})"
        )
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    return _quantize_gguf_tensor_numpy(np.asarray(data), GGMLQuantizationType(int(qtype)))


def native_quantized_loader_enabled() -> bool:
    raw = os.getenv(_INTERNAL_GGUF_QUANTIZED_LOADER_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"", "0", "false", "off", "no"}


def _reader_field_value(reader: "GGUFReader", key: str):
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def inspect_quantized_checkpoint(
    source: "GGUFReader | os.PathLike[str] | str",
) -> GGUFQuantizedCheckpointSpec | None:
    if hasattr(source, "tensors") and hasattr(source, "get_field"):
        reader = source
    else:
        reader = GGUFReader(source)
    architecture = _reader_field_value(reader, "general.architecture")
    if architecture != MODEL_ARCH_QWEN3:
        return None

    linear_qtypes = {
        GGMLQuantizationType(int(tensor.tensor_type))
        for tensor in reader.tensors
        if _QWEN3_LINEAR_TENSOR_RE.fullmatch(tensor.name)
    }
    if len(linear_qtypes) != 1:
        return None

    tensor_qtype = next(iter(linear_qtypes))
    bits_alias = _GGUF_BITS_ALIAS_BY_QTYPE.get(tensor_qtype)
    if bits_alias is None:
        return None

    lm_head_quantized = any(
        tensor.name == "output.weight" and GGMLQuantizationType(int(tensor.tensor_type)) == tensor_qtype
        for tensor in reader.tensors
    )
    return GGUFQuantizedCheckpointSpec(
        model_type=MODEL_ARCH_QWEN3,
        bits_alias=bits_alias,
        tensor_qtype=tensor_qtype,
        lm_head_quantized=lm_head_quantized,
    )


def _resolve_torch_dequant_device() -> torch.device | None:
    raw = os.getenv(_INTERNAL_GGUF_DEQUANT_DEVICE_ENV)
    if raw is None or raw.strip() == "":
        return None

    try:
        device = torch.device(raw.strip())
    except Exception:
        return None

    if device.type == "cuda":
        if not torch.cuda.is_available():
            return None
        if device.index is not None and device.index >= torch.cuda.device_count():
            return None
        return device

    if device.type == "cpu":
        return device

    return None


def _resolve_torch_dequant_chunk_rows(
    *,
    packed_row_bytes: int,
    block_size: int,
    type_size: int,
) -> int:
    output_row_bytes = packed_row_bytes // type_size * block_size * np.dtype(np.float32).itemsize
    if output_row_bytes <= 0:
        return 1

    raw_limit = os.getenv(_INTERNAL_GGUF_DEQUANT_MAX_BYTES_ENV)
    try:
        byte_limit = int(raw_limit) if raw_limit is not None else _INTERNAL_GGUF_DEQUANT_DEFAULT_MAX_BYTES
    except ValueError:
        byte_limit = _INTERNAL_GGUF_DEQUANT_DEFAULT_MAX_BYTES

    return max(1, byte_limit // output_row_bytes)


def _dequantize_sign_only_torch_to_numpy(
    data: np.ndarray,
    *,
    block_size: int,
    type_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = np.asarray(data, dtype=np.uint8)
    if rows.shape[-1] % type_size != 0:
        raise ValueError(
            f"GGUF sign-only row byte width must be divisible by {type_size}, got "
            f"{rows.shape[-1]} for shape {rows.shape}."
        )

    packed_cols = rows.shape[-1]
    output_cols = packed_cols // type_size * block_size
    flat_rows = rows.reshape(-1, packed_cols)
    flat_output = np.empty((flat_rows.shape[0], output_cols), dtype=np.float32)
    chunk_rows = _resolve_torch_dequant_chunk_rows(
        packed_row_bytes=packed_cols,
        block_size=block_size,
        type_size=type_size,
    )

    for start in range(0, flat_rows.shape[0], chunk_rows):
        end = min(start + chunk_rows, flat_rows.shape[0])
        chunk = _dequantize_sign_only_torch(
            flat_rows[start:end],
            block_size=block_size,
            type_size=type_size,
            device=device,
            dtype=torch.float32,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        flat_output[start:end] = chunk.cpu().numpy()

    return flat_output.reshape(*rows.shape[:-1], output_cols)


def dequantize_to_torch(
    data: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    resolved_qtype = GGMLQuantizationType(int(qtype))
    target_device = torch.device("cpu") if device is None else torch.device(device)

    sign_only_info = _TORCH_SIGN_ONLY_QTYPES.get(resolved_qtype)
    if sign_only_info is not None:
        block_size, type_size = sign_only_info
        return _dequantize_sign_only_torch(
            np.asarray(data, dtype=np.uint8),
            block_size=block_size,
            type_size=type_size,
            device=target_device,
            dtype=dtype,
        ).contiguous()

    if resolved_qtype == GGMLQuantizationType.F32:
        tensor = torch.from_numpy(np.array(data, dtype=np.float32, copy=True, order="C"))
    elif resolved_qtype == GGMLQuantizationType.F16:
        tensor = torch.from_numpy(np.array(data, dtype=np.float16, copy=True, order="C")).to(torch.float32)
    elif resolved_qtype == GGMLQuantizationType.BF16:
        rows = np.asarray(data, dtype=np.uint16).astype(np.uint32)
        tensor = torch.from_numpy(np.left_shift(rows, np.uint32(16)).view(np.float32).copy())
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(_dequantize_gguf_tensor_numpy(np.asarray(data), resolved_qtype)))

    if tensor.device != target_device or tensor.dtype != dtype:
        tensor = tensor.to(device=target_device, dtype=dtype)
    return tensor.contiguous()


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    resolved_qtype = GGMLQuantizationType(int(qtype))
    device = _resolve_torch_dequant_device()
    sign_only_info = _TORCH_SIGN_ONLY_QTYPES.get(resolved_qtype)
    if device is not None and sign_only_info is not None:
        block_size, type_size = sign_only_info
        return _dequantize_sign_only_torch_to_numpy(
            np.asarray(data, dtype=np.uint8),
            block_size=block_size,
            type_size=type_size,
            device=device,
        )

    return _dequantize_gguf_tensor_numpy(np.asarray(data), resolved_qtype)


class _MinimalTensorNameMap:
    def __init__(self, arch: str, n_blocks: int):
        self.arch = arch
        self.n_blocks = n_blocks

    def get_name(self, hf_name: str):
        if self.arch != MODEL_ARCH_QWEN3:
            return None

        direct = _QWEN3_DIRECT_NAME_MAP.get(hf_name)
        if direct is not None:
            return direct

        for pattern, template in _QWEN3_BLOCK_PATTERNS:
            match = pattern.fullmatch(hf_name)
            if match is None:
                continue
            block_id = int(match.group(1))
            if block_id >= self.n_blocks:
                return None
            return template.format(bid=block_id)

        return None


def get_tensor_name_map(arch, n_blocks: int):
    return _MinimalTensorNameMap(str(arch), int(n_blocks))


class ReaderField(NamedTuple):
    offset: int
    name: str
    parts: list[npt.NDArray[Any]]
    data: list[int]
    types: list[GGUFValueType]

    def contents(self, index_or_slice: int | slice = slice(None)) -> Any:
        if not self.types:
            return None

        def _to_string(part: npt.NDArray[Any]) -> str:
            return part.tobytes().decode("utf-8")

        main_type = self.types[0]
        if main_type == GGUFValueType.ARRAY:
            sub_type = self.types[-1]
            indices = self.data[index_or_slice]
            if isinstance(index_or_slice, int):
                index_items = [indices]
            else:
                index_items = list(indices)

            if sub_type == GGUFValueType.STRING:
                values = [_to_string(self.parts[idx]) for idx in index_items]
            else:
                values = [self.parts[idx].tolist()[0] for idx in index_items]
            return values[0] if isinstance(index_or_slice, int) else values

        if main_type == GGUFValueType.STRING:
            return _to_string(self.parts[-1])
        return self.parts[-1].tolist()[0]


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: npt.NDArray[np.uint32]
    n_elements: int
    n_bytes: int
    data_offset: int
    data: npt.NDArray[Any]
    field: ReaderField


class GGUFReader:
    byte_order: Literal["I", "S"] = "I"
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    data_offset: int

    _DT = TypeVar("_DT", bound=npt.DTypeLike)

    def __init__(self, path: os.PathLike[str] | str, mode: Literal["r", "r+", "c"] = "r"):
        self.data = np.memmap(path, mode=mode)
        self.fields: OrderedDict[str, ReaderField] = OrderedDict()
        self.tensors: list[ReaderTensor] = []

        offset = 0
        if self._get(offset, np.uint32, override_order="<")[0] != GGUF_MAGIC:
            raise ValueError("GGUF magic invalid")
        offset += 4

        version_array = self._get(offset, np.uint32)
        if version_array[0] & 0xFFFF == 0:
            self.byte_order = "S"
            version_array = version_array.view(version_array.dtype.newbyteorder(self.byte_order))
        version = int(version_array[0])
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version {version}")
        self.endianess = GGUFEndian.BIG if self.byte_order == "S" else GGUFEndian.LITTLE
        offset += self._push_field(ReaderField(offset, "GGUF.version", [version_array], [0], [GGUFValueType.UINT32]))

        counts = self._get(offset, np.uint64, 2)
        offset += self._push_field(ReaderField(offset, "GGUF.tensor_count", [counts[:1]], [0], [GGUFValueType.UINT64]))
        offset += self._push_field(ReaderField(offset, "GGUF.kv_count", [counts[1:]], [0], [GGUFValueType.UINT64]))
        tensor_count, kv_count = (int(counts[0]), int(counts[1]))

        offset = self._read_metadata_fields(offset, kv_count)
        offset, tensor_fields = self._read_tensor_fields(offset, tensor_count)

        alignment_field = self.fields.get("general.alignment")
        if alignment_field is not None:
            if alignment_field.types != [GGUFValueType.UINT32]:
                raise ValueError("Bad type for general.alignment field")
            self.alignment = int(alignment_field.parts[-1][0])
            if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
                raise ValueError("Invalid alignment: expected a non-zero power of two")

        padding = offset % self.alignment
        if padding:
            offset += self.alignment - padding
        self.data_offset = offset
        self._read_tensors(offset, tensor_fields)

    def get_field(self, key: str) -> Union[ReaderField, None]:
        return self.fields.get(key)

    def get_tensor(self, idx: int) -> ReaderTensor:
        return self.tensors[idx]

    def _get(
        self,
        offset: int,
        dtype: npt.DTypeLike,
        count: int = 1,
        override_order: None | Literal["I", "S", "<"] = None,
    ) -> npt.NDArray[Any]:
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        end_offset = offset + itemsize * count
        array = self.data[offset:end_offset].view(dtype=dtype)[:count]
        return array.view(array.dtype.newbyteorder(self.byte_order if override_order is None else override_order))

    def _push_field(self, field: ReaderField, *, include_size: bool = True) -> int:
        if field.name in self.fields:
            raise KeyError(f"Duplicate GGUF field `{field.name}` at offset {field.offset}")
        self.fields[field.name] = field
        return sum(int(part.nbytes) for part in field.parts) if include_size else 0

    def _read_string(self, offset: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
        length = self._get(offset, np.uint64)
        return length, self._get(offset + 8, np.uint8, length[0])

    def _parse_value(
        self,
        offset: int,
        raw_type: int,
    ) -> tuple[int, list[npt.NDArray[Any]], list[int], list[GGUFValueType]]:
        value_type = GGUFValueType(raw_type)
        if value_type == GGUFValueType.STRING:
            parts = list(self._read_string(offset))
            return sum(int(part.nbytes) for part in parts), parts, [1], [value_type]

        scalar_dtype = _GGUF_SCALAR_TO_NP.get(value_type)
        if scalar_dtype is not None:
            value = self._get(offset, scalar_dtype)
            return int(value.nbytes), [value], [0], [value_type]

        if value_type == GGUFValueType.ARRAY:
            inner_type = self._get(offset, np.uint32)
            inner_count = self._get(offset + int(inner_type.nbytes), np.uint64)
            parts: list[npt.NDArray[Any]] = [inner_type, inner_count]
            data_indices: list[int] = []
            types = [value_type]
            cursor = offset + int(inner_type.nbytes + inner_count.nbytes)

            for index in range(int(inner_count[0])):
                size, child_parts, child_data_indices, child_types = self._parse_value(cursor, int(inner_type[0]))
                if index == 0:
                    types.extend(child_types)
                base_index = len(parts)
                parts.extend(child_parts)
                data_indices.extend(base_index + child_index for child_index in child_data_indices)
                cursor += size

            return cursor - offset, parts, data_indices, types

        raise ValueError(f"Unsupported GGUF field type {value_type}")

    def _read_metadata_fields(self, offset: int, count: int) -> int:
        for _ in range(count):
            field_offset = offset
            key_length, key_data = self._read_string(offset)
            offset += int(key_length.nbytes + key_data.nbytes)
            raw_type = self._get(offset, np.uint32)
            offset += int(raw_type.nbytes)
            size, parts, data_indices, types = self._parse_value(offset, int(raw_type[0]))
            field_parts = [key_length, key_data, raw_type, *parts]
            shifted_indices = [3 + index for index in data_indices]
            self._push_field(
                ReaderField(
                    field_offset,
                    key_data.tobytes().decode("utf-8"),
                    field_parts,
                    shifted_indices,
                    types,
                ),
                include_size=False,
            )
            offset += size
        return offset

    def _read_tensor_field(self, offset: int) -> ReaderField:
        field_offset = offset
        name_length, name_data = self._read_string(offset)
        offset += int(name_length.nbytes + name_data.nbytes)
        n_dims = self._get(offset, np.uint32)
        offset += int(n_dims.nbytes)
        dims = self._get(offset, np.uint64, n_dims[0])
        offset += int(dims.nbytes)
        raw_dtype = self._get(offset, np.uint32)
        offset += int(raw_dtype.nbytes)
        tensor_offset = self._get(offset, np.uint64)
        return ReaderField(
            field_offset,
            name_data.tobytes().decode("utf-8"),
            [name_length, name_data, n_dims, dims, raw_dtype, tensor_offset],
            [1, 3, 4, 5],
            [],
        )

    def _read_tensor_fields(self, offset: int, count: int) -> tuple[int, list[ReaderField]]:
        fields: list[ReaderField] = []
        for _ in range(count):
            field = self._read_tensor_field(offset)
            offset += sum(int(part.nbytes) for part in field.parts)
            fields.append(field)
        return offset, fields

    def _read_tensors(self, data_start: int, fields: list[ReaderField]) -> None:
        tensors: list[ReaderTensor] = []
        seen_names: set[str] = set()

        for field in fields:
            _name_length, name_data, _n_dims, dims, raw_dtype, tensor_offset = field.parts
            tensor_name = name_data.tobytes().decode("utf-8")
            if tensor_name in seen_names:
                raise ValueError(f"Duplicate GGUF tensor `{tensor_name}`")
            seen_names.add(tensor_name)

            tensor_type = GGMLQuantizationType(int(raw_dtype[0]))
            n_elements = int(np.prod(dims))
            logical_shape = tuple(reversed(dims.tolist()))
            block_size, type_size = GGML_QUANT_SIZES[tensor_type]
            n_bytes = n_elements * type_size // block_size
            absolute_offset = int(data_start + tensor_offset[0])

            if tensor_type in {
                GGMLQuantizationType.F16,
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F64,
                GGMLQuantizationType.I8,
                GGMLQuantizationType.I16,
                GGMLQuantizationType.I32,
                GGMLQuantizationType.I64,
            }:
                dtype_by_type = {
                    GGMLQuantizationType.F16: np.float16,
                    GGMLQuantizationType.F32: np.float32,
                    GGMLQuantizationType.F64: np.float64,
                    GGMLQuantizationType.I8: np.int8,
                    GGMLQuantizationType.I16: np.int16,
                    GGMLQuantizationType.I32: np.int32,
                    GGMLQuantizationType.I64: np.int64,
                }
                item_dtype = dtype_by_type[tensor_type]
                item_count = n_elements
                storage_shape = logical_shape
            else:
                item_dtype = np.uint8
                item_count = n_bytes
                storage_shape = quant_shape_to_byte_shape(logical_shape, tensor_type)

            tensor_data = self._get(absolute_offset, item_dtype, item_count).reshape(storage_shape)
            tensors.append(
                ReaderTensor(
                    name=tensor_name,
                    tensor_type=tensor_type,
                    shape=dims,
                    n_elements=n_elements,
                    n_bytes=n_bytes,
                    data_offset=absolute_offset,
                    data=tensor_data,
                    field=field,
                )
            )

        self.tensors = tensors


def install_runtime():
    sys.modules["gguf"] = sys.modules[__name__]
    return sys.modules["gguf"]


__all__ = [
    "GGML_QUANT_SIZES",
    "GGMLQuantizationType",
    "GGUFQuantizedCheckpointSpec",
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_MAGIC",
    "GGUF_VERSION",
    "GGUFEndian",
    "GGUFReader",
    "GGUFValueType",
    "MODEL_ARCH_NAMES",
    "ReaderField",
    "ReaderTensor",
    "dequantize",
    "dequantize_to_torch",
    "get_tensor_name_map",
    "inspect_quantized_checkpoint",
    "install_runtime",
    "native_quantized_loader_enabled",
    "quant_shape_to_byte_shape",
    "quantize",
]
