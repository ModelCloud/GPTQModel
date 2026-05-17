# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Safetensor-level dequantization helpers for common quant formats."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..quantization.dtype import (
    available_float4_packed_dtypes,
    available_float8_dtype_names,
    available_float8_dtypes,
    dequantize_f4_e2m1,
    dequantize_fp8,
)
from ..utils.logger import setup_logger


LOG = logging.getLogger(__name__)

# Reuse the shared dtype registries so checkpoint detection stays aligned with
# the CPU dequant kernels and config normalization paths.
_FLOAT8_DTYPES = available_float8_dtypes()
_FLOAT8_FORMAT_NAMES = frozenset(available_float8_dtype_names())
_NVFP4_STORAGE_DTYPES = (torch.uint8, *available_float4_packed_dtypes())

if TYPE_CHECKING:
    from compressed_tensors.compressors.base import BaseCompressor
    from compressed_tensors.quantization.quant_scheme import QuantizationScheme


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def list_safetensor_files(model_path: Path) -> Tuple[list, Optional[dict]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        files = sorted(set(index.get("weight_map", {}).values()))
        return files, index

    files = sorted([p.name for p in model_path.glob("*.safetensors")])
    return files, None


def finalize_for_save(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast to ``target_dtype`` when floating point and move to CPU with optimal layout."""

    if torch.is_floating_point(tensor):
        tensor = tensor.to(target_dtype)

    tensor_cpu = tensor.to("cpu")
    if tensor_cpu.ndim == 4:
        tensor_cpu = tensor_cpu.contiguous(memory_format=torch.channels_last)
    else:
        tensor_cpu = tensor_cpu.contiguous()
    return tensor_cpu


def normalize_device(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    device = device.strip()
    if device.lower() == "cpu":
        return None

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Unsupported device type: {device}")

    if dev.index is None:
        return "cuda:0"
    return f"cuda:{dev.index}"


class _ShardTensorLookup:
    """Resolve tensors by name across sharded safetensors using the HF weight map."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        weight_map: Optional[dict],
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.weight_map = weight_map or {}
        self._stack = ExitStack()
        self._readers: Dict[str, object] = {}
        self._keys: Dict[str, set[str]] = {}

    def close(self) -> None:
        self._stack.close()
        self._readers.clear()
        self._keys.clear()

    def _reader_for_shard(self, shard_name: str):
        reader = self._readers.get(shard_name)
        if reader is None:
            # Keep cross-shard readers open while one dequantization pass runs.
            reader = self._stack.enter_context(
                safe_open(self.model_path / shard_name, framework="pt", device=self.device)
            )
            self._readers[shard_name] = reader
            self._keys[shard_name] = set(reader.keys())
        return reader

    def has_tensor(
        self,
        key: str,
        *,
        local_reader=None,
        local_keys: Optional[set[str]] = None,
    ) -> bool:
        if local_reader is not None and local_keys is not None and key in local_keys:
            return True
        shard_name = self.weight_map.get(key)
        if shard_name is None:
            return False
        reader = self._reader_for_shard(shard_name)
        return key in self._keys[shard_name] and reader is not None

    def get_tensor(
        self,
        key: str,
        *,
        local_reader=None,
        local_keys: Optional[set[str]] = None,
    ) -> torch.Tensor:
        if local_reader is not None and local_keys is not None and key in local_keys:
            return local_reader.get_tensor(key)
        shard_name = self.weight_map.get(key)
        if shard_name is None:
            raise KeyError(key)
        reader = self._reader_for_shard(shard_name)
        if key not in self._keys[shard_name]:
            raise KeyError(key)
        return reader.get_tensor(key)


def _get_compressed_tensors_dependencies() -> dict:
    try:
        from compressed_tensors.compressors.base import BaseCompressor
        from compressed_tensors.compressors.model_compressors.model_compressor import (
            map_module_to_scheme,
        )
        from compressed_tensors.quantization import QuantizationConfig
        from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'compressed-tensors' package. Install it with 'pip install compressed-tensors'."
        ) from exc

    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'accelerate' package. Install it with 'pip install accelerate'."
        ) from exc

    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'transformers' package."
        ) from exc

    return {
        "QuantizationConfig": QuantizationConfig,
        "apply_quantization_config": apply_quantization_config,
        "BaseCompressor": BaseCompressor,
        "map_module_to_scheme": map_module_to_scheme,
        "init_empty_weights": init_empty_weights,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    }


def _get_bitsandbytes_dependencies():
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for bitsandbytes quantized checkpoints requires the "
            "'bitsandbytes' package. Install it with 'pip install bitsandbytes>=0.49.3'."
        ) from exc

    return bnb


def _discover_compressed_tensors_module_schemes(
    model_path: Path,
    quant_config,
    *,
    deps: dict,
) -> Dict[str, "QuantizationScheme"]:
    AutoConfig = deps["AutoConfig"]
    init_empty_weights = deps["init_empty_weights"]
    apply_quantization_config = deps["apply_quantization_config"]
    map_module_to_scheme = deps["map_module_to_scheme"]
    from .hf import prepare_remote_code_compat

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    prepare_remote_code_compat(config)

    loader_candidates = (
        deps["AutoModelForCausalLM"],
        deps["AutoModelForSeq2SeqLM"],
        deps["AutoModel"],
    )

    loader_errors: list[tuple[str, Exception]] = []
    model = None
    for loader in loader_candidates:
        if loader is None:
            continue

        with init_empty_weights(include_buffers=True):
            try:
                model = loader.from_config(
                    config,
                    trust_remote_code=True,
                    dtype=torch.float32,
                )
            except Exception as exc:  # pragma: no cover - depends on available loaders
                loader_errors.append((loader.__name__, exc))
                continue
            else:
                break

    if model is None:
        if loader_errors:
            names = ", ".join(name for name, _ in loader_errors)
            last_error = loader_errors[-1][1]
            raise RuntimeError(
                f"Failed to instantiate model from '{model_path}' while inspecting "
                f"compressed-tensors modules. Loaders attempted: {names}."
            ) from last_error
        raise RuntimeError(
            f"Failed to instantiate model from '{model_path}' while inspecting "
            "compressed-tensors modules."
        )

    try:
        apply_quantization_config(model, quant_config, run_compressed=False)
        module_to_scheme = map_module_to_scheme(model)
    finally:
        del model

    return dict(module_to_scheme)


def _prepare_compressed_tensors_context(
    model_path: Path, quant_cfg: dict
) -> tuple["QuantizationConfig", Dict[str, "QuantizationScheme"], "BaseCompressor"]: # noqa
    deps = _get_compressed_tensors_dependencies()

    QuantizationConfig = deps["QuantizationConfig"]
    quant_config = QuantizationConfig.model_validate(quant_cfg)
    quant_format = (quant_config.format or "").lower()
    if quant_format != "pack-quantized":
        raise ValueError(
            f"Unsupported compressed-tensors format '{quant_config.format}'. "
            "Only 'pack-quantized' is currently supported."
        )

    module_to_scheme = _discover_compressed_tensors_module_schemes(
        model_path, quant_config, deps=deps
    )

    BaseCompressor = deps["BaseCompressor"]
    compressor = BaseCompressor.load_from_registry(quant_format, config=quant_config)
    return quant_config, module_to_scheme, compressor


def resolve_block_size(config: dict) -> Optional[Tuple[int, int]]:
    quant_cfg = config.get("quantization_config", {}) or {}
    block_size = quant_cfg.get("weight_block_size")
    if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])
    return None


def resolve_ignored_layers(config: dict) -> frozenset[str]:
    # Quantization configs name modules without tensor suffixes.
    quant_cfg = config.get("quantization_config", {}) or {}
    ignored_layers = quant_cfg.get("ignored_layers") or ()
    if not isinstance(ignored_layers, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(layer for layer in ignored_layers if isinstance(layer, str) and layer)


def _tensor_key_matches_ignored_layer(key: str, ignored_layers: Iterable[str]) -> bool:
    return any(key == layer or key.startswith(f"{layer}.") for layer in ignored_layers)


def _is_quant_auxiliary_tensor_key(key: str) -> bool:
    # Auxiliary quantization tensors are not useful once the paired weight is kept or dequantized.
    return key.endswith(
        (
            "_scale_inv",
            ".scale",
            ".weight_scale",
            ".weight_absmax",
            ".weight_quant_map",
            ".weight_nested_absmax",
            ".weight_nested_quant_map",
            ".weight_quant_state",
            ".weight_scb",
            ".qweight",
            ".qzeros",
            ".scales",
            ".g_idx",
            ".weight_packed",
            ".weight_zero_point",
            ".weight_g_idx",
            ".weight_shape",
        )
    )


def _handle_ignored_tensor(
    key: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    ignored_layers: Iterable[str],
) -> Optional[torch.Tensor]:
    if not _tensor_key_matches_ignored_layer(key, ignored_layers):
        return None
    if _is_quant_auxiliary_tensor_key(key):
        LOG.debug("Dropping auxiliary quantization tensor '%s' for ignored layer", key)
        return None
    # Ignored layers are already stored as dense tensors in mixed-precision checkpoints.
    return finalize_for_save(tensor, target_dtype)


def _expected_block_grid_shape(
    weight_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
) -> Tuple[int, int]:
    rows, cols = weight_shape
    block_rows, block_cols = block_shape
    return (
        (rows + block_rows - 1) // block_rows,
        (cols + block_cols - 1) // block_cols,
    )


def _block_scale_grid_covers_weight(
    scale_shape: Optional[Tuple[int, ...]],
    weight_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
) -> bool:
    # Accept scale grids padded beyond the real tensor shape.
    if scale_shape is None or len(scale_shape) != 2:
        return False
    rows, cols = weight_shape
    block_rows, block_cols = block_shape
    blocks_r, blocks_c = scale_shape
    return blocks_r * block_rows >= rows and blocks_c * block_cols >= cols


def _expand_padded_block_scale(
    scale_tensor: Optional[torch.Tensor],
    *,
    weight_shape: Tuple[int, int],
    block_shape: Optional[Tuple[int, int]],
) -> Optional[torch.Tensor]:
    """Expand block-grid FP8 scales to dense elementwise scales when edge blocks are padded."""

    if not isinstance(scale_tensor, torch.Tensor):
        return None
    if scale_tensor.ndim != 2 or len(weight_shape) != 2 or block_shape is None:
        return scale_tensor

    rows, cols = weight_shape
    blocks_r, blocks_c = scale_tensor.shape
    if (blocks_r, blocks_c) == (rows, cols):
        return scale_tensor

    block_rows, block_cols = block_shape
    expected_grid = _expected_block_grid_shape(weight_shape, block_shape)
    if (blocks_r, blocks_c) != expected_grid and not _block_scale_grid_covers_weight(
        (blocks_r, blocks_c),
        weight_shape,
        block_shape,
    ):
        return scale_tensor

    # Expand block scales to elementwise scales, then crop padded edge blocks.
    expanded = scale_tensor.repeat_interleave(block_rows, dim=0)
    expanded = expanded.repeat_interleave(block_cols, dim=1)
    return expanded[:rows, :cols].contiguous()


def infer_block_shape(weight_shape: Tuple[int, int], scale_tensor: torch.Tensor) -> Tuple[int, int]:
    rows, cols = weight_shape
    shape = tuple(scale_tensor.shape)

    if scale_tensor.ndim == 0:
        LOG.debug(
            "Inferred block size (%d, %d) from scalar scale tensor for weight shape %s",
            rows,
            cols,
            weight_shape,
        )
        return rows, cols

    if shape == weight_shape:
        LOG.debug(
            "Inferred element-wise scaling (block size 1x1) for weight shape %s",
            weight_shape,
        )
        return 1, 1

    if scale_tensor.ndim == 2:
        block_rows = shape[0]
        block_cols = shape[1]
        if block_rows == 0 or block_cols == 0:
            raise ValueError("scale tensor has zero-sized dimension")
        if rows % block_rows != 0 or cols % block_cols != 0:
            raise ValueError("scale tensor shape incompatible with weight dimensions")
        inferred = (rows // block_rows, cols // block_cols)
        LOG.debug(
            "Inferred block size %s from 2D scale tensor shape %s and weight shape %s",
            inferred,
            shape,
            weight_shape,
        )
        return inferred

    if scale_tensor.ndim == 1:
        count = shape[0]
        if count == 0:
            raise ValueError("scale tensor is empty")

        candidates: list[Tuple[int, int]] = []
        for row_blocks in range(1, count + 1):
            if count % row_blocks != 0:
                continue
            if rows % row_blocks != 0:
                continue
            col_blocks = count // row_blocks
            if cols % col_blocks != 0:
                continue
            block_rows = rows // row_blocks
            block_cols = cols // col_blocks
            candidates.append((block_rows, block_cols))

        if candidates:
            candidates.sort(
                key=lambda bc: (
                    abs(bc[0] - bc[1]),
                    -min(bc),
                    -max(bc),
                    bc[0],
                )
            )
            inferred = candidates[0]
            LOG.debug(
                "Inferred block size %s from 1D scale tensor (count=%d) and weight shape %s",
                inferred,
                count,
                weight_shape,
            )
            return inferred

        if rows % count == 0:
            inferred = (rows // count, cols)
            LOG.debug(
                "Inferred row-only scaling block size %s from 1D scale tensor (count=%d)",
                inferred,
                count,
            )
            return inferred
        if cols % count == 0:
            inferred = (rows, cols // count)
            LOG.debug(
                "Inferred column-only scaling block size %s from 1D scale tensor (count=%d)",
                inferred,
                count,
            )
            return inferred

        raise ValueError("unable to infer block size from 1D scale tensor")

        raise ValueError("unsupported scale tensor rank for block size inference")


def detect_format(model_path: Path, config: dict) -> str:
    quant_cfg = config.get("quantization_config", {}) or {}
    method = (quant_cfg.get("method") or quant_cfg.get("quant_method") or "").lower()
    format_name = (quant_cfg.get("format") or "").lower()

    files, _ = list_safetensor_files(model_path)
    if not files:
        raise FileNotFoundError("No .safetensors files found in model directory")

    with safe_open(model_path / files[0], framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        # Prefer dtype-based detection
        for key in keys:
            if key.endswith(".weight"):
                tensor = reader.get_tensor(key)
                if tensor.dtype in _FLOAT8_DTYPES:
                    LOG.debug("Detected FP8 weights via dtype on tensor '%s'", key)
                    return "fp8"
                if tensor.dtype in _NVFP4_STORAGE_DTYPES and (key + "_scale") in keys:
                    LOG.debug("Detected NVFP4 weights via dtype on tensor '%s'", key)
                    return "nvfp4"
        if any(k.endswith(".weight_packed") for k in keys):
            LOG.debug(
                "Detected compressed-tensors pack-quantized format via '.weight_packed' "
                "metadata in shard '%s'",
                files[0],
            )
            return "compressed-pack"
        if any(k.endswith(".weight_scale") for k in keys):
            LOG.debug("Detected NVFP4 format via '.weight_scale' metadata in shard '%s'", files[0])
            return "nvfp4"
        if any(k.endswith(".weight_scale_inv") for k in keys):
            LOG.debug("Detected FP8 format via '.weight_scale_inv' metadata in shard '%s'", files[0])
            return "fp8"
        if any(k == "weight_quant_state" or k.endswith(".weight_quant_state") for k in keys) or any(
            k == "weight_scb" or k.endswith(".weight_scb") for k in keys
        ):
            LOG.debug("Detected bitsandbytes format via explicit state tensors in shard '%s'", files[0])
            return "bitsandbytes"
        if any(k.endswith(".trellis") for k in keys):
            LOG.debug("Detected EXL3 format via '.trellis' metadata in shard '%s'", files[0])
            return "exl3"
        if any(k.endswith(".qweight") for k in keys):
            has_g = any(k.endswith(".g_idx") for k in keys)
            LOG.debug(
                "Detected %s format via qweight tensors in shard '%s'",
                "gptq" if has_g else "awq",
                files[0],
            )
            return "gptq" if has_g else "awq"

    if format_name in _FLOAT8_FORMAT_NAMES:
        LOG.debug("Detected FP8 format via config format=%s", format_name)
        return "fp8"
    if format_name in {"fp4", "nf4", "int8"}:
        LOG.debug("Detected bitsandbytes format via config format=%s", format_name)
        return "bitsandbytes"
    if method == "fp8":
        LOG.debug("Detected FP8 format via method=%s", method)
        return "fp8"
    if method == "bitsandbytes":
        LOG.debug("Detected bitsandbytes format via method=%s", method)
        return "bitsandbytes"
    if method in ("gptq", "gptqmodel"):
        LOG.debug("Detected GPTQ format via method=%s", method)
        return "gptq"
    if method == "awq":
        LOG.debug("Detected AWQ format via method=%s", method)
        return "awq"
    if method == "exl3":
        LOG.debug("Detected EXL3 format via method=%s", method)
        return "exl3"
    if method == "compressed-tensors":
        fmt_name = (quant_cfg.get("format") or "").lower()
        if fmt_name == "pack-quantized":
            LOG.debug(
                "Detected compressed-tensors format via method=%s and format=%s",
                method,
                fmt_name,
            )
            return "compressed-pack"

    raise ValueError("Unable to detect quantization format for model")


def unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 3:
        return _unpack_cols_3bit(packed)

    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows, cols * pack_factor, dtype=torch.int32)
    for i in range(pack_factor):
        result[:, i::pack_factor] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def pack_cols(values: torch.Tensor, bits: int, *, pack_dtype: torch.dtype) -> torch.Tensor:
    if bits == 3:
        return _pack_cols_3bit(values, pack_dtype=pack_dtype)

    pack_bits = torch.empty((), dtype=pack_dtype).element_size() * 8
    pack_factor = pack_bits // bits
    if pack_factor <= 0:
        raise ValueError(f"Unsupported pack width {pack_bits} for {bits}-bit values")

    rows, cols = values.shape
    if cols % pack_factor != 0:
        raise ValueError(
            f"Column count {cols} is not divisible by the {bits}-bit pack factor {pack_factor}"
        )

    mask = (1 << bits) - 1
    packed = torch.zeros(rows, cols // pack_factor, dtype=pack_dtype, device=values.device)
    values_uint = values.to(torch.int64) & mask
    for i in range(pack_factor):
        packed |= (values_uint[:, i::pack_factor] << (i * bits)).to(pack_dtype)
    return packed


def unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 3:
        return _unpack_rows_3bit(packed)

    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows * pack_factor, cols, dtype=torch.int32)
    for i in range(pack_factor):
        result[i::pack_factor, :] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def _require_int32_words_for_3bit(tensor: torch.Tensor) -> None:
    pack_bits = tensor.element_size() * 8
    if pack_bits != 32:
        raise NotImplementedError(
            f"3-bit GPTQ safetensor dequantization expects 32-bit packed words, got {pack_bits}-bit words."
        )


def _unpack_cols_3bit(packed: torch.Tensor) -> torch.Tensor:
    _require_int32_words_for_3bit(packed)

    rows, cols = packed.shape
    if cols % 3 != 0:
        raise ValueError(f"3-bit GPTQ qzeros expects columns divisible by 3, got shape {tuple(packed.shape)}")

    blocks = cols // 3
    words = packed.to(torch.int64).reshape(rows, blocks, 3)
    word0 = words[:, :, 0]
    word1 = words[:, :, 1]
    word2 = words[:, :, 2]

    result = torch.empty((rows, blocks * 32), dtype=torch.int32, device=packed.device)
    unpacked = result.view(rows, blocks, 32)

    for idx in range(10):
        unpacked[:, :, idx] = ((word0 >> (3 * idx)) & 0x7).to(torch.int32)

    unpacked[:, :, 10] = (
        ((word0 >> 30) & 0x3) | (((word1 >> 0) << 2) & 0x4)
    ).to(torch.int32)

    for idx in range(10):
        unpacked[:, :, 11 + idx] = ((word1 >> (1 + 3 * idx)) & 0x7).to(torch.int32)

    unpacked[:, :, 21] = (
        ((word1 >> 31) & 0x1) | (((word2 >> 0) << 1) & 0x6)
    ).to(torch.int32)

    for idx in range(10):
        unpacked[:, :, 22 + idx] = ((word2 >> (2 + 3 * idx)) & 0x7).to(torch.int32)

    return result


def _pack_cols_3bit(values: torch.Tensor, *, pack_dtype: torch.dtype) -> torch.Tensor:
    pack_bits = torch.empty((), dtype=pack_dtype).element_size() * 8
    if pack_bits != 32:
        raise NotImplementedError(
            f"3-bit GPTQ safetensor dequantization expects 32-bit packed words, got {pack_bits}-bit words."
        )

    rows, cols = values.shape
    if cols % 32 != 0:
        raise ValueError(f"3-bit GPTQ qzeros expects columns divisible by 32, got shape {tuple(values.shape)}")

    blocks = cols // 32
    values_i64 = (values.to(torch.int64) & 0x7).reshape(rows, blocks, 32)
    mask32 = (1 << 32) - 1

    word0 = torch.zeros((rows, blocks), dtype=torch.int64, device=values.device)
    for idx in range(10):
        word0 |= values_i64[:, :, idx] << (3 * idx)
    word0 |= values_i64[:, :, 10] << 30

    word1 = (values_i64[:, :, 10] >> 2) & 0x1
    for idx in range(10):
        word1 |= values_i64[:, :, 11 + idx] << (1 + 3 * idx)
    word1 |= values_i64[:, :, 21] << 31

    word2 = (values_i64[:, :, 21] >> 1) & 0x3
    for idx in range(10):
        word2 |= values_i64[:, :, 22 + idx] << (2 + 3 * idx)

    packed = torch.stack((word0, word1, word2), dim=2).reshape(rows, blocks * 3)
    return (packed & mask32).to(pack_dtype)


def _unpack_rows_3bit(packed: torch.Tensor) -> torch.Tensor:
    _require_int32_words_for_3bit(packed)

    rows, cols = packed.shape
    if rows % 3 != 0:
        raise ValueError(f"3-bit GPTQ qweight expects rows divisible by 3, got shape {tuple(packed.shape)}")

    blocks = rows // 3
    words = packed.to(torch.int64).reshape(blocks, 3, cols)
    word0 = words[:, 0, :]
    word1 = words[:, 1, :]
    word2 = words[:, 2, :]

    result = torch.empty((blocks * 32, cols), dtype=torch.int32, device=packed.device)
    unpacked = result.view(blocks, 32, cols)

    for idx in range(10):
        unpacked[:, idx, :] = ((word0 >> (3 * idx)) & 0x7).to(torch.int32)

    unpacked[:, 10, :] = (
        ((word0 >> 30) & 0x3) | (((word1 >> 0) << 2) & 0x4)
    ).to(torch.int32)

    for idx in range(10):
        unpacked[:, 11 + idx, :] = ((word1 >> (1 + 3 * idx)) & 0x7).to(torch.int32)

    unpacked[:, 21, :] = (
        ((word1 >> 31) & 0x1) | (((word2 >> 0) << 1) & 0x6)
    ).to(torch.int32)

    for idx in range(10):
        unpacked[:, 22 + idx, :] = ((word2 >> (2 + 3 * idx)) & 0x7).to(torch.int32)

    return result


def _uses_gptq_v1_qzeros(config: dict) -> bool:
    checkpoint_format = str(config.get("checkpoint_format") or "gptq").strip().lower()
    return checkpoint_format in {"gptq", "gemm"}


def _shift_gptq_qzeros(qzeros: torch.Tensor, bits: int, *, delta: int) -> torch.Tensor:
    # GPTQ v1 stores qzeros with a per-field -1 offset. For 3-bit checkpoints,
    # some logical values straddle adjacent packed words, so a packed-word add/sub
    # is not equivalent to shifting each decoded zero-point. Decode the fields,
    # shift them in logical space, then repack into the original storage dtype.
    zeros = unpack_cols(qzeros, bits)
    shifted = (zeros + delta) & ((1 << bits) - 1)
    return pack_cols(shifted, bits, pack_dtype=qzeros.dtype)


def _correct_gptq_v1_qzeros(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    return _shift_gptq_qzeros(qzeros, bits, delta=1)


def _revert_gptq_v1_qzeros_correction(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    return _shift_gptq_qzeros(qzeros, bits, delta=-1)


def convert_fp8_shard(
    reader,
    target_dtype: torch.dtype,
    *,
    block_shape: Optional[Tuple[int, int]],
    scale_semantics: str = "heuristic",
    tensor_lookup: Optional[_ShardTensorLookup] = None,
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    reader_keys = set(reader.keys())
    for key in reader.keys():
        tensor = reader.get_tensor(key)
        ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
        if ignored_tensor is not None:
            tensors[key] = ignored_tensor
            continue
        if _tensor_key_matches_ignored_layer(key, ignored_layers):
            continue

        if key.endswith(".weight") and tensor.dtype in _FLOAT8_DTYPES:
            scale_key = key + "_scale_inv"
            scale_tensor = None
            scale_inv = None
            if tensor_lookup is not None and tensor_lookup.has_tensor(
                scale_key, local_reader=reader, local_keys=reader_keys
            ):
                # GPTQModel-native FP8 checkpoints persist inverse scales under
                # the historical `weight_scale_inv` suffix.
                scale_inv = tensor_lookup.get_tensor(
                    scale_key, local_reader=reader, local_keys=reader_keys
                )
                LOG.debug("Using scale_inv tensor '%s' for FP8 weight '%s'", scale_key, key)
            else:
                # Some native FP8 checkpoints (for example DeepSeek V4) store
                # direct scales as `<module>.scale` instead of `weight_scale_inv`.
                scale_key = key[:-len(".weight")] + ".scale"
                if tensor_lookup is None or not tensor_lookup.has_tensor(
                    scale_key, local_reader=reader, local_keys=reader_keys
                ):
                    raise KeyError(f"Missing FP8 scale tensor for {key}")
                scale_tensor = tensor_lookup.get_tensor(
                    scale_key, local_reader=reader, local_keys=reader_keys
                )
                LOG.debug("Using scale tensor '%s' for FP8 weight '%s'", scale_key, key)

            rows, cols = tensor.shape
            effective_block = block_shape
            if effective_block is None:
                try:
                    # Infer the block layout from whichever FP8 scale variant
                    # the checkpoint actually stores.
                    block_source = scale_inv if scale_inv is not None else scale_tensor
                    effective_block = infer_block_shape((rows, cols), block_source)
                    LOG.debug("Inferred block size %s for weight '%s'", effective_block, key)
                except ValueError as exc:
                    LOG.debug(
                        "Falling back to full-tensor block size for weight '%s' (%s)",
                        key,
                        exc,
                    )
                    effective_block = (rows, cols)
            else:
                LOG.debug("Using configured block size %s for weight '%s'", effective_block, key)

            block_rows, block_cols = effective_block
            if block_rows <= 0 or block_cols <= 0:
                raise ValueError(f"Inferred invalid block size {effective_block} for {key}")
            partial_block_grid = _expected_block_grid_shape((rows, cols), effective_block)
            block_source = scale_inv if scale_inv is not None else scale_tensor
            block_source_shape = tuple(block_source.shape) if isinstance(block_source, torch.Tensor) else None
            allows_partial_edge_blocks = block_source_shape == partial_block_grid or _block_scale_grid_covers_weight(
                block_source_shape,
                (rows, cols),
                effective_block,
            )
            if (rows % block_rows != 0 or cols % block_cols != 0) and not allows_partial_edge_blocks:
                raise ValueError(
                    f"Tensor {key} shape {tensor.shape} incompatible with block size {effective_block}"
                )
            if allows_partial_edge_blocks and (rows % block_rows != 0 or cols % block_cols != 0):
                # Accept checkpoints that pad only the scale grid, not the weight.
                LOG.debug(
                    "Allowing partial edge blocks for weight '%s': shape=%s block=%s scale_grid=%s",
                    key,
                    tuple(tensor.shape),
                    effective_block,
                    block_source_shape,
                )

            scale_arg = scale_tensor
            scale_inv_arg = scale_inv
            scale_arg = _expand_padded_block_scale(
                scale_arg,
                weight_shape=(rows, cols),
                block_shape=effective_block,
            )
            scale_inv_arg = _expand_padded_block_scale(
                scale_inv_arg,
                weight_shape=(rows, cols),
                block_shape=effective_block,
            )
            if scale_semantics == "inverse" and scale_inv is not None:
                scale_arg = torch.reciprocal(scale_inv.to(torch.float32))
                scale_inv_arg = None
                scale_arg = _expand_padded_block_scale(
                    scale_arg,
                    weight_shape=(rows, cols),
                    block_shape=effective_block,
                )
            deq = dequantize_fp8(
                tensor,
                scale=scale_arg,
                scale_inv=scale_inv_arg,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = finalize_for_save(deq, target_dtype)
        elif key.endswith("_scale_inv"):
            LOG.debug("Dropping auxiliary FP8 tensor '%s' after dequantization", key)
            continue
        elif key.endswith(".scale"):
            weight_key = key[:-len(".scale")] + ".weight"
            if tensor_lookup is not None and tensor_lookup.has_tensor(
                weight_key, local_reader=reader, local_keys=reader_keys
            ):
                weight_tensor = tensor_lookup.get_tensor(
                    weight_key, local_reader=reader, local_keys=reader_keys
                )
                if weight_tensor.dtype in _FLOAT8_DTYPES:
                    # Mirror the `_scale_inv` handling so exported BF16 checkpoints
                    # keep only dense weights, not FP8 reconstruction metadata.
                    LOG.debug("Dropping auxiliary FP8 tensor '%s' after dequantization", key)
                    continue
            elif weight_key in reader_keys and reader.get_tensor(weight_key).dtype in _FLOAT8_DTYPES:
                # Mirror the `_scale_inv` handling so exported BF16 checkpoints
                # keep only dense weights, not FP8 reconstruction metadata.
                LOG.debug("Dropping auxiliary FP8 tensor '%s' after dequantization", key)
                continue
            tensors[key] = finalize_for_save(tensor, target_dtype)
        else:
            tensors[key] = finalize_for_save(tensor, target_dtype)
    return tensors


def convert_nvfp4_shard(
    reader,
    target_dtype: torch.dtype,
    *,
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        tensor = reader.get_tensor(key)
        ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
        if ignored_tensor is not None:
            tensors[key] = ignored_tensor
            continue
        if _tensor_key_matches_ignored_layer(key, ignored_layers):
            continue

        if key.endswith(".weight") and tensor.dtype in _NVFP4_STORAGE_DTYPES:
            scale_key = key + "_scale"
            if scale_key not in reader.keys():
                raise KeyError(f"Missing scale tensor for {key}")
            scale = reader.get_tensor(scale_key)
            LOG.debug("Using scale tensor '%s' for NVFP4 weight '%s'", scale_key, key)
            deq = dequantize_f4_e2m1(
                tensor,
                scale=scale,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = finalize_for_save(deq, target_dtype)
        elif key.endswith(".weight_scale"):
            LOG.debug("Dropping auxiliary NVFP4 tensor '%s' after dequantization", key)
            continue
        else:
            tensors[key] = finalize_for_save(tensor, target_dtype)
    return tensors


def convert_bitsandbytes_shard(
    reader,
    target_dtype: torch.dtype,
    *,
    quant_cfg: dict,
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    bnb = _get_bitsandbytes_dependencies()

    tensors: Dict[str, torch.Tensor] = {}
    keys = list(reader.keys())
    key_set = set(keys)
    bnb_quant_type = str(
        quant_cfg.get("format")
        or quant_cfg.get("bnb_quant_type")
        or "fp4"
    ).strip().lower()
    if bnb_quant_type == "bitsandbytes":
        bnb_quant_type = "fp4"

    skipped_suffixes = (
        ".weight_absmax",
        ".weight_quant_map",
        ".weight_nested_absmax",
        ".weight_nested_quant_map",
        ".weight_quant_state",
        ".weight_scb",
    )

    for key in keys:
        tensor = reader.get_tensor(key)
        ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
        if ignored_tensor is not None:
            tensors[key] = ignored_tensor
            continue
        if _tensor_key_matches_ignored_layer(key, ignored_layers):
            continue

        if key.endswith(".weight") and (key[:-len(".weight")] + ".weight_quant_state") in key_set:
            prefix = key[:-len(".weight")]
            payload = {
                "absmax": reader.get_tensor(prefix + ".weight_absmax"),
                "quant_map": reader.get_tensor(prefix + ".weight_quant_map"),
                f"quant_state.bitsandbytes__{bnb_quant_type}": reader.get_tensor(prefix + ".weight_quant_state"),
            }
            if prefix + ".weight_nested_absmax" in key_set:
                payload["nested_absmax"] = reader.get_tensor(prefix + ".weight_nested_absmax")
            if prefix + ".weight_nested_quant_map" in key_set:
                payload["nested_quant_map"] = reader.get_tensor(prefix + ".weight_nested_quant_map")

            quant_state = bnb.functional.QuantState.from_dict(payload, device=tensor.device)
            deq = bnb.functional.dequantize_4bit(tensor, quant_state=quant_state)
            tensors[key] = finalize_for_save(deq, target_dtype)
            LOG.debug("Dequantized bitsandbytes 4-bit module '%s' to dtype %s", prefix, target_dtype)
            continue

        if key.endswith(".weight") and (key[:-len(".weight")] + ".weight_scb") in key_set:
            prefix = key[:-len(".weight")]
            deq = bnb.functional.int8_vectorwise_dequant(
                tensor,
                reader.get_tensor(prefix + ".weight_scb"),
            )
            tensors[key] = finalize_for_save(deq, target_dtype)
            LOG.debug("Dequantized bitsandbytes 8-bit module '%s' to dtype %s", prefix, target_dtype)
            continue

        if key.endswith(skipped_suffixes):
            LOG.debug("Dropping auxiliary bitsandbytes tensor '%s' after dequantization", key)
            continue

        tensors[key] = finalize_for_save(tensor, target_dtype)

    return tensors


def convert_awq_file(
    path: Path,
    target_dtype: torch.dtype,
    device: str,
    *,
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
            if ignored_tensor is not None:
                tensors[key] = ignored_tensor
                continue
            if _tensor_key_matches_ignored_layer(key, ignored_layers):
                continue

            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
                LOG.debug("Collected AWQ qweight tensor '%s'", key)
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
                LOG.debug("Collected AWQ qzeros tensor '%s'", key)
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
                LOG.debug("Collected AWQ scale tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete AWQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        bits = 4

        unpacked_weight = unpack_cols(qweight, bits).to(torch.float32)
        unpacked_zeros = unpack_cols(qzeros, bits).to(torch.float32)

        num_groups = scales.shape[0]
        group_size = unpacked_weight.shape[0] // num_groups
        scales_full = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
        zeros_full = unpacked_zeros.repeat_interleave(group_size, dim=0)

        weight = (unpacked_weight - zeros_full) * scales_full
        tensors[prefix + ".weight"] = finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        LOG.debug("Dequantized AWQ module '%s' to dtype %s", prefix, target_dtype)
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def convert_gptq_file(
    path: Path,
    target_dtype: torch.dtype,
    config: dict,
    device: str,
    *,
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
            if ignored_tensor is not None:
                tensors[key] = ignored_tensor
                continue
            if _tensor_key_matches_ignored_layer(key, ignored_layers):
                continue

            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
                LOG.debug("Collected GPTQ qweight tensor '%s'", key)
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
                LOG.debug("Collected GPTQ qzeros tensor '%s'", key)
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
                LOG.debug("Collected GPTQ scale tensor '%s'", key)
            elif key.endswith(".g_idx"):
                prefix = key[:-len(".g_idx")]
                module_buffers[prefix]["g_idx"] = tensor
                LOG.debug("Collected GPTQ g_idx tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales", "g_idx") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete GPTQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        g_idx = buf["g_idx"].to(torch.long)

        bits = config.get("bits", 4)
        if _uses_gptq_v1_qzeros(config):
            qzeros = _correct_gptq_v1_qzeros(qzeros, bits)
        weight_int = unpack_rows(qweight, bits)
        zeros = unpack_cols(qzeros, bits)

        scales_full = scales.to(torch.float32)[g_idx]
        zeros_full = zeros.to(torch.float32)[g_idx]
        weight = (weight_int.to(torch.float32) - zeros_full) * scales_full
        tensors[prefix + ".weight"] = finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        LOG.debug(
            "Dequantized GPTQ module '%s' with %d-bit groups to dtype %s",
            prefix,
            bits,
            target_dtype,
        )
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def convert_compressed_pack_file(
    path: Path,
    target_dtype: torch.dtype,
    *,
    device: str,
    module_to_scheme: Dict[str, "QuantizationScheme"],
    compressor: "BaseCompressor",
    ignored_layers: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            ignored_tensor = _handle_ignored_tensor(key, tensor, target_dtype, ignored_layers)
            if ignored_tensor is not None:
                tensors[key] = ignored_tensor
                continue
            if _tensor_key_matches_ignored_layer(key, ignored_layers):
                continue

            if key.endswith(".weight_packed"):
                prefix = key[: -len(".weight_packed")]
                module_buffers[prefix]["weight_packed"] = tensor
                LOG.debug("Collected compressed weight tensor '%s'", key)
            elif key.endswith(".weight_scale"):
                prefix = key[: -len(".weight_scale")]
                module_buffers[prefix]["weight_scale"] = tensor
                LOG.debug("Collected compressed scale tensor '%s'", key)
            elif key.endswith(".weight_zero_point"):
                prefix = key[: -len(".weight_zero_point")]
                module_buffers[prefix]["weight_zero_point"] = tensor
                LOG.debug("Collected compressed zero-point tensor '%s'", key)
            elif key.endswith(".weight_g_idx"):
                prefix = key[: -len(".weight_g_idx")]
                module_buffers[prefix]["weight_g_idx"] = tensor
                LOG.debug("Collected compressed group-index tensor '%s'", key)
            elif key.endswith(".weight_shape"):
                prefix = key[: -len(".weight_shape")]
                module_buffers[prefix]["weight_shape"] = tensor
                LOG.debug("Collected compressed shape tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        scheme = module_to_scheme.get(prefix)
        if scheme is None:
            raise KeyError(
                f"No quantization scheme registered for compressed module '{prefix}'."
            )

        if scheme.weights is None:
            raise ValueError(
                f"Module '{prefix}' does not define weight quantization parameters."
            )

        required_fields = {"weight_packed", "weight_scale", "weight_shape"}
        missing = required_fields.difference(buf)
        if missing:
            raise KeyError(
                f"Compressed tensors for module '{prefix}' in shard '{path.name}' "
                f"are missing required fields: {sorted(missing)}"
            )

        weight = compressor.decompress_weight(
            compressed_data=buf,
            quantization_args=scheme.weights,
        )
        tensors[prefix + ".weight"] = finalize_for_save(weight, target_dtype)

    return tensors


def copy_aux_files(model_path: Path, output_path: Path, skip: Iterable[str]) -> None:
    for item in model_path.iterdir():
        if item.name in skip:
            continue
        target = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def dequantize_model(
    model_path: Path | str,
    output_path: Path | str,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> None:
    model_path = Path(model_path)
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"Output path {output_path} already exists")

    output_path.mkdir(parents=True)

    config = load_json(model_path / "config.json")
    quant_cfg = config.get("quantization_config", {}) or {}
    fmt = detect_format(model_path, config)

    files, index = list_safetensor_files(model_path)
    if not files:
        raise RuntimeError("No safetensor files to convert")

    device_str = normalize_device(device)
    if device_str is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU dequantization")
        torch.cuda.set_device(torch.device(device_str))
    open_device = device_str or "cpu"

    ignored_layers = resolve_ignored_layers(config)
    block_shape = resolve_block_size(config) if fmt == "fp8" else None
    fp8_scale_semantics = str(quant_cfg.get("weight_scale_semantics") or "heuristic").strip().lower()

    if block_shape is not None:
        LOG.debug("Configured FP8 block size %s found in quantization_config", block_shape)
    else:
        LOG.debug("No explicit FP8 block size found; will infer from scale tensors if needed")

    compressed_module_to_scheme: Dict[str, "QuantizationScheme"] = {}
    compressed_compressor: Optional["BaseCompressor"] = None
    if fmt == "compressed-pack":
        if not quant_cfg:
            raise ValueError(
                "compressed-tensors model requires a populated 'quantization_config' entry."
            )
        _, compressed_module_to_scheme, compressed_compressor = _prepare_compressed_tensors_context(
            model_path, quant_cfg
        )
        LOG.debug(
            "Prepared compressed-tensors context with %d modules",
            len(compressed_module_to_scheme),
        )

    log = setup_logger()
    LOG.debug(
        "Starting dequantization for model '%s' (format=%s, target_dtype=%s, device=%s)",
        model_path,
        fmt,
        target_dtype,
        open_device,
    )
    pb = log.pb(range(len(files))).manual().set(show_left_steps=False).title(f"Dequantizing ({fmt})")
    pb.draw()

    weight_map: Dict[str, str] = {}
    total_size = 0
    tensor_lookup = (
        _ShardTensorLookup(
            model_path=model_path,
            device=open_device,
            weight_map=index.get("weight_map", {}) if isinstance(index, dict) else None,
        )
        if fmt == "fp8"
        else None
    )

    try:
        for idx, filename in enumerate(files):
            path = model_path / filename
            LOG.debug("Processing shard '%s' for format %s on device %s", filename, fmt, open_device)
            if fmt == "fp8":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = convert_fp8_shard(
                        reader,
                        target_dtype,
                        block_shape=block_shape,
                        scale_semantics=fp8_scale_semantics,
                        tensor_lookup=tensor_lookup,
                        ignored_layers=ignored_layers,
                    )
            elif fmt == "bitsandbytes":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = convert_bitsandbytes_shard(
                        reader,
                        target_dtype,
                        quant_cfg=quant_cfg,
                        ignored_layers=ignored_layers,
                    )
            elif fmt == "nvfp4":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = convert_nvfp4_shard(
                        reader,
                        target_dtype,
                        ignored_layers=ignored_layers,
                    )
            elif fmt == "awq":
                tensors = convert_awq_file(
                    path,
                    target_dtype,
                    open_device,
                    ignored_layers=ignored_layers,
                )
            elif fmt == "gptq":
                tensors = convert_gptq_file(
                    path,
                    target_dtype,
                    quant_cfg,
                    open_device,
                    ignored_layers=ignored_layers,
                )
            elif fmt == "compressed-pack":
                if compressed_compressor is None:
                    raise RuntimeError("Compressed-tensors compressor was not initialized")
                tensors = convert_compressed_pack_file(
                    path,
                    target_dtype,
                    device=open_device,
                    module_to_scheme=compressed_module_to_scheme,
                    compressor=compressed_compressor,
                    ignored_layers=ignored_layers,
                )
            else:
                raise ValueError(f"Unsupported format {fmt}")

            if tensors:
                save_file(tensors, str(output_path / filename))
                weight_map.update({str(name): filename for name in tensors})
                total_size += sum(t.element_size() * t.numel() for t in tensors.values())
            else:
                # Auxiliary-only shards disappear once dense weights are emitted elsewhere.
                LOG.debug("Skipping empty output shard '%s' after dequantization", filename)
            pb.subtitle(filename).next().draw()
    finally:
        if tensor_lookup is not None:
            tensor_lookup.close()
        pb.close()

    if index is not None:
        new_index = dict(index)
    else:
        new_index = {}
    metadata = dict(new_index.get("metadata", {}))
    metadata["total_size"] = total_size
    new_index["metadata"] = metadata
    new_index["weight_map"] = weight_map
    write_json(output_path / "model.safetensors.index.json", new_index)

    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config.pop("torch_dtype", None)
    new_config["dtype"] = str(target_dtype).split(".")[-1]
    write_json(output_path / "config.json", new_config)

    skip_files = set(files) | {"config.json", "model.safetensors.index.json"}
    copy_aux_files(model_path, output_path, skip_files)
