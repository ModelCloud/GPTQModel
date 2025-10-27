# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Safetensor-level dequantization helpers for common quant formats."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..quantization.dtype import dequantize_f4_e2m1, dequantize_f8_e4m3
from ..utils.logger import setup_logger


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _list_safetensor_files(model_path: Path) -> Tuple[list, Optional[dict]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        files = sorted(set(index.get("weight_map", {}).values()))
        return files, index

    files = sorted([p.name for p in model_path.glob("*.safetensors")])
    return files, None


def _finalize_for_save(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast to ``target_dtype`` when floating point and move to CPU with optimal layout."""

    if torch.is_floating_point(tensor):
        tensor = tensor.to(target_dtype)

    tensor_cpu = tensor.to("cpu")
    if tensor_cpu.ndim >= 4:
        tensor_cpu = tensor_cpu.contiguous(memory_format=torch.channels_last)
    return tensor_cpu


def _normalize_device(device: Optional[str]) -> Optional[str]:
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


def _resolve_block_size(config: dict) -> tuple[int, int]:
    quant_cfg = config.get("quantization_config", {}) or {}
    block_size = quant_cfg.get("weight_block_size")
    if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])
    return (128, 128)


def _detect_format(model_path: Path, config: dict) -> str:
    quant_cfg = config.get("quantization_config", {}) or {}
    method = (quant_cfg.get("quant_method") or "").lower()
    fmt = (quant_cfg.get("fmt") or "").lower()

    files, _ = _list_safetensor_files(model_path)
    if not files:
        raise FileNotFoundError("No .safetensors files found in model directory")

    with safe_open(model_path / files[0], framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        # Prefer dtype-based detection
        for key in keys:
            if key.endswith(".weight"):
                tensor = reader.get_tensor(key)
                if tensor.dtype == torch.float8_e4m3fn:
                    return "fp8"
                if tensor.dtype == torch.uint8 and (key + "_scale") in keys:
                    return "nvfp4"
        if any(k.endswith(".weight_scale") for k in keys):
            return "nvfp4"
        if any(k.endswith(".weight_scale_inv") for k in keys):
            return "fp8"
        if any(k.endswith(".qweight") for k in keys):
            has_g = any(k.endswith(".g_idx") for k in keys)
            return "gptq" if has_g else "awq"

    if fmt == "float8_e4m3fn":
        return "fp8"
    if method in ("gptq", "gptqmodel"):
        return "gptq"
    if method == "awq":
        return "awq"

    raise ValueError("Unable to detect quantization format for model")


def _unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows, cols * pack_factor, dtype=torch.int32)
    for i in range(pack_factor):
        result[:, i::pack_factor] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def _unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows * pack_factor, cols, dtype=torch.int32)
    for i in range(pack_factor):
        result[i::pack_factor, :] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def _convert_fp8_shard(
    reader,
    target_dtype: torch.dtype,
    *,
    block_shape: tuple[int, int],
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    block_rows, block_cols = block_shape

    for key in reader.keys():
        tensor = reader.get_tensor(key)
        if key.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            if scale_key not in reader.keys():
                raise KeyError(f"Missing scale inverse tensor for {key}")
            scale_inv = reader.get_tensor(scale_key)

            rows, cols = tensor.shape
            if rows % block_rows != 0 or cols % block_cols != 0:
                raise ValueError(
                    f"Tensor {key} shape {tensor.shape} incompatible with block size {block_shape}"
                )

            deq = dequantize_f8_e4m3(
                tensor,
                scale_inv=scale_inv,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = _finalize_for_save(deq, target_dtype)
        elif key.endswith("_scale_inv"):
            continue
        else:
            tensors[key] = _finalize_for_save(tensor, target_dtype)
    return tensors


def _convert_nvfp4_shard(reader, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        tensor = reader.get_tensor(key)
        if key.endswith(".weight") and tensor.dtype == torch.uint8:
            scale_key = key + "_scale"
            if scale_key not in reader.keys():
                raise KeyError(f"Missing scale tensor for {key}")
            scale = reader.get_tensor(scale_key)
            deq = dequantize_f4_e2m1(
                tensor,
                scale=scale,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = _finalize_for_save(deq, target_dtype)
        elif key.endswith("_weight_scale"):
            continue
        else:
            tensors[key] = _finalize_for_save(tensor, target_dtype)
    return tensors


def _convert_awq_file(path: Path, target_dtype: torch.dtype, device: str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
            else:
                tensors[key] = _finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete AWQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        bits = 4

        unpacked_weight = _unpack_cols(qweight, bits).to(torch.float32)
        unpacked_zeros = _unpack_cols(qzeros, bits).to(torch.float32)

        num_groups = scales.shape[0]
        group_size = unpacked_weight.shape[0] // num_groups
        scales_full = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
        zeros_full = unpacked_zeros.repeat_interleave(group_size, dim=0)

        weight = (unpacked_weight - zeros_full) * scales_full
        tensors[prefix + ".weight"] = _finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = _finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def _convert_gptq_file(path: Path, target_dtype: torch.dtype, config: dict, device: str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
            elif key.endswith(".g_idx"):
                prefix = key[:-len(".g_idx")]
                module_buffers[prefix]["g_idx"] = tensor
            else:
                tensors[key] = _finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales", "g_idx") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete GPTQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        g_idx = buf["g_idx"].to(torch.long)

        bits = config.get("bits", 4)
        weight_int = _unpack_rows(qweight, bits)
        zeros = _unpack_cols(qzeros, bits)

        scales_full = scales.to(torch.float32)[g_idx]
        zeros_full = zeros.to(torch.float32)[g_idx]
        weight = (weight_int.to(torch.float32) - zeros_full) * scales_full
        tensors[prefix + ".weight"] = _finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = _finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def _copy_aux_files(model_path: Path, output_path: Path, skip: Iterable[str]) -> None:
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

    config = _load_json(model_path / "config.json")
    quant_cfg = config.get("quantization_config", {}) or {}
    fmt = _detect_format(model_path, config)

    files, index = _list_safetensor_files(model_path)
    if not files:
        raise RuntimeError("No safetensor files to convert")

    device_str = _normalize_device(device)
    if device_str is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU dequantization")
        torch.cuda.set_device(torch.device(device_str))
    open_device = device_str or "cpu"

    block_shape = _resolve_block_size(config) if fmt == "fp8" else None

    log = setup_logger()
    pb = log.pb(range(len(files))).manual().set(show_left_steps=False).title(f"Dequantizing ({fmt})")
    pb.draw()

    weight_map: Dict[str, str] = {}
    total_size = 0

    try:
        for idx, filename in enumerate(files):
            path = model_path / filename
            if fmt == "fp8":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    if block_shape is None:
                        raise RuntimeError("FP8 conversion requires block_shape metadata")
                    tensors = _convert_fp8_shard(reader, target_dtype, block_shape=block_shape)
            elif fmt == "nvfp4":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = _convert_nvfp4_shard(reader, target_dtype)
            elif fmt == "awq":
                tensors = _convert_awq_file(path, target_dtype, open_device)
            elif fmt == "gptq":
                tensors = _convert_gptq_file(path, target_dtype, quant_cfg, open_device)
            else:
                raise ValueError(f"Unsupported format {fmt}")

            save_file(tensors, str(output_path / filename))
            weight_map.update({str(name): filename for name in tensors})
            total_size += sum(t.element_size() * t.numel() for t in tensors.values())
            pb.subtitle(filename).next().draw()
    finally:
        pb.close()

    if index is not None:
        new_index = dict(index)
    else:
        new_index = {}
    metadata = dict(new_index.get("metadata", {}))
    metadata["total_size"] = total_size
    new_index["metadata"] = metadata
    new_index["weight_map"] = weight_map
    _write_json(output_path / "model.safetensors.index.json", new_index)

    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config["torch_dtype"] = str(target_dtype).split(".")[-1]
    _write_json(output_path / "config.json", new_config)

    skip_files = set(files) | {"config.json", "model.safetensors.index.json"}
    _copy_aux_files(model_path, output_path, skip_files)


# Backwards compatibility with older imports.
dequantize = dequantize_model
