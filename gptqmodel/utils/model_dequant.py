# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for converting FP8-quantized models to higher precision."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..quantization.dtype import dequantize_f8_e4m3
from ..utils.logger import setup_logger


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _is_fp8_format(config: dict) -> bool:
    quant_cfg = config.get("quantization_config", {})
    fmt = quant_cfg.get("fmt")
    return fmt == "float8_e4m3fn"


def _resolve_block_size(config: dict) -> tuple[int, int]:
    quant_cfg = config.get("quantization_config", {})
    block_size = quant_cfg.get("weight_block_size")
    if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])
    return (128, 128)


def _dequantize_shard(
    shard_path: Path,
    output_path: Path,
    *,
    target_dtype: torch.dtype,
    block_shape: tuple[int, int],
    device: Optional[str],
) -> None:
    tensors = {}
    open_device = device or "cpu"

    with safe_open(shard_path, framework="pt", device=open_device) as reader:
        for name in reader.keys():
            tensor = reader.get_tensor(name)

            if tensor.dtype is torch.float8_e4m3fn:
                scale_inv_name = name + "_scale_inv"
                if scale_inv_name not in reader.keys():
                    tensors[name] = tensor.to(target_dtype).to("cpu")
                    continue

                rows, cols = tensor.shape
                block_rows, block_cols = block_shape
                if rows % block_rows != 0 or cols % block_cols != 0:
                    raise ValueError(
                        f"Tensor {name} shape {tensor.shape} incompatible with block size {block_shape}"
                    )

                scale_inv = reader.get_tensor(scale_inv_name)
                deq = dequantize_f8_e4m3(
                    tensor,
                    scale_inv=scale_inv,
                    axis=None,
                    target_dtype=target_dtype,
                )
                if deq.ndimension() >= 4:
                    tensors[name] = deq.to("cpu", memory_format=torch.channels_last)
                else:
                    tensors[name] = deq.to("cpu")
            elif tensor.dtype is torch.uint8 and name.endswith(".weight"):
                converted = tensor.to(target_dtype)
                if converted.ndimension() >= 4:
                    tensors[name] = converted.to("cpu", memory_format=torch.channels_last)
                else:
                    tensors[name] = converted.to("cpu")
            else:
                if tensor.ndimension() >= 4 and (device is None or tensor.device.type == "cpu"):
                    tensors[name] = tensor.to("cpu", memory_format=torch.channels_last)
                else:
                    tensors[name] = tensor.to("cpu")

    save_file(tensors, str(output_path))


def dequantize(
    model_path: str | Path,
    model_output_path: str | Path,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> None:
    """Dequantize an FP8 E4M3 model into the requested ``target_dtype``.

    Parameters
    ----------
    model_path:
        Directory containing ``model.safetensors`` shards and ``config.json``.
    model_output_path:
        Destination directory for the dequantized model.
    target_dtype:
        Desired floating point dtype (defaults to ``torch.bfloat16``).
    """

    model_path = Path(model_path)
    model_output_path = Path(model_output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    if model_output_path.exists():
        raise FileExistsError(f"Output path {model_output_path} already exists")

    config = _load_json(model_path / "config.json")
    if not _is_fp8_format(config):
        raise ValueError("Model does not advertise float8_e4m3fn quantization")

    block_shape = _resolve_block_size(config)

    device_str = _normalize_device(device)
    if device_str is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU dequantization")
        torch.cuda.set_device(torch.device(device_str))

    output_path = model_output_path
    output_path.mkdir(parents=True, exist_ok=False)

    index_path = model_path / "model.safetensors.index.json"
    index = _load_json(index_path)
    weight_map: dict[str, str] = index.get("weight_map", {})

    shard_names = sorted(set(weight_map.values()))

    new_weight_map = {}
    log = setup_logger()
    pb = (
        log.pb(range(len(shard_names)))
           .manual()
           .set(show_left_steps=False)
           .title("Dequantizing FP8 shards")
    )
    pb.draw()

    for shard in shard_names:
        shard_path = model_path / shard
        output_shard = output_path / shard
        output_shard.parent.mkdir(parents=True, exist_ok=True)

        _dequantize_shard(
            shard_path,
            output_shard,
            target_dtype=target_dtype,
            block_shape=block_shape,
            device=device_str,
        )

        new_weight_map.update({name: shard for name, shard in weight_map.items() if weight_map[name] == shard})
        pb.subtitle(shard).next().draw()

    pb.close()

    new_index = dict(index)
    new_index["weight_map"] = new_weight_map
    (output_path / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))

    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config["torch_dtype"] = target_dtype.__repr__().split(".")[-1]
    (output_path / "config.json").write_text(json.dumps(new_config, indent=2))

    skip_files = {"config.json", "model.safetensors.index.json"}.union(shard_names)

    for entry in model_path.iterdir():
        if entry.name in skip_files:
            continue
        target = output_path / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
