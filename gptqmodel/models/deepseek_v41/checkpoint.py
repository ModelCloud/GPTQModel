# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Indexed loading of the original FP8/FP4 DeepSeek V4.1 shards."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import infer_auto_device_map
from safetensors import safe_open
from transformers.utils.hub import cached_file, cached_files, extract_commit_hash

from ...quantization.dtype import is_fp4_packed_dtype
from ...utils.torch import as_torch_device
from .configuration import DeepSeekV41FlashConfig
from .modeling import DeepSeekV41FlashForCausalLM


def _runtime_key(name):
    if name.startswith("mtp."):
        return None
    if name.startswith("layers."):
        layer, index, leaf = name.split(".", 2)
        if leaf.startswith("engram."):
            return None
        for branch in ("attn", "ffn"):
            prefix = f"hc_{branch}_"
            if leaf.startswith(prefix):
                return f"{layer}.{index}.{branch}_hc.{leaf[len(prefix):]}"
    return name


def _checkpoint_tensors(index, expected):
    shards = defaultdict(dict)
    mapped = set()
    for source, shard in index["weight_map"].items():
        target = _runtime_key(source)
        if target is None:
            continue
        if Path(shard).name != shard or "\\" in shard or not shard.endswith(".safetensors"):
            raise ValueError(f"Invalid DeepSeek V4.1 shard name {shard!r}")
        if target in mapped or target not in expected:
            raise ValueError(f"Unexpected DeepSeek V4.1 checkpoint tensor {source}")
        shards[shard][source] = target
        mapped.add(target)
    missing = expected.keys() - mapped
    if missing:
        raise ValueError(f"Missing DeepSeek V4.1 checkpoint tensor {min(missing)} ({len(missing)} missing)")
    return shards


def _get_placements(model, expected, device, device_map, max_memory):
    if device_map is None:
        device_map = {"": device} if device is not None else "auto"
    if max_memory is not None and device_map != "auto":
        raise ValueError('DeepSeek V4.1 max_memory requires device_map="auto"')
    if device_map == "auto":
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=model._no_split_modules,
        )
    if not isinstance(device_map, dict):
        raise ValueError('DeepSeek V4.1 device_map must be a mapping or "auto"')

    placement = {}
    layer_devices = {}
    for name in expected:
        path = name
        while path not in device_map and path:
            path = path.rpartition(".")[0]
        if path not in device_map:
            raise ValueError(f"DeepSeek V4.1 device map has no device for {name}")
        target = device_map[path]
        if target == "disk":
            raise ValueError("DeepSeek V4.1 requires memory for its packed weights. Disk offload is unavailable.")
        target = as_torch_device(target)
        placement[name] = target
        parts = name.split(".")
        unit = ".".join(parts[:2]) if parts[0] == "layers" else parts[0]
        if layer_devices.setdefault(unit, target) != target:
            raise ValueError(f"DeepSeek V4.1 requires a single device for {unit}")
    return device_map, placement


def _read_tensor(shard, source, template, device):
    value = shard.get_tensor(source)
    if value.shape != template.shape:
        raise ValueError(f"DeepSeek V4.1 tensor {source} has shape {tuple(value.shape)}, expected {tuple(template.shape)}")
    target_dtype = template.dtype
    if target_dtype == torch.int8 and is_fp4_packed_dtype(value.dtype):
        value = value.view(torch.int8)
    elif target_dtype == torch.float8_e8m0fnu and value.dtype == torch.uint8:
        value = value.view(torch.float8_e8m0fnu)
    if value.dtype != target_dtype and not (
        target_dtype in (torch.bfloat16, torch.float32) and value.dtype in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(f"DeepSeek V4.1 tensor {source} has dtype {value.dtype}, expected {target_dtype}")
    return value.to(device=device, dtype=target_dtype)


def load_checkpoint(checkpoint, *, selection, dtype, device, device_map, max_memory):
    index_name = "model.safetensors.index.json"
    index_path = Path(cached_file(checkpoint, index_name, **selection))
    revision = extract_commit_hash(str(index_path), None)
    if revision is not None:
        selection = {**selection, "revision": revision}
    config = DeepSeekV41FlashConfig.from_pretrained(checkpoint, **selection)
    format_config = getattr(config, "quantization_config", {})
    if (
        format_config.get("quant_method") != "fp8"
        or format_config.get("expert_dtype") != "fp4"
        or format_config.get("scale_fmt") != "ue8m0"
        or tuple(format_config.get("weight_block_size", ())) != (32, 32)
        or format_config.get("activation_scheme") != "dynamic"
    ):
        raise ValueError("DeepSeek V4.1 Flash requires the published block-32 FP8 / FP4 checkpoint format")

    if dtype in (None, "auto"):
        dtype = config.dtype or torch.bfloat16
    if dtype != torch.bfloat16:
        raise ValueError("DeepSeek V4.1 Flash uses the checkpoint's bfloat16 compute dtype")
    config.dtype = dtype
    config.text_config.engram_layer_ids = []
    config.text_config.engram_num_embeddings = []
    config.text_config.num_nextn_predict_layers = 0
    config.text_config.dspark_target_layer_ids = []

    with torch.device("meta"):
        model = DeepSeekV41FlashForCausalLM(config, dtype=dtype)
    expected = model.state_dict(keep_vars=True)
    shards = _checkpoint_tensors(json.loads(index_path.read_text(encoding="utf-8")), expected)
    device_map, placements = _get_placements(model, expected, device, device_map, max_memory)
    filenames = cached_files(checkpoint, list(shards), **selection)
    for filename in filenames:
        with safe_open(filename, framework="pt", device="cpu") as shard:
            for source, target in shards[Path(filename).name].items():
                value = _read_tensor(shard, source, expected[target], placements[target])
                parent, _, leaf = target.rpartition(".")
                module = model.get_submodule(parent)
                if isinstance(expected[target], torch.nn.Parameter):
                    value = torch.nn.Parameter(value, requires_grad=False)
                setattr(module, leaf, value)
    model.hf_device_map = device_map
    return model.eval(), str(index_path.parent), selection
