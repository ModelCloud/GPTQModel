# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Write DeepSeek V4.1 checkpoints with unchanged source tensors by range."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ...quantization.config import FORMAT
from ...utils.model import OffloadTensorRef, TensorSource, streaming_state_dict_to_shards
from ...utils.offload import _read_safetensors_header
from ..writer import _parse_max_shard_size, _torch_dtype_from_safetensors_name


def _checkpoint_sources(model):
    replaced = model.config.gptqmodel_v41_projections or {}
    payload = {}
    headers = {}
    for name, shard in model.source_weight_map.items():
        owner = name.rpartition(".")[0]
        if owner in replaced or owner.rpartition(".projections.")[0] in replaced:
            continue
        path = model.source_shards[shard]
        if path not in headers:
            headers[path] = _read_safetensors_header(path)
        base, header = headers[path]
        entry = header[name]
        dtype = _torch_dtype_from_safetensors_name(entry["dtype"])
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        ref = OffloadTensorRef(path, dtype, shape, "safetensors", name, (base + start, base + end))
        if start < 0 or end - start != ref.num_bytes:
            raise ValueError(f"Invalid DeepSeek V4.1 checkpoint extent for {name}")
        payload[name] = TensorSource(name, dtype, shape, ref)
    for path in replaced:
        for leaf, tensor in model.get_submodule(path).state_dict().items():
            name = f"{path}.{leaf}"
            tensor = tensor.detach().contiguous()
            payload[name] = TensorSource(name, tensor.dtype, tuple(tensor.shape), tensor)
    return {name: payload[name] for name in sorted(payload)}


def save_checkpoint(wrapper, save_dir, *, safetensors_metadata, max_shard_size, meta_quantizer):
    model = wrapper.model
    destination = Path(save_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("DeepSeek V4.1 checkpoint destination must be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = model.config.to_dict()
    config.pop("auto_map", None)
    metadata = {str(key): str(value) for key, value in (safetensors_metadata or {}).items()}
    metadata.update(format="pt", quant_method="fp8")
    if model.config.gptqmodel_v41_projections:
        config["gptqmodel_v41_source_quantization"] = (
            config.get("gptqmodel_v41_source_quantization") or config["quantization_config"]
        )
        quantization = wrapper.quantize_config.to_dict()
        quantization["format"] = FORMAT.GPTQ_V2.value
        quantization["checkpoint_format"] = FORMAT.GPTQ_V2.value
        config["quantization_config"] = quantization
        metadata.update(quant_method="gptq", checkpoint_format=FORMAT.GPTQ_V2.value)
    if meta_quantizer is not None:
        metadata["quantizer"] = str(meta_quantizer)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as staging:
        _, weight_map, total_size = streaming_state_dict_to_shards(
            _checkpoint_sources(model), staging, "model", "model.safetensors", metadata,
            _parse_max_shard_size(max_shard_size),
        )
        root = Path(staging)
        for filename, data in (
            ("config.json", config),
            ("model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        ):
            (root / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        wrapper.tokenizer.save_pretrained(staging)
        if model.generation_config is not None:
            model.generation_config.save_pretrained(staging)
        os.replace(root, destination)
