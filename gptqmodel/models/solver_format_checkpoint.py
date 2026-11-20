# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Safetensors persistence of resolved solver/format solutions, not match rules."""

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, GenerationConfig

from ..nn_modules.qlinear.mxfp4 import MXFP4Linear
from ..quantization.formats import MXFP4


METHOD = "solver_format"


def read_manifest(path):
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open() as handle:
        value = json.load(handle).get("quantization_config", {})
    return value if isinstance(value, dict) else {}


def _state_and_aliases(model):
    tensors, aliases, owners = {}, {}, {}
    for name, value in model.state_dict().items():
        key = (value.untyped_storage().data_ptr(), value.storage_offset(), value.shape, value.stride(), value.dtype)
        if value.numel() and key in owners:
            aliases[name] = owners[key]
        else:
            owners[key] = name
            tensors[name] = value
    return tensors, aliases


def save_checkpoint(wrapper, save_dir, *, max_shard_size="5GB", metadata=None):
    from ..quantization.targeting import replay_buffer_errors

    if errors := replay_buffer_errors(wrapper.model):
        raise ValueError("; ".join(errors))
    destination = Path(save_dir).absolute()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Solver/format save requires a new or empty directory for atomic publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        state, aliases = _state_and_aliases(wrapper.model)
        manifest = {"quant_method": METHOD, "schema_version": 1,
                    "modules": copy.deepcopy(wrapper._format_solutions), "aliases": aliases}
        for name, spec in manifest["modules"].items():
            module = wrapper.model.get_submodule(name)
            if not isinstance(module, MXFP4Linear) or spec["format"] != MXFP4().to_dict():
                raise ValueError(f"Solution manifest does not own a supported packed module: {name}")
            if (module.qweight.dtype != torch.uint8 or module.scales.dtype != torch.uint8
                    or module.qweight.shape != (spec["out_features"], spec["in_features"] // 2)
                    or module.scales.shape != (spec["out_features"], spec["in_features"] // 32)
                    or (module.scales == 255).any()):
                raise ValueError(f"Invalid MXFP4 storage: {name}")
        header = {} if metadata is None else {str(key): str(value) for key, value in metadata.items()}
        header.update({"format": "pt", "gptqmodel_format": METHOD, "schema_version": "1",
                       "solutions": json.dumps(manifest["modules"], sort_keys=True)})
        shards = split_torch_state_dict_into_shards(state, max_shard_size=max_shard_size or "5GB")
        for filename, names in shards.filename_to_tensors.items():
            save_file({name: state[name].contiguous() for name in names}, staging / filename, metadata=header)
        if shards.is_sharded:
            with (staging / "model.safetensors.index.json").open("w") as handle:
                json.dump({"metadata": shards.metadata, "weight_map": shards.tensor_to_filename}, handle)
        config = copy.deepcopy(wrapper.model.config)
        config.quantization_config = manifest
        config.dtype = next(wrapper.model.parameters()).dtype
        config.save_pretrained(staging)
        if wrapper.tokenizer is not None:
            wrapper.tokenizer.save_pretrained(staging)
        generation = getattr(wrapper.model, "generation_config", None)
        if generation is not None:
            generation.save_pretrained(staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _read_state(path, manifest):
    root = Path(path).resolve()
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        with index_path.open() as handle:
            weight_map = json.load(handle)["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = ["model.safetensors"]
    state = {}
    for filename in filenames:
        shard = (root / filename).resolve()
        if shard.parent != root:
            raise ValueError("Checkpoint shard path escapes its model directory")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
            if (header.get("gptqmodel_format") != METHOD or header.get("schema_version") != "1"
                    or json.loads(header.get("solutions", "{}")) != manifest["modules"]):
                raise ValueError("Safetensors format metadata does not match the solution manifest")
            for name in handle.keys():
                if name in state or (weight_map is not None and weight_map.get(name) != filename):
                    raise ValueError(f"Duplicate or incorrectly indexed checkpoint tensor: {name}")
                state[name] = handle.get_tensor(name)
    if weight_map is not None and state.keys() != weight_map.keys():
        raise ValueError("Checkpoint index has missing tensor payloads")
    for alias, owner in manifest["aliases"].items():
        if alias in state or owner not in state:
            raise ValueError("Invalid checkpoint tensor alias")
        state[alias] = state[owner]
    return state


def load_checkpoint(model_cls, path, *, dtype=None):
    manifest = read_manifest(path)
    if (set(manifest) != {"quant_method", "schema_version", "modules", "aliases"}
            or manifest["quant_method"] != METHOD or manifest["schema_version"] != 1 or not manifest["modules"]):
        raise ValueError("Unsupported solver/format checkpoint schema")
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if not model_cls.__dict__.get("solver_format_support"):
        raise ValueError("Model lifecycle is not qualified for solver/format checkpoints")
    stored_dtype = config.dtype
    if isinstance(stored_dtype, str):
        stored_dtype = getattr(torch, stored_dtype)
    if isinstance(dtype, torch.dtype) and dtype != stored_dtype:
        raise ValueError("Solver/format reload currently preserves checkpoint precision; omit dtype")
    del config.quantization_config
    config._attn_implementation = "eager"
    from .loader import model_load_lock

    with model_load_lock(), init_empty_weights(include_buffers=False):
        model = model_cls.loader.from_config(config, trust_remote_code=False, dtype=stored_dtype)
    for name, spec in manifest["modules"].items():
        if spec["format"] != MXFP4().to_dict():
            raise ValueError(f"Unsupported target format/layout for {name}")
        module = model.get_submodule(name)
        if (not isinstance(module, torch.nn.Linear)
                or (module.in_features, module.out_features, module.bias is not None)
                != (spec["in_features"], spec["out_features"], spec["bias"])):
            raise ValueError(f"Checkpoint geometry does not match model: {name}")
        bias = torch.zeros(module.out_features, dtype=stored_dtype) if spec["bias"] else None
        model.set_submodule(name, MXFP4Linear(module.in_features, module.out_features, bias))
    state = _read_state(path, manifest)
    for name in manifest["modules"]:
        if (state[name + ".qweight"].dtype != torch.uint8 or state[name + ".scales"].dtype != torch.uint8
                or (state[name + ".scales"] == 255).any()):
            raise ValueError(f"Invalid packed MXFP4 storage: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    del state
    model.eval()
    if (Path(path) / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(path)
    tokenizer = None
    if (Path(path) / "tokenizer_config.json").exists():
        from ..utils.hf import load_hf_tokenizer

        tokenizer = load_hf_tokenizer(path, model_config=config, trust_remote_code=False)
    model.seqlen = getattr(config, "max_position_embeddings", 4096)
    wrapper = model_cls(model=model, quantized=True, quantize_config=None, load_quantized_model=True,
                        tokenizer=tokenizer, model_local_path=str(path))
    wrapper._format_solutions = manifest["modules"]
    return wrapper
