# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4387 item 3: FP8 checkpoint load must not widen packed weights.

Accelerate casts every floating checkpoint tensor to the requested model
dtype.  For an FP8 checkpoint that materializes each packed ``weight`` as
FP16/BF16 on the target device until the loader re-reads the FP8 bytes, so
the load peak is roughly twice the resident footprint.  The loader now hands
Accelerate a temporary checkpoint without the packed weights and restores
those bytes straight from the source checkpoint.
"""

from __future__ import annotations

import gc
import json
import os

import accelerate
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.utils.model import (
    _restore_checkpoint_buffers,
    load_checkpoint_in_model_then_tie_weights,
)


class _FP8LoadModel(torch.nn.Module):
    def __init__(self, *, count: int = 2, size: int = 32, fmt: str = "float8_e4m3fn"):
        super().__init__()
        self.fmt = fmt
        self.projs = torch.nn.ModuleList(
            [
                TorchFP8Linear(
                    bits=8,
                    group_size=-1,
                    sym=True,
                    desc_act=False,
                    in_features=size,
                    out_features=size,
                    bias=True,
                    register_buffers=True,
                    format=fmt,
                    weight_scale_method="row",
                )
                for _ in range(count)
            ]
        )
        self.dense = torch.nn.Linear(size, size, bias=True)
        self.embed = torch.nn.Embedding(size, size)

    def tie_weights(self, **kwargs):
        del kwargs

    def forward(self, x):
        return self.dense(self.projs[0](x))


def _fp8_modules(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TorchFP8Linear)
    ]


def _payload(model, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in _fp8_modules(model):
        prefix = f"{name}."
        tensors[prefix + "weight"] = (
            torch.randn(module.out_features, module.in_features, generator=generator) * 8
        ).to(module.fp8_dtype)
        tensors[prefix + "weight_scale_inv"] = (
            torch.rand(module.out_features, generator=generator) + 0.5
        ).to(torch.float32)
        tensors[prefix + "bias"] = (
            torch.randn(module.out_features, generator=generator) * 100
        ).to(torch.float32)
    for name, module in model.named_modules():
        if not name or isinstance(module, TorchFP8Linear):
            continue
        if isinstance(module, torch.nn.Linear):
            tensors[f"{name}.weight"] = torch.randn(
                module.out_features, module.in_features, generator=generator
            ).to(torch.bfloat16)
            tensors[f"{name}.bias"] = torch.randn(
                module.out_features, generator=generator
            ).to(torch.bfloat16)
        elif isinstance(module, torch.nn.Embedding):
            tensors[f"{name}.weight"] = torch.randn(
                module.num_embeddings, module.embedding_dim, generator=generator
            ).to(torch.bfloat16)
    return tensors


def _load_options(model):
    fp8_dtypes = {f"{name}.weight": module.fp8_dtype for name, module in _fp8_modules(model)}
    metadata = {
        f"{name}.{buffer}"
        for name, _ in _fp8_modules(model)
        for buffer in ("weight_scale_inv", "bias")
    }
    return fp8_dtypes, sorted(metadata)


def _raw(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def _assert_bitwise(actual, expected):
    actual_state = actual.state_dict()
    assert set(actual_state) == set(expected)
    for name, expected_tensor in expected.items():
        loaded = actual_state[name]
        assert loaded.dtype == expected_tensor.dtype, name
        assert torch.equal(_raw(loaded), _raw(expected_tensor)), name


def _checkpoint_keys(path):
    path = os.fspath(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            return set(json.load(handle)["weight_map"])
    with safe_open(path, framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _capture_accelerate_checkpoint(monkeypatch):
    seen = {"calls": 0}
    real = accelerate.load_checkpoint_in_model

    def wrapper(model, *args, **kwargs):
        seen["calls"] += 1
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is None and args:
            checkpoint = args[0]
        seen["keys"] = _checkpoint_keys(checkpoint)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(accelerate, "load_checkpoint_in_model", wrapper)
    return seen


def _old_path_load(model, checkpoint, fp8_dtypes, metadata_names, device_map):
    """Reproduce the pre-fix load: Accelerate widens FP8, then the loader restores it."""

    accelerate.load_checkpoint_in_model(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    restore = dict(fp8_dtypes)
    restore.update(dict.fromkeys(metadata_names, torch.float32))
    _restore_checkpoint_buffers(
        model,
        checkpoint,
        restore,
        exact_source_names=set(fp8_dtypes),
    )


def _save_sharded(tmp_path, tensors, fp8_names):
    metadata_shard = "model-00001-of-00002.safetensors"
    packed_shard = "model-00002-of-00002.safetensors"
    metadata_tensors = {f"hf_{name}": value for name, value in tensors.items() if name not in fp8_names}
    packed_tensors = {f"hf_{name}": value for name, value in tensors.items() if name in fp8_names}
    save_file(metadata_tensors, tmp_path / metadata_shard, metadata={"format": "pt"})
    save_file(packed_tensors, tmp_path / packed_shard, metadata={"format": "pt"})
    weight_map = dict.fromkeys(metadata_tensors, metadata_shard)
    weight_map.update(dict.fromkeys(packed_tensors, packed_shard))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return index


def test_fp8_load_keeps_packed_weights_out_of_accelerate(tmp_path, monkeypatch):
    model = _FP8LoadModel()
    tensors = _payload(model)
    fp8_dtypes, metadata_names = _load_options(model)
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint, metadata={"format": "pt"})

    seen = _capture_accelerate_checkpoint(monkeypatch)
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        keep_in_fp32_modules=metadata_names,
        fp32_bf16_buffer_names=metadata_names,
        packed_fp8_buffer_dtypes=fp8_dtypes,
    )

    # #5584: an FP8 checkpoint now takes the lazy placement path, so Accelerate
    # is not called at all; the copy fallback still hands it a checkpoint
    # without the packed weights. Either way it never sees them.
    accelerate_keys = seen.get("keys", set())
    assert set(fp8_dtypes).isdisjoint(accelerate_keys)
    if seen["calls"]:
        assert set(tensors) - set(fp8_dtypes) <= accelerate_keys
    assert all(
        module.weight.dtype == module.fp8_dtype for _, module in _fp8_modules(model)
    )
    _assert_bitwise(model, tensors)

    reference = _FP8LoadModel()
    _old_path_load(
        reference, checkpoint, fp8_dtypes, metadata_names, device_map={"": "cpu"}
    )
    _assert_bitwise(model, reference.state_dict())

    # The temporary Accelerate-only checkpoint must not outlive the load.
    assert sorted(os.listdir(tmp_path)) == ["model.safetensors"]


def test_fp8_load_subset_handles_shards_and_key_conversion(tmp_path, monkeypatch):
    model = _FP8LoadModel()
    tensors = _payload(model)
    fp8_dtypes, metadata_names = _load_options(model)
    index = _save_sharded(tmp_path, tensors, set(fp8_dtypes))

    seen = _capture_accelerate_checkpoint(monkeypatch)
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(index),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        keep_in_fp32_modules=metadata_names,
        fp32_bf16_buffer_names=metadata_names,
        packed_fp8_buffer_dtypes=fp8_dtypes,
        key_converter=lambda name: name.removeprefix("hf_"),
    )

    # #5584: an FP8 checkpoint now takes the lazy placement path, so Accelerate
    # is not called at all; the copy fallback still hands it a checkpoint
    # without the packed weights. Either way it never sees them.
    accelerate_keys = seen.get("keys", set())
    assert set(fp8_dtypes).isdisjoint(accelerate_keys)
    if seen["calls"]:
        assert set(tensors) - set(fp8_dtypes) <= accelerate_keys
    assert all(
        module.weight.dtype == module.fp8_dtype for _, module in _fp8_modules(model)
    )
    _assert_bitwise(model, tensors)

    # The packed-only shard is skipped and the temporary directory is removed.
    assert not [
        name for name in os.listdir(tmp_path) if name.startswith("gptqmodel-key-remap-")
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_fp8_load_peak_drops_and_skips_model_dtype_materialization(tmp_path):
    count, size = 64, 512
    model = _FP8LoadModel(count=count, size=size)
    tensors = _payload(model)
    fp8_dtypes, metadata_names = _load_options(model)
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint, metadata={"format": "pt"})
    fp8_bytes = sum(
        tensors[name].numel() * tensors[name].element_size() for name in fp8_dtypes
    )

    old_model = _FP8LoadModel(count=count, size=size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _old_path_load(old_model, checkpoint, fp8_dtypes, metadata_names, {"": "cuda"})
    old_peak = torch.cuda.max_memory_allocated()
    del old_model
    gc.collect()
    torch.cuda.empty_cache()

    new_model = _FP8LoadModel(count=count, size=size)
    torch.cuda.reset_peak_memory_stats()
    load_checkpoint_in_model_then_tie_weights(
        new_model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        keep_in_fp32_modules=metadata_names,
        fp32_bf16_buffer_names=metadata_names,
        packed_fp8_buffer_dtypes=fp8_dtypes,
    )
    new_peak = torch.cuda.max_memory_allocated()

    assert all(
        module.weight.dtype == module.fp8_dtype
        and module.weight.device.type == "cuda"
        for _, module in _fp8_modules(new_model)
    )
    assert new_peak < old_peak
    # The old path held the widened copy next to the packed shard.
    assert old_peak - new_peak >= int(0.8 * fp8_bytes)
    _assert_bitwise(new_model, tensors)
