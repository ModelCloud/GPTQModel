# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 native passthrough must not republish unloaded source state (#5029).

The source index describes the whole on-disk checkpoint, not the loaded model.
A reduced `num_hidden_layers` copy of an 8B checkpoint still indexes every layer
above the cut, and MTP / next-token heads are commonly present even though the
model never instantiates them. `_build_exllamav3_checkpoint_passthrough_state_dict`
used `set(weight_map) - replaced`, so those tensors were copied into the saved
checkpoint in their original dense dtype: 14086 MB / 375 tensors instead of
2453 MB / 123 on the issue's 4-layer Llama-3.1-8B fixture, and a 2.5x reload
peak.

Ownership is now decided from the loaded model: a source tensor is republished
only when its owning module exists (the model's own state covers it) or when a
runtime parameter/buffer resolves to it through the checkpoint source's
runtime -> checkpoint alias chain.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.models import writer
from gptqmodel.utils.structure import LazyTurtle


class _FakeEXL3(torch.nn.Module):
    QUANT_TYPE = "exl3"


PACKED_SUFFIXES = ("trellis", "suh", "svh", "mcg")

# Source entries a 2-layer model built over a 4-layer checkpoint never owns.
UNOWNED_SOURCE_NAMES = (
    "model.layers.1.proj.weight",
    "model.layers.2.norm.weight",
    "model.layers.2.proj.weight",
    "model.layers.3.norm.weight",
    "model.layers.3.proj.weight",
    "mtp.0.main_norm.weight",
    "mtp.0.proj.weight",
)

OWNED_NATIVE_NAMES = (
    "model.layers.0.norm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)


def _tensor_bytes(sources) -> int:
    total = 0
    for source in sources.values():
        element_size = torch.empty(0, dtype=source.torch_dtype).element_size()
        total += element_size * math.prod(int(dim) for dim in source.shape)
    return total


def _write_source(tmp_path):
    """A sharded source checkpoint with owned and unowned native entries."""

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shard = snapshot / "model-00001-of-00001.safetensors"
    native_norm = torch.tensor([1.25, -2.5], dtype=torch.float32)
    native_embed = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    native_head = torch.tensor([[0.5, -0.25], [1.0, 2.0], [-1.5, 0.75], [0.0, 3.0]], dtype=torch.float16)
    tensors = {
        "model.layers.0.proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "model.layers.0.norm.weight": native_norm,
        "model.embed_tokens.weight": native_embed,
        "lm_head.weight": native_head,
    }
    for name in UNOWNED_SOURCE_NAMES:
        tensors[name] = torch.full((2, 2), 7.0, dtype=torch.float32)
    save_file(tensors, shard)
    weight_map = dict.fromkeys(tensors, shard.name)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return snapshot, weight_map, tensors, {
        "model.layers.0.norm.weight": native_norm,
        "model.embed_tokens.weight": native_embed,
        "lm_head.weight": native_head,
    }


def _build_model():
    """A 2-layer model: layer 0 packed by EXL3, everything else native."""

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].proj = _FakeEXL3()
    for suffix in PACKED_SUFFIXES:
        dtype = torch.int16 if suffix == "trellis" else (
            torch.int32 if suffix == "mcg" else torch.float16
        )
        shape = () if suffix == "mcg" else (1,)
        model.model.layers[0].proj.register_buffer(
            suffix, torch.ones(shape, dtype=dtype)
        )
    model.model.layers[0].norm = torch.nn.LayerNorm(2)
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.lm_head = torch.nn.Linear(2, 4, bias=False)
    return model


def _storage():
    return {
        "model.layers.0.proj": {
            "stored_tensors": {
                f"model.layers.0.proj.{suffix}": {} for suffix in PACKED_SUFFIXES
            }
        }
    }


def _owner(snapshot, model, *, aliases=None):
    turtle = LazyTurtle(
        model_local_path=str(snapshot),
        config=SimpleNamespace(model_type="llama"),
        module_tree=None,
        target_model=None,
    )
    if aliases is not None:
        turtle.checkpoint_source_aliases = aliases
    return SimpleNamespace(model=model, turtle_model=turtle)


def test_unloaded_source_layers_are_not_republished(tmp_path):
    snapshot, weight_map, tensors, native = _write_source(tmp_path)
    owner = _owner(snapshot, _build_model())

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, _storage())
    assert plan is not None
    state = writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)

    packed = {f"model.layers.0.proj.{suffix}" for suffix in PACKED_SUFFIXES}
    assert set(state) == packed | set(OWNED_NATIVE_NAMES)
    # Nothing the 2-layer model does not have may leak into the checkpoint.
    assert not set(state).intersection(UNOWNED_SOURCE_NAMES)

    # Byte-for-byte fidelity for every retained native tensor.
    published = tmp_path / "published"
    published.mkdir()
    writer.streaming_state_dict_to_shards(
        state,
        save_dir=str(published),
        model_base_name="published",
        single_file_name="published.safetensors",
        metadata={"format": "pt"},
        max_shard_size=None,
    )
    with safe_open(
        published / "published.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        assert set(checkpoint.keys()) == packed | set(OWNED_NATIVE_NAMES)
        for name, expected in native.items():
            stored = checkpoint.get_tensor(name)
            assert stored.dtype == expected.dtype
            assert torch.equal(stored, expected), name

    # The saved payload is the model's own state, not the whole source index.
    def _raw_bytes(values):
        return sum(
            torch.empty(0, dtype=tensor.dtype).element_size() * tensor.numel()
            for tensor in values
        )

    source_bytes = _raw_bytes(tensors.values())
    owned_source_bytes = _raw_bytes(native.values())
    packed_bytes = sum(
        buffer.numel() * buffer.element_size()
        for _name, buffer in owner.model.named_buffers()
    )
    assert _tensor_bytes(state) == owned_source_bytes + packed_bytes
    assert _tensor_bytes(state) < source_bytes


def test_converted_source_names_are_kept_through_the_alias_chain(tmp_path):
    """A converted checkpoint spells owned tensors differently (#5029)."""

    snapshot, weight_map, _tensors, _native = _write_source(tmp_path)

    # The packed module keeps its source spelling, while the native tensors live
    # under the legacy `language_model` prefix a converted checkpoint uses.
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].proj = _FakeEXL3()
    for suffix in PACKED_SUFFIXES:
        dtype = torch.int16 if suffix == "trellis" else (
            torch.int32 if suffix == "mcg" else torch.float16
        )
        shape = () if suffix == "mcg" else (1,)
        model.model.layers[0].proj.register_buffer(
            suffix, torch.ones(shape, dtype=dtype)
        )
    model.language_model = torch.nn.Module()
    model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.language_model.layers[0].norm = torch.nn.LayerNorm(2)
    model.language_model.embed_tokens = torch.nn.Embedding(4, 2)
    model.lm_head = torch.nn.Linear(2, 4, bias=False)

    def aliases(runtime_name: str):
        # Runtime spelling is `language_model.*`; the checkpoint stores the same
        # tensors as `model.*`.
        candidates = [runtime_name]
        marker = "language_model."
        if runtime_name.startswith(marker):
            candidates.append("model." + runtime_name[len(marker):])
        return tuple(candidates)

    owner = _owner(snapshot, model, aliases=aliases)

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, _storage())
    assert plan is not None
    state = writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)

    # The renamed native tensors are still owned, so they must survive.
    assert "model.layers.0.norm.weight" in state
    assert "model.embed_tokens.weight" in state
    assert "lm_head.weight" in state
    assert not set(state).intersection(UNOWNED_SOURCE_NAMES)


def test_ownership_unknown_keeps_every_source_entry():
    """Fail closed: an unenumerable model must not drop native state."""

    assert (
        writer._exllamav3_owned_native_source_names(
            SimpleNamespace(), SimpleNamespace(), {"model.layers.0.norm.weight"}
        )
        is None
    )


def test_builder_keeps_every_entry_when_ownership_is_unknown(tmp_path, monkeypatch):
    snapshot, weight_map, _tensors, _native = _write_source(tmp_path)
    owner = _owner(snapshot, _build_model())
    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, _storage())
    assert plan is not None

    monkeypatch.setattr(
        writer, "_exllamav3_owned_native_source_names", lambda *args, **kwargs: None
    )
    state = writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)

    packed = {f"model.layers.0.proj.{suffix}" for suffix in PACKED_SUFFIXES}
    replaced = set(plan["replaced_source_names"])
    assert set(state) == packed | (set(weight_map) - replaced)
