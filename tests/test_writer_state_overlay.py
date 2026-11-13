# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel import ShardConfig, reshard
from gptqmodel.models import writer
from gptqmodel.utils import model as model_utils


class _Owner:
    def __init__(
        self,
        prefixes,
        expected_suffixes=None,
        remove_prefixes=None,
        remove_expected_suffixes=None,
    ):
        self.prefixes = prefixes
        self.expected_suffixes = expected_suffixes
        self.remove_prefixes = remove_prefixes
        self.remove_expected_suffixes = remove_expected_suffixes
        self.model = torch.nn.Linear(1, 1)

    def save_state_overlay(self):
        contract = {
            "model": torch.nn.Linear(1, 1),
            "offload_root": "/offload",
            "replace_prefixes": self.prefixes,
        }
        if self.expected_suffixes is not None:
            contract["expected_suffixes"] = self.expected_suffixes
        if self.remove_prefixes is not None:
            contract["remove_prefixes"] = self.remove_prefixes
        if self.remove_expected_suffixes is not None:
            contract["remove_expected_suffixes"] = self.remove_expected_suffixes
        return contract


class _FakeEXL3(torch.nn.Module):
    QUANT_TYPE = "exl3"


def test_exl3_publication_guard_rejects_native_recovery_target():
    owner = SimpleNamespace(
        model=torch.nn.Module(),
        quant_log=[
            {
                "exl3_error_ledger_record": {
                    "module": "experts.0.gate_proj"
                }
            }
        ],
    )
    owner.model.experts = torch.nn.ModuleList([torch.nn.Module()])
    owner.model.experts[0].gate_proj = torch.nn.Linear(2, 2, device="meta")

    with pytest.raises(RuntimeError, match="publication tree is missing"):
        writer._validate_exllamav3_publication_modules(owner)


def test_exl3_publication_guard_accepts_packed_recovery_target():
    owner = SimpleNamespace(
        model=torch.nn.Module(),
        quant_log=[
            {
                "exl3_error_ledger_record": {
                    "module": "experts.0.gate_proj"
                }
            }
        ],
    )
    owner.model.experts = torch.nn.ModuleList([torch.nn.Module()])
    owner.model.experts[0].gate_proj = _FakeEXL3()

    assert writer._validate_exllamav3_publication_modules(owner) == 1


@pytest.mark.parametrize("fused", [False, True])
def test_exl3_checkpoint_passthrough_preserves_native_source_identity(tmp_path, fused):
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    blob_path = blobs / "payload"
    source_path = snapshot / "source.safetensors"
    native = torch.tensor([1.25, -2.5], dtype=torch.float32)
    source_weight = "fused.gate_up_proj" if fused else "expert.weight"
    save_file(
        {
            source_weight: torch.ones((2, 2), dtype=torch.bfloat16),
            "expert.weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            "original.hc_attn_base": native,
        },
        blob_path,
    )
    source_path.symlink_to(blob_path)

    model = torch.nn.Module()
    model.expert = _FakeEXL3()
    model.expert.register_buffer("trellis", torch.ones((1,), dtype=torch.int16))
    model.expert.register_buffer("suh", torch.ones((1,), dtype=torch.float16))
    model.expert.register_buffer("svh", torch.ones((1,), dtype=torch.float16))
    model.expert.register_buffer("mcg", torch.ones((), dtype=torch.int32))
    # This runtime-only spelling must never leak into the published checkpoint.
    model.runtime_alias = torch.nn.Parameter(torch.zeros((2,), dtype=torch.float32))
    owner = SimpleNamespace(
        model=model,
        turtle_model=SimpleNamespace(
            model_local_path=str(snapshot),
            _weight_map={
                source_weight: source_path.name,
                "expert.weight_scale_inv": source_path.name,
                "original.hc_attn_base": source_path.name,
            },
        ),
    )
    storage = {
        "expert": {
            "stored_tensors": {
                f"expert.{suffix}": {}
                for suffix in ("trellis", "suh", "svh", "mcg")
            }
        }
    }

    if fused:
        owner.exllamav3_checkpoint_replaced_tensors = lambda storage: {
            source_weight, "expert.weight_scale_inv"
        }

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, storage)
    assert plan["replaced_source_names"] == frozenset(
        {source_weight, "expert.weight_scale_inv"}
    )
    state = writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)

    assert set(state) == {
        "expert.trellis",
        "expert.suh",
        "expert.svh",
        "expert.mcg",
        "original.hc_attn_base",
    }
    assert "runtime_alias" not in state
    writer.streaming_state_dict_to_shards(
        state,
        save_dir=str(snapshot),
        model_base_name="published",
        single_file_name="published.safetensors",
        metadata={"format": "pt"},
        max_shard_size=None,
    )
    with safe_open(
        snapshot / "published.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        assert set(checkpoint.keys()) == {
            "expert.trellis",
            "expert.suh",
            "expert.svh",
            "expert.mcg",
            "original.hc_attn_base",
        }
        assert torch.equal(checkpoint.get_tensor("original.hc_attn_base"), native)


def test_exl3_checkpoint_passthrough_accepts_reshard_per_layer_shards(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    native_norm = torch.tensor([1.25, -2.5], dtype=torch.float32)
    save_file(
        {
            "model.layers.0.proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "model.layers.0.norm.weight": native_norm,
            "model.embed_tokens.weight": torch.ones((4, 2), dtype=torch.float32),
        },
        source / "model.safetensors",
    )
    resharded = tmp_path / "resharded"
    reshard(source, resharded, ShardConfig(layer_prefixes=["model.layers"]))
    weight_map = json.loads(
        (resharded / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    # The public reshard() per-layer layout nests shards in subdirectories.
    assert weight_map["model.layers.0.norm.weight"] == (
        "model.layers.0/layer.safetensors"
    )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].proj = _FakeEXL3()
    model.model.layers[0].proj.register_buffer(
        "trellis", torch.ones((1,), dtype=torch.int16)
    )
    owner = SimpleNamespace(
        model=model,
        turtle_model=SimpleNamespace(
            model_local_path=str(resharded),
            _weight_map=weight_map,
        ),
    )
    storage = {
        "model.layers.0.proj": {
            "stored_tensors": {"model.layers.0.proj.trellis": {}},
        }
    }

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, storage)
    assert plan is not None
    state = writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)
    assert set(state) == {
        "model.layers.0.proj.trellis",
        "model.layers.0.norm.weight",
        "model.embed_tokens.weight",
    }

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
        assert set(checkpoint.keys()) == {
            "model.layers.0.proj.trellis",
            "model.layers.0.norm.weight",
            "model.embed_tokens.weight",
        }
        assert torch.equal(
            checkpoint.get_tensor("model.layers.0.norm.weight"), native_norm
        )
        assert torch.equal(
            checkpoint.get_tensor("model.embed_tokens.weight"),
            torch.ones((4, 2), dtype=torch.float32),
        )


@pytest.mark.parametrize(
    "shard_name",
    [
        "/absolute/layer.safetensors",
        "../escape/layer.safetensors",
        "nested/../escape.safetensors",
        "./layer.safetensors",
        "nested//layer.safetensors",
        "nested\\layer.safetensors",
    ],
)
def test_exl3_checkpoint_passthrough_rejects_unsafe_shard_paths(
    tmp_path, shard_name
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file(
        {
            "expert.weight": torch.ones((2, 2)),
            "native.norm.weight": torch.ones((2,)),
        },
        snapshot / "source.safetensors",
    )
    model = torch.nn.Module()
    model.expert = _FakeEXL3()
    model.expert.register_buffer("trellis", torch.ones((1,), dtype=torch.int16))
    owner = SimpleNamespace(
        model=model,
        turtle_model=SimpleNamespace(
            model_local_path=str(snapshot),
            _weight_map={
                "expert.weight": shard_name,
                "native.norm.weight": shard_name,
            },
        ),
    )
    storage = {"expert": {"stored_tensors": {"expert.trellis": {}}}}

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, storage)
    assert plan is not None
    with pytest.raises(RuntimeError, match="Checkpoint shard name is unsafe"):
        writer._build_exllamav3_checkpoint_passthrough_state_dict(owner, plan)


def test_exl3_checkpoint_passthrough_requires_source_weight_identity(tmp_path):
    owner = SimpleNamespace(
        turtle_model=SimpleNamespace(
            model_local_path=str(tmp_path),
            _weight_map={"converted.expert.weight": "source.safetensors"},
        )
    )
    storage = {
        "expert": {"stored_tensors": {"expert.trellis": {}}},
    }

    assert writer._exllamav3_checkpoint_passthrough_plan(owner, storage) is None


def test_exl3_checkpoint_passthrough_resolves_converted_source_module(tmp_path):
    owner = SimpleNamespace(
        turtle_model=SimpleNamespace(
            model_local_path=str(tmp_path),
            _weight_map={
                "layers.0.ffn.experts.0.w1.weight": "source.safetensors",
                "layers.0.ffn.experts.0.w1.scale": "source.safetensors",
                "native.attention.weight": "source.safetensors",
            },
        ),
        exllamav3_checkpoint_module_candidates=lambda module: (
            module,
            "layers.0.ffn.experts.0.w1",
        ),
    )
    storage = {
        "model.layers.0.mlp.experts.0.gate_proj": {
            "stored_tensors": {
                "model.layers.0.mlp.experts.0.gate_proj.trellis": {},
            }
        }
    }

    plan = writer._exllamav3_checkpoint_passthrough_plan(owner, storage)

    assert plan["replaced_source_names"] == frozenset(
        {
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
        }
    )


def test_save_state_overlay_replaces_only_declared_prefixes(monkeypatch):
    old = SimpleNamespace(value="old")
    native = SimpleNamespace(value="native")
    trellis = SimpleNamespace(value="trellis")
    suh = SimpleNamespace(value="suh")
    state = {
        "mtp.0.mlp.experts.7.gate_proj.weight": old,
        "mtp.0.mlp.experts.7.gate_proj.weight_scale_inv": old,
        "mtp.0.main_norm.weight": native,
    }
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda model, offload_root=None, include_prefixes=None: {
            "mtp.0.mlp.experts.7.gate_proj.trellis": trellis,
            "mtp.0.mlp.experts.7.gate_proj.suh": suh,
            "mtp.0.main_norm.weight": SimpleNamespace(value="not-selected"),
        },
    )

    writer._apply_save_state_overlay(
        _Owner(["mtp.0.mlp.experts.7.gate_proj"]), state
    )

    assert state == {
        "mtp.0.main_norm.weight": native,
        "mtp.0.mlp.experts.7.gate_proj.trellis": trellis,
        "mtp.0.mlp.experts.7.gate_proj.suh": suh,
    }


def test_save_state_overlay_removes_checkpoint_alias_prefixes(monkeypatch):
    native = object()
    state = {
        "mtp.0.ffn.experts.7.w1.weight": native,
        "mtp.0.ffn.experts.7.w1.scale": native,
        "mtp.0.main_norm.weight": native,
    }
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda *args, **kwargs: {
            "mtp.0.mlp.experts.7.gate_proj.trellis": object(),
            "mtp.0.mlp.experts.7.gate_proj.suh": object(),
        },
    )

    writer._apply_save_state_overlay(
        _Owner(
            ["mtp.0.mlp.experts.7.gate_proj"],
            remove_prefixes=["mtp.0.ffn.experts.7.w1"],
            remove_expected_suffixes=["weight", "scale"],
        ),
        state,
    )

    assert set(state) == {
        "mtp.0.main_norm.weight",
        "mtp.0.mlp.experts.7.gate_proj.trellis",
        "mtp.0.mlp.experts.7.gate_proj.suh",
    }


def test_save_state_overlay_removal_fails_closed_on_missing_source_suffix(
    monkeypatch,
):
    state = {"mtp.0.ffn.experts.7.w1.weight": object()}
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda *args, **kwargs: {
            "mtp.0.mlp.experts.7.gate_proj.trellis": object(),
        },
    )

    with pytest.raises(ValueError, match="source-removal contract differs"):
        writer._apply_save_state_overlay(
            _Owner(
                ["mtp.0.mlp.experts.7.gate_proj"],
                remove_prefixes=["mtp.0.ffn.experts.7.w1"],
                remove_expected_suffixes=["weight", "scale"],
            ),
            state,
        )


def test_save_tensor_storage_includes_exact_overlay_modules(monkeypatch):
    base = {"model.layers.0.expert": {"stored_tensors": {}}}
    mtp_module = "mtp.0.mlp.experts.7.gate_proj"
    overlay = {
        mtp_module: {
            "stored_tensors": {
                f"{mtp_module}.trellis": {},
                f"{mtp_module}.suh": {},
                f"{mtp_module}.svh": {},
                f"{mtp_module}.mcg": {},
            }
        }
    }
    storage_by_model = iter((base, overlay))
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: next(storage_by_model),
    )

    storage = writer._build_exllamav3_tensor_storage_for_save(
        _Owner([mtp_module], ["trellis", "suh", "svh", "mcg"])
    )

    assert storage == {**base, **overlay}


def test_save_tensor_storage_fails_closed_on_missing_overlay_module(monkeypatch):
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: {},
    )
    with pytest.raises(ValueError, match="no EXL3 tensor-storage metadata"):
        writer._build_exllamav3_tensor_storage_for_save(
            _Owner(["mtp.0.missing"])
        )


def test_save_tensor_storage_fails_closed_on_tensor_contract(monkeypatch):
    module = "mtp.0.expert"
    storage_by_model = iter(
        (
            {},
            {module: {"stored_tensors": {f"{module}.trellis": {}}}},
        )
    )
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: next(storage_by_model),
    )
    with pytest.raises(ValueError, match="tensor-storage contract differs"):
        writer._build_exllamav3_tensor_storage_for_save(
            _Owner([module], ["trellis", "suh", "svh", "mcg"])
        )


def test_save_state_overlay_fails_closed_on_missing_prefix(monkeypatch):
    monkeypatch.setattr(writer, "get_state_dict_for_save", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="no replacement tensors"):
        writer._apply_save_state_overlay(_Owner(["mtp.0.missing"]), {})


def test_save_state_overlay_fails_closed_on_tensor_contract(monkeypatch):
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda *args, **kwargs: {"mtp.0.expert.weight": object()},
    )
    with pytest.raises(ValueError, match="tensor contract differs"):
        writer._apply_save_state_overlay(
            _Owner(["mtp.0.expert"], ["trellis", "suh", "svh", "mcg"]),
            {},
        )


@pytest.mark.parametrize("prefixes", ([], [""], ["mtp.0", "mtp.0"]))
def test_save_state_overlay_rejects_ambiguous_prefix_contract(prefixes):
    with pytest.raises(ValueError, match="unique non-empty"):
        writer._apply_save_state_overlay(_Owner(prefixes), {})


def test_prefixed_state_collection_does_not_resolve_unrelated_meta_tensors(
    monkeypatch,
):
    tree = torch.nn.Module()
    tree.selected = torch.nn.Linear(2, 2, device="meta")
    tree.unrelated = torch.nn.Linear(2, 2, device="meta")
    resolved = []

    def fake_resolve(root, module_path, leaf, dtype, shape, cache):
        resolved.append((module_path, leaf))
        return SimpleNamespace(path=f"{root}/{module_path}/{leaf}")

    monkeypatch.setattr(model_utils, "_resolve_offload_entry", fake_resolve)
    state = model_utils.get_state_dict_for_save(
        tree,
        offload_root="/offload",
        include_prefixes=("selected.",),
    )

    assert set(state) == {"selected.weight", "selected.bias"}
    assert resolved == [("selected", "weight"), ("selected", "bias")]
