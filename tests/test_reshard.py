# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel import ShardStrategy, reshard
from gptqmodel.models.definitions.mixtral import MixtralQModel
from gptqmodel.utils.reshard import (
    _group_per_layer_names,
    _match_module_template,
    _pack_routed_module_subgroups,
    _routed_module_identity,
    routed_module_templates_from_model_definition,
)


def _build_source_dir(tmp: str, *, prefix: str = "model.layers", config: dict = None) -> str:
    src = os.path.join(tmp, "source")
    os.makedirs(src, exist_ok=True)

    layer0_self_attn = f"{prefix}.0.self_attn.q_proj.weight"
    layer0_mlp = f"{prefix}.0.mlp.gate_proj.weight"
    layer1_self_attn = f"{prefix}.1.self_attn.q_proj.weight"
    layer1_mlp = f"{prefix}.1.mlp.gate_proj.weight"

    if prefix.startswith("model."):
        embed = "model.embed_tokens.weight"
        norm = "model.norm.weight"
    else:
        # For nested/custom prefixes keep non-layer tensors directly under the parent.
        parent = prefix.rsplit(".", 1)[0]
        embed = f"{parent}.embed_tokens.weight"
        norm = f"{parent}.norm.weight"

    t1 = {
        embed: torch.randn(10, 4),
        layer0_self_attn: torch.randn(4, 4),
        layer0_mlp: torch.randn(8, 4),
    }
    t2 = {
        layer1_self_attn: torch.randn(4, 4),
        layer1_mlp: torch.randn(8, 4),
        norm: torch.randn(4),
        "lm_head.weight": torch.randn(10, 4),
    }
    save_file(t1, os.path.join(src, "model-00001-of-00002.safetensors"))
    save_file(t2, os.path.join(src, "model-00002-of-00002.safetensors"))

    all_names = list(t1.keys()) + list(t2.keys())
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            name: "model-00001-of-00002.safetensors" if name in t1 else "model-00002-of-00002.safetensors"
            for name in all_names
        },
    }
    with open(os.path.join(src, "model.safetensors.index.json"), "w", encoding="utf-8") as fp:
        json.dump(index, fp)

    config = config if config is not None else {}
    with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(config, fp)
    return src


def _build_source_dir_with_prefix(tmp: str, prefix: str) -> str:
    """Compatibility helper used by pre-existing tests."""
    return _build_source_dir(tmp, prefix=prefix, config={})


def _load_source_state_dict(src_path: str) -> dict:
    """Load the complete source state dict from a safetensors checkpoint."""
    index_path = os.path.join(src_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fp:
            index = json.load(fp)
        weight_map = index.get("weight_map", {})
        state = {}
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(os.path.join(src_path, shard)))
        return state

    # Single-file checkpoint.
    shard_path = os.path.join(src_path, "model.safetensors")
    if os.path.isfile(shard_path):
        return load_file(shard_path)

    files = [f for f in os.listdir(src_path) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {src_path}")
    if len(files) == 1:
        return load_file(os.path.join(src_path, files[0]))

    raise ValueError(f"Multiple unindexed safetensors files in {src_path}: {files}")


def test_reshard_per_layer_creates_index_and_shards():
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_layers"] == 2
        assert result["num_non_layer_tensors"] == 3
        assert result["num_shards"] == 3
        assert os.path.isfile(os.path.join(dst, "model.safetensors.index.json"))

        with open(os.path.join(dst, "model.safetensors.index.json"), encoding="utf-8") as fp:
            new_index = json.load(fp)
        assert len(new_index["weight_map"]) == result["num_tensors"]


def test_reshard_per_layer_preserves_tensor_values():
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer")

        original = {}
        for fn in os.listdir(src):
            if fn.endswith(".safetensors"):
                original.update(load_file(os.path.join(src, fn)))

        reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        for fn in os.listdir(dst):
            if not fn.endswith(".safetensors"):
                continue
            for name, tensor in load_file(os.path.join(dst, fn)).items():
                diff = (tensor - original[name]).abs().max().item()
                assert diff == 0.0, f"{name} differs after reshard"


def test_reshard_per_layer_custom_prefix():
    """Explicit layer_prefixes can shard non-LLaMA style checkpoints."""
    prefix = "language_model.model.layers"
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir_with_prefix(tmp, prefix)
        dst = os.path.join(tmp, "per-layer")
        result = reshard(
            src,
            dst,
            strategy=ShardStrategy.PER_LAYER,
            layer_prefixes=[prefix],
            progress=False,
        )

        assert result["num_layers"] == 2
        assert result["num_non_layer_tensors"] == 3
        assert result["num_shards"] == 3

        with open(os.path.join(dst, "model.safetensors.index.json"), encoding="utf-8") as fp:
            new_index = json.load(fp)
        assert len(new_index["weight_map"]) == result["num_tensors"]


def test_reshard_per_layer_no_layers_raises():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        os.makedirs(src, exist_ok=True)
        save_file({"foo.weight": torch.randn(4)}, os.path.join(src, "model.safetensors"))
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fp:
            fp.write("{}")

        with open(os.path.join(src, "model.safetensors.index.json"), "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {"foo.weight": "model.safetensors"},
                },
                fp,
            )

        with tempfile.TemporaryDirectory() as tmp2:
            dst = os.path.join(tmp2, "per-layer")
            try:
                reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)
                assert False, "expected ValueError for checkpoint without layers"
            except ValueError as exc:
                assert "No layer tensors matched" in str(exc)


def test_reshard_auto_detects_nested_layer_prefix():
    """Layer prefixes are inferred from tensor names when config has no model type."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(
            tmp,
            prefix="language.model.layers",
            config={},
        )
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_layers"] == 2
        assert result["num_non_layer_tensors"] == 3


def test_reshard_uses_model_definition_for_gpt2():
    """A valid ``model_type`` in config.json lets reshard use extract_layers_node()."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(
            tmp,
            prefix="transformer.h",
            config={"model_type": "gpt2"},
        )
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_layers"] == 2
        assert result["num_non_layer_tensors"] == 3


def test_reshard_explicit_layer_prefixes_override_auto():
    """Callers can pass explicit layer prefixes, e.g. for custom/niche checkpoints."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(
            tmp,
            prefix="custom.blocks",
            config={},
        )
        dst = os.path.join(tmp, "per-layer")
        result = reshard(
            src,
            dst,
            strategy=ShardStrategy.PER_LAYER,
            layer_prefixes=["custom.blocks"],
            progress=False,
        )

        assert result["num_layers"] == 2
        assert result["num_non_layer_tensors"] == 3


def test_reshard_accepts_string_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy="per_layer", progress=False)

        assert result["strategy"] == ShardStrategy.PER_LAYER.value
        assert result["num_shards"] == 3


def test_per_layer_moe_planner_uses_explicit_templates_and_keeps_shared_dense():
    names = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        # A misleading name is dense unless an explicit module-tree template declares it routed.
        "model.layers.0.not_experts.0.gate_proj.weight",
        "model.embed_tokens.weight",
    }
    groups, group_is_layer = _group_per_layer_names(
        names,
        layer_prefixes=["model.layers"],
        strategy=ShardStrategy.PER_LAYER_MOE,
        routed_module_templates=["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        moe_modules_per_shard=2,
    )

    assert groups["model.layers.0"] == [
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.not_experts.0.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert groups["model.layers.0.routed.0000"] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]
    assert groups["model.layers.0.routed.0001"] == [
        "model.layers.0.mlp.experts.1.gate_proj.weight",
    ]
    assert groups["non_layer"] == ["model.embed_tokens.weight"]
    assert group_is_layer["model.layers.0.routed.0000"] is True
    assert group_is_layer["non_layer"] is False


def test_per_layer_moe_planner_expands_runtime_and_checkpoint_aliases_from_module_tree():
    templates = routed_module_templates_from_model_definition(MixtralQModel)
    assert "mlp.experts.{expert_index}.gate_proj" in templates
    assert "block_sparse_moe.experts.{expert_index}.w1" in templates

    runtime_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    checkpoint_alias = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
    groups, group_is_layer = _group_per_layer_names(
        {
            "model.layers.0.self_attn.q_proj.weight",
            runtime_name,
            checkpoint_alias,
            "model.embed_tokens.weight",
        },
        layer_prefixes=["model.layers"],
        strategy=ShardStrategy.PER_LAYER_MOE,
        routed_module_templates=templates,
    )

    assert groups["model.layers.0"] == ["model.layers.0.self_attn.q_proj.weight"]
    assert groups["model.layers.0.routed.0000"] == [checkpoint_alias, runtime_name]
    assert group_is_layer["model.layers.0.routed.0000"] is True
    assert groups["non_layer"] == ["model.embed_tokens.weight"]
    assert group_is_layer["non_layer"] is False


def test_per_layer_moe_planner_rejects_missing_tags_and_invalid_bound():
    names = ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    with pytest.raises(ValueError, match="module_tree"):
        _group_per_layer_names(
            names,
            layer_prefixes=["model.layers"],
            strategy=ShardStrategy.PER_LAYER_MOE,
        )
    with pytest.raises(ValueError, match="positive"):
        _group_per_layer_names(
            names,
            layer_prefixes=["model.layers"],
            strategy=ShardStrategy.PER_LAYER_MOE,
            routed_module_templates=["mlp.experts.{expert_index}.gate_proj"],
            moe_modules_per_shard=0,
        )


def test_routed_template_and_atomic_packer_fail_closed_corner_cases():
    assert _match_module_template(
        "mlp.experts.not_an_index.gate_proj.weight", "mlp.experts.#.gate_proj"
    ) is None
    assert _routed_module_identity(
        "other.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0",
        ["mlp.experts.#.gate_proj"],
    ) is None
    assert _pack_routed_module_subgroups(
        [],
        {},
        64,
        layer_group="model.layers.0",
        routed_module_templates=["mlp.experts.#.gate_proj"],
    ) == []
    with pytest.raises(RuntimeError, match="untagged tensor"):
        _pack_routed_module_subgroups(
            ["model.layers.0.mlp.shared_experts.gate_proj.weight"],
            {"model.layers.0.mlp.shared_experts.gate_proj.weight": 16},
            64,
            layer_group="model.layers.0",
            routed_module_templates=["mlp.experts.#.gate_proj"],
        )

    groups, _ = _group_per_layer_names(
        ["model.layers.0.mlp.experts.0.gate_proj.weight"],
        layer_prefixes=["model.layers"],
        strategy=ShardStrategy.PER_LAYER_MOE,
        routed_module_templates=["mlp.experts.#.gate_proj"],
    )
    assert list(groups) == ["model.layers.0.routed.0000"]


def test_reshard_per_layer_moe_rejects_model_definition_failure(monkeypatch):
    from gptqmodel.models import auto

    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer-moe")
        monkeypatch.setattr(
            auto,
            "check_and_get_model_definition",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported")),
        )
        with pytest.raises(ValueError, match="supported model definition"):
            reshard(src, dst, strategy=ShardStrategy.PER_LAYER_MOE, progress=False)


def test_reshard_per_layer_moe_preserves_values_and_original(monkeypatch):
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
    from gptqmodel.models import auto

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        dst = os.path.join(tmp, "per-layer-moe")
        os.makedirs(src)
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(2, 2),
        }
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"] = torch.randn(2, 2)
        source_file = os.path.join(src, "model.safetensors")
        save_file(tensors, source_file)
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fp:
            json.dump({"model_type": "deepseek_v4"}, fp)
        with open(source_file, "rb") as fp:
            source_bytes = fp.read()
        monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: DeepSeekV4QModel)

        result = reshard(
            src,
            dst,
            strategy=ShardStrategy.PER_LAYER_MOE,
            moe_modules_per_shard=2,
            progress=False,
        )

        assert result["strategy"] == "per_layer_moe"
        assert result["num_shards"] == 4  # one dense/shared plus three routed shards
        with open(source_file, "rb") as fp:
            assert fp.read() == source_bytes
        restored = _load_source_state_dict(dst)
        assert set(restored) == set(tensors)
        for name, original in tensors.items():
            assert torch.equal(restored[name], original)

        with open(os.path.join(dst, "model.safetensors.index.json"), encoding="utf-8") as fp:
            weight_map = json.load(fp)["weight_map"]
        dense_file = weight_map["model.layers.0.self_attn.q_proj.weight"]
        assert weight_map["model.layers.0.mlp.shared_experts.gate_proj.weight"] == dense_file
        assert all(
            weight_map[name] != dense_file
            for name in tensors
            if ".mlp.experts." in name
        )


def test_reshard_per_layer_moe_routes_checkpoint_aliases(monkeypatch):
    from gptqmodel.models import auto

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        dst = os.path.join(tmp, "per-layer-moe")
        os.makedirs(src)
        dense_name = "model.layers.0.self_attn.q_proj.weight"
        routed_prefix = "model.layers.0.block_sparse_moe.experts.0.w1"
        non_layer_name = "model.embed_tokens.weight"
        tensors = {
            dense_name: torch.randn(2, 2),
            f"{routed_prefix}.qweight": torch.randint(0, 16, (2, 2), dtype=torch.int32),
            f"{routed_prefix}.scales": torch.randn(2, 1),
            non_layer_name: torch.randn(4, 2),
        }
        save_file(tensors, os.path.join(src, "model.safetensors"))
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fp:
            json.dump({"model_type": "mixtral"}, fp)
        monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: MixtralQModel)

        result = reshard(
            src,
            dst,
            strategy=ShardStrategy.PER_LAYER_MOE,
            progress=False,
        )

        assert result["num_shards"] == 3
        with open(os.path.join(dst, "model.safetensors.index.json"), encoding="utf-8") as fp:
            weight_map = json.load(fp)["weight_map"]
        assert weight_map[f"{routed_prefix}.qweight"] == weight_map[f"{routed_prefix}.scales"]
        assert weight_map[f"{routed_prefix}.qweight"] != weight_map[dense_name]
        assert weight_map[non_layer_name] not in {
            weight_map[dense_name],
            weight_map[f"{routed_prefix}.qweight"],
        }

        restored = _load_source_state_dict(dst)
        assert set(restored) == set(tensors)
        for name, original in tensors.items():
            assert torch.equal(restored[name], original)


def test_reshard_per_layer_moe_max_size_keeps_module_state_atomic(monkeypatch):
    from gptqmodel.models import auto
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        dst = os.path.join(tmp, "per-layer-moe")
        os.makedirs(src)
        tensors = {"model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2)}
        for expert in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert}.gate_proj"
            tensors[f"{prefix}.qweight"] = torch.randint(0, 16, (4, 4), dtype=torch.int32)
            tensors[f"{prefix}.scales"] = torch.randn(4, 2)
            tensors[f"{prefix}.qzeros"] = torch.randint(0, 16, (4, 1), dtype=torch.int32)
        save_file(tensors, os.path.join(src, "model.safetensors"))
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fp:
            json.dump({"model_type": "deepseek_v4"}, fp)
        monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: DeepSeekV4QModel)

        result = reshard(
            src,
            dst,
            strategy=ShardStrategy.PER_LAYER_MOE,
            moe_modules_per_shard=2,
            max_shard_size_gb=64 / 1024**3,
            progress=False,
        )

        assert result["num_shards"] == 3
        with open(os.path.join(dst, "model.safetensors.index.json"), encoding="utf-8") as fp:
            weight_map = json.load(fp)["weight_map"]
        for expert in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert}.gate_proj"
            assert len(
                {
                    weight_map[f"{prefix}.qweight"],
                    weight_map[f"{prefix}.scales"],
                    weight_map[f"{prefix}.qzeros"],
                }
            ) == 1


def test_reshard_rejects_source_target_collision():
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)

        try:
            reshard(src, src, strategy=ShardStrategy.PER_LAYER, overwrite=True, progress=False)
            assert False, "expected ValueError for same source/target path"
        except ValueError as exc:
            assert "same as or a parent of" in str(exc)

    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        parent = os.path.dirname(src)

        try:
            reshard(src, parent, strategy=ShardStrategy.PER_LAYER, overwrite=True, progress=False)
            assert False, "expected ValueError for target parent of source"
        except ValueError as exc:
            assert "same as or a parent of" in str(exc)


def test_reshard_allows_source_and_target_on_different_windows_drives(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer")
        src_real = os.path.realpath(src)
        dst_real = os.path.realpath(dst)
        real_commonpath = os.path.commonpath
        real_splitdrive = os.path.splitdrive

        def cross_drive_commonpath(paths):
            if list(paths) == [src_real, dst_real]:
                raise ValueError("Paths don't have the same drive")
            return real_commonpath(paths)

        def cross_drive_splitdrive(path):
            if path == src_real:
                return "C:", path
            if path == dst_real:
                return "D:", path
            return real_splitdrive(path)

        monkeypatch.setattr(os.path, "commonpath", cross_drive_commonpath)
        monkeypatch.setattr(os.path, "splitdrive", cross_drive_splitdrive)

        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_tensors"] > 0
        assert os.path.isfile(os.path.join(dst, "model.safetensors.index.json"))


def test_reshard_propagates_unexpected_commonpath_value_error_on_same_drive(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        src = _build_source_dir(tmp)
        dst = os.path.join(tmp, "per-layer")
        src_real = os.path.realpath(src)
        dst_real = os.path.realpath(dst)
        real_commonpath = os.path.commonpath
        real_splitdrive = os.path.splitdrive

        def failing_commonpath(paths):
            if list(paths) == [src_real, dst_real]:
                raise ValueError("simulated same-drive commonpath failure")
            return real_commonpath(paths)

        def same_drive_splitdrive(path):
            if path in {src_real, dst_real}:
                return "C:", path
            return real_splitdrive(path)

        monkeypatch.setattr(os.path, "commonpath", failing_commonpath)
        monkeypatch.setattr(os.path, "splitdrive", same_drive_splitdrive)

        with pytest.raises(ValueError, match="simulated same-drive commonpath failure"):
            reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)


def test_reshard_preserves_safetensors_metadata():
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        os.makedirs(src, exist_ok=True)
        t = {
            "model.embed_tokens.weight": torch.randn(10, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
            "model.norm.weight": torch.randn(4),
            "lm_head.weight": torch.randn(10, 4),
        }
        save_file(
            t,
            os.path.join(src, "model.safetensors"),
            metadata={"format": "legacy", "custom_key": "custom_value"},
        )
        index = {
            "metadata": {"total_size": 0},
            "weight_map": dict.fromkeys(t, "model.safetensors"),
        }
        with open(os.path.join(src, "model.safetensors.index.json"), "w", encoding="utf-8") as fp:
            json.dump(index, fp)

        dst = os.path.join(tmp, "per-layer")
        reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        for fn in os.listdir(dst):
            if not fn.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(dst, fn), framework="pt", device="cpu") as handler:
                meta = handler.metadata()
            assert meta.get("format") == "pt"
            assert meta.get("custom_key") == "custom_value"


LLAMA_3_2_1B_INSTRUCT_PATH = "/monster/data/model/Llama-3.2-1B-Instruct"
QWEN3_MOE_LAYERS_1_PATH = "/monster/data/model/Qwen3-30B-A3B-layers-1"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isdir(LLAMA_3_2_1B_INSTRUCT_PATH),
    reason=f"{LLAMA_3_2_1B_INSTRUCT_PATH} not available",
)
def test_reshard_real_llama_3_2_1b_instruct():
    """End-to-end per-layer reshard of the dense Llama-3.2-1B-Instruct checkpoint."""
    src = LLAMA_3_2_1B_INSTRUCT_PATH
    with open(os.path.join(src, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    expected_layers = config["num_hidden_layers"]

    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_layers"] == expected_layers
        assert os.path.isfile(os.path.join(dst, "model.safetensors.index.json"))

        # Verify every tensor is present and unchanged.
        original = _load_source_state_dict(src)
        for fn in os.listdir(dst):
            if not fn.endswith(".safetensors"):
                continue
            for name, tensor in load_file(os.path.join(dst, fn)).items():
                diff = (tensor - original[name]).abs().max().item()
                assert diff == 0.0, f"{name} differs after reshard"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isdir(QWEN3_MOE_LAYERS_1_PATH),
    reason=f"{QWEN3_MOE_LAYERS_1_PATH} not available",
)
def test_reshard_real_qwen3_moe_single_layer():
    """End-to-end per-layer reshard of a 1-layer Qwen3-MoE checkpoint.

    This exercises the MoE path where a single layer contains many expert
    weight tensors (``mlp.experts.{idx}.*``) but still maps to one shard.
    """
    src = QWEN3_MOE_LAYERS_1_PATH
    with open(os.path.join(src, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    expected_layers = config["num_hidden_layers"]

    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "per-layer")
        result = reshard(src, dst, strategy=ShardStrategy.PER_LAYER, progress=False)

        assert result["num_layers"] == expected_layers
        assert os.path.isfile(os.path.join(dst, "model.safetensors.index.json"))

        # Verify at least one expert tensor exists in the (single) layer shard.
        index_path = os.path.join(dst, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as fp:
            index = json.load(fp)
        expert_names = [n for n in index["weight_map"] if ".mlp.experts." in n]
        assert expert_names, "expected MoE expert tensors in the sharded checkpoint"

        # Verify every tensor is present and unchanged.
        original = _load_source_state_dict(src)
        for fn in os.listdir(dst):
            if not fn.endswith(".safetensors"):
                continue
            for name, tensor in load_file(os.path.join(dst, fn)).items():
                diff = (tensor - original[name]).abs().max().item()
                assert diff == 0.0, f"{name} differs after reshard"
