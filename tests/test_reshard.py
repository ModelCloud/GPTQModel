# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

from gptqmodel import ShardStrategy, reshard


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
