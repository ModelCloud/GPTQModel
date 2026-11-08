# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import mmap
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel import ShardConfig, ShardLayout, reshard


def _make_state():
    return {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
        "model.layers.0.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.layers.0.self_attn.scales": torch.ones(4, 2, dtype=torch.float16),
        "model.layers.1.mlp.down_proj.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.layers.1.mlp.down_proj.qzeros": torch.zeros(4, 1, dtype=torch.int32),
        "model.norm.weight": torch.ones(4, dtype=torch.float16),
        "lm_head.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
        "model.rotary_emb.inv_freq": torch.ones(2, dtype=torch.float32),
    }


def _build_source(root: Path, *, with_config: bool = True, state=None):
    state = _make_state() if state is None else state
    root.mkdir(parents=True)

    names = sorted(state)
    split = len(names) // 2
    shard_names = [names[:split], names[split:]]
    weight_map = {}
    total_size = 0

    for index, group in enumerate(shard_names, start=1):
        filename = f"model-{index:05d}-of-{len(shard_names):05d}.safetensors"
        save_file(
            {name: state[name].contiguous() for name in group},
            str(root / filename),
            metadata={"format": "pt", "gptq_bits": "4"},
        )
        total_size += (root / filename).stat().st_size
        for name in group:
            weight_map[name] = filename

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    if with_config:
        (root / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
            encoding="utf-8",
        )
    (root / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")
    return state


def _read_index(target: Path):
    return json.loads((target / "model.safetensors.index.json").read_text(encoding="utf-8"))


def _assert_tensors_equal(loaded, expected):
    assert set(loaded) == set(expected)
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, expected[name])


def test_reshard_per_layer_with_single_global(tmp_path):
    source = tmp_path / "source"
    state = _build_source(source)
    target = tmp_path / "target"

    result = reshard(source, target)

    assert result == str(target)
    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.layers.1" / "layer.safetensors").is_file()
    assert (target / "global.safetensors").is_file()

    # Every non-layer tensor lives in the single global shard, not in per-module files.
    assert not (target / "model.embed_tokens.safetensors").exists()
    assert not (target / "lm_head.safetensors").exists()
    assert not (target / "model.norm.safetensors").exists()
    assert not (target / "model.rotary_emb.safetensors").exists()
    assert not (target / "model.embed_tokens").exists()

    assert (target / "config.json").is_file()
    assert (target / "tokenizer_config.json").is_file()

    weight_map = _read_index(target)["weight_map"]
    assert weight_map["model.layers.0.self_attn.qweight"] == "model.layers.0/layer.safetensors"
    assert weight_map["model.layers.0.self_attn.scales"] == "model.layers.0/layer.safetensors"
    assert weight_map["model.layers.1.mlp.down_proj.qweight"] == "model.layers.1/layer.safetensors"
    assert weight_map["model.embed_tokens.weight"] == "global.safetensors"
    assert weight_map["model.norm.weight"] == "global.safetensors"
    assert weight_map["lm_head.weight"] == "global.safetensors"
    assert weight_map["model.rotary_emb.inv_freq"] == "global.safetensors"

    global_state = load_file(str(target / "global.safetensors"))
    assert set(global_state) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.rotary_emb.inv_freq",
    }
    _assert_tensors_equal(global_state, {name: state[name] for name in global_state})
    for layer in ("model.layers.0", "model.layers.1"):
        loaded = load_file(str(target / layer / "layer.safetensors"))
        _assert_tensors_equal(loaded, {name: state[name] for name in loaded})

    total_size = sum(path.stat().st_size for path in target.rglob("*.safetensors"))
    assert _read_index(target)["metadata"]["total_size"] == total_size


def test_reshard_preserves_safetensors_metadata(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"

    reshard(source, target)

    with safe_open(str(target / "global.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["gptq_bits"] == "4"
    assert metadata["format"] == "pt"

    with safe_open(str(target / "model.layers.0" / "layer.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["gptq_bits"] == "4"


def test_reshard_keeps_source_untouched(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    before = {path.name: path.stat().st_size for path in source.iterdir()}

    reshard(source, tmp_path / "target")

    after = {path.name: path.stat().st_size for path in source.iterdir()}
    assert after == before


def test_reshard_does_not_shadow_index_with_single_file_name(tmp_path):
    # `transformers.from_pretrained` prefers a root `model.safetensors` over
    # `model.safetensors.index.json`. A per-layer layout must keep the global
    # shard under a different name or loaders silently skip every layer shard.
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"

    reshard(source, target)

    assert (target / "model.safetensors.index.json").is_file()
    assert not (target / "model.safetensors").exists()
    assert (target / "global.safetensors").is_file()
    assert _read_index(target)["weight_map"]["lm_head.weight"] == "global.safetensors"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_reshard_mirrors_source_directory_permissions(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    os.chmod(source, 0o750)
    target = tmp_path / "target"

    reshard(source, target)

    assert (target.stat().st_mode & 0o777) == 0o750


def test_reshard_survives_directory_mode_failure(tmp_path, monkeypatch):
    # Mirroring the source mode is best-effort; a filesystem that rejects the
    # chmod must not abort an otherwise valid reshard.
    import importlib

    reshard_module = importlib.import_module("gptqmodel.reshard")

    source = tmp_path / "source"
    _build_source(source)
    real_chmod = os.chmod

    def _fail_staging_chmod(path, mode, *args, **kwargs):
        # Only the staging root, not files copied into it.
        if os.path.basename(str(path)).startswith(".gptqmodel-reshard-"):
            raise OSError("simulated chmod failure")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(reshard_module.os, "chmod", _fail_staging_chmod)
    target = tmp_path / "target"

    reshard(source, target)

    assert (target / "global.safetensors").is_file()
    assert (target / "model.layers.0" / "layer.safetensors").is_file()


def test_reshard_splits_large_layer_but_not_global_shard(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"

    reshard(source, target, ShardConfig(max_shard_size=64))

    layer_shards = sorted(path.name for path in (target / "model.layers.0").glob("*.safetensors"))
    assert len(layer_shards) > 1
    assert all(name.startswith("layer-") for name in layer_shards)

    root_shards = sorted(path.name for path in target.glob("*.safetensors"))
    assert root_shards == ["global.safetensors"]

    weight_map = _read_index(target)["weight_map"]
    assert weight_map["model.embed_tokens.weight"] == "global.safetensors"
    assert weight_map["model.norm.weight"] == "global.safetensors"
    assert weight_map["model.layers.0.self_attn.qweight"].startswith("model.layers.0/layer-")


def test_reshard_single_layout_writes_one_file(tmp_path):
    source = tmp_path / "source"
    state = _build_source(source)
    target = tmp_path / "target"

    reshard(source, target, ShardConfig(layout=ShardLayout.SINGLE))

    weight_files = sorted(target.glob("*.safetensors"))
    assert len(weight_files) == 1
    assert not (target / "model.layers.0").exists()
    assert not (target / "model.safetensors.index.json").exists()

    _assert_tensors_equal(load_file(str(weight_files[0])), state)


def test_reshard_requires_overwrite_for_non_empty_destination(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        reshard(source, target)

    reshard(source, target, overwrite=True)
    assert not (target / "stale.txt").exists()
    assert (target / "global.safetensors").is_file()


def test_reshard_accepts_loaded_model_like_object(tmp_path):
    source = tmp_path / "source"
    _build_source(source)

    class _ModelLike:
        def __init__(self, path):
            self.model_local_path = str(path)

        @classmethod
        def extract_layers_node(cls):
            return ["model.layers"]

    target = tmp_path / "target"
    reshard(_ModelLike(source), target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()


def test_reshard_explicit_layer_prefixes(tmp_path):
    source = tmp_path / "source"
    _build_source(source)

    target = tmp_path / "target"
    reshard(source, target, ShardConfig(layer_prefixes=["model.layers"]))

    assert (target / "model.layers.1" / "layer.safetensors").is_file()


def test_reshard_detects_layers_without_config(tmp_path):
    source = tmp_path / "source"
    _build_source(source, with_config=False)

    target = tmp_path / "target"
    reshard(source, target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "global.safetensors").is_file()


def test_reshard_is_repeatable_on_nested_layout(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    reshard(source, first)
    reshard(first, second)

    assert (second / "model.layers.0" / "layer.safetensors").is_file()
    assert _read_index(second)["weight_map"] == _read_index(first)["weight_map"]
    assert (second / "global.safetensors").is_file()


def test_shard_config_rejects_unknown_layout():
    with pytest.raises(ValueError):
        ShardConfig(layout="not-a-layout")


def test_reshard_groups_all_layer_stacks_and_keeps_numeric_non_layer_global(tmp_path):
    # A model definition may only describe the text decoder. Reshard must still
    # give the vision transformer blocks their own layer directories, while a
    # numerically indexed projector such as `mlp1.0` stays in the global shard.
    state = {
        "model.text_model.layers.0.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.text_model.layers.1.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.vision_model.encoder.layers.0.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.vision_model.encoder.layers.1.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "mlp1.0.weight": torch.ones(4, 4, dtype=torch.float16),
        "mlp1.1.weight": torch.ones(4, 4, dtype=torch.float16),
        "lm_head.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
    }
    source = tmp_path / "source"
    _build_source(source, state=state)

    class _TextOnlyModelLike:
        def __init__(self, path):
            self.model_local_path = str(path)

        @classmethod
        def extract_layers_node(cls):
            return ["model.text_model.layers"]

    target = tmp_path / "target"
    reshard(_TextOnlyModelLike(source), target)

    assert (target / "model.text_model.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.text_model.layers.1" / "layer.safetensors").is_file()
    assert (target / "model.vision_model.encoder.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.vision_model.encoder.layers.1" / "layer.safetensors").is_file()
    assert not (target / "mlp1.0").exists()

    global_state = load_file(str(target / "global.safetensors"))
    assert set(global_state) == {"mlp1.0.weight", "mlp1.1.weight", "lm_head.weight"}


def test_reshard_detects_multiple_single_layer_stacks_without_config(tmp_path):
    state = {
        "model.text.layers.0.attn.qweight": torch.ones(4, 4, dtype=torch.int32),
        "model.vision.layers.0.attn.qweight": torch.full((4, 4), 2, dtype=torch.int32),
        "lm_head.weight": torch.ones(4, 4, dtype=torch.float16),
    }
    source = tmp_path / "source"
    _build_source(source, with_config=False, state=state)
    target = tmp_path / "target"

    reshard(source, target)

    assert (target / "model.text.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.vision.layers.0" / "layer.safetensors").is_file()
    assert set(load_file(str(target / "global.safetensors"))) == {"lm_head.weight"}


def test_reshard_does_not_mmap_source_weights(tmp_path, monkeypatch):
    import safetensors

    source = tmp_path / "source"
    _build_source(source)

    def _fail_safe_open(*args, **kwargs):
        raise AssertionError("reshard must not open source weights with safetensors.safe_open")

    def _fail_mmap(*args, **kwargs):
        raise AssertionError("reshard must not mmap source weights")

    monkeypatch.setattr(safetensors, "safe_open", _fail_safe_open)
    monkeypatch.setattr(mmap, "mmap", _fail_mmap)
    target = tmp_path / "target"
    reshard(source, target)

    assert (target / "global.safetensors").is_file()


def test_reshard_keeps_at_most_one_source_shard_open(tmp_path, monkeypatch):
    import builtins

    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"
    source_prefix = f"{source}{os.sep}"

    real_open = builtins.open
    open_source_shards = set()
    peak_open = 0

    class _TrackedHandle:
        def __init__(self, handle, key):
            self._handle = handle
            self._key = key

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc_info):
            try:
                return self._handle.__exit__(*exc_info)
            finally:
                self._release()

        def close(self):
            try:
                self._handle.close()
            finally:
                self._release()

        def _release(self):
            open_source_shards.discard(self._key)

    def _tracked_open(file, mode="r", *args, **kwargs):
        nonlocal peak_open
        handle = real_open(file, mode, *args, **kwargs)
        try:
            path = os.fspath(file)
        except TypeError:
            path = ""
        if (
            isinstance(path, str)
            and path.startswith(source_prefix)
            and path.endswith(".safetensors")
            and "r" in mode
            and "+" not in mode
        ):
            key = object()
            open_source_shards.add(key)
            peak_open = max(peak_open, len(open_source_shards))
            return _TrackedHandle(handle, key)
        return handle

    monkeypatch.setattr(builtins, "open", _tracked_open)
    reshard(source, target)
    monkeypatch.undo()

    assert peak_open == 1
    assert not open_source_shards


def test_reshard_copies_nested_auxiliary_files(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    nested = source / "assets"
    nested.mkdir()
    (nested / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    target = tmp_path / "target"
    reshard(source, target)

    assert (target / "assets" / "chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}"


def test_reshard_rejects_destination_inside_source(tmp_path):
    source = tmp_path / "source"
    _build_source(source)

    with pytest.raises(ValueError, match="inside the source"):
        reshard(source, source / "nested-target")


def test_reshard_restores_destination_when_publish_fails(tmp_path, monkeypatch):
    import importlib

    reshard_module = importlib.import_module("gptqmodel.reshard")

    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    real_replace = reshard_module.os.replace

    def _fail_staging_publish(src, dst):
        if ".gptqmodel-reshard-" in str(src):
            raise OSError("simulated publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(reshard_module.os, "replace", _fail_staging_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        reshard(source, target, overwrite=True)

    assert (target / "stale.txt").read_text(encoding="utf-8") == "stale"


def test_reshard_overwrite_preserves_unrelated_backup_path(tmp_path):
    source = tmp_path / "source"
    state = _build_source(source)
    target = tmp_path / "target"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    # The former deterministic backup name can already exist after a failed
    # operation in this process. It must never be removed by a new reshard.
    existing_backup = tmp_path / f"target.bak-{os.getpid()}"
    existing_backup.mkdir()
    (existing_backup / "keep.txt").write_text("keep", encoding="utf-8")

    reshard(source, target, ShardConfig(layout=ShardLayout.SINGLE), overwrite=True)

    assert (existing_backup / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "stale.txt").exists()
    _assert_tensors_equal(load_file(str(target / "model.safetensors")), state)


def test_reshard_discovers_unindexed_multi_shard_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    first = {"model.layers.0.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4)}
    second = {
        "model.layers.1.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "lm_head.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
    }
    save_file(first, str(source / "part-a.safetensors"), metadata={"format": "pt"})
    save_file(second, str(source / "part-b.safetensors"), metadata={"format": "pt"})

    target = tmp_path / "target"
    reshard(source, target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.layers.1" / "layer.safetensors").is_file()
    assert set(load_file(str(target / "global.safetensors"))) == {"lm_head.weight"}


def test_reshard_accepts_hf_cache_symlink_shards(tmp_path):
    # Hugging Face cache snapshots store shards as symlinks into ../../blobs.
    # Resharding a cached checkpoint must accept those links.
    hub = tmp_path / "hub"
    blobs = hub / "blobs"
    snapshot = hub / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    state = _make_state()
    shard = blobs / "blob.safetensors"
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(shard), metadata={"format": "pt"})
    shard_name = "model-00001-of-00001.safetensors"
    os.symlink(os.path.join("..", "..", "blobs", "blob.safetensors"), snapshot / shard_name)
    index = {
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": dict.fromkeys(state, shard_name),
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    target = tmp_path / "target"
    reshard(snapshot, target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "global.safetensors").is_file()


def test_reshard_rejects_index_shard_outside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {"model.layers.0.qweight": "../outside.safetensors"},
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the source directory"):
        reshard(source, tmp_path / "target")


def test_shard_config_accepts_layout_aliases():
    assert ShardConfig(layout="per-layer-with-single-global").layout is ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL
