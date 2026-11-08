"""Regression tests for the targeted module-map optimization in LazyTurtle.materialize_submodule."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.utils.structure import LazyTurtle


class _TransposedCheckpointInner(nn.Module):
    """Nested module whose checkpoint weight is stored in (in, out) layout."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Runtime Linear expects (out, in). Checkpoint stores (in, out) and is
        # marked transposed on the parent container.
        self.layer = nn.Linear(in_features, out_features, bias=False)


class _TransposedCheckpointShell(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.inner = _TransposedCheckpointInner(in_features, out_features)


def _write_index(model_dir: Path, shard_name: str, tensors: dict[str, torch.Tensor]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard_name)}), encoding="utf-8"
    )


def test_materialize_submodule_recurse_true_uses_descendant_and_ancestor_map(tmp_path, monkeypatch):
    """With recurse=True, module_path uses ancestor+descendant map instead of target_model.named_modules()."""

    in_features, out_features = 16, 32
    model_dir = tmp_path / "source"
    model_dir.mkdir()

    # Checkpoint stores (in, out); the shell Linear expects (out, in).
    source = {"inner.layer.weight": torch.randn(in_features, out_features, dtype=torch.float32)}
    save_file(source, str(model_dir / "model.safetensors"))
    _write_index(model_dir, "model.safetensors", source)

    shell = _TransposedCheckpointShell(in_features, out_features)
    for p in shell.parameters():
        p.requires_grad = False

    # Mark the parent container as transposed so _resolve_prefer_transposed_hint
    # must be able to walk from the descendant up to this ancestor.
    shell.inner.is_transposed = True

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    original_named_modules = nn.Module.named_modules

    def _raising_named_modules(self, *args, **kwargs):
        # Scanning the root model would be the old full-model-map behavior.
        if self is shell:
            raise AssertionError("target_model.named_modules() should not be called")
        return original_named_modules(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "named_modules", _raising_named_modules)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
        recurse=True,
    )

    expected = source["inner.layer.weight"].transpose(0, 1).contiguous()
    loaded = shell.inner.layer.weight
    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype
    assert torch.equal(loaded, expected)


def test_materialize_submodule_recurse_false_uses_ancestor_map_for_leaf(tmp_path, monkeypatch):
    """With recurse=False, module_path uses the ancestor chain to resolve transpose hints on a leaf."""

    in_features, out_features = 16, 32
    model_dir = tmp_path / "source"
    model_dir.mkdir()

    source = {"inner.layer.weight": torch.randn(in_features, out_features, dtype=torch.float32)}
    save_file(source, str(model_dir / "model.safetensors"))
    _write_index(model_dir, "model.safetensors", source)

    shell = _TransposedCheckpointShell(in_features, out_features)
    for p in shell.parameters():
        p.requires_grad = False

    shell.inner.is_transposed = True

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    original_named_modules = nn.Module.named_modules

    def _raising_named_modules(self, *args, **kwargs):
        if self is shell:
            raise AssertionError("target_model.named_modules() should not be called")
        return original_named_modules(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "named_modules", _raising_named_modules)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner.layer,
        device=torch.device("cpu"),
        module_path="inner.layer",
        recurse=False,
    )

    expected = source["inner.layer.weight"].transpose(0, 1).contiguous()
    loaded = shell.inner.layer.weight
    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype
    assert torch.equal(loaded, expected)


def test_materialize_submodule_descendant_map_respects_local_is_transposed(tmp_path, monkeypatch):
    """A descendant's own is_transposed hint must be found via the subtree map."""

    in_features, out_features = 16, 32
    model_dir = tmp_path / "source"
    model_dir.mkdir()

    source = {"inner.layer.weight": torch.randn(in_features, out_features, dtype=torch.float32)}
    save_file(source, str(model_dir / "model.safetensors"))
    _write_index(model_dir, "model.safetensors", source)

    shell = _TransposedCheckpointShell(in_features, out_features)
    for p in shell.parameters():
        p.requires_grad = False

    # Set the hint on the leaf module itself, not on the parent.
    shell.inner.layer.is_transposed = True

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    original_named_modules = nn.Module.named_modules

    def _raising_named_modules(self, *args, **kwargs):
        if self is shell:
            raise AssertionError("target_model.named_modules() should not be called")
        return original_named_modules(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "named_modules", _raising_named_modules)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
        recurse=True,
    )

    expected = source["inner.layer.weight"].transpose(0, 1).contiguous()
    loaded = shell.inner.layer.weight
    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype
    assert torch.equal(loaded, expected)


def test_active_source_staging_is_layer_scoped_and_pruned_after_boundary(tmp_path):
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    source = {"model.layers.0.weight": torch.randn(4, 4)}
    shard_name = "model.safetensors"
    save_file(source, str(model_dir / shard_name))
    _write_index(model_dir, shard_name, source)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None
    staging = tmp_path / "active"
    turtle.configure_active_source_staging(
        str(staging), provenance={"plan_sha256": "a" * 64}
    )

    first = Path(
        turtle._checkpoint_shard_path(
            shard_name, module_path="model.layers.0"
        )
    )
    second = Path(
        turtle._checkpoint_shard_path(
            shard_name, module_path="model.layers.0.mlp.experts.0"
        )
    )

    assert first == second
    assert first.parent == staging / "base-000000"
    assert first.read_bytes() == (model_dir / shard_name).read_bytes()
    turtle.prune_active_source_through(0)
    assert not first.parent.exists()


def test_active_source_staging_can_be_suspended_for_direct_checkpoint_reads(tmp_path):
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    source = {"model.layers.0.weight": torch.randn(4, 4)}
    shard_name = "model.safetensors"
    save_file(source, str(model_dir / shard_name))
    _write_index(model_dir, shard_name, source)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None
    staging = tmp_path / "active"
    turtle.configure_active_source_staging(
        str(staging), provenance={"plan_sha256": "a" * 64}
    )

    with turtle.suspend_active_source_staging():
        direct = Path(
            turtle._checkpoint_shard_path(
                shard_name, module_path="model.layers.0"
            )
        )
        assert direct == model_dir / shard_name
        assert not (staging / "base-000000").exists()

    staged = Path(
        turtle._checkpoint_shard_path(
            shard_name, module_path="model.layers.0"
        )
    )
    assert staged == staging / "base-000000" / shard_name
    assert staged.is_file()
