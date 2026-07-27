import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import gptqmodel.utils.structure as structure
from gptqmodel.utils.structure import LazyTurtle, print_module_tree


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class DummyStackModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock() for _ in range(num_layers)])
        self.heads = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
        self.lm_head = nn.Linear(4, 4)


def test_print_module_tree_caps_layer_stacks_by_default(capsys):
    model = DummyStackModel(num_layers=6)

    print_module_tree(model, color=False, show_all=False)
    captured = capsys.readouterr()
    output = captured.out

    assert "model.layers.0: DummyBlock" in output
    assert "model.layers.1: DummyBlock" in output
    assert "model.layers.2: DummyBlock" in output
    assert "model.layers.3: DummyBlock" in output
    assert "model.layers.4: DummyBlock" not in output
    assert "model.layers.5: DummyBlock" not in output
    assert "collapsed (repeats 4..5, per-layer" in output
    assert "model.heads.4: Linear" in output
    assert "model.lm_head: Linear" in output


def test_print_module_tree_can_show_all_layers(capsys):
    model = DummyStackModel(num_layers=6)

    print_module_tree(model, color=False, show_all=False, layers_show=None)
    captured = capsys.readouterr()
    output = captured.out

    assert "model.layers.4: DummyBlock" in output
    assert "model.layers.5: DummyBlock" in output
    assert "collapsed (repeats" not in output


class _LazyTurtleInner(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(8):
            self.add_module(f"layer{i}", nn.Linear(16, 32, bias=False))


class _LazyTurtleShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _LazyTurtleInner()


def _write_lazy_turtle_index(model_dir: Path, shard_name: str, tensors: dict[str, torch.Tensor]) -> None:
    weight_map = {k: shard_name for k in tensors}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_lazy_turtle_parallel_materialization_preserves_src_dst_values(tmp_path, monkeypatch, workers):
    """Verify parallel and sequential LazyTurtle loading produce the same correct values."""

    monkeypatch.setenv("GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS", str(workers))

    model_dir = tmp_path / "source"
    model_dir.mkdir()

    # Store checkpoint weights in the transposed (in, out) layout. Target Linear
    # modules expect (out, in), so _transform_checkpoint_tensor must transpose.
    source = {}
    expected = {}
    for i in range(8):
        name = f"inner.layer{i}.weight"
        w_in_out = torch.randn(16, 32, dtype=torch.float32)
        source[name] = w_in_out
        expected[name] = w_in_out.transpose(0, 1).contiguous()

    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    shell = _LazyTurtleShell()
    for p in shell.parameters():
        p.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
    )

    for i in range(8):
        loaded = getattr(shell.inner, f"layer{i}").weight
        exp = expected[f"inner.layer{i}.weight"]
        assert loaded.shape == exp.shape
        assert loaded.dtype == torch.float32
        assert torch.equal(loaded, exp)

    # Confirm the source checkpoint values were not mutated by reloading it.
    reloaded = {}
    from safetensors import safe_open
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in source:
            reloaded[name] = f.get_tensor(name)
    for name in source:
        assert torch.equal(reloaded[name], source[name])


def _make_simple_checkpoint(tmp_path):
    """Create a tiny safetensors checkpoint for a _LazyTurtleShell and return the model dir."""
    model_dir = tmp_path / "source"
    model_dir.mkdir()

    source = {"inner.layer0.weight": torch.randn(16, 32, dtype=torch.float32)}
    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)
    return model_dir, source


def test_lazy_turtle_materialize_submodule_uses_module_path(tmp_path, monkeypatch):
    """Providing module_path should skip the whole-model _get_qualified_name scan."""

    model_dir, source = _make_simple_checkpoint(tmp_path)
    shell = _LazyTurtleShell()
    for p in shell.parameters():
        p.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("_get_qualified_name should not be called when module_path is provided")

    monkeypatch.setattr(structure, "_get_qualified_name", _should_not_be_called)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
    )

    expected = source["inner.layer0.weight"].transpose(0, 1).contiguous()
    loaded = shell.inner.layer0.weight
    assert loaded.shape == expected.shape
    assert torch.equal(loaded, expected)


def test_lazy_turtle_checkpoint_tensors_for_submodule_uses_module_path(tmp_path, monkeypatch):
    """Providing module_path should skip the whole-model _get_qualified_name scan."""

    model_dir, source = _make_simple_checkpoint(tmp_path)
    shell = _LazyTurtleShell()

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("_get_qualified_name should not be called when module_path is provided")

    monkeypatch.setattr(structure, "_get_qualified_name", _should_not_be_called)

    tensors = turtle.checkpoint_tensors_for_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        recurse=True,
        module_path="inner",
    )

    assert "layer0.weight" in tensors
    assert torch.equal(tensors["layer0.weight"], source["inner.layer0.weight"])


def test_lazy_turtle_materialize_submodule_falls_back_to_get_qualified_name(tmp_path, monkeypatch):
    """Without module_path the helper should still resolve the path automatically."""

    model_dir, source = _make_simple_checkpoint(tmp_path)
    shell = _LazyTurtleShell()
    for p in shell.parameters():
        p.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    called = []
    original = structure._get_qualified_name

    def _spy(*args, **kwargs):
        called.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(structure, "_get_qualified_name", _spy)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
    )

    assert len(called) == 1
    expected = source["inner.layer0.weight"].transpose(0, 1).contiguous()
    assert torch.equal(shell.inner.layer0.weight, expected)
