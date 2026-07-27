import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


class _LagunaExpert(nn.Module):
    """One MoE expert with the three projection weights used by Laguna-S-2.1."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, out_features, bias=False)
        self.up_proj = nn.Linear(in_features, out_features, bias=False)
        self.down_proj = nn.Linear(out_features, in_features, bias=False)


class _LagunaMoELayer(nn.Module):
    """One transformer layer with a 256-expert MoE MLP block."""

    def __init__(self, num_experts: int = 256, in_features: int = 512, hidden_features: int = 1408):
        super().__init__()
        self.mlp = nn.ModuleDict({
            "experts": nn.ModuleList([
                _LagunaExpert(in_features, hidden_features)
                for _ in range(num_experts)
            ])
        })


class _LagunaShellModel(nn.Module):
    """Tiny shell model that mirrors the Laguna-S-2.1 MoE layout."""

    def __init__(self, num_layers: int = 1, num_experts: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            _LagunaMoELayer(num_experts=num_experts)
            for _ in range(num_layers)
        ])


@pytest.mark.parametrize("workers", [1, 4])
def test_lazy_turtle_materialize_submodules_batch_moe_256_experts(tmp_path, monkeypatch, workers):
    """Batch loading should load an entire MoE projection at once with multiple workers."""

    monkeypatch.setenv("GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS", str(workers))

    num_experts = 256
    model_dir = tmp_path / "laguna_moe"
    model_dir.mkdir()

    # Build the shell on meta so LazyTurtle can populate real weights.
    prev_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _LagunaShellModel(num_layers=1, num_experts=num_experts)
    finally:
        torch.set_default_device(prev_device)
    for p in shell.parameters():
        p.requires_grad = False

    source = {}
    expected = {}
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expert = shell.layers[0].mlp["experts"][i]
            linear = getattr(expert, proj)
            # Checkpoint stores (in, out); _transform_checkpoint_tensor transposes to (out, in).
            w_in_out = torch.randn(linear.in_features, linear.out_features, dtype=torch.float32)
            name = f"layers.0.mlp.experts.{i}.{proj}.weight"
            source[name] = w_in_out
            expected[f"experts.{i}.{proj}"] = w_in_out.transpose(0, 1).contiguous()

    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    # Build the batch request: load every `up_proj` for layer 0 in one call.
    submodules = [
        (shell.layers[0].mlp["experts"][i].up_proj, f"layers.0.mlp.experts.{i}.up_proj", torch.device("cpu"))
        for i in range(num_experts)
    ]

    captured_logs = []
    original_log_info = structure._log_info

    def _capture_log(msg, *args, **kwargs):
        captured_logs.append(msg % args if args else msg)
        original_log_info(msg, *args, **kwargs)

    monkeypatch.setattr(structure, "_log_info", _capture_log)

    turtle.materialize_submodules(
        target_model=shell,
        submodules=submodules,
        non_blocking=False,
    )

    # Confirm the batched log mentions all tensors and multiple workers.
    batch_logs = [m for m in captured_logs if "loading" in m and "grouped tensors" in m]
    assert len(batch_logs) == 1, f"expected one batch log, got {len(batch_logs)}: {captured_logs}"
    assert f"loading {num_experts} grouped tensors" in batch_logs[0]
    assert f"using {workers} worker(s)" in batch_logs[0]

    for i in range(num_experts):
        loaded = shell.layers[0].mlp["experts"][i].up_proj.weight
        exp = expected[f"experts.{i}.up_proj"]
        assert loaded.shape == exp.shape
        assert loaded.dtype == torch.float32
        assert torch.equal(loaded, exp)

    # The gate/down projections were not touched by the filtered batch.
    for i in range(num_experts):
        assert shell.layers[0].mlp["experts"][i].gate_proj.weight.device.type == "meta"
        assert shell.layers[0].mlp["experts"][i].down_proj.weight.device.type == "meta"


def test_lazy_turtle_materialize_submodules_holds_turtle_lock(tmp_path, monkeypatch):
    """The batch materialize path must keep the LazyTurtle lock while opening shard handlers."""

    num_experts = 4
    model_dir = tmp_path / "laguna_lock"
    model_dir.mkdir()

    prev_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _LagunaShellModel(num_layers=1, num_experts=num_experts)
    finally:
        torch.set_default_device(prev_device)
    for p in shell.parameters():
        p.requires_grad = False

    source = {}
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expert = shell.layers[0].mlp["experts"][i]
            linear = getattr(expert, proj)
            w_in_out = torch.randn(linear.in_features, linear.out_features, dtype=torch.float32)
            source[f"layers.0.mlp.experts.{i}.{proj}.weight"] = w_in_out

    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    lock_states: list[bool] = []
    original_get_shard_handler = LazyTurtle._get_shard_handler

    def _patched_get_shard_handler(self, shard_path: str) -> Any:
        lock_states.append(self._lock._is_owned())
        return original_get_shard_handler(self, shard_path)

    monkeypatch.setattr(LazyTurtle, "_get_shard_handler", _patched_get_shard_handler)

    submodules = [
        (shell.layers[0].mlp["experts"][i].up_proj, f"layers.0.mlp.experts.{i}.up_proj", torch.device("cpu"))
        for i in range(num_experts)
    ]
    turtle.materialize_submodules(
        target_model=shell,
        submodules=submodules,
        non_blocking=False,
    )

    assert len(lock_states) > 0, "expected shard handler(s) to be opened during batch load"
    assert all(lock_states), "LazyTurtle lock must be held while opening shard handlers"
