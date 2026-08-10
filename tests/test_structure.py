import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import gptqmodel.utils.structure as structure
from gptqmodel.models.base import BaseQModel
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


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value
        self.result_calls = 0

    def result(self):
        self.result_calls += 1
        return self.value


class _ImmediatePool:
    def __init__(self):
        self.futures = []

    def submit(self, _device, fn, *args):
        future = _ImmediateFuture(fn(*args))
        self.futures.append(future)
        return future


def test_run_lazy_turtle_jobs_drains_all_lanes_before_reraising(monkeypatch):
    pool = _ImmediatePool()
    monkeypatch.setattr("gptqmodel.DEVICE_THREAD_POOL", pool)
    completed = []

    def worker(job):
        if job == 0:
            raise OSError("broken shard")
        completed.append(job)
        return job

    with pytest.raises(OSError, match="broken shard"):
        structure._run_lazy_turtle_jobs([0, 1, 2, 3], worker, max_workers=2)

    assert completed == [1, 3]
    assert [future.result_calls for future in pool.futures] == [1, 1]


def test_run_lazy_turtle_jobs_preserves_input_order(monkeypatch):
    pool = _ImmediatePool()
    monkeypatch.setattr("gptqmodel.DEVICE_THREAD_POOL", pool)

    assert structure._run_lazy_turtle_jobs([3, 1, 2, 0], lambda value: value * 2, max_workers=3) == [6, 2, 4, 0]


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
    weight_map = dict.fromkeys(tensors, shard_name)
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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lazy_turtle_cuda_materialization_uses_only_device_owner_thread(tmp_path, monkeypatch):
    """Parallel loader lanes may prepare CPU sources but only ThreadX may dispatch CUDA copies."""

    from gptqmodel import DEVICE_THREAD_POOL

    monkeypatch.setenv("GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS", "4")
    model_dir = tmp_path / "cuda_owner"
    model_dir.mkdir()
    source = {
        f"inner.layer{i}.weight": torch.randn(16, 32, dtype=torch.float32)
        for i in range(8)
    }
    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    previous_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _LazyTurtleShell()
    finally:
        torch.set_default_device(previous_device)
    for parameter in shell.parameters():
        parameter.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    # Reproduce the prefetch handoff: checkpoint lanes first materialize CPU
    # sources, then the CUDA request moves those already-loaded tensors.
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
    )

    copy_threads = []
    original_copy = structure._copy_tensor_to_target

    def copy_on_owner(source_tensor, target_tensor, *, non_blocking):
        copy_threads.append(
            (
                threading.get_ident(),
                str(target_tensor.device),
                DEVICE_THREAD_POOL.is_device_owner_thread(target_tensor.device),
            )
        )
        original_copy(source_tensor, target_tensor, non_blocking=non_blocking)
        # Exercise the caller's pinned-source lifetime path. The real helper
        # returns True when cudaHostRegister makes this copy non-blocking.
        return True

    monkeypatch.setattr(structure, "_copy_tensor_to_target", copy_on_owner)

    target_device = torch.device("cuda", 0)
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=target_device,
        module_path="inner",
    )

    assert len(copy_threads) == len(source)
    assert all(device == "cuda:0" and is_owner for _, device, is_owner in copy_threads)
    assert len({thread_id for thread_id, _, _ in copy_threads}) == 1

    loaded = DEVICE_THREAD_POOL.do(
        target_device,
        lambda: {
            f"inner.layer{i}.weight": getattr(shell.inner, f"layer{i}").weight.detach().cpu()
            for i in range(8)
        },
    )
    for name, expected_source in source.items():
        torch.testing.assert_close(loaded[name], expected_source.transpose(0, 1).contiguous(), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_copy_helper_routes_off_owner_cuda_call_through_threadx(monkeypatch):
    """The lowest copy boundary repairs an off-owner call instead of dispatching CUDA directly."""

    from gptqmodel import get_device_thread_pool

    pool = get_device_thread_pool()
    device = torch.device("cuda", 0)
    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    real_do = pool.do
    target = real_do(device, torch.empty_like, source, device=device)
    dispatches = []
    caller_thread = threading.get_ident()

    def record_do(routed_device, fn, *args, **kwargs):
        dispatches.append((threading.get_ident(), str(torch.device(routed_device))))
        return real_do(routed_device, fn, *args, **kwargs)

    monkeypatch.setattr(pool, "do", record_do)
    source_is_async = structure._copy_tensor_to_target(source, target, non_blocking=False)

    assert source_is_async is False
    assert dispatches == [(caller_thread, "cuda:0")]
    loaded = real_do(device, lambda: target.cpu())
    torch.testing.assert_close(loaded, source, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_lazy_turtle_multi_cuda_batch_has_one_owner_per_device(tmp_path, monkeypatch):
    """A mixed-device batch must split into one FIFO ThreadX task per physical CUDA device."""

    from gptqmodel import DEVICE_THREAD_POOL

    model_dir = tmp_path / "multi_cuda_owner"
    model_dir.mkdir()
    source = {
        "inner.layer0.weight": torch.randn(16, 32, dtype=torch.float32),
        "inner.layer1.weight": torch.randn(16, 32, dtype=torch.float32),
        "inner.layer2.weight": torch.randn(16, 32, dtype=torch.float32),
        "inner.layer3.weight": torch.randn(16, 32, dtype=torch.float32),
    }
    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    previous_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _LazyTurtleShell()
    finally:
        torch.set_default_device(previous_device)
    for parameter in shell.parameters():
        parameter.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    initial_batch = [
        (shell.inner.layer0, "inner.layer0", torch.device("cpu")),
        (shell.inner.layer1, "inner.layer1", torch.device("cpu")),
        (shell.inner.layer2, "inner.layer2", torch.device("cpu")),
        (shell.inner.layer3, "inner.layer3", torch.device("cpu")),
    ]
    turtle.materialize_submodules(target_model=shell, submodules=initial_batch)

    copy_threads_by_device = {}
    tie_calls = []
    original_copy = structure._copy_tensor_to_target

    def copy_on_owner(source_tensor, target_tensor, *, non_blocking):
        device = str(target_tensor.device)
        assert DEVICE_THREAD_POOL.is_device_owner_thread(target_tensor.device)
        copy_threads_by_device.setdefault(device, set()).add(threading.get_ident())
        original_copy(source_tensor, target_tensor, non_blocking=non_blocking)
        return True

    monkeypatch.setattr(structure, "_copy_tensor_to_target", copy_on_owner)
    monkeypatch.setattr(shell, "tie_weights", lambda: tie_calls.append(1), raising=False)
    DEVICE_THREAD_POOL.do(
        torch.device("cuda", 0),
        turtle.materialize_submodules,
        target_model=shell,
        submodules=[
            (shell.inner.layer0, "inner.layer0", torch.device("cuda", 0)),
            (shell.inner.layer1, "inner.layer1", torch.device("cuda", 1)),
            (shell.inner.layer2, "inner.layer2", torch.device("cuda", 0)),
            (shell.inner.layer3, "inner.layer3", torch.device("cpu")),
        ],
    )

    assert set(copy_threads_by_device) == {"cuda:0", "cuda:1"}
    assert all(len(thread_ids) == 1 for thread_ids in copy_threads_by_device.values())
    assert copy_threads_by_device["cuda:0"].isdisjoint(copy_threads_by_device["cuda:1"])
    assert tie_calls == [1]

    # Also cover the explicit no-tie route; an already materialized tensor is
    # a no-op but must still traverse and drain the CUDA owner queue.
    turtle.materialize_submodules(
        target_model=shell,
        submodules=[(shell.inner.layer1, "inner.layer1", torch.device("cuda", 1))],
        tie_weights=False,
    )

    loaded0 = DEVICE_THREAD_POOL.do(torch.device("cuda", 0), lambda: shell.inner.layer0.weight.detach().cpu())
    loaded2 = DEVICE_THREAD_POOL.do(torch.device("cuda", 0), lambda: shell.inner.layer2.weight.detach().cpu())
    loaded1 = DEVICE_THREAD_POOL.do(torch.device("cuda", 1), lambda: shell.inner.layer1.weight.detach().cpu())
    torch.testing.assert_close(loaded0, source["inner.layer0.weight"].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(loaded2, source["inner.layer2.weight"].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(loaded1, source["inner.layer1.weight"].transpose(0, 1), rtol=0, atol=0)


def test_lazy_turtle_cuda_routing_drains_failed_future(monkeypatch):
    """A routed device failure is re-raised only after its future is consumed."""

    import gptqmodel

    expected_error = RuntimeError("synthetic CUDA materialization failure")

    class _FailedFuture:
        def __init__(self):
            self.result_calls = 0

        def result(self):
            self.result_calls += 1
            raise expected_error

    failed_future = _FailedFuture()

    class _FailedPool:
        @staticmethod
        def is_device_owner_thread(_device):
            return False

        @staticmethod
        def submit(_device, _fn, *args, **kwargs):
            return failed_future

    monkeypatch.setattr(gptqmodel, "DEVICE_THREAD_POOL", _FailedPool())
    turtle = object.__new__(LazyTurtle)
    with pytest.raises(RuntimeError, match="synthetic CUDA materialization failure"):
        turtle.materialize_submodules(
            target_model=nn.Module(),
            submodules=[(nn.Linear(1, 1), "leaf", torch.device("cuda", 0))],
        )
    assert failed_future.result_calls == 1


def test_lazy_turtle_mixed_routing_collects_direct_and_cpu_errors(monkeypatch):
    """Mixed batches drain both inline CUDA and non-CUDA groups before raising."""

    import gptqmodel

    owner_checks = []

    class _OwnerPool:
        @staticmethod
        def is_device_owner_thread(device):
            owner_checks.append(str(device))
            return True

    monkeypatch.setattr(gptqmodel, "DEVICE_THREAD_POOL", _OwnerPool())
    turtle = object.__new__(LazyTurtle)
    target = nn.Linear(1, 1)
    with pytest.raises(AttributeError, match="_lock"):
        turtle.materialize_submodules(
            target_model=nn.Module(),
            submodules=[
                (target, "cuda_leaf", torch.device("cuda", 0)),
                (target, "cpu_leaf", torch.device("cpu")),
            ],
        )
    assert owner_checks == ["cuda:0", "cuda:0", "cuda:0"]


def test_lazy_turtle_loader_casts_cpu_sources_before_single_and_batch_copy(tmp_path):
    """Loader lanes cast checkpoint fp32 to the target bf16 before either copy path."""

    model_dir = tmp_path / "loader_dtype_cast"
    model_dir.mkdir()
    source = {
        f"inner.layer{i}.weight": torch.randn(16, 32, dtype=torch.float32)
        for i in range(8)
    }
    save_file(source, str(model_dir / "model.safetensors"))
    _write_lazy_turtle_index(model_dir, "model.safetensors", source)

    def make_bf16_shell():
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.bfloat16)
        try:
            result = _LazyTurtleShell()
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.set_default_device(previous_device)
        for parameter in result.parameters():
            parameter.requires_grad = False
        return result

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    single_shell = make_bf16_shell()
    turtle.materialize_submodule(
        target_model=single_shell,
        target_submodule=single_shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
    )

    batch_shell = make_bf16_shell()
    turtle.materialize_submodules(
        target_model=batch_shell,
        submodules=[
            (getattr(batch_shell.inner, f"layer{i}"), f"inner.layer{i}", torch.device("cpu"))
            for i in range(8)
        ],
    )

    for i in range(8):
        expected = source[f"inner.layer{i}.weight"].transpose(0, 1).to(torch.bfloat16)
        single_weight = getattr(single_shell.inner, f"layer{i}").weight
        batch_weight = getattr(batch_shell.inner, f"layer{i}").weight
        assert single_weight.dtype == batch_weight.dtype == torch.bfloat16
        torch.testing.assert_close(single_weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(batch_weight, expected, rtol=0, atol=0)


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


def test_lazy_turtle_exact_checkpoint_key_skips_general_alias_resolution(monkeypatch):
    turtle = object.__new__(LazyTurtle)
    exact_name = "model.layers.0.mlp.experts.7.down_proj.weight"
    turtle._weight_map = {exact_name: "model.safetensors"}

    def fail_general_resolution(*_args, **_kwargs):
        raise AssertionError("exact checkpoint keys must not enter general alias resolution")

    monkeypatch.setattr(turtle, "_resolve_checkpoint_tensor_name", fail_general_resolution)

    assert turtle._resolve_direct_checkpoint_tensor_source(
        "model.layers.0.mlp.experts.7.down_proj",
        "weight",
    ) == (exact_name, None, None, None)


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


def test_lazy_turtle_materialize_submodule_recurse_false_avoids_model_scan(tmp_path, monkeypatch):
    """recurse=False should only build a minimal modules_by_name map and not scan the model."""

    model_dir, _ = _make_simple_checkpoint(tmp_path)

    prev_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _LazyTurtleShell()
    finally:
        torch.set_default_device(prev_device)
    for p in shell.parameters():
        p.requires_grad = False

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("target_model.named_modules() should not be called when recurse=False")

    monkeypatch.setattr(shell, "named_modules", _should_not_be_called)

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
        recurse=False,
    )

    # With recurse=False the submodule's own parameters are not loaded; children stay meta.
    assert shell.inner.layer0.weight.device.type == "meta"


def test_lazy_turtle_materialize_submodule_recurse_true_loads_children(tmp_path):
    """recurse=True should descend into children and load their parameters."""

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

    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=shell.inner,
        device=torch.device("cpu"),
        module_path="inner",
        recurse=True,
    )

    expected = source["inner.layer0.weight"].transpose(0, 1).contiguous()
    assert torch.equal(shell.inner.layer0.weight, expected)


class _TestBaseQModel(BaseQModel):
    """Minimal BaseQModel instance for pre_quantize unit tests."""

    def __init__(self, model: nn.Module, turtle_model: LazyTurtle, quantize_config: Any) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.turtle_model = turtle_model
        self.quantize_config = quantize_config


def test_base_qmodel_pre_quantize_batches_leaf_modules(tmp_path, monkeypatch):
    """pre_quantize with skip_module_names should batch load leaves and recurse=False on containers."""

    num_experts = 4
    model_dir = tmp_path / "pre_quant"
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
            linear = getattr(shell.layers[0].mlp["experts"][i], proj)
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

    gptq_model = _TestBaseQModel(
        model=shell,
        turtle_model=turtle,
        quantize_config=SimpleNamespace(device="cpu"),
    )

    skip_names = {
        f"mlp.experts.{i}.{proj}"
        for i in range(num_experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }

    submodule_calls = []
    submodules_calls = []

    orig_submodule = turtle.materialize_submodule
    orig_submodules = turtle.materialize_submodules

    tie_weights_calls = []

    def _patched_submodule(*, target_model, target_submodule, device, non_blocking=False, module_path=None, recurse=True, tie_weights=True, show_progress=True):
        submodule_calls.append({"module_path": module_path, "recurse": recurse, "device": str(device), "tie_weights": tie_weights, "show_progress": show_progress})
        return orig_submodule(
            target_model=target_model,
            target_submodule=target_submodule,
            device=device,
            non_blocking=non_blocking,
            module_path=module_path,
            recurse=recurse,
            tie_weights=tie_weights,
            show_progress=show_progress,
        )

    def _patched_submodules(*, target_model, submodules, non_blocking=False, tie_weights=True, show_progress=True):
        submodules_calls.append({"submodules": submodules, "tie_weights": tie_weights, "show_progress": show_progress})
        return orig_submodules(
            target_model=target_model,
            submodules=submodules,
            non_blocking=non_blocking,
            tie_weights=tie_weights,
            show_progress=show_progress,
        )

    def _patched_tie_weights():
        tie_weights_calls.append(1)

    monkeypatch.setattr(shell, "tie_weights", _patched_tie_weights, raising=False)

    monkeypatch.setattr(turtle, "materialize_submodule", _patched_submodule)
    monkeypatch.setattr(turtle, "materialize_submodules", _patched_submodules)

    gptq_model.pre_quantize(
        shell.layers[0],
        skip_module_names=skip_names,
        layer_name="layers.0",
    )

    assert len(submodules_calls) == 1
    batch = submodules_calls[0]["submodules"]
    assert submodules_calls[0]["tie_weights"] is False
    assert len(batch) == len(skip_names)
    for _, path, device in batch:
        assert path.startswith("layers.0.mlp.experts.")
        assert str(device) == "cpu"

    # Every non-leaf submodule call must use recurse=False so containers do not duplicate leaf loads.
    for call in submodule_calls:
        assert call["recurse"] is False

    # No per-submodule weight tying; the whole layer should tie exactly once.
    for call in submodule_calls:
        assert call["tie_weights"] is False
        assert call["show_progress"] is False
    for call in submodules_calls:
        assert call["tie_weights"] is False
        assert call["show_progress"] is False
    assert len(tie_weights_calls) == 1

    # All skipped leaf projections should be materialized as one batch.
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            full = f"layers.0.mlp.experts.{i}.{proj}"
            assert any(path == full for _, path, _ in batch)
            loaded = getattr(shell.layers[0].mlp["experts"][i], proj).weight
            assert loaded.device.type == "cpu"
            assert torch.equal(
                loaded,
                source[f"layers.0.mlp.experts.{i}.{proj}.weight"].transpose(0, 1).contiguous(),
            )


def test_lazy_turtle_parallel_workers_heuristic(monkeypatch):
    """The worker heuristic should scale with target devices, be capped, and honor env overrides."""
    env_key = "GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS"
    import sys

    def _is_gil_enabled():
        return False

    monkeypatch.setattr(sys, "_is_gil_enabled", _is_gil_enabled, raising=False)
    monkeypatch.setattr(structure, "_is_gil_enabled", _is_gil_enabled, raising=False)

    # No env override: CPU path uses os.cpu_count(), capped at 8.
    monkeypatch.delenv(env_key, raising=False)
    with monkeypatch.context() as m:
        m.setattr(structure.os, "cpu_count", lambda: 16)
        assert structure._lazy_turtle_parallel_workers(20, 0) == 8

    # One CUDA device: small pool (2) is enough for async H2D.
    monkeypatch.delenv(env_key, raising=False)
    assert structure._lazy_turtle_parallel_workers(20, 1) == 2

    # Four CUDA devices: pool scales to 8 but is capped.
    monkeypatch.delenv(env_key, raising=False)
    assert structure._lazy_turtle_parallel_workers(20, 4) == 8

    # Fewer jobs than the heuristic still clamps to num_jobs.
    monkeypatch.delenv(env_key, raising=False)
    assert structure._lazy_turtle_parallel_workers(1, 4) == 1

    # Env override wins.
    monkeypatch.setenv(env_key, "3")
    assert structure._lazy_turtle_parallel_workers(20, 4) == 3


def test_lazy_turtle_host_register_shard_uses_safetensors_api(tmp_path, monkeypatch):
    """_host_register_shard must read tensor keys with the real safetensors API and call cudaHostRegister."""

    model_dir, source = _make_simple_checkpoint(tmp_path)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shard_path = str(model_dir / "model.safetensors")
    handler = turtle._get_shard_handler(shard_path)

    calls = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            calls.append((ptr, size, flags))
            return 0

    monkeypatch.setattr(structure.torch.cuda, "cudart", lambda: FakeCudart())
    turtle._host_register_shard(shard_path, handler)

    assert shard_path in turtle._shard_pin_ranges
    start, size = turtle._shard_pin_ranges[shard_path]
    assert size > 0
    assert len(calls) == 1
    assert calls[0][0] == start
    assert calls[0][1] == size
    assert calls[0][2] == 0


def test_lazy_turtle_skips_shard_larger_than_total_pin_budget(tmp_path, monkeypatch):
    """An oversized mmap must fall back to pageable copies without retrying registration."""

    model_dir, _ = _make_simple_checkpoint(tmp_path)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shard_path = str(model_dir / "model.safetensors")
    real_handler = turtle._get_shard_handler(shard_path)
    key_calls = []

    class _CountingHandler:
        def keys(self):
            key_calls.append(1)
            return real_handler.keys()

        def get_tensor(self, key):
            return real_handler.get_tensor(key)

    def reject_registration():
        raise AssertionError("oversized shard unexpectedly reached cudaHostRegister")

    logs = []
    monkeypatch.setattr(structure.torch.cuda, "cudart", reject_registration)
    monkeypatch.setattr(structure, "_log_info", lambda message, *args: logs.append(message % args))
    turtle._max_pinned_bytes = 1

    handler = _CountingHandler()
    turtle._host_register_shard(shard_path, handler)
    turtle._host_register_shard(shard_path, handler)

    assert key_calls == [1]
    assert turtle._shard_pin_ineligible == {shard_path}
    assert shard_path not in turtle._shard_pin_ranges
    assert len(logs) == 1
    assert "exceeds 0.00 GB pin budget" in logs[0]
