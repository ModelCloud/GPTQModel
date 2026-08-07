# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import gptqmodel.models.moe_lifecycle as moe_lifecycle
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    HessianConfig,
    MoEConfig,
    MoEExecutionConfig,
    QuantizeConfig,
)


@pytest.mark.parametrize(
    ("configured", "gil_disabled", "visible_gpu_count", "expected"),
    [
        pytest.param(True, True, 2, True, id="enabled-free-threaded-multi-gpu"),
        pytest.param(True, True, 1, False, id="single-visible-gpu"),
        pytest.param(True, False, 2, False, id="gil-enabled"),
        pytest.param(False, True, 2, False, id="explicit-opt-out"),
    ],
)
def test_moe_parallel_input_capture_runtime_eligibility(
    monkeypatch,
    configured: bool,
    gil_disabled: bool,
    visible_gpu_count: int,
    expected: bool,
):
    monkeypatch.setattr(moe_lifecycle, "has_gil_disabled", lambda: gil_disabled)
    monkeypatch.setattr(moe_lifecycle.torch.cuda, "device_count", lambda: visible_gpu_count)

    quantize_config = SimpleNamespace(
        moe=MoEConfig(
            routing=ExpertsRoutingBypass(),
            execution=MoEExecutionConfig(parallel_input_capture=configured),
        )
    )
    assert moe_lifecycle._moe_parallel_input_capture_eligible(quantize_config) is expected


def test_moe_parallel_input_capture_missing_config_uses_default(monkeypatch):
    monkeypatch.setattr(moe_lifecycle, "has_gil_disabled", lambda: True)
    monkeypatch.setattr(moe_lifecycle.torch.cuda, "device_count", lambda: 2)

    assert moe_lifecycle._moe_parallel_input_capture_eligible(SimpleNamespace()) is True


def test_moe_execution_config_serialization_and_round_trip():
    qcfg = QuantizeConfig(
        moe=MoEConfig(
            routing=ExpertsRoutingBypass(),
            execution=MoEExecutionConfig(batch_size=4, parallel_input_capture=False),
        )
    )
    payload = qcfg.to_dict()
    serialized_moe = payload["meta"]["moe"]
    assert serialized_moe == {
        "routing": {"class": "ExpertsRoutingBypass"},
        "execution": {"batch_size": 4, "parallel_input_capture": False},
    }
    assert "moe_parallel_input_capture" not in payload["meta"]

    restored = QuantizeConfig.from_quant_config(copy.deepcopy(payload))
    assert isinstance(restored.moe.routing, ExpertsRoutingBypass)
    assert restored.moe.execution.batch_size == 4
    assert restored.moe.execution.parallel_input_capture is False


def test_removed_moe_parallel_constructor_field_is_rejected():
    with pytest.raises(ValueError, match="moe.execution.parallel_input_capture"):
        QuantizeConfig(
            moe=MoEConfig(routing=ExpertsRoutingBypass()),
            moe_parallel_input_capture=False,
        )


def _make_moe_bypass_processor(
    *,
    num_experts: int = 3,
    hidden_size: int = 16,
    intermediate_size: int = 12,
    expert_gate_declaration: str | None = "expert_gate=experts._apply_gate",
):
    """Build a GPTQProcessor with a small MoE bypass subset for unit testing."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        sym=True,
        desc_act=False,
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
        hessian=HessianConfig(staging_dtype=torch.float32),
        enable_shared_hessian_cache=True,
    )

    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # Processor normally gets gptq_model from ModuleLooper; attach a minimal fake.
    processor.gptq_model = type("FakeModel", (), {"moe_lifecycle_hooks": GateUpDownMoELifecycleHooks()})()

    subset = {}
    for expert_idx in range(num_experts):
        for proj, in_features in (
            ("gate_proj", hidden_size),
            ("up_proj", hidden_size),
            ("down_proj", intermediate_size),
        ):
            name = f"mlp.experts.{expert_idx}.{proj}"
            module = nn.Linear(in_features, 8, bias=False)
            named = NamedModule(
                module,
                name=name,
                full_name=f"model.layers.0.{name}",
                layer_index=0,
            )
            # Module-tree flags are normally parsed from the model definition by
            # ``ModuleLooper.create_named_modules``; set them explicitly here so
            # the processor can identify the routed expert and gate/up/down role
            # without relying on model attribute names or lifecycle hooks.
            flags = {"routed", proj.split("_")[0]}
            if expert_gate_declaration is not None:
                flags.add(expert_gate_declaration)
            named.state["module_tree_flags"] = frozenset(flags)
            named.state["module_tree_expert_group"] = f"routed-specialist-{expert_idx}"
            processor.preprocess(named)
            subset[name] = named

    return processor, subset


def test_moe_bypass_down_proj_is_isolated_per_expert():
    """Per-expert down projections must not share a Hessian with other experts."""

    processor, subset = _make_moe_bypass_processor(num_experts=3)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    down_keys = set()
    for name, named_module in subset.items():
        if "down_proj" in name:
            task = processor.tasks[name]
            # Each down expert forms a group of one, so the shared Hessian path is skipped.
            assert task._shared_hessian_accum_key is None
            assert task._shared_hessian_state is None
            assert task._shared_hessian_logical_key is False
            down_keys.add(name)

    assert len(down_keys) == 3


def test_moe_bypass_gate_and_up_share_one_logical_group():
    """All MoE gate/up projections see the same hidden state and share one Hessian."""

    processor, subset = _make_moe_bypass_processor(num_experts=3)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    gate_up_tasks = [processor.tasks[name] for name in subset if name.endswith((".gate_proj", ".up_proj"))]
    assert len(gate_up_tasks) == 6

    accum_keys = {task._shared_hessian_accum_key for task in gate_up_tasks}
    assert len(accum_keys) == 1
    assert list(gate_up_tasks)[0]._shared_hessian_logical_key is True

    # The cache key must not depend on the physical storage pointer, otherwise
    # per-device copies of the same hidden state create inflated sample counts.
    cache_source_a = torch.randn(2, 8, 16)
    cache_source_b = torch.randn(2, 8, 16)
    key_a = processor._shared_hessian_cache_key(gate_up_tasks[0], cache_source_a, 0, None)
    key_b = processor._shared_hessian_cache_key(gate_up_tasks[1], cache_source_b, 0, None)
    assert key_a == key_b


def test_moe_bypass_sample_counts_match_dense_reference():
    """After forwarding one batch, every MoE expert projection records the same sample count."""

    processor, subset = _make_moe_bypass_processor(num_experts=2, hidden_size=16, intermediate_size=12)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    batch_tokens = 2 * 8  # (batch, seq) = (2, 8)
    hidden = torch.randn(2, 8, 16)

    for expert_idx in range(2):
        gate_name = f"mlp.experts.{expert_idx}.gate_proj"
        up_name = f"mlp.experts.{expert_idx}.up_proj"
        down_name = f"mlp.experts.{expert_idx}.down_proj"

        gate_hook = processor.pre_process_fwd_hook(gate_name)
        gate_hook(subset[gate_name].module, (hidden,), subset[gate_name].module(hidden))

        up_hook = processor.pre_process_fwd_hook(up_name)
        up_hook(subset[up_name].module, (hidden,), subset[up_name].module(hidden))

        # Each expert's down projection receives a distinct intermediate activation.
        down_hidden = torch.randn(2, 8, 12)
        down_hook = processor.pre_process_fwd_hook(down_name)
        down_hook(subset[down_name].module, (down_hidden,), subset[down_name].module(down_hidden))

    for name in subset:
        task = processor.tasks[name]
        task.finalize_hessian(target_device=torch.device("cpu"))
        assert task.nsamples == batch_tokens, f"{name} has {task.nsamples} samples, expected {batch_tokens}"


def _build_replica_and_run_bypass(
    processor,
    subset,
    hidden,
    keep_mask,
    *,
    expert_devices=None,
    forward_counts=None,
    activation=None,
    captured_tensors=None,
):
    """Attach forward hooks and run ``forward_to_all_experts`` through a fake MoE block."""

    # Match the intermediate size used by the processor's tasks.  In
    # `_make_moe_bypass_processor` the gate/up projections map hidden_size -> 8
    # and the down projection maps 8 -> hidden_size.
    gate_name = next(k for k in subset if k.endswith(".gate_proj"))
    intermediate_size = int(processor.tasks[gate_name].module.weight.shape[0])

    class FakeExpert(nn.Module):
        def __init__(self, device=None):
            super().__init__()
            # Use HookedLinear so the `forward_hook` attribute fires during forward.
            self.gate_proj = HookedLinear.from_linear(nn.Linear(hidden.shape[-1], intermediate_size, bias=False))
            self.up_proj = HookedLinear.from_linear(nn.Linear(hidden.shape[-1], intermediate_size, bias=False))
            self.down_proj = HookedLinear.from_linear(nn.Linear(intermediate_size, hidden.shape[-1], bias=False))
            self.act_fn = activation or nn.SiLU()
            if device is not None:
                self.gate_proj = self.gate_proj.to(device)
                self.up_proj = self.up_proj.to(device)
                self.down_proj = self.down_proj.to(device)

        def forward(self, value):
            return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))

    class FakeExperts(nn.ModuleList):
        def __init__(self, experts):
            super().__init__(experts)
            self.act_fn = activation or nn.SiLU()

        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return self.act_fn(gate) * up

    class FakeMoEBlock(nn.Module):
        def __init__(self, num_experts, devices=None):
            super().__init__()
            devices = devices or [None] * num_experts
            self.experts = FakeExperts([FakeExpert(d) for d in devices])

    class FakeLayer(nn.Module):
        def __init__(self, num_experts, devices=None):
            super().__init__()
            self.mlp = FakeMoEBlock(num_experts, devices)

    num_experts = len([k for k in subset if k.endswith(".gate_proj")])
    devices = expert_devices or [None] * num_experts
    replica_module = FakeLayer(num_experts, devices)
    moe_block = replica_module.mlp

    # Attach the processor's per-name forward hooks to the replica modules.
    for name in subset:
        parts = name.split(".")
        mod = replica_module
        for p in parts:
            mod = getattr(mod, p)
        mod.forward_hook = processor.pre_process_fwd_hook(name)
        if captured_tensors is not None:
            original_hook = mod.forward_hook

            def capture_tensor(module, inputs, output, *, module_name=name, hook=original_hook):
                captured_tensors[module_name] = {
                    "input": inputs[0].detach().clone(),
                    "output": output.detach().clone() if isinstance(output, torch.Tensor) else output,
                }
                return hook(module, inputs, output)

            mod.forward_hook = capture_tensor
        if forward_counts is not None:
            forward_counts[name] = 0

            def count_forward(_module, _inputs, *, module_name=name):
                forward_counts[module_name] += 1

            mod.register_forward_pre_hook(count_forward)

    class FakeModuleLooper:
        def __init__(self, mask):
            self._mask = mask
            self.paused = False

        def _set_processor_hooks_paused(self, processor, value):
            self.paused = value

        def _get_processor_mask(self, processor):
            return self._mask

        def _set_processor_mask(self, processor, mask):
            processor._mask_tls.value = mask

    processor._mask_tls = threading.local()
    processor._mask_tls.value = keep_mask
    processor._set_current_batch_index(0)

    hooks = GateUpDownMoELifecycleHooks()
    ordered_module_names = list(subset.keys())

    class FakeModelClass:
        @staticmethod
        def get_moe_module_name():
            return ("mlp",)

    def original_forward(hidden_states, **kwargs):
        return hidden_states

    return hooks.forward_to_all_experts(
        moe_block=moe_block,
        hidden_states=hidden,
        processor=processor,
        subset=subset,
        ordered_module_names=ordered_module_names,
        original_forward=original_forward,
        model_class=FakeModelClass,
        module_looper=FakeModuleLooper(keep_mask),
        moe_block_prefix="mlp",
        replica_module=replica_module,
    )


@pytest.mark.parametrize(
    "activation_declaration",
    ["expert_activation=expert.act_fn", "expert_activation=experts.act_fn"],
)
def test_moe_bypass_down_input_uses_module_tree_declared_activation(activation_declaration):
    """Down replay resolves the declared activation from the exact expert owner."""

    class SquareActivation(nn.Module):
        def forward(self, value):
            return value.square()

    processor, subset = _make_moe_bypass_processor(num_experts=2, hidden_size=16, intermediate_size=8)
    for named_module in subset.values():
        role = next(flag for flag in named_module.state["module_tree_flags"] if flag in {"gate", "up", "down"})
        named_module.state["module_tree_flags"] = frozenset(
            {"routed", role, activation_declaration}
        )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    hidden = torch.randn(2, 8, 16)
    keep_mask = torch.ones((2, 8), dtype=torch.bool)
    captured = {}

    _build_replica_and_run_bypass(
        processor,
        subset,
        hidden,
        keep_mask,
        activation=SquareActivation(),
        captured_tensors=captured,
    )

    for expert_idx in range(2):
        prefix = f"mlp.experts.{expert_idx}"
        gate = captured[f"{prefix}.gate_proj"]["output"]
        up = captured[f"{prefix}.up_proj"]["output"]
        down_input = captured[f"{prefix}.down_proj"]["input"]
        torch.testing.assert_close(down_input, gate.square() * up, rtol=0, atol=0)
        assert not torch.allclose(down_input, torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize("replay_declaration", [None, "expert_forward=expert.forward"])
def test_moe_bypass_down_input_exact_expert_forward(replay_declaration):
    """Fallback and explicitly declared forward replay both preserve exact expert math."""

    class SquareActivation(nn.Module):
        def forward(self, value):
            return value.square()

    processor, subset = _make_moe_bypass_processor(
        num_experts=1,
        hidden_size=16,
        intermediate_size=8,
        expert_gate_declaration=replay_declaration,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    hidden = torch.randn(2, 8, 16)
    keep_mask = torch.ones((2, 8), dtype=torch.bool)
    captured = {}

    _build_replica_and_run_bypass(
        processor,
        subset,
        hidden,
        keep_mask,
        activation=SquareActivation(),
        captured_tensors=captured,
    )

    gate = captured["mlp.experts.0.gate_proj"]["output"]
    up = captured["mlp.experts.0.up_proj"]["output"]
    down_input = captured["mlp.experts.0.down_proj"]["input"]
    torch.testing.assert_close(down_input, gate.square() * up, rtol=0, atol=0)
    assert not torch.allclose(down_input, torch.nn.functional.silu(gate) * up)


def test_moe_bypass_padded_sample_counts_single_device():
    """Routing=bypass must drop padding positions and count only valid tokens."""

    hidden_size = 16
    intermediate_size = 8
    processor, subset = _make_moe_bypass_processor(
        num_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    batch, seq_len = 2, 8
    hidden = torch.randn(batch, seq_len, hidden_size)
    # Mark half of the first sequence as padding; second sequence is fully valid.
    keep_mask = torch.tensor([[True] * 4 + [False] * 4, [True] * seq_len], dtype=torch.bool)
    valid_tokens = int(keep_mask.sum().item())

    _build_replica_and_run_bypass(processor, subset, hidden, keep_mask)

    for name in subset:
        task = processor.tasks[name]
        task.finalize_hessian(target_device=torch.device("cpu"))
        assert task.nsamples == valid_tokens, (
            f"{name} recorded {task.nsamples} samples, expected {valid_tokens}"
        )


def test_moe_bypass_gptq_skips_unused_projection_outputs():
    """Input-only GPTQ hooks must not execute projection GEMMs whose outputs are discarded."""

    processor, full_subset = _make_moe_bypass_processor(
        num_experts=2,
        hidden_size=16,
        intermediate_size=8,
    )
    hidden = torch.randn(2, 8, 16)
    keep_mask = torch.ones((2, 8), dtype=torch.bool)

    processor.prepare_subset(full_subset, subset_index=0, subset_total=1)
    down_counts = {}
    _build_replica_and_run_bypass(
        processor,
        full_subset,
        hidden,
        keep_mask,
        forward_counts=down_counts,
    )
    assert all(count == 1 for name, count in down_counts.items() if not name.endswith(".down_proj"))
    assert all(count == 0 for name, count in down_counts.items() if name.endswith(".down_proj"))
    processor.cleanup_subset(full_subset, subset_index=0, subset_total=1)

    gate_up_subset = {
        name: module for name, module in full_subset.items() if not name.endswith(".down_proj")
    }
    processor.prepare_subset(gate_up_subset, subset_index=0, subset_total=1)
    counters_before = {
        name: (processor.tasks[name].nsamples, processor.tasks[name].fwd_counter)
        for name in gate_up_subset
    }
    stats_before = processor.shared_hessian_stats()
    gate_up_counts = {}
    _build_replica_and_run_bypass(
        processor,
        gate_up_subset,
        hidden,
        keep_mask,
        forward_counts=gate_up_counts,
    )
    assert all(count == 0 for count in gate_up_counts.values())
    for name in gate_up_subset:
        task = processor.tasks[name]
        nsamples_before, fwd_counter_before = counters_before[name]
        assert task.nsamples - nsamples_before == 16
        assert task.fwd_counter - fwd_counter_before == 2

    stats = processor.shared_hessian_stats()
    assert stats["batch_misses"] - stats_before["batch_misses"] == 2
    assert stats["batch_hits"] - stats_before["batch_hits"] == (len(gate_up_subset) - 1) * 2


def test_moe_bypass_reuses_hidden_state_transfer_per_device(monkeypatch):
    """Down-input replay must copy a calibration batch at most once per expert device."""

    processor, subset = _make_moe_bypass_processor(
        num_experts=3,
        hidden_size=16,
        intermediate_size=8,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    hidden = torch.randn(2, 8, 16)
    keep_mask = torch.ones((2, 8), dtype=torch.bool)

    original_move_to = moe_lifecycle.move_to
    transfers = []

    def record_move_to(value, device, dtype=None):
        transfers.append(str(device))
        return original_move_to(value, device, dtype=dtype)

    monkeypatch.setattr(moe_lifecycle, "move_to", record_move_to)
    _build_replica_and_run_bypass(processor, subset, hidden, keep_mask)

    assert transfers == ["cpu"]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to place experts on different devices")
def test_moe_bypass_logical_cache_across_devices():
    """Per-expert device copies must not create duplicate Hessian cache entries."""

    hidden_size = 16
    intermediate_size = 8
    processor, subset = _make_moe_bypass_processor(
        num_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    batch, seq_len = 2, 8
    hidden = torch.randn(batch, seq_len, hidden_size)
    keep_mask = torch.tensor([[True] * 4 + [False] * 4, [True] * seq_len], dtype=torch.bool)
    valid_tokens = int(keep_mask.sum().item())

    # Place each expert on a different GPU. `forward_to_all_experts` will move
    # the same hidden state to each device, producing distinct storage pointers.
    # The logical shared-Hessian cache key must collapse these copies.
    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    _build_replica_and_run_bypass(processor, subset, hidden, keep_mask, expert_devices=devices)

    for name in subset:
        task = processor.tasks[name]
        task.finalize_hessian(target_device=torch.device("cpu"))
        assert task.nsamples == valid_tokens, (
            f"{name} recorded {task.nsamples} samples, expected {valid_tokens}"
        )


def _assert_down_nsamples_preserved_after_mock_recursion(processor, subset, valid_tokens):
    """Each down projection must keep its sample count after a mock-quantization recursion."""

    for name, named_module in subset.items():
        if not name.endswith(".down_proj"):
            continue
        task = processor.tasks[name]
        task.finalize_hessian(target_device=torch.device("cpu"))
        assert task.nsamples == valid_tokens, (
            f"{name}: pre-release nsamples {task.nsamples} != {valid_tokens}"
        )
        assert task.H is not None, f"{name}: H should be materialized before release"

        # Simulate the peak-memory release path inside ``GPTQ.quantize()``: the dense
        # Hessian is dropped after Hinv is computed, and a NaN loss triggers a mock
        # quantization recursion. The partials have already been merged and freed, so a
        # second ``finalize_hessian()`` call has no statistics left to rebuild ``H``.
        # Without preserving per-device sample counts, that re-materialized empty
        # Hessian reports zero samples ("using fail safe mode"); with the counts
        # preserved but no valid Hessian, the run must not zero the weight and must
        # keep reporting the real token count.
        del task.H
        task.H = None
        task._device_hessian_partials.clear()
        task.qcfg.mock_quantization = True

        Q, _, _, _, _, avg_loss, _, nsamples = task.quantize(blocksize=8)
        assert nsamples == valid_tokens, (
            f"{name}: mock-recursion nsamples {nsamples} != {valid_tokens}"
        )
        assert not torch.allclose(Q, torch.zeros_like(Q)), (
            f"{name}: mock recursion produced all-zero weights"
        )
        assert not torch.isnan(Q).any(), (
            f"{name}: mock recursion produced NaN weights"
        )
        # The returned loss is either a finite float (successful mock retry) or a
        # fallback string (no valid Hessian left to retry with); both are OK as
        # long as the weight is not silently zeroed.
        assert avg_loss is not None and avg_loss != 999999999, (
            f"{name}: mock recursion should report a real loss, got {avg_loss!r}"
        )


def test_moe_bypass_down_proj_nsamples_preserved_after_hessian_release():
    """Routing=bypass down projections must not lose sample counts during mock recursion."""

    hidden_size = 16
    intermediate_size = 8
    processor, subset = _make_moe_bypass_processor(
        num_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    batch, seq_len = 2, 8
    hidden = torch.randn(batch, seq_len, hidden_size)
    keep_mask = torch.tensor([[True] * 4 + [False] * 4, [True] * seq_len], dtype=torch.bool)
    valid_tokens = int(keep_mask.sum().item())

    _build_replica_and_run_bypass(processor, subset, hidden, keep_mask)
    _assert_down_nsamples_preserved_after_mock_recursion(processor, subset, valid_tokens)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to place experts on different devices")
def test_moe_bypass_down_proj_nsamples_preserved_after_hessian_release_multi_gpu():
    """Per-device down-projection copies must keep sample counts during mock recursion."""

    hidden_size = 16
    intermediate_size = 8
    processor, subset = _make_moe_bypass_processor(
        num_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    batch, seq_len = 2, 8
    hidden = torch.randn(batch, seq_len, hidden_size)
    keep_mask = torch.tensor([[True] * 4 + [False] * 4, [True] * seq_len], dtype=torch.bool)
    valid_tokens = int(keep_mask.sum().item())

    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    _build_replica_and_run_bypass(processor, subset, hidden, keep_mask, expert_devices=devices)
    _assert_down_nsamples_preserved_after_mock_recursion(processor, subset, valid_tokens)
