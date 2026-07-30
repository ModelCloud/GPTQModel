# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    HessianConfig,
    MoEConfig,
    QuantizeConfig,
)


def _make_moe_bypass_processor(*, num_experts: int = 3, hidden_size: int = 16, intermediate_size: int = 12):
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
            # the processor can identify gate/up/down roles without relying on
            # the model lifecycle hooks.
            named.state["module_tree_flags"] = frozenset({proj.split("_")[0]})
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
            if device is not None:
                self.gate_proj = self.gate_proj.to(device)
                self.up_proj = self.up_proj.to(device)
                self.down_proj = self.down_proj.to(device)

    class FakeMoEBlock(nn.Module):
        def __init__(self, num_experts, devices=None):
            super().__init__()
            devices = devices or [None] * num_experts
            self.experts = nn.ModuleList([FakeExpert(d) for d in devices])

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

    class FakeModuleLooper:
        def __init__(self, mask):
            self._mask = mask
            self.paused = False

        def _set_processor_hooks_paused(self, processor, value):
            self.paused = value

        def _get_processor_mask(self, processor):
            return self._mask

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
