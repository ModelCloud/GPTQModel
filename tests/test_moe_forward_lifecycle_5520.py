# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5520: leaving the MoE bypass forward context must not leave a reference cycle.

`ModuleLooper.MoELifecycleContext` patches ``forward`` on the MoE block while a
bypass-routed forward runs. Restoring an *inherited* ``forward`` by assigning the
saved bound method back into the block's ``__dict__`` creates
``block -> __dict__ -> bound method -> block``. Every data-parallel replica built
by ``ForwardExecutor.run_parallel`` then survives until the cyclic GC runs, so its
expert weights stay resident on the second GPU (6.1/7.4 GiB peak on a 2-layer
OLMoE versus 1.26/2.7 GiB once the cycles are collected).
"""

import copy
import gc
import weakref

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.module_looper import ModuleLooper


pytestmark = [pytest.mark.cpu]


class _Expert(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _MoeBlock(nn.Module):
    """MoE block whose ``forward`` is a class method (inherited instance lookup)."""

    def __init__(self, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(num_experts)])

    def forward(self, hidden_states):
        return sum(expert(hidden_states) for expert in self.experts)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _MoeBlock()


class _Hooks:
    gate_proj_name = "gate_proj"
    up_proj_name = "up_proj"
    down_proj_name = "down_proj"

    def get_moe_block_for_subset(self, layer_module, model_class, current_subset=None):
        return layer_module.moe

    def _extract_moe_block_prefix(self, subset, moe_block):
        return None

    def forward_to_all_experts(
        self,
        moe_block,
        hidden_states,
        processor,
        subset,
        ordered_module_names,
        original_forward,
        model_class,
        module_looper,
        moe_block_prefix=None,
        replica_module=None,
        **kwargs,
    ):
        return original_forward(hidden_states, **kwargs)

    @staticmethod
    def apply_expert_activation(experts_module, expert, gate_out, up_out):
        return nn.functional.silu(gate_out) * up_out


class _Model:
    moe_lifecycle_hooks = _Hooks()


class _Looper:
    gptq_model = _Model()

    def _should_use_moe_lifecycle(self, module, processor, current_subset=None):
        return True


def test_lifecycle_context_removes_inherited_forward():
    layer = _Layer()
    block = layer.moe
    assert "forward" not in vars(block)

    ctx = ModuleLooper.MoELifecycleContext(_Looper(), layer, object(), None, None)
    with ctx:
        assert "forward" in vars(block)
        assert block(torch.zeros(2, 8)).shape == (2, 8)

    assert "forward" not in vars(block), (
        "exiting the context left an instance-level forward (bound method) on the block"
    )

    layer_ref = weakref.ref(layer)
    gc.disable()
    try:
        del layer, block, ctx
        assert layer_ref() is None, (
            "MoE replica is only freed by the cyclic GC; the forward context left a cycle"
        )
    finally:
        gc.enable()
        gc.collect()


def test_lifecycle_context_restores_existing_instance_forward():
    layer = _Layer()
    block = layer.moe

    def custom_forward(hidden_states):
        return hidden_states + 1

    block.forward = custom_forward
    ctx = ModuleLooper.MoELifecycleContext(_Looper(), layer, object(), None, None)
    with ctx:
        assert vars(block)["forward"] is not custom_forward
    assert vars(block)["forward"] is custom_forward


def test_lifecycle_context_leaves_no_bound_forward_for_deepcopy():
    layer = _Layer()
    with ModuleLooper.MoELifecycleContext(_Looper(), layer, object(), None, None):
        pass

    replica = copy.deepcopy(layer)
    assert "forward" not in vars(replica.moe)
    assert replica.moe(torch.zeros(2, 8)).shape == (2, 8)


def test_awq_all_experts_replay_removes_inherited_forward():
    block = _MoeBlock()
    targets = [
        (f"experts.{index}.gate_proj", expert.gate_proj)
        for index, expert in enumerate(block.experts)
    ]

    assert "forward" not in vars(block)
    with AWQProcessor._all_experts_replay_forward(
        None, _Hooks(), block, block.experts, targets
    ):
        assert "forward" in vars(block)
        assert block(torch.zeros(2, 8)).shape == (2, 8)
    assert "forward" not in vars(block), (
        "AWQ all-experts replay left an instance-level forward (bound method) on the block"
    )

    block_ref = weakref.ref(block)
    gc.disable()
    try:
        del block
        assert block_ref() is None, (
            "AWQ replay left a reference cycle on the MoE block"
        )
    finally:
        gc.enable()
        gc.collect()
