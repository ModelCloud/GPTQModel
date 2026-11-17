# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4636: PhiMoE MoE routing must be honoured or rejected, never silently ignored.

`PhimoeTopKRouter.forward` passes `top_k` to `sparsemixer`, but transformers
5.22's `sparsemixer` always selects two experts.  Writing `router.top_k` was
therefore a silent no-op for `ExpertsRoutingOverride`.  The definition also
declared no `moe_lifecycle_hooks`, so `ExpertsRoutingBypass` warned once per
forward and then calibrated with native top-2 routing.

These tests pin the two observable contracts on CPU: an override changes the
routed width (and restores it), the bypass lifecycle runs every defused expert,
and a definition without lifecycle hooks rejects bypass instead of falling back.
"""

import copy
import os
import types


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from transformers.models.phimoe.modeling_phimoe import PhimoeConfig, PhimoeForCausalLM

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.phi3 import PhiMoEGPTQForCausalLM
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_moe_routing_modules, restore_moe_topk, set_moe_topk


pytestmark = [pytest.mark.cpu]

_NUM_EXPERTS = 4
_HIDDEN_SIZE = 32
_NATIVE_TOP_K = 2


def _tiny_phimoe_model() -> PhimoeForCausalLM:
    config = PhimoeConfig(
        vocab_size=128,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=_NUM_EXPERTS,
        num_experts_per_tok=_NATIVE_TOP_K,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return PhimoeForCausalLM(config).eval()


def _phimoe_router(model: torch.nn.Module):
    routers = [
        module
        for module in find_moe_routing_modules(model)
        if module.__class__.__name__ == "PhimoeTopKRouter"
    ]
    assert len(routers) == 1
    return routers[0]


def test_phimoe_routing_override_changes_and_restores_routed_width():
    torch.manual_seed(0)
    model = _tiny_phimoe_model()
    router = _phimoe_router(model)
    hidden_states = torch.randn(6, _HIDDEN_SIZE)

    with torch.inference_mode():
        _, native_weights, native_indices = router(hidden_states)
    assert native_indices.shape[-1] == _NATIVE_TOP_K
    assert native_weights.shape == native_indices.shape

    state = set_moe_topk(model, _NUM_EXPERTS)
    try:
        assert router.top_k == _NUM_EXPERTS
        with torch.inference_mode():
            _, weights, indices = router(hidden_states)
        assert indices.shape[-1] == _NUM_EXPERTS, (
            "ExpertsRoutingOverride set router.top_k but the router still "
            "selected the native top-2 experts"
        )
        assert weights.shape == indices.shape
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(weights.shape[0], dtype=weights.dtype),
        )
    finally:
        restore_moe_topk(state)

    with torch.inference_mode():
        _, restored_weights, restored_indices = router(hidden_states)
    assert restored_indices.shape[-1] == _NATIVE_TOP_K
    assert restored_weights.shape == restored_indices.shape


class _Processor:
    def __init__(self):
        self.paused = []

    def record_moe_root_input_feature(self, _name, _hidden_states):
        pass


class _ModuleLooperStub:
    def _set_processor_hooks_paused(self, processor, paused):
        processor.paused.append(paused)


def test_phimoe_bypass_lifecycle_feeds_every_defused_expert():
    torch.manual_seed(0)
    model = _tiny_phimoe_model()
    assert _convert_model_with_defuser(PhiMoEGPTQForCausalLM, model, cleanup_original=True)

    hooks = getattr(PhiMoEGPTQForCausalLM, "moe_lifecycle_hooks", None)
    assert hooks is not None, (
        "PhiMoEGPTQForCausalLM must declare moe_lifecycle_hooks; otherwise "
        "ExpertsRoutingBypass silently falls back to native routing"
    )
    hooks = copy.copy(hooks)
    # BaseQModel.after_model_load binds the compiled plan to the per-instance
    # hooks; mirror that here so the experts container resolves exactly as it
    # does during a real quantization run.
    hooks.module_plan = ModulePlan.from_tree(PhiMoEGPTQForCausalLM.module_tree)

    layer = model.model.layers[0]
    moe_block = hooks.get_moe_block_for_subset(layer, PhiMoEGPTQForCausalLM)
    assert moe_block is not None
    experts = hooks.get_experts_module(moe_block, PhiMoEGPTQForCausalLM)
    assert experts is not None
    assert len(experts) == _NUM_EXPERTS

    subset = {}
    for expert_index in range(_NUM_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            subset[f"mlp.experts.{expert_index}.{projection}"] = getattr(
                experts[expert_index], projection
            )

    hidden_states = torch.randn(2, 5, _HIDDEN_SIZE)
    with torch.inference_mode():
        native_output = moe_block(hidden_states)
    assert native_output.shape == hidden_states.shape

    calls = {}
    handles = []
    for expert_index in range(_NUM_EXPERTS):
        gate = getattr(experts[expert_index], "gate_proj")

        def record(_module, _inputs, _output, index=expert_index):
            calls[index] = calls.get(index, 0) + 1

        handles.append(gate.register_forward_hook(record))

    # Replace only the final routed forward: the assertion below must count
    # calls made by the all-experts replay, not the natively routed pass.
    original_calls = []

    def original_forward(inputs, **kwargs):
        original_calls.append(inputs.shape)
        return torch.zeros_like(inputs)

    processor = _Processor()
    try:
        with torch.inference_mode():
            output = hooks.forward_to_all_experts(
                moe_block=moe_block,
                hidden_states=hidden_states,
                processor=processor,
                subset=subset,
                ordered_module_names=list(subset),
                original_forward=original_forward,
                model_class=PhiMoEGPTQForCausalLM,
                module_looper=_ModuleLooperStub(),
                moe_block_prefix="mlp",
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == hidden_states.shape
    assert calls == dict.fromkeys(range(_NUM_EXPERTS), 1), (
        "the bypass lifecycle must run every expert on the block input"
    )
    assert original_calls == [hidden_states.shape]
    assert processor.paused == [True, False]


class _DummyQModel:
    """Minimal GPT-QModel double accepted by ModuleLooper.__init__."""

    capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
    capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
    prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
    prepare_layer_replay_keep_mask = BaseQModel.prepare_layer_replay_keep_mask

    def __init__(self, lifecycle_hooks):
        self.quantize_config = types.SimpleNamespace(
            device=torch.device("cpu"),
            calibration_data_device=None,
            compute_device_filter=None,
            auto_forward_data_parallel=True,
            dense_vram_strategy="exclusive",
            dense_vram_strategy_devices=None,
            moe_vram_strategy="exclusive",
            moe_vram_strategy_devices=None,
            moe_routing_bypass=lambda: True,
            moe=None,
            gc_mode=None,
        )
        self.support_batch_quantize = False
        self.layer_callback = None
        self.lm_head = None
        self.quant_region_timer = None
        self.moe_lifecycle_hooks = lifecycle_hooks
        self.model = torch.nn.Module()


def test_bypass_without_lifecycle_hooks_is_rejected_at_looper_construction():
    with pytest.raises(ValueError, match="moe_lifecycle_hooks"):
        ModuleLooper(model=_DummyQModel(lifecycle_hooks=None), processors=[])

    looper = ModuleLooper(model=_DummyQModel(lifecycle_hooks=object()), processors=[])
    assert looper.moe_routing_bypass is True
