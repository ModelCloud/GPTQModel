# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import dynamic_module_utils

from gptqmodel.models import auto
from gptqmodel.models.definitions.k2_horizon import (
    K2HorizonMoELifecycleHooks,
    K2HorizonQModel,
)
from gptqmodel.utils.model import (
    find_moe_routing_modules,
    restore_moe_topk,
    set_moe_topk,
)


MOVA_MODEL_PATH = Path("/monster/data/model/K2-Horizon-MoVA-36B-A4B")
DENSE_MODEL_PATH = Path("/monster/data/model/K2-Horizon-0.9B")


def _quantize_config():
    return SimpleNamespace(dynamic=None)


def test_k2_horizon_model_type_selects_definition(monkeypatch):
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="k2_horizon"),
    )

    assert (
        auto.check_and_get_model_definition(
            "k2-horizon-fixture", trust_remote_code=True
        )
        is K2HorizonQModel
    )


def test_k2_horizon_expands_mlp_and_mova_experts_with_independent_counts():
    config = SimpleNamespace(num_experts=100, mova_num_experts=64)
    blocks = K2HorizonQModel.simple_layer_modules(config, _quantize_config())
    modules = {name for block in blocks for name in block}

    assert K2HorizonQModel.dynamic_expert_index == "num_experts"
    assert K2HorizonQModel.dynamic_expert_indices == {
        "mlp.experts": "num_experts",
        "self_attn.v_experts": "mova_num_experts",
    }
    assert "self_attn.v_experts.0" in modules
    assert "self_attn.v_experts.63" in modules
    assert "self_attn.v_experts.64" not in modules
    assert "mlp.experts.0.gate_proj" in modules
    assert "mlp.experts.99.down_proj" in modules
    assert "mlp.experts.100.gate_proj" not in modules
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.gate_proj",
        "self_attn.o_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= modules
    assert "self_attn.v_router" not in modules
    assert "mlp.gate" not in modules


def test_k2_horizon_dense_config_does_not_expand_expert_placeholders():
    config = SimpleNamespace(num_experts=0, mova_num_experts=0)
    modules = {
        name
        for block in K2HorizonQModel.simple_layer_modules(config, _quantize_config())
        for name in block
    }

    assert not any(name.startswith("self_attn.v_experts.") for name in modules)
    assert not any(name.startswith("mlp.experts.") for name in modules)
    assert {
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= modules


class _MLPRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 8
        self.num_experts = 100


class _MoVARouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts_per_tok = 4
        self.v_experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(64)])


def test_k2_horizon_routing_override_clamps_each_router_to_its_local_count():
    model = torch.nn.Module()
    model.mlp = _MLPRouter()
    model.self_attn = _MoVARouter()

    assert find_moe_routing_modules(model) == [model.mlp, model.self_attn]
    state = set_moe_topk(model, 100)
    assert model.mlp.top_k == 100
    assert model.self_attn.num_experts_per_tok == 64

    restore_moe_topk(state)
    assert model.mlp.top_k == 8
    assert model.self_attn.num_experts_per_tok == 4


def test_k2_horizon_lifecycle_selects_expert_family_from_subset():
    layer = torch.nn.Module()
    layer.self_attn = _MoVARouter()
    layer.mlp = _MLPRouter()
    hooks = K2HorizonMoELifecycleHooks()

    assert (
        hooks.get_moe_block_for_subset(
            layer,
            K2HorizonQModel,
            current_subset={"self_attn.v_experts.0": object()},
        )
        is layer.self_attn
    )
    assert (
        hooks.get_moe_block_for_subset(
            layer,
            K2HorizonQModel,
            current_subset={"mlp.experts.0.gate_proj": object()},
        )
        is layer.mlp
    )


def test_k2_horizon_lifecycle_replays_mova_inputs_through_each_value_expert():
    class _MoVABlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_experts = torch.nn.ModuleList(
                [torch.nn.Linear(4, 3, bias=False), torch.nn.Linear(4, 3, bias=False)]
            )

    class _Looper:
        def __init__(self):
            self.paused = []

        def _set_processor_hooks_paused(self, processor, paused):
            self.paused.append(paused)

    block = _MoVABlock()
    calls = [0, 0]
    handles = []
    for index, expert in enumerate(block.v_experts):
        handles.append(
            expert.register_forward_pre_hook(
                lambda module, args, index=index: calls.__setitem__(
                    index, calls[index] + 1
                )
            )
        )

    hidden_states = torch.randn(2, 3, 4)
    looper = _Looper()
    result = K2HorizonMoELifecycleHooks().forward_to_all_experts(
        moe_block=block,
        hidden_states=hidden_states,
        processor=object(),
        subset={
            "self_attn.v_experts.0": block.v_experts[0],
            "self_attn.v_experts.1": block.v_experts[1],
        },
        ordered_module_names=[
            "self_attn.v_experts.0",
            "self_attn.v_experts.1",
        ],
        original_forward=lambda inputs, **kwargs: inputs + 1,
        model_class=K2HorizonQModel,
        module_looper=looper,
        moe_block_prefix="self_attn",
    )

    for handle in handles:
        handle.remove()
    assert calls == [1, 1]
    assert looper.paused == [True, False]
    torch.testing.assert_close(result, hidden_states + 1)


@pytest.mark.parametrize("model_path", [MOVA_MODEL_PATH, DENSE_MODEL_PATH])
def test_local_k2_horizon_remote_config_matches_registered_definition(
    model_path, monkeypatch, tmp_path
):
    if not model_path.is_dir():
        pytest.skip(f"local K2 fixture is unavailable: {model_path}")

    modules_cache = tmp_path / "hf_modules"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache))
    config = auto.AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    assert config.model_type == "k2_horizon"
    assert auto.MODEL_MAP[config.model_type] is K2HorizonQModel
