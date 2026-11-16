# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4373: remote Ling MLA projections must resolve in the module plan.

The remote-code ``bailing_hybrid`` checkpoint (Ling-2.6-flash) keeps its
attention projections under ``model.layers.N.attention`` and, on MLA layers,
owns ``q_a_proj``/``q_b_proj``/``kv_a_proj_with_mqa``/``kv_b_proj`` plus
``g_proj``/``dense``. ``layer_modules_strict = False`` means an undeclared
projection silently stays in the source dtype, so the plan is checked against
``state_dict()`` owners rather than module names alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.module_plan import ModulePlan


QC = SimpleNamespace(dynamic=None)
CONFIG = SimpleNamespace(num_experts=2, num_local_experts=2, num_shared_experts=1)


class _RemoteMLAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, rank: int = 4):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden_size, rank, bias=False)
        self.q_b_proj = nn.Linear(rank, hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, rank, bias=False)
        self.kv_b_proj = nn.Linear(rank, hidden_size, bias=False)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_a_layernorm = nn.LayerNorm(rank)
        self.kv_a_layernorm = nn.LayerNorm(rank)


class _RemoteLinearAttention(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.query_key_value = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layernorm = nn.LayerNorm(hidden_size)
        self.key_layernorm = nn.LayerNorm(hidden_size)
        self.g_norm = nn.LayerNorm(hidden_size)


class _NativeAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, rank: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size + rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size + rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size + rank, hidden_size, bias=False)


class _Layer(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.attention = attention
        self.post_attention_layernorm = nn.LayerNorm(8)


class _NativeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = _NativeAttention()
        self.post_attention_layernorm = nn.LayerNorm(8)


def _shell(*layers: nn.Module) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(layers)
    shell = nn.Module()
    shell.model = model
    return shell


def _weight_owners(layer: nn.Module):
    for key in layer.state_dict():
        yield key.rsplit(".", 1)[0]


def _plan() -> ModulePlan:
    return ModulePlan.from_tree(BailingMoeQModel.module_tree)


def test_remote_mla_state_dict_owners_resolve_to_quantizable_specs():
    shell = _shell(_Layer(_RemoteMLAttention()), _Layer(_RemoteLinearAttention()))
    plan = _plan()

    expected_remote = {
        "attention.q_a_proj",
        "attention.q_b_proj",
        "attention.kv_a_proj_with_mqa",
        "attention.kv_b_proj",
        "attention.dense",
        "attention.query_key_value",
        "attention.g_proj",
    }
    resolved = set()
    non_quantized = set()
    for layer in shell.model.layers:
        for owner in _weight_owners(layer):
            if not owner.startswith("attention."):
                continue
            spec = plan.find(owner)
            assert spec is not None, f"{owner} is not declared"
            if isinstance(layer.get_submodule(owner), nn.Linear):
                assert spec.quantize, (owner, spec.flags)
                resolved.add(owner)
            else:
                assert not spec.quantize, (owner, spec.flags)
                non_quantized.add(owner)

    assert expected_remote <= resolved
    assert {
        "attention.query_layernorm",
        "attention.key_layernorm",
        "attention.g_norm",
        "attention.q_a_layernorm",
        "attention.kv_a_layernorm",
    } <= non_quantized


def test_remote_mla_projection_stages_follow_the_checkpoint_forward():
    blocks = [
        set(block)
        for block in BailingMoeQModel.simple_layer_modules(
            model_config=CONFIG, quantize_config=QC
        )
    ]

    def block_with(name: str) -> set[str]:
        matches = [block for block in blocks if name in block]
        assert len(matches) == 1, (name, matches)
        return matches[0]

    # The MLA input projections consume the block input norm together with the
    # linear-attention qkv/gate branch.
    first = block_with("attention.q_a_proj")
    assert {
        "attention.query_key_value",
        "attention.q_a_proj",
        "attention.kv_a_proj_with_mqa",
        "attention.g_proj",
    } <= first

    # q_b/kv_b consume q_a/kv_a and are quantized in the next stage.
    second = block_with("attention.q_b_proj")
    assert second == {"attention.q_b_proj", "attention.kv_b_proj"}

    # dense consumes the attention output after the low-rank pair.
    assert block_with("attention.dense") == {"attention.dense"}


def test_native_bailing_attention_branch_still_resolves():
    shell = _shell(_NativeLayer())
    plan = _plan()

    resolved = {
        owner
        for owner in _weight_owners(shell.model.layers[0])
        if owner.startswith("self_attn.")
    }
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.g_proj",
    } <= resolved

    for relative in resolved:
        spec = plan.find(relative)
        assert spec is not None, f"{relative} is not declared"
        assert spec.quantize, (relative, spec.flags)

    # The Ling 2.0 remote `attention.query_key_value`/`dense` branch remains.
    blocks = {
        name
        for block in BailingMoeQModel.simple_layer_modules(
            model_config=CONFIG, quantize_config=QC
        )
        for name in block
    }
    assert "attention.query_key_value" in blocks
    assert "attention.dense" in blocks
