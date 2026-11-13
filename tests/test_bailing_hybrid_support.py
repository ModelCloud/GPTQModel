# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def _config():
    return SimpleNamespace(num_experts=2, num_local_experts=2, num_shared_experts=1)


def test_bailing_hybrid_model_type_selects_definition(monkeypatch):
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="bailing_hybrid"),
    )

    assert auto.check_and_get_model_definition("/tmp/bailing-hybrid") is BailingMoeQModel
    assert auto.MODEL_MAP["bailing_hybrid"] is BailingMoeQModel


def test_bailing_hybrid_simple_layer_modules_cover_attention_and_mlp():
    layer_modules = BailingMoeQModel.simple_layer_modules(
        model_config=_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    expected = {
        # self_attn projections shared by delta/full-attention layers.
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.g_proj",
        # delta/linear-attention extras.
        "self_attn.forget_gate.f_proj",
        "self_attn.b_proj",
        # MLA extras.
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
        # Dense (delta/linear-attention) MLP.
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        # Shared experts plus routed expert projections expanded from the module
        # tree (they resolve only after the loader's Defuser splits the packed
        # expert tensors into per-expert Linears).
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    }
    for expert in (0, 1):
        expected |= {
            f"mlp.experts.{expert}.gate_proj",
            f"mlp.experts.{expert}.up_proj",
            f"mlp.experts.{expert}.down_proj",
        }

    assert expected <= flat_modules
    assert not any(name.startswith("attention.") for name in flat_modules)
    assert "attention.query_key_value" not in flat_modules
    # Regression: only the plural `shared_experts` container exists.
    assert not any(name.startswith("mlp.shared_expert.") for name in flat_modules)

    assert BailingMoeQModel.layer_modules_strict is False
    assert isinstance(BailingMoeQModel.moe_lifecycle_hooks, GateUpDownMoELifecycleHooks)
