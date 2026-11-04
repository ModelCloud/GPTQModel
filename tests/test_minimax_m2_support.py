# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.minimax_m2 import MiniMaxM2GPTQ, MiniMaxQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def test_minimax_m2_model_types_select_definition(monkeypatch):
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="minimax_m2"),
    )

    assert auto.check_and_get_model_definition("/tmp/minimax-m2") is MiniMaxM2GPTQ
    assert auto.MODEL_MAP["minimax"] is MiniMaxQModel
    assert auto.MODEL_MAP["minimax_m2"] is MiniMaxM2GPTQ


def test_minimax_m2_module_tree_targets_defused_expert_projections():
    tree = MiniMaxM2GPTQ.module_tree[-1]
    moe_tree = tree["mlp:moe"]

    assert "block_sparse_moe" not in tree
    assert "e_score_correction_bias" not in moe_tree
    assert moe_tree["gate"] == ("gate:!",)
    assert moe_tree["experts"]["#"] == ("gate_proj:0", "up_proj:0", "down_proj:1")
    assert isinstance(MiniMaxM2GPTQ.moe_lifecycle_hooks, GateUpDownMoELifecycleHooks)


def test_minimax_m2_simple_layer_modules_cover_expert_projections():
    layer_modules = MiniMaxM2GPTQ.simple_layer_modules(
        model_config=SimpleNamespace(num_local_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    # The router gate is a non-Linear module marked with `:!`.
    assert "mlp.gate" not in flat_modules
    for expert in (0, 1):
        assert f"mlp.experts.{expert}.gate_proj" in flat_modules
        assert f"mlp.experts.{expert}.up_proj" in flat_modules

    # Only the down projection is part of the second (out) projection group.
    assert any("mlp.experts.0.down_proj" in block for block in layer_modules)
    assert not any("block_sparse_moe" in name for name in flat_modules)
    assert not any(name.endswith((".w1", ".w2", ".w3")) for name in flat_modules)


def test_minimax_original_targets_full_and_linear_attention():
    """MiniMax-Text-01 interleaves full and linear ("lightning") attention.

    The linear-attention layers expose a fused ``qkv_proj`` plus ``out_proj``
    and ``output_gate``; the M2 tree did not target them, so they were left
    unquantized.
    """

    layer_modules = MiniMaxQModel.simple_layer_modules(
        model_config=SimpleNamespace(num_local_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    # Full-attention layers.
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
        assert name in flat_modules
    # Linear-attention layers.
    for name in ("self_attn.qkv_proj", "self_attn.out_proj", "self_attn.output_gate"):
        assert name in flat_modules
    # Routed experts (defused by Defuser).
    for expert in (0, 1):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"mlp.experts.{expert}.{proj}" in flat_modules
    assert "mlp.gate" not in flat_modules
    assert not any("block_sparse_moe" in name for name in flat_modules)

    assert MiniMaxQModel.layer_modules_strict is False
    assert isinstance(MiniMaxQModel.moe_lifecycle_hooks, GateUpDownMoELifecycleHooks)
