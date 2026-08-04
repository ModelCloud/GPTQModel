# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import tempfile

import pytest
import torch


pytest.importorskip("defuser")
pytest.importorskip("transformers.models.axk2")

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.axk2.configuration_axk2 import AXK2Config  # noqa: E402

from gptqmodel.models.auto import MODEL_MAP, check_and_get_model_definition  # noqa: E402
from gptqmodel.models.definitions.axk2 import AXK2QModel  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.utils.model import find_modules  # noqa: E402


def _tiny_axk2_config() -> AXK2Config:
    return AXK2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_model_map_registration():
    assert MODEL_MAP.get("axk2") is AXK2QModel


def test_check_and_get_model_definition():
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "model_type": "axk2",
            "architectures": ["AXK2ForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "index_n_heads": 2,
            "index_head_dim": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "vocab_size": 128,
            "rms_norm_eps": 1e-06,
            "hidden_act": "silu",
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        definition = check_and_get_model_definition(d)
        assert definition is AXK2QModel


def test_simple_layer_modules_expand():
    cfg = _tiny_axk2_config()
    qcfg = QuantizeConfig(bits=4, group_size=128)
    layer_modules = AXK2QModel.simple_layer_modules(cfg, qcfg)
    flat = [name for block in layer_modules for name in block]
    # Attention projections
    assert "self_attn.q_a_proj" in flat
    assert "self_attn.kv_a_proj_with_mqa" in flat
    assert "self_attn.q_gate_proj" in flat
    assert "self_attn.kv_b_proj" in flat
    assert "self_attn.o_proj" in flat
    # MoE projections
    assert any("mlp.experts.0.gate_proj" in name for name in flat)
    assert any("mlp.experts.3.down_proj" in name for name in flat)
    assert "mlp.shared_experts.gate_proj" in flat


@pytest.mark.parametrize("num_experts", [2, 4, 8])
def test_dynamic_expert_index(num_experts: int):
    cfg = _tiny_axk2_config()
    cfg.n_routed_experts = num_experts
    qcfg = QuantizeConfig(bits=4, group_size=128)
    layer_modules = AXK2QModel.simple_layer_modules(cfg, qcfg)
    flat = [name for block in layer_modules for name in block]
    assert any(f"mlp.experts.{num_experts - 1}.gate_proj" in name for name in flat)
    assert not any(f"mlp.experts.{num_experts}.gate_proj" in name for name in flat)


def test_defuser_unfuses_axk2_experts():
    cfg = _tiny_axk2_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, trust_remote_code=False)

    # Defuser registration is performed when AXK2QModel is imported.
    import defuser

    converted = defuser.convert_model(model, cleanup_original=False)
    assert converted is True

    modules = find_modules(model)
    # Routed experts should be per-expert nn.Linear modules.
    assert "model.layers.1.mlp.experts.0.gate_proj" in modules
    assert "model.layers.1.mlp.experts.0.up_proj" in modules
    assert "model.layers.1.mlp.experts.0.down_proj" in modules
    # Shared experts should stay reachable.
    assert "model.layers.1.mlp.shared_experts.gate_proj" in modules


def test_quantizable_modules_match_tree():
    cfg = _tiny_axk2_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, trust_remote_code=False)

    import defuser

    defuser.convert_model(model, cleanup_original=False)
    qcfg = QuantizeConfig(bits=4, group_size=128)
    slm = AXK2QModel.simple_layer_modules(cfg, qcfg)
    modules = find_modules(model)
    matched = {name for name in modules if any(name.endswith(suffix) for sublist in slm for suffix in sublist)}
    # Core attention and MoE modules should be matched.
    assert any("self_attn.q_a_proj" in n for n in matched)
    assert any("self_attn.q_gate_proj" in n for n in matched)
    assert any("mlp.experts.0.gate_proj" in n for n in matched)
    # Indexer and gated-RMSNorm internals are not part of the tree.
    assert not any("self_attn.indexer" in n for n in matched)
    assert not any("input_layernorm.mlp" in n for n in matched)
