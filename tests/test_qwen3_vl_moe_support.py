# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VL_MoeQModel


def test_qwen3_vl_moe_registry_and_module_tree(monkeypatch):
    fake_config = SimpleNamespace(model_type="qwen3_vl_moe")
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda _path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_config,
    )

    assert auto.MODEL_MAP["qwen3_vl_moe"] is Qwen3_VL_MoeQModel
    assert (
        auto.check_and_get_model_definition("qwen3-vl-moe-fixture")
        is Qwen3_VL_MoeQModel
    )


def test_qwen3_vl_moe_module_tree_expands_all_experts():
    config = SimpleNamespace(model_type="qwen3_vl_moe", text_config=SimpleNamespace(num_experts=3))
    quantize_config = SimpleNamespace(dynamic=None)

    simple = Qwen3_VL_MoeQModel.simple_layer_modules(config, quantize_config)
    simple_names = {name for block in simple for name in block}

    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.1.up_proj",
        "mlp.experts.2.down_proj",
    } <= simple_names
    full_names = {
        name for block in Qwen3_VL_MoeQModel.full_layer_modules(config) for name in block
    }
    assert "mlp.gate:!" in full_names
    assert Qwen3_VL_MoeQModel.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]
    assert Qwen3_VL_MoeQModel.dynamic_expert_index == "num_experts"
