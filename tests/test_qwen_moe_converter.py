# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from defuser import convert_model
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

from gptqmodel.nn_modules.converter import MODULE_CONVERTER_MAP, convert_qwen3_omni_moe_expert_converter
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear

import torch


def _make_tiny_moe_config(config_cls):
    return config_cls(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
    )


def _assert_converted_experts(layer, hidden_size: int, *, dtype: torch.dtype = torch.float32):
    assert isinstance(layer.mlp.experts, torch.nn.Module)
    assert not hasattr(layer.mlp.experts, "gate_up_proj")
    assert len([name for name, _ in layer.mlp.experts.named_children() if name.isdigit()]) == 4

    expert0 = layer.mlp.experts[0]
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")

    output = layer.mlp(torch.randn(2, 3, hidden_size, dtype=dtype))
    assert output.shape == (2, 3, hidden_size)


def test_qwen2_moe_uses_defuser_for_fused_experts():
    assert "qwen2_moe" not in MODULE_CONVERTER_MAP

    model = Qwen2MoeForCausalLM(_make_tiny_moe_config(Qwen2MoeConfig))
    convert_model(model, cleanup_original=False)
    layer = model.model.layers[0]

    assert hasattr(layer.mlp, "shared_expert")
    assert hasattr(layer.mlp, "shared_expert_gate")
    _assert_converted_experts(layer, hidden_size=model.config.hidden_size)


def test_qwen3_moe_uses_defuser_for_fused_experts():
    assert "qwen3_moe" not in MODULE_CONVERTER_MAP

    model = Qwen3MoeForCausalLM(_make_tiny_moe_config(Qwen3MoeConfig))
    convert_model(model, cleanup_original=False)
    layer = model.model.layers[0]

    assert not hasattr(layer.mlp, "shared_expert")
    _assert_converted_experts(layer, hidden_size=model.config.hidden_size)


def test_qwen3_next_uses_defuser_for_fused_experts():
    assert "qwen3_next" not in MODULE_CONVERTER_MAP

    model = Qwen3NextForCausalLM(_make_tiny_moe_config(Qwen3NextConfig))
    convert_model(model, cleanup_original=False)
    layer = model.model.layers[0]

    assert hasattr(layer.mlp, "shared_expert")
    assert hasattr(layer.mlp, "shared_expert_gate")
    _assert_converted_experts(layer, hidden_size=model.config.hidden_size)


def test_qwen3_omni_converter_uses_nested_text_config():
    text_config = Qwen3OmniMoeTextConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
    )

    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Qwen3OmniMoeThinkerTextSparseMoeBlock(text_config)

    layer = DummyLayer()
    layer.mlp = layer.mlp.to(dtype=torch.bfloat16)
    convert_qwen3_omni_moe_expert_converter(
        layer,
        SimpleNamespace(thinker_config=SimpleNamespace(text_config=text_config)),
    )

    _assert_converted_experts(layer, hidden_size=text_config.hidden_size, dtype=torch.bfloat16)
    assert layer.mlp.experts[0].gate_proj.weight.dtype == torch.bfloat16


def test_awq_single_bit_validation_allows_skip_only_dynamic_rules():
    ok, err = AwqTorchQuantLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic={"-:^model\\.layers\\.4\\.mlp$": {}},
    )

    assert ok, err
    assert err is None
