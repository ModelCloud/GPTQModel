# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from defuser import convert_model
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

from gptqmodel.nn_modules.converter import MODULE_CONVERTER_MAP
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


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


def _make_tiny_qwen3_omni_config():
    return Qwen3OmniMoeConfig(
        enable_audio_output=False,
        thinker_config={
            "text_config": {
                "num_hidden_layers": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": 128,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "out_hidden_size": 64,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": [0],
            },
            "audio_config": {
                "num_mel_bins": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 128,
                "d_model": 64,
                "output_dim": 64,
                "max_source_positions": 32,
                "n_window": 4,
                "n_window_infer": 4,
                "conv_chunksize": 16,
                "downsample_hidden_size": 32,
            },
        },
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


def test_qwen3_omni_uses_defuser_for_fused_experts():
    assert "qwen3_omni_moe" not in MODULE_CONVERTER_MAP

    model = Qwen3OmniMoeForConditionalGeneration(_make_tiny_qwen3_omni_config())
    convert_model(model, cleanup_original=False, max_layers=1)
    layer = model.thinker.model.layers[0]

    _assert_converted_experts(
        layer,
        hidden_size=model.config.get_text_config().hidden_size,
        dtype=next(layer.mlp.experts[0].gate_proj.parameters()).dtype,
    )

def test_awq_single_bit_validation_allows_skip_only_dynamic_rules():
    ok, err = AwqTorchLinear.validate(
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
