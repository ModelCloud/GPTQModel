# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.ovis2_6_moe import Ovis2_6_MoeQModel


def test_ovis2_6_moe_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="ovis2_6_moe")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/ovis2_6_moe") is Ovis2_6_MoeQModel
    assert Ovis2_6_MoeQModel.extract_layers_node() == ["llm.model.layers"]


def test_ovis2_6_moe_module_tree_expands_qwen3_moe_paths():
    layer_modules = Ovis2_6_MoeQModel.simple_layer_modules(
        model_config=SimpleNamespace(num_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.q_norm" not in flat_modules
    assert "self_attn.k_norm" not in flat_modules
    assert "mlp.gate" not in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert Ovis2_6_MoeQModel.defuser_module_paths == ("llm",)


def test_ovis2_6_moe_materializes_missing_vision_post_layernorm_defaults():
    layernorm = nn.LayerNorm(4, device="meta", dtype=torch.bfloat16)

    Ovis2_6_MoeQModel._materialize_layernorm_defaults(layernorm, torch.device("cpu"))

    assert layernorm.weight.device.type == "cpu"
    assert layernorm.bias.device.type == "cpu"
    assert layernorm.weight.dtype == torch.bfloat16
    assert layernorm.bias.dtype == torch.bfloat16
    torch.testing.assert_close(layernorm.weight, torch.ones(4, dtype=torch.bfloat16))
    torch.testing.assert_close(layernorm.bias, torch.zeros(4, dtype=torch.bfloat16))
