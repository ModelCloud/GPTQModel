# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.xing4_0 import Xing4_0QModel


def test_xing4_0_registry_and_loader_contract(monkeypatch):
    fake_config = SimpleNamespace(model_type="xing4_0")
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda _path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda _path: None)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_config,
    )

    assert auto.MODEL_MAP["xing4_0"] is Xing4_0QModel
    assert auto.check_and_get_model_definition("xing4-fixture", trust_remote_code=True) is Xing4_0QModel
    assert Xing4_0QModel.require_trust_remote_code is True
    assert Xing4_0QModel.require_fast_init is True
    assert Xing4_0QModel.layer_modules_strict is False
    assert Xing4_0QModel.dynamic_expert_index == "n_routed_experts"
    assert Xing4_0QModel.pre_lm_head_norm_module == "model.norm"
    assert Xing4_0QModel.rotary_embedding == "model.rotary_emb"
    assert Xing4_0QModel.out_of_model_tensors == {"prefixes": ["model.layers.40"]}


def test_xing4_0_module_tree_covers_mla_dense_and_moe_paths():
    config = SimpleNamespace(n_routed_experts=3)
    quantize_config = SimpleNamespace(dynamic=None)
    simple = Xing4_0QModel.simple_layer_modules(config, quantize_config)
    simple_names = {name for block in simple for name in block}

    assert {
        "self_attn.q_proj",
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.2.up_proj",
        "mlp.shared_experts.down_proj",
    } <= simple_names

    full_names = {
        name for block in Xing4_0QModel.full_layer_modules(config) for name in block
    }
    assert {
        "input_layernorm:!",
        "self_attn.q_a_layernorm:!",
        "self_attn.kv_a_layernorm:!",
        "post_attention_layernorm:!",
        "mlp.gate:!",
        "mlp.gate.e_score_correction_bias:!",
        "attn_hc.hc_fn:!",
        "attn_hc.hc_base:!",
        "attn_hc.hc_scale:!",
        "ffn_hc.hc_fn:!",
        "ffn_hc.hc_base:!",
        "ffn_hc.hc_scale:!",
    } <= full_names
    assert not any(name.endswith(":!") for name in simple_names)


def test_xing4_0_lm_head_hook_reduces_hyperconnection_streams():
    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(4)

    qmodel = object.__new__(Xing4_0QModel)
    nn.Module.__init__(qmodel)
    qmodel.model = _Root()
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    stream_4d = torch.arange(24, dtype=torch.float32).square().reshape(1, 2, 3, 4)
    stream_3d = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    inputs = [[stream_4d.clone(), stream_3d.clone()]]

    result = qmodel.lm_head_pre_quantize_generate_hook(inputs)

    norm = qmodel.model.model.norm
    torch.testing.assert_close(result[0][0], norm(stream_4d.mean(dim=2)))
    torch.testing.assert_close(result[0][1], norm(stream_3d))
    assert not torch.allclose(result[0][0], norm(stream_4d).mean(dim=2))

