# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Live MiMo owners, including its full-precision auxiliary MTP checkpoint."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_mimo_local_port import ASSETS, tiny_config, tiny_model

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.mimo import MimoQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.mimo.configuration_mimo import MiMoConfig
from gptqmodel.models.ports.mimo.modeling_mimo import MiMoForCausalLM


EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]],
}


def expected_shapes(config):
    """Independent checkpoint inventory from the inspected upstream equations."""
    h, i, v = config.hidden_size, config.intermediate_size, config.vocab_size
    dim = getattr(config, "head_dim", h // config.num_attention_heads)
    q, kv = config.num_attention_heads * dim, config.num_key_value_heads * dim
    block = {
        "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
        "self_attn.q_proj.weight": (q, h), "self_attn.q_proj.bias": (q,),
        "self_attn.k_proj.weight": (kv, h), "self_attn.k_proj.bias": (kv,),
        "self_attn.v_proj.weight": (kv, h), "self_attn.v_proj.bias": (kv,),
        "self_attn.o_proj.weight": (h, q), "mlp.gate_proj.weight": (i, h),
        "mlp.up_proj.weight": (i, h), "mlp.down_proj.weight": (h, i),
    }
    shapes = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,), "lm_head.weight": (v, h)}
    for index in range(config.num_hidden_layers):
        shapes.update({f"model.layers.{index}.{key}": shape for key, shape in block.items()})
    auxiliary = {**block, "token_layernorm.weight": (h,), "hidden_layernorm.weight": (h,),
                 "final_layernorm.weight": (h,), "input_proj.weight": (h, 2 * h)}
    for index in range(config.num_nextn_predict_layers):
        shapes.update({f"model.mtp_layers.{index}.{key}": shape for key, shape in auxiliary.items()})
    return shapes


@pytest.mark.parametrize("filename", ["config.json", "base_config.json", "sft_config.json", "rl_zero_config.json",
                                      "rl_0530_config.json"])
def test_pinned_real_checkpoint_has_exact_live_meta_owners(filename):
    config = MiMoConfig.from_dict(json.loads((ASSETS / filename).read_text()))
    index = json.loads((ASSETS / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("meta"):
        model = MiMoForCausalLM(config)
    assert len(index) == 451
    assert set(index) == set(model.state_dict()) == set(expected_shapes(config))
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected_shapes(config)
    assert sum(key.startswith("model.mtp_layers.") for key in index) == 16
    assert config.rope_parameters == {"rope_theta": 640000, "rope_type": "default"}
    assert MODEL_MAP["mimo"] is MimoQModel
    del model


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("mtp", [0, 1, 2])
def test_live_decoder_plan_roles_external_mtp_and_exact_shapes(kv, mtp):
    config = tiny_config(num_key_value_heads=kv, num_nextn_predict_layers=mtp)
    model = tiny_model(num_key_value_heads=kv, num_nextn_predict_layers=mtp)
    _convert_model_with_defuser(MimoQModel, model, cleanup_original=False)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected_shapes(config)
    tree = MimoQModel._resolve_effective_module_tree(model, SimpleNamespace(method="awq"))
    plan = ModulePlan.from_tree(tree)
    assert MimoQModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert MimoQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    for layer in model.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == ("layernorm" not in owner)
            assert not spec.capture_only
            if spec.quantize:
                quantized.add(owner)
            else:
                assert "norm" in spec.flags
        assert quantized == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    for name in ["q", "k", "v", "o"]:
        assert plan.roles(f"self_attn.{name}_proj") == (name,)
    assert "in=x" in plan.find("self_attn.q_proj").flags
    for name in ["self_attn.q_proj.extra", "other.self_attn.q_proj", "self_attn.q_proj_suffix"]:
        assert plan.find(name) is None
    assert not plan.moe_roots()
    # The native causal decoder loop never calls MTP. It remains dense, just
    # like embed_tokens/norm/lm_head, and must not become a second loop branch.
    assert len(model.model.mtp_layers) == mtp
    for key in model.state_dict():
        if not key.startswith("model.layers."):
            assert key.startswith(("model.mtp_layers.", "model.embed_tokens.", "model.norm.", "lm_head."))
            assert plan.find(key.rsplit(".", 1)[0]) is None
    assert MimoQModel.fusable is None


def test_plain_awq_policy_is_definition_owned():
    model = tiny_model()
    wrapper = MimoQModel.__new__(MimoQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    assert wrapper.awq_get_modules_for_scaling(model.model.layers[0], {}, {}) == []
