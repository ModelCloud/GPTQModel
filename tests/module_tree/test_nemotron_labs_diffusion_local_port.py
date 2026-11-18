# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.nemotron_labs_diffusion import NemotronLabsDiffusionQModel as Definition
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.nemotron_labs_diffusion import NemotronLabsDiffusionConfig as Config
from gptqmodel.models.ports.nemotron_labs_diffusion import NemotronLabsDiffusionModel as Model
from tests.test_nemotron_labs_diffusion_local_port import tiny_config


EXPECTED_PLAN = {
    "layers": ["encoder.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]],
}


# nvidia/Nemotron-Labs-Diffusion-3B@0d51902da1f8869f83413ce642fab402fa5641e0 config.json geometry.
RELEASED_CONFIG = {
    "hidden_size": 3072, "intermediate_size": 9216, "num_attention_heads": 32, "num_hidden_layers": 26,
    "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 131072, "max_position_embeddings": 262144,
    "mask_token_id": 100, "eos_token_id": 11, "rms_norm_eps": 1e-05, "tie_word_embeddings": False,
    "rope_parameters": {
        "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1, "mscale": 1.0,
        "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384, "rope_theta": 1000000.0,
        "rope_type": "yarn",
    },
}
# sha256 of the released model.safetensors header inventory {name: shape} (237 tensors),
# serialized with json.dumps(sort_keys=True, separators=(",", ":")).
RELEASED_SHAPES_SHA256 = "61d774d045d4ef7d633bb877dfe5fd8225f57bbff5ce655f41914493972eba1f"


def test_pinned_released_checkpoint_header_has_exact_owners_on_meta():
    cfg = Config(**RELEASED_CONFIG)
    with torch.device("meta"):
        model = Model(cfg)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(actual) == 237 and digest == RELEASED_SHAPES_SHA256
    assert actual["encoder.layers.0.self_attn.q_proj.weight"] == [4096, 3072]
    assert all(not key.endswith("bias") for key in actual)


@pytest.mark.parametrize("paradigm", ["bidirectional", "autoregressive", "block_diff"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_real_owner_inventory_and_plain_awq_policy(paradigm, kv, device):
    cfg = tiny_config(dlm_paradigm=paradigm, num_key_value_heads=kv)
    with torch.device(device):
        model = Model(cfg)
    plan = ModulePlan.from_tree(Definition.module_tree)
    names = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    for layer in model.encoder.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert layer.get_submodule(owner) is not None
            if spec.quantize:
                quantized.add(owner)
            else:
                assert "!" in spec.flags
        assert quantized == names
        assert Definition.awq_get_modules_for_scaling(None, layer, {}, {}) == []
    assert plan.find("self_attn.q_proj.fake") is None
    assert plan.find("mlp.gate_proj_extra") is None
    assert Definition.fusable is None
    assert Definition.build_layer_modules(Definition.module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    assert Definition.extract_layers_node() == EXPECTED_PLAN["layers"]
    entries = Definition.simple_layer_modules(model_config=cfg, quantize_config=SimpleNamespace(method="gptq", dynamic=None))
    assert set(sum(entries, [])) == names


def test_aliases_and_no_shape_based_substitution():
    assert {name for name, definition in MODEL_MAP.items() if definition is Definition} == {"nemotron_labs_diffusion"}
    assert Definition.lm_head == "diffusion_head"
    assert Definition.pre_lm_head_norm_module == "encoder.norm"
    assert Definition.require_trust_remote_code is False


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_source_ignores_bias_metadata_and_optional_tied_embeddings(bias, tied):
    # Pinned source constructors always use bias=False, regardless of these legacy config fields.
    model = Model(tiny_config(attention_bias=bias, mlp_bias=bias, tie_word_embeddings=tied))
    assert all(not name.endswith("bias") for name in model.state_dict())
    assert (model.diffusion_head.weight is model.encoder.embed_tokens.weight) == tied
