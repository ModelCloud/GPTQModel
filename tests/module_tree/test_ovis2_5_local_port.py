# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual Ovis2.5 Qwen3/NaViT weight owners, released headers and aliases."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ovis2_5 import Ovis2_5QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.ovis2_5.configuration_ovis2_5 import Ovis2_5_Config
from gptqmodel.models.ports.ovis2_5.modeling_ovis2_5 import Ovis2_5
from tests.test_ovis2_5_local_port import _config


EXPECTED_PLAN = {
    "layers": ["llm.model.layers"],
    "blocks": [["input_layernorm:!"],
               ["self_attn.q_proj", "self_attn.q_norm:!", "self_attn.k_proj", "self_attn.k_norm:!", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"],
               ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}


_VIT = {"model_type": "siglip2_navit", "hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27,
        "num_attention_heads": 16, "patch_size": 16, "image_size": 512, "hidden_stride": 2, "window_size": 112,
        "num_patches": -1, "preserve_original_pe": True, "use_rope": True, "fullatt_block_indexes": None}
_LLM = {"model_type": "qwen3", "head_dim": 128, "num_key_value_heads": 8, "vocab_size": 151936,
        "rms_norm_eps": 1e-06, "rope_theta": 1000000, "max_position_embeddings": 40960}
# Released AIDC-AI/Ovis2.5 config.json decoder dimensions (2B @ 393c932b, 9B @ d73b2283) and the sha256
# of the sorted (name, shape) inventory of all original safetensors headers. The 2B index omits its tied
# output alias, which is added to its inventory; the 9B index contains an untied lm_head.
RELEASES = {
    "2B": ({"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28, "num_attention_heads": 16,
            "tie_word_embeddings": True},
           "63d2ee343d491840509737266c591998f9fdd1c7b49252b87d90dc9aff9b9c96", 752),
    "9B": ({"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "num_attention_heads": 32},
           "ff1b33b35e51a37d75cb05e05994fee2cb4b9e91014d22b42e525a7afb3098f6", 840),
}


@pytest.mark.parametrize("release", ["2B", "9B"])
def test_exact_released_headers_and_tied_embedding_alias(release):
    llm, digest, count = RELEASES[release]
    config = Ovis2_5_Config(llm_config={**_LLM, **llm}, vit_config=dict(_VIT), hidden_size=llm["hidden_size"],
                            visual_vocab_size=65536)
    with torch.device("meta"):
        model = Ovis2_5(config)
    state = model.state_dict()
    assert len(state) == count
    inventory = sorted((name, list(value.shape)) for name, value in state.items())
    assert hashlib.sha256(json.dumps(inventory).encode()).hexdigest() == digest
    tied = model.llm.lm_head.weight is model.llm.model.embed_tokens.weight
    assert tied == (release == "2B")
    assert model.config.max_pixels == (1344 * 1792 if release == "2B" else 1792 * 1792)


def test_central_registered_plan_is_unchanged():
    fixtures = Path(__file__).parent / "fixtures/registered_module_plans.json"
    assert json.loads(fixtures.read_text())["cases"]["Ovis2_5QModel/gptq"] == EXPECTED_PLAN


@pytest.mark.parametrize("kv", [4, 2, 1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_all_decoder_weight_owners_and_native_head_variants(kv, bias, device):
    config = _config(text={"num_key_value_heads": kv, "attention_bias": bias})
    with torch.device(device):
        model = Ovis2_5(config)
    _convert_model_with_defuser(Ovis2_5QModel, model, cleanup_original=False)
    wrapper = Ovis2_5QModel.__new__(Ovis2_5QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    declared = {p for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for p in block}
    expected = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert declared == expected
    for layer in model.llm.model.layers:
        assert {name for name, item in layer.named_modules() if isinstance(item, nn.Linear)} == expected
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in expected)
            if not spec.quantize:
                assert "!" in spec.flags
        for name in expected:
            assert plan.find(name + ".foreign") is None
    assert model.get_submodule(wrapper.pre_lm_head_norm_module) is model.llm.model.norm
    assert model.get_submodule(wrapper.lm_head) is model.llm.lm_head
    assert wrapper.awq_get_modules_for_scaling(model.llm.model.layers[0], {}, {}) == []
    assert wrapper.build_layer_modules(wrapper.module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    assert wrapper.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert {key for key, value in MODEL_MAP.items() if value is Ovis2_5QModel} == {"ovis2_5"}
    assert Ovis2_5QModel.fusable is None


def test_meta_shell_vision_rotary_buffer_is_rebuilt_by_nested_vit_init():
    # The NaViT inverse frequencies are a nonpersistent buffer: no checkpoint carries them, so the
    # nested Siglip2 PreTrainedModel init (dispatched by Transformers' smart_apply) must rebuild them.
    with torch.device("meta"):
        model = Ovis2_5(_config())
    model.to_empty(device="cpu")
    rotary = model.visual_tokenizer.vit.vision_model.encoder.rotary_pos_emb
    model.initialize_weights()
    expected = 1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32) / rotary.dim))
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
