# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual Ovis2.5 Qwen3/NaViT weight owners, released headers and aliases."""

import gzip
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


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/ovis2_5"
EXPECTED_PLAN = {
    "layers": ["llm.model.layers"],
    "blocks": [["input_layernorm:!"],
               ["self_attn.q_proj", "self_attn.q_norm:!", "self_attn.k_proj", "self_attn.k_norm:!", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"],
               ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}


@pytest.mark.parametrize("release", ["2B", "9B"])
def test_exact_released_headers_and_tied_embedding_alias(release):
    # Read all six original safetensors headers via byte ranges, independent of the model.
    headers = json.loads(gzip.decompress((PORT / "released_headers.json.gz").read_bytes()))[release]
    config = Ovis2_5_Config(**json.loads((PORT / f"release_{release}_config.json").read_text()))
    with torch.device("meta"):
        model = Ovis2_5(config)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    expected = {k: v["shape"] for k, v in headers.items()}
    if release == "2B":
        # The 2B safetensors index omits its tied output alias; the 9B index contains it.
        assert model.llm.lm_head.weight is model.llm.model.embed_tokens.weight
        expected["llm.lm_head.weight"] = expected["llm.model.embed_tokens.weight"]
    assert actual == expected
    assert model.config.max_pixels == (1344 * 1792 if release == "2B" else 1792 * 1792)
    proposal = json.loads((PORT / "central_fixture_proposals.json").read_text())
    assert proposal["registered_module_plans.cases.Ovis2_5QModel/gptq"] == EXPECTED_PLAN


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
