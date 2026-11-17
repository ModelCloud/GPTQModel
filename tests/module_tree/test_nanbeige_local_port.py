# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.nanbeige import __file__ as PORT_FILE
from gptqmodel.models.ports.nanbeige.configuration_nanbeige import NanbeigeConfig
from gptqmodel.models.ports.nanbeige.modeling_nanbeige import NanbeigeForCausalLM
from gptqmodel.utils.model import find_modules


PROJECTIONS = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


def tiny_config(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_loops": 2,
        "pad_token_id": 0,
        "max_position_embeddings": 32,
    }
    values.update(kwargs)
    return NanbeigeConfig(**values)


@pytest.mark.parametrize("source", ["Nanbeige/Nanbeige4.2-3B", "Nanbeige/Nanbeige4.2-3B-Base"])
def test_complete_real_checkpoint_headers_match_exact_meta_owners(source):
    contract = json.loads(Path(PORT_FILE).with_name("checkpoint_contract.json").read_text())
    config = NanbeigeConfig(**contract["sources"][source]["config"])
    with torch.device("meta"):
        model = NanbeigeForCausalLM(config)
    observed = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert observed == {key: value["shape"] for key, value in contract["tensors"].items()}
    assert len(observed) == 201
    assert len(model.model.layers) == 22
    assert config.num_attention_heads * config.head_dim != config.hidden_size
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (6144, 3072)
    assert model.model.layers[0].self_attn.o_proj.weight.shape == (3072, 6144)


@pytest.mark.parametrize("source", ["Nanbeige/Nanbeige4.2-3B-FP8", "Nanbeige/Nanbeige4.2-3B-GPTQ-Int8"])
def test_released_compressed_metadata_preserves_exact_dense_architecture_contract(source, tmp_path):
    contract = json.loads(Path(PORT_FILE).with_name("checkpoint_contract.json").read_text())
    metadata = contract["layout_aliases"][source]["config"]
    config = NanbeigeConfig(**metadata)
    config.save_pretrained(tmp_path)
    reloaded = NanbeigeQModel.loader.load_config(str(tmp_path), trust_remote_code=True, code_revision="ignored")
    assert type(reloaded) is NanbeigeConfig
    assert reloaded.quantization_config == metadata["quantization_config"]
    assert reloaded.model_type == "nanbeige"
    with torch.device("meta"):
        model = NanbeigeForCausalLM(reloaded)
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == {
        key: value["shape"] for key, value in contract["tensors"].items()
    }


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("variant", ["plain", "qknorm", "hc", "mhc", "split_mhc"])
def test_real_layouts_have_complete_owned_weight_plan(device, kv_heads, bias, variant):
    options = {
        "num_key_value_heads": kv_heads,
        "attention_bias": bias,
        "mlp_bias": bias,
        "qk_layernorm": variant == "qknorm",
        "enable_hyper_connection": variant in {"hc", "mhc", "split_mhc"},
        "enable_mhc": variant in {"mhc", "split_mhc"},
        "mhc_diff_for_loop": variant == "split_mhc",
        "enable_double_loop_split": variant == "split_mhc",
    }
    with torch.device(device):
        model = NanbeigeForCausalLM(tiny_config(**options))
    _convert_model_with_defuser(NanbeigeQModel, model, cleanup_original=False)
    wrapper = NanbeigeQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert MODEL_MAP["nanbeige"] is NanbeigeQModel
    assert NanbeigeQModel.extract_layers_node() == ["model.layers"]
    for layer in model.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if "." in key else ""
            spec = plan.find(owner)
            assert spec is not None, (variant, key, owner)
            if owner in PROJECTIONS:
                assert spec.quantize
                assert spec.roles == (PROJECTIONS[owner],)
                quantized.add(owner)
            else:
                assert not spec.quantize, key
        assert quantized == set(PROJECTIONS)
        assert set(find_modules(layer)) >= set(PROJECTIONS)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    for bad in ["self_attn.q_proj_extra", "self_attn.q_proj.weight", "self_attn.unknown", "mlp.down_proj.extra"]:
        assert plan.find(bad) is None
    if variant == "plain":
        assert NanbeigeQModel.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) == [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]
