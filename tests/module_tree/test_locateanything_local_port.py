# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real decoder owners and the pinned 770-tensor released checkpoint inventory."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_locateanything_local_port import tiny_config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.locateanything.configuration_locateanything import LocateAnythingConfig
from gptqmodel.models.ports.locateanything.modeling_locateanything import LocateAnythingForConditionalGeneration
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.model import get_layers_with_prefixes


FIXTURES = Path(__file__).parents[1] / "fixtures/ports/locateanything"
RELEASED_INVENTORY_SHA256 = "ad24aadd8935a6853b5cb17c1b0c1daf8a4a895764ab8c44ae34781aba7aeca1"

EXPECTED_PLAN = {
    "layers": ["language_model.model.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]],
}


def test_no_exact_native_mapping_and_all_registry_aliases():
    assert "locateanything" not in CONFIG_MAPPING
    assert {key for key, value in MODEL_MAP.items() if value is LocateAnythingQModel} == {"locateanything"}
    assert LocateAnythingQModel.fusable is None


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_real_plan_covers_every_decoder_weight_and_concrete_consumer(architecture, kv_heads, method):
    with torch.device("meta"):
        model = LocateAnythingQModel.loader.from_config(tiny_config(architecture=architecture, kv_heads=kv_heads))
    _convert_model_with_defuser(LocateAnythingQModel, model, cleanup_original=False)
    config = QuantizeConfig(method=method, group_size=16)
    tree = LocateAnythingQModel._resolve_effective_module_tree(model, config)
    model.gptqmodel_effective_module_tree = tree
    assert LocateAnythingQModel._resolve_effective_module_tree(model, config) == tree
    wrapper = LocateAnythingQModel(model=model, quantized=False, quantize_config=config)
    plan = wrapper.module_plan()
    layers, prefixes = get_layers_with_prefixes(model, wrapper.extract_layers_node())
    assert len(layers) == 2 and prefixes == [f"language_model.model.layers.{i}" for i in range(2)]
    expected = {"input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    if "3" in architecture:
        expected |= {"self_attn.q_norm", "self_attn.k_norm"}
    for layer in layers:
        owners = {key.rsplit(".", 1)[0] for key in layer.state_dict()}
        assert owners == expected
        for owner in owners:
            spec = plan.find(owner)
            assert spec is not None, owner
            assert spec.quantize is ("norm" not in owner), owner
            assert layer.get_submodule(owner) is not None
        for name in ("self_attn.q_proj_extra", "self_attn.extra_q_proj", "mlp.gate_proj_extra"):
            assert plan.find(name) is None
    assert wrapper.lm_head == "language_model.lm_head"
    assert {"vision_model", "mlp1"} <= set(wrapper.get_base_modules(model))
    assert wrapper.awq_get_modules_for_scaling(layers[0], {}, {}) == []
    if "2" in architecture:
        assert wrapper.build_layer_modules(wrapper.effective_module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]


def test_full_released_checkpoint_keys_shapes_and_alias_ownership_on_meta():
    config = LocateAnythingConfig(**json.loads((FIXTURES / "checkpoint_config.json").read_text()))
    with torch.device("meta"):
        model = LocateAnythingForConditionalGeneration(config)
    actual = model.state_dict()
    # sha256 of the released safetensors index ("name:shape" lines, sorted; 770 BF16 tensors).
    inventory = "\n".join(f"{name}:{','.join(map(str, actual[name].shape))}" for name in sorted(actual))
    assert len(actual) == 770
    assert hashlib.sha256(inventory.encode()).hexdigest() == RELEASED_INVENTORY_SHA256
    assert tuple(actual["language_model.model.embed_tokens.weight"].shape) == (152681, 2048)
    assert model.language_model.lm_head.weight is model.language_model.model.embed_tokens.weight
    wrapper = LocateAnythingQModel(model=model, quantized=False, quantize_config=QuantizeConfig())
    plan = wrapper.module_plan()
    for name in actual:
        if name.startswith("language_model.model.layers."):
            owner = '.'.join(name.split('.')[4:-1])
            assert plan.find(owner) is not None, name
        else:
            assert name.startswith(("vision_model.", "mlp1.", "language_model.model.embed_tokens.",
                                    "language_model.model.norm.", "language_model.lm_head.")), name
