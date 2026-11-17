# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inventories and real MiMo-V2 owners, independent of tree generation."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.mimo_v2 import MimoV2QModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.mimo_v2.configuration_mimo_v2 import MiMoV2Config
from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM
from tests.test_mimo_v2_local_port import tiny_config


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/mimo_v2"
EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [
        ["input_layernorm:!"],
        ["self_attn.qkv_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.attention_sink_bias:!"],
        ["self_attn.o_proj"],
        ["post_attention_layernorm:!"],
        ["mlp:?"],
        ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"], ["mlp.gate:!"],
        ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        ["mlp.experts.{expert_index}.down_proj"],
    ],
}


def test_exact_local_registry_plan_and_native_distinction():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    assert "mimo_v2" not in CONFIG_MAPPING_NAMES
    assert "mimo_v2" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "mimo_v2_flash" in CONFIG_MAPPING_NAMES
    assert MODEL_MAP["mimo_v2"] is MimoV2QModel
    tree = MimoV2QModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert MimoV2QModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    assert MimoV2QModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert not MimoV2QModel.require_trust_remote_code
    assert json.loads((PORT / "module_plan_case.json").read_text()) == EXPECTED_PLAN


@pytest.mark.parametrize("layout", ["split", "fused_qkv"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("moe", [[0, 0], [0, 1], [1, 1]])
def test_actual_decoder_tensor_ownership_layout_and_consumer_tags(layout, kv, bias, moe):
    config = MiMoV2Config(**tiny_config(attention_projection_layout=layout, num_key_value_heads=kv,
                                      swa_num_key_value_heads=kv, attention_bias=bias, moe_layer_freq=moe))
    with torch.device("meta"):
        model = MiMoV2ForCausalLM(config).eval()
    plan = ModulePlan.from_tree(MimoV2QModel.module_tree)
    quantized = 0
    for layer in model.model.layers:
        for name, tensor in layer.state_dict().items():
            owner = name.rsplit(".", 1)[0]
            if name.endswith("attention_sink_bias"):
                owner = name
            spec = plan.find(owner)
            assert spec is not None, name
            if spec.quantize:
                assert isinstance(layer.get_submodule(owner), nn.Linear)
                quantized += name.endswith("weight")
            else:
                assert "!" in spec.flags
            assert tensor.device.type == "meta"
        assert MimoV2QModel.awq_get_modules_for_scaling(None, layer, {}, {}) == []
        a = layer.self_attn
        qdim = config.num_attention_heads * config.head_dim
        if layout == "fused_qkv":
            assert a.qkv_proj.weight.shape == (qdim + kv * (config.head_dim + config.v_head_dim), config.hidden_size)
        else:
            assert a.q_proj.weight.shape == (qdim, config.hidden_size)
        assert a.o_proj.weight.shape == (config.hidden_size, config.num_attention_heads * config.v_head_dim)
    assert quantized == 2 * (2 if layout == "fused_qkv" else 4) + sum(12 if is_moe else 3 for is_moe in moe)
    assert "model.rotary_emb.inv_freq" not in model.state_dict()
    assert not any(k.startswith("model.mtp.") for k in model.state_dict())


@pytest.mark.parametrize("variant", ["MiMo-V2.5-Base", "MiMo-V2.5-Pro-Base", "MiMo-V2.6-Pro-RL", "MiMo-V2.6-Flash-RL"])
def test_pinned_complete_checkpoint_key_inventory(tmp_path, variant):
    fixture = json.loads(gzip.decompress((PORT / "fixtures" / (variant + ".json.gz")).read_bytes()))
    values = dict(fixture["config"])
    values.pop("quantization_config", None)
    with torch.device("meta"):
        model = MiMoV2ForCausalLM(MiMoV2Config(**values))
    expected = set(fixture["keys"])
    # Run the actual omission policy against the independent checkpoint index.
    # This exercises real visual/audio owners rather than fake child modules.
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(expected, "unused")}))
    MimoV2QModel._drop_visual_merger_biases_if_checkpoint_omits_them(model, str(tmp_path))
    MimoV2QModel._drop_checkpoint_omitted_audio_tensors(model, str(tmp_path))
    state = model.state_dict()
    actual = set(state)
    mtp = {name for name in expected if name.startswith("model.mtp.")}
    source_scales = {name for name in expected if name.endswith(("weight_scale_inv", "weight_scale"))}
    assert actual == expected - mtp - source_scales
    assert len(expected) == fixture["key_count"]
    assert mtp == set(fixture["mtp_keys"])
    for name, shape in fixture.get("header_shapes", {}).items():
        if name not in mtp | source_scales:
            assert list(state[name].shape) == shape, name
