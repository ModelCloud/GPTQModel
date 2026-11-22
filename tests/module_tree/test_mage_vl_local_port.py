# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Live Mage-VL checkpoint owners, independently pinned to Microsoft's index."""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.mage_vl import MageVLQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.mage_vl.configuration_mage_vl import MageVLConfig
from gptqmodel.models.ports.mage_vl.modeling_mage_vl import MageVLForConditionalGeneration
from tests.test_mage_vl_local_port import _config, _model


def _checkpoint_shapes():
    # Complete 696-key index inspected at microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
    shapes = {"lm_head.weight": (151936, 2560), "model.language_model.embed_tokens.weight": (151936, 2560),
              "model.language_model.norm.weight": (2560,),
              "model.visual.embeddings.patch_embedding.weight": (1024, 3, 16, 16),
              "model.visual.layernorm_pre.weight": (1024,), "model.visual.layernorm_pre.bias": (1024,),
              "model.visual.merger.ln_q.weight": (1024,), "model.visual.merger.ln_q.bias": (1024,),
              "model.visual.merger.mlp.0.weight": (4096, 4096), "model.visual.merger.mlp.0.bias": (4096,),
              "model.visual.merger.mlp.2.weight": (2560, 4096), "model.visual.merger.mlp.2.bias": (2560,)}
    text = {"input_layernorm.weight": (2560,), "post_attention_layernorm.weight": (2560,),
            "self_attn.q_proj.weight": (4096, 2560), "self_attn.k_proj.weight": (1024, 2560),
            "self_attn.v_proj.weight": (1024, 2560), "self_attn.o_proj.weight": (2560, 4096),
            "self_attn.q_norm.weight": (128,), "self_attn.k_norm.weight": (128,),
            "mlp.gate_proj.weight": (9728, 2560), "mlp.up_proj.weight": (9728, 2560),
            "mlp.down_proj.weight": (2560, 9728)}
    vision = {f"layer_norm{n}.{kind}": (1024,) for n in (1, 2) for kind in ("weight", "bias")}
    for name, dims in (("self_attn.qkv", (3072, 1024)), ("self_attn.proj", (1024, 1024)),
                       ("mlp.fc1", (4096, 1024)), ("mlp.fc2", (1024, 4096))):
        vision[name + ".weight"] = dims
        vision[name + ".bias"] = dims[:1]
    shapes.update({f"model.language_model.layers.{i}.{key}": value for i in range(36) for key, value in text.items()})
    shapes.update({f"model.visual.encoder.layers.{i}.{key}": value for i in range(24) for key, value in vision.items()})
    assert len(shapes) == 696
    return shapes


def test_real_pinned_checkpoint_complete_meta_inventory_and_shapes():
    config = MageVLConfig(text_config={"model_type": "qwen3", "hidden_size": 2560, "head_dim": 128,
                                           "intermediate_size": 9728, "num_attention_heads": 32, "num_key_value_heads": 8,
                                           "num_hidden_layers": 36, "vocab_size": 151936, "max_position_embeddings": 262144},
                          vision_config={"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
                                             "num_attention_heads": 16, "patch_size": 16, "out_hidden_size": 2560})
    with torch.device("meta"):
        model = MageVLForConditionalGeneration(config)
    assert {key: tuple(t.shape) for key, t in model.state_dict().items()} == _checkpoint_shapes()
    assert MageVLQModel.get_base_modules(model) == ["model.visual"]
    assert {k for k, definition in MODEL_MAP.items() if definition is MageVLQModel} == {"mage_vl"}


@pytest.mark.parametrize("kv", [4, 2, 1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_all_declared_leaves_bind_actual_decoder_owners(kv, bias, device):
    config = _config(text={"num_key_value_heads": kv, "attention_bias": bias})
    with torch.device(device):
        model = MageVLForConditionalGeneration(config)
    _convert_model_with_defuser(MageVLQModel, model, cleanup_original=False)
    wrapper = MageVLQModel.__new__(MageVLQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    expected = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert declared == expected
    for layer in model.model.language_model.layers:
        assert {name for name, m in layer.named_modules() if isinstance(m, nn.Linear)} == expected
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in expected)
            if not spec.quantize:
                assert "!" in spec.flags
        for name in expected:
            assert isinstance(layer.get_submodule(name), nn.Linear)
            assert plan.find(name + ".not_declared") is None


@pytest.mark.parametrize("kv,bias", [(4, False), (2, True)])
def test_awq_text_decoder_folds_preserve_layer_output(kv, bias):
    # The Qwen3 text decoder owns ordinary pre-norm folds; the vision tower is
    # outside the quantized tree and only changes the decoder's input rows.
    model = _model(text={"num_key_value_heads": kv, "attention_bias": bias})
    language = model.model.language_model
    ids = torch.randint(3, 59, (1, 8))

    def forward():
        return language(input_ids=ids).last_hidden_state

    for layer in language.layers:
        checked = assert_awq_folds_neutral(MageVLQModel, model, layer, forward)
        assert [(source, targets) for source, targets, _ in checked] == [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("mlp.up_proj", ["mlp.down_proj"]),
        ]
