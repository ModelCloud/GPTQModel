# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.llava_qwen2 import LlavaQwen2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.llava_qwen2.configuration_llava_qwen2 import LlavaConfig
from gptqmodel.models.ports.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM
from tests.test_llava_qwen2_local_port import _config, _model


# Released apple/FastVLM config.json fields that differ between sizes; the rest are shared.
FASTVLM_BASE = {
    "hidden_act": "silu", "image_aspect_ratio": "pad", "max_position_embeddings": 32768,
    "mm_hidden_size": 3072, "mm_patch_merge_type": "flat", "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_feature": "patch", "mm_vision_select_layer": -2, "mm_vision_tower": "mobileclip_l_1024",
    "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "tokenizer_model_max_length": 8192,
    "torch_dtype": "bfloat16", "use_sliding_window": False, "unfreeze_mm_vision_tower": True, "bos_token_id": 151643, "eos_token_id": 151645,
}
FASTVLM = {
    "0.5B": {"hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14, "num_hidden_layers": 24, "max_window_layers": 24,
                 "num_key_value_heads": 2, "tie_word_embeddings": True, "vocab_size": 151936},
    "1.5B": {"hidden_size": 1536, "intermediate_size": 8960, "num_attention_heads": 12, "num_hidden_layers": 28, "max_window_layers": 28,
                 "num_key_value_heads": 2, "tie_word_embeddings": True, "vocab_size": 151936},
    "7B": {"hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28, "num_hidden_layers": 28, "max_window_layers": 28,
               "num_key_value_heads": 4, "tie_word_embeddings": False, "vocab_size": 152064},
}
# sha256 of the sorted (name, shape) inventory read from the apple/FastVLM-0.5B safetensors
# header (revision 16375720c2d673fa583e57e9876afde27549c7d0): all 924 tensors, and the 629 vision ones.
FASTVLM_05B_SHA256 = "4024f327788efc53db0b9484d6ff5669e692670e74dcc575315b111ba7dfb1d8"
FASTVLM_VISION_SHA256 = "797aacb57a603f383e545381a43cd54265183334c6de2933d9ab3113bb7d0444"


def _inventory_sha256(items):
    return hashlib.sha256(json.dumps(sorted((name, list(shape)) for name, shape in items)).encode()).hexdigest()


LINEARS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]],
}


def test_exact_native_mapping_is_absent_and_all_registered_aliases_select_owned_definition():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "llava_qwen2" not in CONFIG_MAPPING_NAMES
    assert "llava_qwen2" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert [name for name, definition in MODEL_MAP.items() if definition is LlavaQwen2QModel] == ["llava_qwen2"]


@pytest.mark.parametrize("size", ["0.5B", "1.5B", "7B"])
def test_pinned_real_checkpoint_config_has_exact_flat_vision_and_decoder_owners_on_meta(size):
    config = LlavaConfig(**FASTVLM_BASE, **FASTVLM[size])
    with torch.device("meta"):
        model = LlavaQwen2ForCausalLM(config)
    _convert_model_with_defuser(LlavaQwen2QModel, model, cleanup_original=False)
    state = model.state_dict()
    vision = [(name, value.shape) for name, value in state.items() if name.startswith("model.vision_tower.")]
    assert len(vision) == 629
    assert _inventory_sha256(vision) == FASTVLM_VISION_SHA256
    if size == "0.5B":
        assert len(state) == 924
        assert _inventory_sha256((name, value.shape) for name, value in state.items()) == FASTVLM_05B_SHA256
    plan = ModulePlan.from_tree(LlavaQwen2QModel.module_tree)
    declared = {name for block in LlavaQwen2QModel.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == LINEARS
    expected = {name + ".weight" for name in LINEARS} | {"input_layernorm.weight", "post_attention_layernorm.weight"}
    expected |= {f"self_attn.{name}_proj.bias" for name in ("q", "k", "v")}
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, name
            if spec.quantize:
                assert owner in LINEARS
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert "!" in spec.flags
    assert {"model.embed_tokens", "model.norm", "model.rotary_emb", "model.mm_projector", "model.vision_tower"} <= set(
        LlavaQwen2QModel.get_base_modules(model))
    assert not any(name.startswith("language_model.") for name in state)


@pytest.mark.parametrize("tower", ["mobileclip", "clip"])
@pytest.mark.parametrize("merge", ["flat", "spatial", "spatial_unpad"])
def test_live_tiny_plan_excludes_vision_projector_norms_and_preserves_all_seven_decoder_linears(tower, merge):
    config = _config(tower, mm_patch_merge_type=merge)
    with torch.device("meta"):
        model = LlavaQwen2ForCausalLM(config)
    wrapper = LlavaQwen2QModel.__new__(LlavaQwen2QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    assert wrapper.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) == [
        [name for name in group if not name.endswith(":!")] for group in EXPECTED_PLAN["blocks"]
        if not all(name.endswith(":!") for name in group)]
    assert wrapper.pre_lm_head_norm_module == "model.norm"
    assert wrapper.rotary_embedding == "model.rotary_emb"
    assert ("model.image_newline" in model.state_dict()) == (merge == "spatial_unpad")


def test_central_registered_plan_is_unchanged():
    fixtures = Path(__file__).parent / "fixtures/registered_module_plans.json"
    assert json.loads(fixtures.read_text())["cases"]["LlavaQwen2QModel/gptq"] == EXPECTED_PLAN


# FastVLM ships GQA (v_proj is narrower than o_proj, so that fold is not
# emitted); equal KV heads additionally qualify the v_proj -> o_proj fold.
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_scale_folds_preserve_text_decoder_output(kv_heads):
    model = _model(num_key_value_heads=kv_heads)
    ids = torch.tensor([[1, 7, 9, 4, 12, 11, 5, 3]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    expected = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])]
    if kv_heads == 4:
        expected.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    expected += [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]), ("mlp.up_proj", ["mlp.down_proj"])]
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(LlavaQwen2QModel, model, layer, forward)
        assert [(source, targets) for source, targets, _ in checked] == expected
