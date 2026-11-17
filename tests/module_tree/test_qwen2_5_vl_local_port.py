# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native weight owners for the distinct multimodal and text tasks."""

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_qwen2_5_vl_local_port import composite_config, text_config
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import get_layers_with_prefixes


PROJECTIONS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
BLOCKS = [
    ["input_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_actual_native_weight_owners_and_layout_isolation(text_only, kv_heads, device):
    import defuser

    config = text_config(kv_heads) if text_only else composite_config()
    if not text_only:
        config.text_config.num_key_value_heads = kv_heads
    original_tree = copy.deepcopy(Qwen2_5_VLQModel.module_tree)
    with torch.device(device):
        model = Qwen2_5_VLQModel.loader.from_config(config, attn_implementation="eager")
    defuser.convert_model(model, cleanup_original=False)
    tree = Qwen2_5_VLQModel._resolve_effective_module_tree(model, None)
    wrapper = Qwen2_5_VLQModel.__new__(Qwen2_5_VLQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.effective_module_tree = tree
    wrapper._module_plan = ModulePlan.from_tree(tree)
    prefix = "model.layers" if text_only else "model.language_model.layers"
    assert wrapper.extract_layers_node()[0] == prefix
    assert wrapper.build_layer_modules(tree, include_capture_only=True) == BLOCKS
    layers, names = get_layers_with_prefixes(model, wrapper.extract_layers_node())
    assert names == [f"{prefix}.{index}" for index in range(2)]
    for layer in layers:
        assert {name for name, module in layer.named_modules() if isinstance(module, nn.Linear)} == PROJECTIONS
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = wrapper.module_plan().find(owner)
            assert spec is not None, key
            assert spec.quantize is (owner in PROJECTIONS)
            if owner in PROJECTIONS:
                assert spec.roles
            assert wrapper.module_plan().find(owner + ".undeclared") is None
    expected_bases = {"model.embed_tokens", "model.norm", "model.rotary_emb"} if text_only else {"model.visual"}
    assert set(wrapper.get_base_modules(model)) == expected_bases
    assert "model.norm" in wrapper.pre_lm_head_norm_module
    assert wrapper.require_load_processor is (not text_only)
    assert Qwen2_5_VLQModel.module_tree == original_tree
    if text_only:
        assert not any("visual" in key or "language_model" in key for key in model.state_dict())
        assert len(model.state_dict()) == 27
        assert wrapper.awq_get_modules_for_scaling(layers[0], {}, {}) == []
    assert MODEL_MAP[config.model_type] is Qwen2_5_VLQModel


def test_text_fusion_is_explicitly_unqualified_and_composite_policy_unchanged():
    assert require_fusable_config(Qwen2_5_VLQModel, "qwen2_5_vl") is Qwen2_5_VLQModel.fusable
    with pytest.raises(ValueError):
        require_fusable_config(Qwen2_5_VLQModel, "qwen2_5_vl_text")
    # Preserve the registered default case; the runtime text tree is independent.
    tree = Qwen2_5_VLQModel._resolve_effective_module_tree(SimpleNamespace(), None)
    with Qwen2_5_VLQModel._module_tree_context(tree):
        assert Qwen2_5_VLQModel.extract_layers_node() == ["model.language_model.layers", "language_model.layers"]


def test_pinned_released_checkpoint_all_shapes_and_text_extraction_schema():
    from transformers import Qwen2_5_VLConfig

    import gptqmodel.models.ports.qwen2_5_vl as port

    fixtures = Path(port.__file__).parent / "fixtures"
    config = Qwen2_5_VLConfig.from_json_file(fixtures / "qwen3b_config.json")
    with torch.device("meta"):
        composite = Qwen2_5_VLQModel.loader.from_config(config)
        text = Qwen2_5_VLQModel.loader.from_config(config.text_config)
    released = json.loads(gzip.decompress((fixtures / "qwen3b_shapes.json.gz").read_bytes()))
    assert len(released) == 824
    expected = {}
    for key, shape in released.items():
        if key.startswith("visual."):
            key = "model." + key
        elif key.startswith("model."):
            key = key.replace("model.", "model.language_model.", 1)
        expected[key] = shape
    # The 3B release omits its tied head, which native HF reconstructs.
    assert "lm_head.weight" not in expected and config.tie_word_embeddings
    expected["lm_head.weight"] = expected["model.language_model.embed_tokens.weight"]
    assert {key: list(value.shape) for key, value in composite.state_dict().items()} == expected
    extracted = {
        key.replace("model.language_model.", "model.", 1): shape
        for key, shape in expected.items()
        if key.startswith("model.language_model.") or key == "lm_head.weight"
    }
    assert {key: list(value.shape) for key, value in text.state_dict().items()} == extracted
