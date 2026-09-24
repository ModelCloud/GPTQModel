# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.internvl import InternVLQModel


class TestInternVL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/InternVL3-1B-hf" # OpenGVLab/InternVL3-1B-hf
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3276, "floor_pct": 0.04},
            "acc_norm": {"value": 0.3464, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_internvl3(self):
        self.quantize_and_evaluate()


def test_internvl3_definition_matches_transformers_layout():
    assert InternVLQModel.extract_layers_node() == ["model.language_model.layers"]
    assert InternVLQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert InternVLQModel.rotary_embedding == "model.language_model.rotary_emb"

    conversion_pairs = {
        (entry.source_patterns[0], entry.target_patterns[0])
        for entry in InternVLQModel.resolve_hf_conversion_map_reversed()
    }
    assert (
        r"^model\.language_model\.(.+)$",
        r"^language_model.model.\1",
    ) in conversion_pairs

    modules = InternVLQModel.simple_layer_modules(
        model_config=None,
        quantize_config=type("QuantizeConfig", (), {"dynamic": None})(),
    )
    flattened = {name for block in modules for name in block}
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= flattened


def test_internvl_model_type_selects_definition(monkeypatch):
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: type("Config", (), {"model_type": "internvl"})(),
    )

    assert auto.check_and_get_model_definition("/tmp/InternVL3-1B-hf") is InternVLQModel


def test_internvl_base_modules_cover_language_roots_and_vision_modules():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 4)
            self.layers = nn.ModuleList([nn.Identity()])
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _CoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Identity()
            self.multi_modal_projector = nn.Identity()
            self.language_model = _LanguageModel()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _CoreModel()

    base_modules = set(InternVLQModel.get_base_modules(_Wrapper()))
    assert {
        "model.vision_tower",
        "model.multi_modal_projector",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
    } <= base_modules
