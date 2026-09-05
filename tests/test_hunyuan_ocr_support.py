from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions import hunyuan_vl as hunyuan_vl_module
from gptqmodel.models.definitions.hunyuan_vl import HunYuanVLQModel
from gptqmodel.utils.hf import normalize_hf_config_compat
from gptqmodel.utils.model import MODALITY
from transformers import AutoConfig


def test_hunyuan_vl_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="hunyuan_vl")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("tencent/HunyuanOCR") is HunYuanVLQModel


def test_hunyuan_vl_module_tree_matches_dense_text_backbone():
    layer_modules = HunYuanVLQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert HunYuanVLQModel.__bases__ == (BaseQModel,)
    assert HunYuanVLQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert HunYuanVLQModel.require_load_processor is True
    assert HunYuanVLQModel.extract_layers_node() == ["model.language_model.layers"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "self_attn.query_layernorm" not in flat_modules
    assert "self_attn.key_layernorm" not in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules


def test_hunyuan_vl_keeps_multimodal_and_embedding_modules_in_base_dtype():
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.embed_tokens = nn.Embedding(8, 4)
    model.model.language_model.layers = nn.ModuleList([nn.Identity()])
    model.model.language_model.norm = nn.LayerNorm(4)
    model.model.language_model.rotary_emb = nn.Identity()
    model.model.vision_tower = nn.Linear(4, 4)

    base_modules = set(HunYuanVLQModel.get_base_modules(model))

    assert base_modules == {
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
        "model.vision_tower",
    }


def test_hunyuan_vl_materialize_uses_canonical_device_and_module_path():
    qmodel = object.__new__(HunYuanVLQModel)
    nn.Module.__init__(qmodel)
    qmodel.quantize_config = SimpleNamespace(device="cuda")
    parent = nn.Module()
    parent.embed_tokens = nn.Embedding(8, 4, device="meta")
    calls = []

    def fake_materialize(module, device, *, module_path):
        calls.append((module, device, module_path))
        return nn.Embedding(8, 4)

    qmodel.shell_module_materialize = fake_materialize
    qmodel._materialize_module(
        parent,
        "embed_tokens",
        "model.language_model.embed_tokens",
    )

    assert parent.embed_tokens.weight.device == torch.device("cpu")
    assert calls[0][1:] == (
        torch.device("cuda:0"),
        "model.language_model.embed_tokens",
    )


def test_hunyuan_vl_prepare_dataset_uses_processor_chat_template(monkeypatch):
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return {"input_ids": torch.ones((len(batch), 4), dtype=torch.long)}

    monkeypatch.setattr(
        hunyuan_vl_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: FakeProcessor(),
    )

    qmodel = object.__new__(HunYuanVLQModel)
    nn.Module.__init__(qmodel)
    qmodel.model_local_path = "tencent/HunyuanOCR"
    samples = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]

    prepared = qmodel.prepare_dataset(samples, batch_size=2)

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].shape == (2, 4)
    assert calls == [
        (
            samples,
            {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "processor_kwargs": {"padding": True},
            },
        )
    ]


def test_hunyuan_vl_rope_normalization_stays_in_text_config(tmp_path):
    mrope_section = [16, 16, 16, 16]
    config = AutoConfig.for_model(
        "hunyuan_vl",
        text_config={
            "head_dim": 128,
            "rope_parameters": {
                "alpha": 1000.0,
                "factor": 1.0,
                "mrope_section": mrope_section,
                "rope_theta": 10000.0,
                "rope_type": "dynamic",
            },
        },
    )

    normalize_hf_config_compat(config)

    assert "rope_parameters" not in config.to_dict()
    assert config.text_config.rope_parameters["mrope_section"] == mrope_section

    config.save_pretrained(tmp_path)
    reloaded = AutoConfig.from_pretrained(tmp_path)

    assert reloaded.text_config.rope_parameters["mrope_section"] == mrope_section
