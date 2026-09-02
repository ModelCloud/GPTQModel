# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from torch import nn
from transformers import AutoModelForMultimodalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.apertus import Apertus1p5QModel, Apertus1p5TextQModel


class _FakeApertus1p5LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()
        self.rotary_emb = nn.Identity()


class _FakeApertus1p5Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeApertus1p5LanguageModel()
        self.vision_tokenizer = nn.Identity()
        self.audio_tokenizer = nn.Identity()


class _FakeApertus1p5Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeApertus1p5Core()
        self.lm_head = nn.Linear(4, 16, bias=False)


def _select_definition(monkeypatch, model_type):
    fake_config = SimpleNamespace(model_type=model_type)
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    return auto.check_and_get_model_definition(f"/tmp/{model_type}")


def test_apertus_v15_model_types_select_definitions(monkeypatch):
    assert _select_definition(monkeypatch, "apertus1p5") is Apertus1p5QModel
    assert _select_definition(monkeypatch, "apertus1p5_text") is Apertus1p5TextQModel


def test_apertus_v15_module_tree_matches_multimodal_checkpoint_layout():
    layer_modules = Apertus1p5QModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert Apertus1p5QModel.loader is AutoModelForMultimodalLM
    assert Apertus1p5QModel.require_load_processor is True
    assert Apertus1p5QModel.extract_layers_node() == ["model.language_model.layers"]
    assert Apertus1p5QModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert Apertus1p5QModel.rotary_embedding == "model.language_model.rotary_emb"
    assert flat_modules == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }


def test_apertus_v15_base_modules_preserve_bundled_tokenizers():
    base_modules = set(Apertus1p5QModel.get_base_modules(_FakeApertus1p5Wrapper()))

    assert base_modules == {
        "model.vision_tokenizer",
        "model.audio_tokenizer",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
    }


def test_apertus_v15_calibration_uses_native_multimodal_processor():
    class RecordingProcessor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, conversations, **kwargs):
            self.calls.append((conversations, kwargs))
            return {"input_ids": [[1, 2, 3]], "pixel_values": [[0.0]]}

    model = object.__new__(Apertus1p5QModel)
    nn.Module.__init__(model)
    processor = RecordingProcessor()
    model.processor = processor
    conversations = [
        [{"role": "user", "content": [{"type": "image", "url": "image.jpg"}]}],
        [{"role": "user", "content": "describe the image"}],
    ]

    prepared = model.prepare_dataset(conversations, batch_size=2)

    assert prepared == [{"input_ids": [[1, 2, 3]], "pixel_values": [[0.0]]}]
    assert processor.calls == [
        (
            conversations,
            {
                "add_generation_prompt": False,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "processor_kwargs": {"padding": True},
            },
        )
    ]
