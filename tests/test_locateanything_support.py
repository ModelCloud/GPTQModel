import sys
from types import ModuleType, SimpleNamespace

import torch
import transformers
from torch import nn
from transformers import AutoModel

from gptqmodel.models import auto
from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.hf import normalize_hf_config_compat, prepare_remote_model_init_compat
from gptqmodel.utils.model import MODALITY


def test_locateanything_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="locateanything")
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code: True)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert (
        auto.check_and_get_model_definition(
            "/tmp/LocateAnything-3B",
            trust_remote_code=True,
        )
        is LocateAnythingQModel
    )


def test_locateanything_definition_contract():
    assert LocateAnythingQModel.loader is AutoModel
    assert LocateAnythingQModel.require_trust_remote_code is True
    assert LocateAnythingQModel.require_load_processor is False
    assert LocateAnythingQModel.out_of_model_tensors == {
        "files": [
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.json",
        ],
    }
    assert LocateAnythingQModel.support_batch_quantize is False
    assert LocateAnythingQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert LocateAnythingQModel.lm_head == "language_model.lm_head"
    assert LocateAnythingQModel.pre_lm_head_norm_module == "language_model.model.norm"
    assert LocateAnythingQModel.extract_layers_node() == ["language_model.model.layers"]
    assert LocateAnythingQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=QuantizeConfig(),
    ) == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def test_locateanything_config_restores_remote_qwen_rope_theta():
    text_config = SimpleNamespace(
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )
    config = SimpleNamespace(
        model_type="locateanything",
        text_config=text_config,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    )

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert text_config.rope_theta == 1_000_000.0


def test_locateanything_remote_attention_compat(monkeypatch):
    calls = []
    module_root = "transformers_modules.fake_locateanything"
    outer_module = ModuleType(f"{module_root}.modeling_locateanything")
    qwen_module = ModuleType(f"{module_root}.modeling_qwen2")

    class DummyOuterModel:
        _supports_flash_attn_2 = True

        def __init__(self):
            self.config = SimpleNamespace(
                model_type="locateanything",
                vision_config=SimpleNamespace(_attn_implementation="eager"),
                text_config=SimpleNamespace(_attn_implementation="eager"),
            )

        def _check_and_adjust_attn_implementation(self, implementation, is_init_check=False):
            calls.append((implementation, is_init_check))
            return implementation

    class DummyQwenBase:
        _supports_flash_attn_2 = True

    class DummyQwenForCausalLM(DummyQwenBase):
        pass

    DummyOuterModel.__module__ = outer_module.__name__
    DummyQwenBase.__module__ = qwen_module.__name__
    DummyQwenForCausalLM.__module__ = qwen_module.__name__
    outer_module.DummyOuterModel = DummyOuterModel
    qwen_module.DummyQwenBase = DummyQwenBase
    qwen_module.DummyQwenForCausalLM = DummyQwenForCausalLM

    monkeypatch.setitem(sys.modules, outer_module.__name__, outer_module)
    monkeypatch.setitem(sys.modules, qwen_module.__name__, qwen_module)
    monkeypatch.setattr(transformers.utils, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: DummyOuterModel,
    )
    config = SimpleNamespace(
        model_type="locateanything",
        auto_map={"AutoModel": "modeling_locateanything.DummyOuterModel"},
    )

    prepare_remote_model_init_compat("/tmp/LocateAnything-3B", config)
    model = DummyOuterModel()
    results = [
        model._check_and_adjust_attn_implementation(
            requested,
            is_init_check=True,
            allow_all_kernels=False,
        )
        for requested in ("eager", "flash_attention_2")
    ]

    assert results == ["sdpa", "sdpa"]
    assert calls == [("sdpa", True), ("sdpa", True)]
    assert model.config.vision_config._attn_implementation == "sdpa"
    assert model.config.text_config._attn_implementation == "sdpa"
    assert DummyOuterModel._supports_flash_attn is True
    assert DummyQwenBase._supports_flash_attn is True
    assert DummyQwenForCausalLM._supports_flash_attn is True
    assert getattr(DummyOuterModel, "_gptqmodel_attn_adjust_kwargs_patch", False) is True


def _fake_model_tree():
    text_model = nn.Module()
    text_model.embed_tokens = nn.Embedding(8, 4)
    text_model.layers = nn.ModuleList([nn.Linear(4, 4)])
    text_model.norm = nn.LayerNorm(4)

    language_model = nn.Module()
    language_model.model = text_model
    language_model.lm_head = nn.Linear(4, 8, bias=False)

    model = nn.Module()
    model.language_model = language_model
    model.vision_model = nn.Linear(4, 4)
    model.mlp1 = nn.Sequential(nn.Linear(4, 4))
    return model


def test_locateanything_keeps_multimodal_modules_in_base_dtype():
    model = _fake_model_tree()

    assert set(LocateAnythingQModel.get_base_modules(model)) == {
        "language_model.lm_head",
        "language_model.model.embed_tokens",
        "language_model.model.norm",
        "vision_model",
        "mlp1",
    }


def test_locateanything_prepare_dataset_uses_remote_processor(monkeypatch):
    calls = []

    class FakeProcessor:
        def py_apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            return "rendered prompt"

        def process_vision_info(self, messages):
            calls.append(("vision", messages))
            return ["image"], []

        def __call__(self, **kwargs):
            calls.append(("processor", kwargs))
            return {"input_ids": torch.tensor([[1, 2]])}

    qmodel = object.__new__(LocateAnythingQModel)
    monkeypatch.setattr(qmodel, "load_processor", lambda: FakeProcessor())
    messages = [{"role": "user", "content": [{"type": "image", "image": "url"}]}]

    result = qmodel.prepare_dataset([messages], batch_size=8)

    assert result[0]["input_ids"].tolist() == [[1, 2]]
    assert calls == [
        (
            "template",
            messages,
            {"tokenize": False, "add_generation_prompt": False},
        ),
        ("vision", messages),
        (
            "processor",
            {
                "text": ["rendered prompt"],
                "images": ["image"],
                "videos": [],
                "return_tensors": "pt",
            },
        ),
    ]


def test_locateanything_text_forward_bypasses_visual_wrapper():
    class LanguageModel(nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return ("text", input_ids, kwargs)

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

        def forward(self, *args, **kwargs):
            return ("visual", args, kwargs)

    qmodel = object.__new__(LocateAnythingQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = OuterModel()

    input_ids = torch.tensor([[1, 2]])
    assert qmodel(input_ids=input_ids)[0] == "text"
    assert qmodel(input_ids)[0] == "text"
    assert qmodel(pixel_values=torch.ones(1), input_ids=input_ids)[0] == "visual"


def test_locateanything_move_input_capture_casts_pixels_to_vision_dtype():
    qmodel = object.__new__(LocateAnythingQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = SimpleNamespace(vision_model=nn.Linear(2, 2).to(dtype=torch.bfloat16))

    result = qmodel.move_input_capture_example(
        {
            "pixel_values": torch.ones(1, 2, dtype=torch.float32),
            "image_grid_hws": [[18, 26]],
        },
        torch.device("cpu"),
    )

    assert result["pixel_values"].dtype is torch.bfloat16
    assert result["image_grid_hws"].dtype is torch.int32
    assert result["image_grid_hws"].tolist() == [[18, 26]]


def test_locateanything_input_capture_uses_decoder_visual_features_path():
    calls = []

    class LanguageModel(nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            return "captured"

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()
            self.mlp1 = nn.Identity()
            self.image_token_index = 42

        def extract_feature(self, pixel_values, image_grid_hws):
            assert pixel_values.shape == (1, 2)
            assert image_grid_hws.tolist() == [[1, 2]]
            return [torch.ones(2, 4)]

    qmodel = object.__new__(LocateAnythingQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = OuterModel()
    input_ids = torch.tensor([[42, 42]])
    attention_mask = torch.ones_like(input_ids)

    result = qmodel.run_input_capture(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.ones(1, 2),
            "image_grid_hws": torch.tensor([[1, 2]], dtype=torch.int32),
        },
        use_cache=False,
        data_device=torch.device("cpu"),
    )

    assert result == "captured"
    assert calls[0]["input_ids"] is input_ids
    assert calls[0]["attention_mask"] is attention_mask
    assert calls[0]["visual_features"].shape == (2, 4)
    assert calls[0]["image_token_index"] == 42
    assert calls[0]["use_cache"] is False
