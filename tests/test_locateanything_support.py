from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.locateanything.configuration_locateanything import LocateAnythingConfig
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.hf import load_model_config, normalize_hf_config_compat, prepare_remote_model_init_compat
from gptqmodel.utils.model import MODALITY


@pytest.fixture
def owned_config_dir(tmp_path):
    config = LocateAnythingConfig(
        text_config={"rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0}},
        attn_implementation="eager",
        auto_map={"AutoConfig": "forbidden.Config", "AutoModel": "forbidden.Model"},
    )
    config.save_pretrained(tmp_path)
    return tmp_path


def test_locateanything_model_type_selects_definition(owned_config_dir, monkeypatch):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *_args, **_kwargs: pytest.fail("Package-owned selection imported checkpoint Python"),
    )

    assert (
        auto.check_and_get_model_definition(
            owned_config_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        is LocateAnythingQModel
    )


def test_locateanything_definition_contract():
    assert isinstance(LocateAnythingQModel.loader, LocalModelLoader)
    assert LocateAnythingQModel.loader.config_class.endswith(".configuration_locateanything.LocateAnythingConfig")
    assert LocateAnythingQModel.loader.model_class.endswith(".modeling_locateanything.LocateAnythingForConditionalGeneration")
    assert LocateAnythingQModel.loader.processor_class.endswith(".processing_locateanything.LocateAnythingProcessor")
    assert LocateAnythingQModel.require_trust_remote_code is False
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


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa", "magi"])
def test_locateanything_config_owns_rope_and_attention_policy(trust_remote_code, implementation):
    config = LocateAnythingConfig(
        text_config={"rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0}},
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        attn_implementation=implementation,
    )

    normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)

    assert config.text_config.rope_parameters["rope_theta"] == 1_000_000.0
    expected = "sdpa" if implementation == "magi" else implementation
    assert config._attn_implementation == expected
    assert config.text_config._attn_implementation == expected
    assert config.vision_config._attn_implementation == expected


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_locateanything_remote_compat_cannot_override_owned_config(
    owned_config_dir, monkeypatch, trust_remote_code, bypass,
):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *_args, **_kwargs: pytest.fail("Package-owned config imported checkpoint Python"),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.hf._patch_transformers_remote_code_compat",
        lambda *_args, **_kwargs: pytest.fail("Package-owned config installed remote compatibility shims"),
    )
    config = load_model_config(
        owned_config_dir, trust_remote_code=trust_remote_code,
        local_files_only=True, attn_implementation="eager",
    )
    assert config._attn_implementation == "eager"
    prepare_remote_model_init_compat(owned_config_dir, config, trust_remote_code=trust_remote_code)

    assert type(config) is LocateAnythingConfig
    assert config._attn_implementation == "eager"
    assert config.text_config.rope_parameters["rope_theta"] == 1_000_000.0


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


def test_locateanything_prepare_dataset_uses_declared_processor(monkeypatch):
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
    nn.Module.__init__(qmodel)
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
