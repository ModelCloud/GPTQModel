from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)

from gptqmodel.models.definitions import base_qwen3_vl
from gptqmodel.models.definitions.base_qwen3_vl import BaseQwen3VLGPTQ
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
from gptqmodel.utils.model import MODALITY


class _FakeProcessor:
    def __init__(self):
        self.template_calls = []
        self.processor_calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.template_calls.append((conversations, kwargs))
        return [f"rendered-{index}" for index, _ in enumerate(conversations)]

    def __call__(self, *, text, images, videos, padding, return_tensors):
        batch_size = len(text) if isinstance(text, list) else 1
        image_count = len(images or [])
        self.processor_calls.append(
            {
                "text": text,
                "images": images,
                "videos": videos,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )
        output = {"input_ids": torch.ones((batch_size, 4), dtype=torch.long)}
        if image_count:
            output.update(
                {
                    "pixel_values": torch.arange(image_count, dtype=torch.float32),
                    "image_grid_thw": torch.ones((image_count, 3), dtype=torch.long),
                    "mm_token_type_ids": torch.zeros((batch_size, 4), dtype=torch.long),
                }
            )
        return output


def _qmodel():
    model = object.__new__(Qwen3_5_MoeQModel)
    nn.Module.__init__(model)
    model.model_local_path = "qwen3.5-moe"
    return model


def _image_content(image):
    return {"type": "image", "image": image}


def test_qwen3_5_moe_load_processor_uses_model_path(monkeypatch):
    calls = []
    processor = object()

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return processor

    monkeypatch.setattr(
        base_qwen3_vl.AutoProcessor, "from_pretrained", fake_from_pretrained
    )
    model = _qmodel()

    assert model.load_processor() is processor
    assert calls == [
        (
            ("qwen3.5-moe",),
            {},
        )
    ]


def test_qwen3_5_moe_uses_shared_multimodal_base_contract():
    assert Qwen3_5_MoeQModel.__bases__ == (BaseQwen3VLGPTQ,)
    assert Qwen3_5_MoeQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert Qwen3_5_MoeQModel.loader is AutoModelForImageTextToText
    assert Qwen3_5_MoeQModel.require_load_processor is True


def test_qwen3_5_moe_get_base_modules_only_keeps_visual_module():
    language_model = nn.Module()
    language_model.layers = nn.ModuleList([nn.Identity()])

    core_model = nn.Module()
    core_model.language_model = language_model
    core_model.visual = nn.Identity()

    wrapper = nn.Module()
    wrapper.model = core_model
    wrapper.lm_head = nn.Linear(4, 4)

    assert Qwen3_5_MoeQModel.get_base_modules(wrapper) == ["model.visual"]


@pytest.mark.parametrize(
    "calibration_dataset,batch_size,expected_image_count,expected_batch_size",
    [
        (
            [
                [{"role": "user", "content": "describe this text"}],
            ],
            1,
            0,
            1,
        ),
        (
            [
                [
                    {
                        "role": "user",
                        "content": [_image_content(Image.new("RGB", (2, 2)))],
                    }
                ],
            ],
            1,
            1,
            1,
        ),
        (
            [
                [
                    {
                        "role": "user",
                        "content": [
                            _image_content(Image.new("RGB", (2, 2))),
                            {"type": "text", "text": "compare"},
                            _image_content(Image.new("RGB", (2, 2))),
                        ],
                    }
                ],
            ],
            1,
            2,
            1,
        ),
        (
            [
                [
                    {
                        "role": "user",
                        "content": [_image_content(Image.new("RGB", (2, 2)))],
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": [_image_content(Image.new("RGB", (2, 2)))],
                    }
                ],
            ],
            2,
            2,
            2,
        ),
    ],
)
def test_qwen3_5_moe_prepare_dataset_supports_text_images_and_batches(
    monkeypatch,
    calibration_dataset,
    batch_size,
    expected_image_count,
    expected_batch_size,
):
    processor = _FakeProcessor()
    monkeypatch.setattr(Qwen3_5_MoeQModel, "load_processor", lambda self: processor)
    model = _qmodel()

    prepared = model.prepare_dataset(calibration_dataset, batch_size=batch_size)

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].shape == (expected_batch_size, 4)
    call = processor.processor_calls[0]
    assert len(call["images"] or []) == expected_image_count
    assert call["padding"] is True
    assert call["return_tensors"] == "pt"
    if expected_image_count:
        assert prepared[0]["pixel_values"].shape == (expected_image_count,)
        assert prepared[0]["image_grid_thw"].shape == (expected_image_count, 3)
    else:
        assert "pixel_values" not in prepared[0]
        assert "image_grid_thw" not in prepared[0]


def test_qwen3_5_moe_prepare_dataset_rejects_invalid_image_entry(monkeypatch):
    processor = _FakeProcessor()
    monkeypatch.setattr(Qwen3_5_MoeQModel, "load_processor", lambda self: processor)
    model = _qmodel()
    invalid = [[{"role": "user", "content": [{"type": "image", "image": None}]}]]

    with pytest.raises(
        ValueError, match="Invalid Qwen3-VL image content.*content item 0"
    ):
        model.prepare_dataset(invalid)


@pytest.mark.parametrize(
    "invalid_dataset", [{"text": "not a conversation"}, "not a conversation", []]
)
def test_qwen3_5_moe_prepare_dataset_rejects_invalid_top_level_data(
    monkeypatch, invalid_dataset
):
    processor = _FakeProcessor()
    monkeypatch.setattr(Qwen3_5_MoeQModel, "load_processor", lambda self: processor)
    model = _qmodel()

    with pytest.raises(ValueError, match="calibration"):
        model.prepare_dataset(invalid_dataset)


class _FakeVision(nn.Module):
    def forward(self, pixel_values, image_grid_thw):
        del image_grid_thw
        values = pixel_values.float().mean(dim=tuple(range(1, pixel_values.ndim)))
        return values.unsqueeze(-1).expand(-1, 4)


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states)


class _FakeMultimodalCore(nn.Module):
    image_token_id = 1

    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.visual = _FakeVision()
        with torch.no_grad():
            self.language_model.embed_tokens.weight.copy_(
                torch.arange(64).reshape(16, 4)
            )

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None):
        hidden_states = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_mask = input_ids.eq(self.image_token_id)
            image_embeds = self.visual(pixel_values, image_grid_thw)
            hidden_states[image_mask] = image_embeds
        return self.language_model(inputs_embeds=hidden_states)


def test_qwen3_5_moe_multimodal_embeddings_reach_first_language_layer():
    core = _FakeMultimodalCore()
    captures = []
    hook = core.language_model.layers[0].register_forward_pre_hook(
        lambda _module, inputs: captures.append(inputs[0].detach().clone())
    )
    try:
        input_ids = torch.tensor([[3, core.image_token_id, 5]])
        core(input_ids)
        text_hidden_states = captures[-1]
        core(
            input_ids,
            pixel_values=torch.tensor([[2.0, 2.0]]),
            image_grid_thw=torch.ones((1, 3), dtype=torch.long),
        )
        first_image_hidden_states = captures[-1]
        core(
            input_ids,
            pixel_values=torch.tensor([[7.0, 7.0]]),
            image_grid_thw=torch.ones((1, 3), dtype=torch.long),
        )
        second_image_hidden_states = captures[-1]
    finally:
        hook.remove()

    for hidden_states in (
        text_hidden_states,
        first_image_hidden_states,
        second_image_hidden_states,
    ):
        assert torch.isfinite(hidden_states).all()
    assert not torch.allclose(text_hidden_states[:, 1], first_image_hidden_states[:, 1])
    assert not torch.allclose(
        first_image_hidden_states[:, 1], second_image_hidden_states[:, 1]
    )
    assert torch.equal(text_hidden_states[:, 0], first_image_hidden_states[:, 0])
    assert torch.equal(text_hidden_states[:, 2], second_image_hidden_states[:, 2])


def test_qwen3_5_moe_configs_save_reload_model_and_vision_fields(tmp_path):
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    vision_config = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=16,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    multimodal_config = Qwen3_5MoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=31,
    )

    multimodal_path = tmp_path / "qwen3_5_moe"
    text_path = tmp_path / "qwen3_5_moe_text"
    multimodal_config.save_pretrained(multimodal_path)
    text_config.save_pretrained(text_path)

    reloaded_multimodal = Qwen3_5MoeConfig.from_pretrained(multimodal_path)
    reloaded_text = Qwen3_5MoeTextConfig.from_pretrained(text_path)

    assert reloaded_multimodal.model_type == "qwen3_5_moe"
    assert reloaded_multimodal.image_token_id == 31
    assert reloaded_multimodal.vision_config.model_type == "qwen3_5_moe_vision"
    assert reloaded_multimodal.vision_config.patch_size == 2
    assert reloaded_multimodal.vision_config.out_hidden_size == 16
    assert reloaded_text.model_type == "qwen3_5_moe_text"
    assert reloaded_text.vocab_size == 32
