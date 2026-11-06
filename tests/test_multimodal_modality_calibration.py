# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for multimodal calibration modality declarations.

The language towers of these wrappers consume image embeddings at inference
time. They must therefore declare ``IMAGE_TO_TEXT`` and build a calibration
stream that contains ``pixel_values`` instead of falling back to text-only
tokenization.
"""

import torch
from PIL import Image
from torch import nn

from gptqmodel.models.definitions.base_qwen3_vl import BaseQwen3VLGPTQ
from gptqmodel.models.definitions.glm_ocr import GlmOCRGPTQ
from gptqmodel.models.definitions.minimax_m3_vl import MiniMaxM3VLGPTQ
from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel
from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
from gptqmodel.utils.model import MODALITY
from tests.models.ovis import image_to_test_dataset


def _conversation():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8))},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


class _FakeTokenizer:
    def apply_chat_template(self, conversations, **kwargs):
        return "rendered"


class _FakeVisionProcessor:
    """Processor that mirrors the fields each family needs for calibration."""

    def __init__(self, *, tokenizer=None, image_fields, accepts_videos=False):
        self.tokenizer = tokenizer
        self.image_fields = image_fields
        self.accepts_videos = accepts_videos
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        return "rendered"

    def __call__(self, *, text, images, padding, return_tensors, videos=None):
        self.calls.append(
            {
                "text": text,
                "images": images,
                "videos": videos,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )
        output = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
        if images:
            output["pixel_values"] = torch.zeros((len(images), 3, 4, 4))
            for name, value in self.image_fields.items():
                output[name] = value(len(images))
        return output


def _make_qmodel(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.model_local_path = "fake-model"
    return model


def test_multimodal_models_declare_image_to_text_modality():
    for cls in (
        GlmOCRGPTQ,
        MiniMaxM3VLGPTQ,
        Qwen3OmniMoeGPTQ,
        NemotronOmniQModel,
    ):
        assert MODALITY.IMAGE_TO_TEXT in cls.modality, cls.__name__
        assert MODALITY.TEXT in cls.modality, cls.__name__


def test_glm_ocr_uses_shared_qwen3_vl_multimodal_path():
    assert issubclass(GlmOCRGPTQ, BaseQwen3VLGPTQ)
    assert GlmOCRGPTQ.require_mm_token_type_ids is True
    assert GlmOCRGPTQ.require_load_processor is True
    assert GlmOCRGPTQ.rotary_embedding == "model.language_model.rotary_emb"


def test_minimax_prepare_dataset_keeps_image_fields(monkeypatch):
    processor = _FakeVisionProcessor(
        tokenizer=_FakeTokenizer(),
        image_fields={"image_grid_thw": lambda n: torch.ones((n, 3), dtype=torch.long)},
    )
    monkeypatch.setattr(MiniMaxM3VLGPTQ, "load_processor", lambda self: processor)
    model = _make_qmodel(MiniMaxM3VLGPTQ)

    prepared = model.prepare_dataset([_conversation()], batch_size=1)

    assert len(prepared) == 1
    assert prepared[0]["pixel_values"].shape == (1, 3, 4, 4)
    assert prepared[0]["image_grid_thw"].shape == (1, 3)
    assert processor.calls[0]["text"] == "rendered"
    assert len(processor.calls[0]["images"]) == 1


def test_qwen3_omni_prepare_dataset_keeps_image_fields(monkeypatch):
    processor = _FakeVisionProcessor(
        image_fields={"image_grid_thw": lambda n: torch.ones((n, 3), dtype=torch.long)},
        accepts_videos=True,
    )
    monkeypatch.setattr(Qwen3OmniMoeGPTQ, "load_processor", lambda self: processor)
    model = _make_qmodel(Qwen3OmniMoeGPTQ)

    prepared = model.prepare_dataset([_conversation()], batch_size=1)

    assert prepared[0]["pixel_values"].shape == (1, 3, 4, 4)
    assert prepared[0]["image_grid_thw"].shape == (1, 3)
    assert processor.calls[0]["videos"] is None


def test_nemotron_prepare_dataset_keeps_image_fields(monkeypatch):
    processor = _FakeVisionProcessor(
        image_fields={
            "num_patches": lambda n: torch.ones((n,), dtype=torch.long),
            "num_tokens": lambda n: torch.ones((n,), dtype=torch.long),
            "imgs_sizes": lambda n: torch.ones((n, 2), dtype=torch.long),
        },
    )
    monkeypatch.setattr(NemotronOmniQModel, "load_processor", lambda self: processor)
    model = _make_qmodel(NemotronOmniQModel)

    prepared = model.prepare_dataset([_conversation()], batch_size=1)

    assert prepared[0]["pixel_values"].shape == (1, 3, 4, 4)
    for name in ("num_patches", "num_tokens", "imgs_sizes"):
        assert name in prepared[0]


def test_nemotron_batch_quantize_disabled_for_variable_image_sizes():
    # Nemotron's dynamic-resolution processor returns a list of pixel tensors
    # when image sizes differ, which its generate() cannot consume.
    assert NemotronOmniQModel.support_batch_quantize is False


def test_glm_ocr_prepare_dataset_requires_mm_token_type_ids(monkeypatch):
    processor = _FakeVisionProcessor(
        image_fields={
            "image_grid_thw": lambda n: torch.ones((n, 3), dtype=torch.long),
            "mm_token_type_ids": lambda n: torch.zeros((1, 4), dtype=torch.long),
        },
    )
    monkeypatch.setattr(GlmOCRGPTQ, "load_processor", lambda self: processor)
    model = _make_qmodel(GlmOCRGPTQ)

    prepared = model.prepare_dataset([_conversation()], batch_size=1)
    assert "mm_token_type_ids" in prepared[0]

    processor.image_fields.pop("mm_token_type_ids")
    try:
        model.prepare_dataset([_conversation()], batch_size=1)
    except ValueError as exc:
        assert "mm_token_type_ids" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("missing mm_token_type_ids must be rejected")


class _FakeNemotronModel:
    def __init__(self):
        self.language_model = self
        self.generate_calls = []
        self.language_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.zeros((1, 1), dtype=torch.long)

    def __call__(self, **kwargs):
        self.language_calls.append(kwargs)
        return None


def test_nemotron_multimodal_capture_uses_generate_without_extra_fields():
    wrapper = _make_qmodel(NemotronOmniQModel)
    fake_model = _FakeNemotronModel()
    wrapper.model = fake_model

    wrapper.run_input_capture(
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long, device="cpu"),
            "attention_mask": torch.ones((1, 4), dtype=torch.long, device="cpu"),
            "pixel_values": torch.zeros((1, 3, 4, 4), device="cpu"),
            "imgs_sizes": torch.ones((1, 2), dtype=torch.long, device="cpu"),
            "num_patches": torch.ones((1,), dtype=torch.long, device="cpu"),
            "num_tokens": torch.ones((1,), dtype=torch.long, device="cpu"),
        },
        use_cache=False,
        data_device=torch.device("cpu"),
    )

    assert len(fake_model.generate_calls) == 1
    call = fake_model.generate_calls[0]
    assert set(call) == {"input_ids", "attention_mask", "pixel_values", "max_new_tokens"}
    assert call["max_new_tokens"] == 1
    assert not fake_model.language_calls


def test_nemotron_text_capture_stays_on_language_model():
    wrapper = _make_qmodel(NemotronOmniQModel)
    fake_model = _FakeNemotronModel()
    wrapper.model = fake_model

    wrapper.run_input_capture(
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long, device="cpu"),
            "attention_mask": torch.ones((1, 4), dtype=torch.long, device="cpu"),
        },
        use_cache=False,
        data_device=torch.device("cpu"),
    )

    assert len(fake_model.language_calls) == 1
    assert not fake_model.generate_calls


def test_get_calib_dataset_dispatches_new_multimodal_models(monkeypatch):
    sentinel = object()
    calls = []

    def fake_prepare_dataset(format_func, n_sample=20):
        calls.append(format_func)
        return sentinel

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    for cls in (MiniMaxM3VLGPTQ, NemotronOmniQModel):
        calls.clear()
        result = image_to_test_dataset.get_calib_dataset(object.__new__(cls))
        assert result is sentinel
        assert calls == [image_to_test_dataset.format_qwen2_vl_dataset]

    calls.clear()
    result = image_to_test_dataset.get_calib_dataset(object.__new__(Qwen3OmniMoeGPTQ))
    assert result is sentinel
    assert calls == [image_to_test_dataset.format_qwen2_5_omni_dataset]

    calls.clear()
    result = image_to_test_dataset.get_calib_dataset(object.__new__(GlmOCRGPTQ))
    assert result is sentinel
    assert calls == [image_to_test_dataset.format_qwen2_vl_dataset]
