# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared consumers follow the native generative task and media layout."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from test_unlimited_ocr_local_port import source_fixture, tiny_config
from torch import nn
from transformers import UnlimitedOcrConfig

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.unlimited_ocr import UnlimitedOCRQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.utils.model import MODALITY


UNLIMITED_OCR_MODEL_TYPES = ("unlimited_ocr", "unlimited-ocr")


@pytest.mark.parametrize("model_type", UNLIMITED_OCR_MODEL_TYPES)
@pytest.mark.parametrize("trust", [False, True])
def test_unlimited_ocr_model_type_selects_definition(tmp_path, monkeypatch, model_type, trust):
    calls = []
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    config = SimpleNamespace(model_type=model_type)
    monkeypatch.setattr(auto, "load_model_config", lambda *args, **kwargs: calls.append(kwargs) or config)
    assert auto.check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is UnlimitedOCRQModel
    assert calls[-1]["trust_remote_code"] is False
    assert isinstance(UnlimitedOCRQModel.loader, LocalModelLoader)
    assert UnlimitedOCRQModel.require_trust_remote_code is False


def test_unlimited_ocr_module_tree_expands_mha_and_moe_paths():
    modules = UnlimitedOCRQModel.simple_layer_modules(
        model_config=tiny_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat = {name for block in modules for name in block}
    assert UnlimitedOCRQModel.__bases__ == (BaseQModel,)
    assert UnlimitedOCRQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert UnlimitedOCRQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert UnlimitedOCRQModel.extract_layers_node() == ["model.language_model.layers"]
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"self_attn.{projection}" in flat
    for prefix in ("mlp", "mlp.shared_experts", "mlp.experts.0", "mlp.experts.3"):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert f"{prefix}.{projection}" in flat


@pytest.mark.parametrize("model_type", UNLIMITED_OCR_MODEL_TYPES)
def test_unlimited_ocr_native_config_preserves_legacy_values(tmp_path, model_type):
    payload = tiny_config().to_dict()
    payload["model_type"] = model_type
    payload["language_config"] = payload.pop("text_config")
    payload["language_config"].pop("rope_parameters", None)
    payload["language_config"].update(attention_dropout=0.25, rope_theta=500000.0, sliding_window=64)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(payload))
    before = config_file.read_bytes()
    config = UnlimitedOCRQModel.loader.load_config(tmp_path, trust_remote_code=False)
    assert isinstance(config, UnlimitedOcrConfig)
    assert config.model_type == "unlimited_ocr"
    assert config.text_config.attention_dropout == 0.25
    assert config.text_config.rope_parameters["rope_theta"] == 500000.0
    assert config.text_config.sliding_window == 64
    assert config_file.read_bytes() == before


@pytest.fixture
def native_wrapper():
    wrapper = object.__new__(UnlimitedOCRQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = UnlimitedOCRQModel.loader.from_config(tiny_config()).eval()
    return wrapper


def test_unlimited_ocr_keeps_multimodal_modules_in_base_dtype(native_wrapper):
    assert set(UnlimitedOCRQModel.get_base_modules(native_wrapper.model)) == {
        "model.vision_tower",
        "model.multi_modal_projector",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
    }
    assert UnlimitedOCRQModel.get_modules_with_direct_meta_tensors(native_wrapper.model) == ["model"]


def test_unlimited_ocr_prepares_native_image_layout_and_retries(tmp_path):
    source_fixture(tmp_path, released_vocab=True)
    wrapper = GPTQModel.load(str(tmp_path), dtype=torch.float32, backend=BACKEND.TORCH)
    image = Image.fromarray(torch.arange(32 * 32 * 3).remainder(256).byte().reshape(32, 32, 3).numpy())
    with pytest.raises(ValueError, match="exactly one"):
        wrapper.prepare_dataset([{"image": image, "text": "No image placeholder"}])
    prepared = wrapper.prepare_dataset([{"image": image, "text": "<image>\nFree OCR."}])[0]
    assert (prepared["input_ids"] == 128815).sum().item() == 7
    assert prepared["pixel_values"].shape == (1, 3, 32, 32)
    assert prepared["pixel_values"].std() > 0
    with torch.inference_mode():
        output = wrapper.model(**prepared, use_cache=False).logits
    assert output.shape[:2] == prepared["input_ids"].shape
    assert torch.isfinite(output).all()


def test_unlimited_ocr_move_input_capture_casts_native_pixels(native_wrapper):
    native_wrapper.model.model.vision_tower.to(dtype=torch.bfloat16)
    example = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 32, 32),
        "pixel_values_local": torch.randn(2, 3, 32, 32),
    }
    moved = native_wrapper.move_input_capture_example(example, torch.device("cpu"))
    assert moved["pixel_values"].dtype == moved["pixel_values_local"].dtype == torch.bfloat16
    assert moved["input_ids"].dtype == moved["attention_mask"].dtype == torch.long


def test_unlimited_ocr_restores_checkpoint_omitted_vision_position_ids(native_wrapper):
    core = native_wrapper.model.model
    embeddings = core.vision_tower.vision_encoder.embeddings
    expected = torch.arange(embeddings.position_embedding.num_embeddings).expand(1, -1)
    embeddings.position_ids = None
    UnlimitedOCRQModel._restore_vision_position_ids(core)
    torch.testing.assert_close(embeddings.position_ids, expected)
    assert "position_ids" not in embeddings.state_dict()


def test_unlimited_ocr_moves_direct_multimodal_parameters(native_wrapper):
    core = native_wrapper.model.model
    core.image_newline.requires_grad_(False)
    before = core.image_newline
    native_wrapper._move_direct_parameters(torch.device("meta"))
    assert core.image_newline is not before
    assert core.image_newline.device.type == core.view_separator.device.type == "meta"
    assert core.image_newline.requires_grad is False
