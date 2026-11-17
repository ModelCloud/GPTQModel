# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Capture must not override a multimodal model's own mRoPE position ids (#4933).

#4930 made `run_input_capture` pass mask-derived 2-D `position_ids` to every
model whose `forward` accepts them. Qwen3-VL/Qwen3.5/GLM-4V/Hunyuan-VL-class
models compute their 3-D/4-D image-grid ids inside `forward` only when
`position_ids is None`, so the derived ids replaced the mRoPE grid with
sequential text positions and calibrated image rows on hidden states the model
never produces at inference.

The derivation must stay for models with learned absolute position embeddings
(#4861) and be skipped for a model that owns the computation.
"""

import torch
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from gptqmodel.models.base import BaseQModel, _with_capture_position_ids


class _CaptureQModel:
    """Minimal stand-in exposing the production capture forward."""

    INPUT_EMBEDDING_EXTRA_ARGS = None
    _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings
    run_input_capture = BaseQModel.run_input_capture

    def __init__(self, model):
        self.model = model


class _MRopeInner(nn.Module):
    """Stand-in for the inner multimodal model that owns the grid ids."""

    def compute_3d_position_ids(self, input_ids, attention_mask=None):
        batch, seq = input_ids.shape
        return torch.arange(seq).view(1, 1, -1).expand(3, batch, -1).clone()


class _MRopeModel(nn.Module):
    """Outer wrapper: computes mRoPE ids only when `position_ids` is None."""

    def __init__(self):
        super().__init__()
        self.model = _MRopeInner()
        self.received = "unset"

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=None):
        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(input_ids, attention_mask)
        self.received = position_ids
        return input_ids


class _TextModel(nn.Module):
    """Plain absolute-position model: no mRoPE capability."""

    def __init__(self):
        super().__init__()
        self.received = "unset"

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=None):
        self.received = position_ids
        return input_ids


def _capture(model, example, *, derive=None):
    qmodel = _CaptureQModel(model)
    if derive is not None:
        qmodel.CAPTURE_DERIVE_POSITION_IDS = derive
    qmodel.run_input_capture(example, use_cache=False, data_device=torch.device("cpu"))


LEFT_PADDED = {
    "input_ids": torch.tensor([[0, 0, 5, 6], [7, 8, 9, 0]]),
    "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]]),
}


def test_mrope_model_keeps_position_ids_unset():
    """An mRoPE model computes its own grid ids; capture must pass None."""

    model = _MRopeModel()
    example = dict(LEFT_PADDED)
    example["input_ids"] = example["input_ids"].clone()
    example["attention_mask"] = example["attention_mask"].clone()

    _capture(model, example)

    assert model.received is not None
    assert model.received.ndim == 3, "capture must let the model build t/h/w ids"
    assert "position_ids" not in example, "capture must not mutate the caller's batch"
    assert _with_capture_position_ids(model, example) is example


def test_plain_text_model_still_gets_left_pad_positions():
    """The #4861 fix must keep deriving 1-D ids for absolute-position models."""

    model = _TextModel()
    example = dict(LEFT_PADDED)

    _capture(model, example)

    assert model.received is not None
    assert model.received.shape == example["attention_mask"].shape
    real = example["attention_mask"].bool()
    running = example["attention_mask"].long().cumsum(-1) - 1
    torch.testing.assert_close(model.received[real], running[real])
    assert "position_ids" not in example


def test_explicit_definition_flag_overrides_auto_detection():
    """A definition can force the derivation on or off."""

    model = _MRopeModel()
    _capture(model, dict(LEFT_PADDED), derive=True)
    assert model.received.ndim == 2

    model = _TextModel()
    _capture(model, dict(LEFT_PADDED), derive=False)
    assert model.received is None


def _tiny_qwen3_vl():
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 64,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": (),
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def _image_example():
    # One image with a 4x4 patch grid merges to four image tokens, whose
    # t/h/w grid positions differ from the sequential text positions.
    return {
        "input_ids": torch.tensor([[62, 60, 60, 60, 60, 63, 5, 6, 7]]),
        "attention_mask": torch.ones(1, 9, dtype=torch.long),
        "pixel_values": torch.randn(16, 12),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0]]),
    }


def test_image_row_capture_matches_eager_mrope_ids():
    """The language model must see the same ids in capture and in inference."""

    model = _tiny_qwen3_vl()
    seen = []
    example = _image_example()

    def hook(module, args, kwargs):
        position_ids = kwargs.get("position_ids")
        seen.append(None if position_ids is None else position_ids.detach().clone())

    handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(**example)
            eager = seen[-1]
            _capture(model, example)
            captured = seen[-1]
    finally:
        handle.remove()

    assert eager is not None and eager.ndim == 3
    assert captured is not None, "capture passed no position_ids, so the model built mRoPE ids"
    assert captured.ndim == 3
    torch.testing.assert_close(captured, eager)
    # Guard against the pre-#4933 behavior: the derived 2-D ids were sequential
    # and the t/h/w channels differed from each other for image tokens.
    assert not torch.equal(captured[0], captured[1])
