# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Per-layer replay checks for multimodal calibration (issue #4015).

Each test builds a tiny random model from the real Transformers config, records
what the native forward hands to every decoder layer, and replays the captured
first-layer state through the model definition's replay hooks. The replay must
reproduce the native layer inputs; end-to-end logits alone cannot catch the
silent DeepStack / image-block-mask bugs.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
from gptqmodel.models.definitions.mllama import MLlamaQModel
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel


class _StopCapture(Exception):
    """Abort a capture forward from the first-layer hook."""


def _cpu():
    return torch.device("cpu")


def _qwen3_vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 4,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "deepstack_visual_indexes": (1, 2, 3),
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    return model, _qwen3_vl_inputs(config)


def _qwen3_vl_inputs(config, batch_size=1):
    image_token = config.image_token_id
    input_ids = torch.tensor(
        [[5, 6, 58, image_token, image_token, image_token, image_token, 59, 7, 8]]
        * batch_size
    )
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token] = 1
    return {
        "input_ids": input_ids,
        "pixel_values": torch.randn(batch_size * 16, 3 * 2 * 16 * 16),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * batch_size),
        "mm_token_type_ids": mm_token_type_ids,
    }


def _gemma3_vl():
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

    torch.manual_seed(0)
    text_config = Gemma3TextConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    config = Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        mm_tokens_per_image=4,
        boi_token_index=201,
        eoi_token_index=202,
        image_token_index=200,
    )
    model = Gemma3ForConditionalGeneration(config).eval()
    image_token = config.image_token_index
    input_ids = torch.tensor([[5, 6, 201, image_token, image_token, image_token, image_token, 202, 7, 8]])
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[input_ids == image_token] = 1
    return model, {
        "input_ids": input_ids,
        "pixel_values": torch.randn(1, 3, 28, 28),
        "token_type_ids": token_type_ids,
    }


def _qwen2_vl_inputs(config, batch_size=2):
    image_token = config.image_token_id
    input_ids = torch.tensor(
        [[5, 6, 58, image_token, image_token, image_token, image_token, 59, 7, 8]]
        * batch_size
    )
    attention_mask = torch.ones_like(input_ids)
    # Left padding makes the 2-D mask non-trivial, which is what crashed the
    # first-layer capture before it materialized the 4-D causal mask.
    attention_mask[1:, -2:] = 0
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.randn(batch_size * 16, 3 * 2 * 14 * 14),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * batch_size),
        "mm_token_type_ids": mm_token_type_ids,
    }


def _qwen2_vl(kind):
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLConfig,
        Qwen2VLForConditionalGeneration,
    )

    torch.manual_seed(0)
    text_config = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 64,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "patch_size": 14,
    }
    if kind == "qwen2_vl":
        config = Qwen2VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=58,
            vision_end_token_id=59,
        )
        model_cls = Qwen2VLForConditionalGeneration
        qmodel_cls = Qwen2VLQModel
    else:
        config = Qwen2_5_VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=58,
            vision_end_token_id=59,
        )
        model_cls = Qwen2_5_VLForConditionalGeneration
        qmodel_cls = Qwen2_5_VLQModel
    model = model_cls(config).eval()
    return model, qmodel_cls, _qwen2_vl_inputs(config)


def _mllama():
    from transformers import MllamaConfig, MllamaForConditionalGeneration

    torch.manual_seed(0)
    config = MllamaConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "cross_attention_layers": [],
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
            "max_num_tiles": 4,
            "num_global_layers": 0,
        },
        image_token_index=60,
    )
    model = MllamaForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] * 2)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1:, -2:] = 0
    return model, {"input_ids": input_ids, "attention_mask": attention_mask}


def _record_native_layers(layers, forward):
    recorded = {}
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs, index=index: recorded.__setitem__(
                index,
                {
                    "hidden_states": args[0].detach().clone(),
                    "attention_mask": kwargs.get("attention_mask"),
                    "position_embeddings": kwargs.get("position_embeddings"),
                },
            ),
            with_kwargs=True,
        )
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.inference_mode():
            forward()
    finally:
        for handle in handles:
            handle.remove()
    return recorded


def _replay_layers(
    model,
    qmodel_cls,
    layers,
    forward,
    example,
    start_capture=None,
    stop_capture=None,
):
    """Capture the first-layer state, then replay every layer through the definition hooks."""

    qmodel = object.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    captured = {}

    def capture_first(module, args, kwargs):
        replay_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("hidden_states", "attention_mask", "position_ids")
        }
        replay_kwargs = qmodel.capture_first_layer_input_kwargs(
            args=args,
            kwargs=kwargs,
            batch_device=_cpu(),
            layer_input_kwargs=replay_kwargs,
        )
        capture_layer_type = getattr(qmodel, "capture_layer_type_replay_kwargs", None)
        if callable(capture_layer_type):
            replay_kwargs = capture_layer_type(
                args=args,
                kwargs=kwargs,
                batch_device=_cpu(),
                layer_input_kwargs=replay_kwargs,
            )
        captured["hidden_states"] = args[0].detach().clone()
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["kwargs"] = replay_kwargs
        raise _StopCapture

    handle = layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)
    if callable(start_capture):
        start_capture()
    qmodel.begin_input_capture_example(
        {"input_ids": example.get("input_ids"), "attention_mask": example.get("attention_mask")},
        _cpu(),
    )
    try:
        forward()
    except _StopCapture:
        pass
    finally:
        handle.remove()
        if callable(stop_capture):
            stop_capture()
        qmodel.end_input_capture_example()

    recorded = {}
    hidden_states = captured["hidden_states"]
    for layer in layers:
        index = getattr(getattr(layer, "self_attn", None), "layer_idx", None)
        additional_inputs = {
            key: value
            for key, value in captured["kwargs"].items()
            if key not in ("past_key_values", "past_key_value")
        }
        additional_inputs["attention_mask"] = captured["attention_mask"]
        additional_inputs["position_ids"] = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        additional_inputs["use_cache"] = False
        additional_inputs = qmodel.prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=[hidden_states],
            additional_inputs=additional_inputs,
            target_device=_cpu(),
        )
        assert not [key for key in additional_inputs if key.startswith("_gptqmodel")], (
            f"private replay key leaked into {type(layer).__name__} kwargs"
        )
        recorded[index] = {
            "hidden_states": hidden_states.detach().clone(),
            "attention_mask": additional_inputs.get("attention_mask"),
            "position_embeddings": additional_inputs.get("position_embeddings"),
        }
        # `update_layer_replay_kwargs_from_output` mutates the output in place,
        # which is not allowed for inference-mode tensors.
        with torch.no_grad():
            output = layer(hidden_states, **additional_inputs)
        update_replay = getattr(qmodel, "update_layer_replay_kwargs_from_output", None)
        if callable(update_replay):
            update_replay(
                layer=layer,
                layer_output=output,
                layer_input_kwargs=captured["kwargs"],
                target_device=_cpu(),
            )
        hidden_states = output[0] if isinstance(output, tuple) else output
    return recorded


def _assert_masks_equal(index, expected, actual):
    if expected is None or actual is None:
        assert expected is None and actual is None, f"layer {index}: mask presence differs"
        return
    assert torch.equal(expected, actual), f"layer {index}: replay attention_mask differs from HF"


@pytest.mark.parametrize("batch_size", [1, 2], ids=["bs1", "bs2"])
def test_qwen3_vl_replay_applies_deepstack_residual(batch_size):
    model, example = _qwen3_vl()
    if batch_size == 2:
        example = _qwen3_vl_inputs(model.config, batch_size=2)
    layers = model.model.language_model.layers
    native = _record_native_layers(layers, lambda: model(**example, use_cache=False))

    qmodel = object.__new__(Qwen3_VLQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    replayed = _replay_layers(
        model,
        Qwen3_VLQModel,
        layers,
        lambda: model(**example, use_cache=False),
        example,
        start_capture=qmodel._start_deepstack_input_capture,
        stop_capture=qmodel._stop_deepstack_input_capture,
    )

    # The tiny model applies three DeepStack residuals, so layers 1..3 must be
    # affected; layer 0 must already match without the residual.
    assert len(model.config.vision_config.deepstack_visual_indexes) == 3
    for index in native:
        torch.testing.assert_close(
            replayed[index]["hidden_states"],
            native[index]["hidden_states"],
            rtol=0,
            atol=0,
            msg=lambda msg: f"layer {index}: {msg}",
        )


def test_gemma3_vl_replay_keeps_bidirectional_image_mask():
    model, example = _gemma3_vl()
    layers = model.model.language_model.layers
    native = _record_native_layers(layers, lambda: model(**example, use_cache=False))

    qmodel = object.__new__(Gemma3ForConditionalGenerationGPTQ)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    replayed = _replay_layers(
        model,
        Gemma3ForConditionalGenerationGPTQ,
        layers,
        lambda: model(**example, use_cache=False),
        example,
        start_capture=qmodel._start_replay_mask_capture,
        stop_capture=qmodel._stop_replay_mask_capture,
    )

    # The image block must be bidirectional: a plain causal mask forbids an
    # image token from attending to a later image token. Assert that forward
    # attention inside the block is allowed, so a replay that drops the block
    # overlay cannot pass by matching a causal mask.
    # Layer 1 is a full-attention layer, so the block overlay is not clipped by
    # the sliding window.
    block_mask = native[1]["attention_mask"]
    assert block_mask is not None
    image_positions = torch.nonzero(example["token_type_ids"][0]).flatten()
    assert len(image_positions) == 4
    assert bool(block_mask[0, 0, image_positions[0], image_positions[-1]])
    assert bool(block_mask[0, 0, image_positions[-1], image_positions[0]])

    for index in native:
        _assert_masks_equal(index, native[index]["attention_mask"], replayed[index]["attention_mask"])
        torch.testing.assert_close(
            replayed[index]["hidden_states"],
            native[index]["hidden_states"],
            rtol=0,
            atol=0,
            msg=lambda msg: f"layer {index}: {msg}",
        )


@pytest.mark.parametrize("kind", ["qwen2_vl", "qwen2_5_vl"])
def test_qwen_vl_batch_capture_builds_the_4d_mask(kind):
    model, qmodel_cls, example = _qwen2_vl(kind)
    language_model = model.model.language_model
    native = _record_native_layers(
        [language_model.layers[0]],
        lambda: model(**example, use_cache=False),
    )

    qmodel = object.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    captured = {}

    def capture_first(module, args, kwargs):
        captured["hidden_states"] = args[0].detach().clone()
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["position_embeddings"] = kwargs.get("position_embeddings")
        raise _StopCapture

    handle = language_model.layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)
    try:
        qmodel.run_input_capture(example, use_cache=False, data_device=_cpu())
    except _StopCapture:
        pass
    finally:
        handle.remove()

    torch.testing.assert_close(
        captured["hidden_states"],
        native[0]["hidden_states"],
        rtol=0,
        atol=0,
    )
    _assert_masks_equal(0, native[0]["attention_mask"], captured["attention_mask"])
    for expected, actual in zip(native[0]["position_embeddings"], captured["position_embeddings"]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_mllama_batch_capture_builds_the_4d_mask():
    model, example = _mllama()
    language_model = model.model.language_model
    native = _record_native_layers(
        [language_model.layers[0]],
        lambda: model(**example, use_cache=False),
    )

    qmodel = object.__new__(MLlamaQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    captured = {}

    def capture_first(module, args, kwargs):
        captured["hidden_states"] = args[0].detach().clone()
        captured["attention_mask"] = kwargs.get("attention_mask")
        raise _StopCapture

    handle = language_model.layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)
    try:
        qmodel.run_input_capture(example, use_cache=False, data_device=_cpu())
    except _StopCapture:
        pass
    finally:
        handle.remove()

    torch.testing.assert_close(
        captured["hidden_states"],
        native[0]["hidden_states"],
        rtol=0,
        atol=0,
    )
    _assert_masks_equal(0, native[0]["attention_mask"], captured["attention_mask"])


class _PaddingProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [[1, 2, 3]]}


def _batched_conversations():
    return [
        [{"role": "user", "content": [{"type": "text", "text": "first"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "second"}]}],
    ]


def test_llava_prepare_dataset_pads_batched_conversations():
    from gptqmodel.models.definitions.llava import LlavaQModel

    model = object.__new__(LlavaQModel)
    nn.Module.__init__(model)
    processor = _PaddingProcessor()
    model.processor = processor

    model.prepare_dataset(_batched_conversations(), batch_size=2)

    assert len(processor.calls) == 1
    assert processor.calls[0]["padding"] is True


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("gptqmodel.models.definitions.lfm2_vl", "LFM2VLQModel"),
        ("gptqmodel.models.definitions.smolvlm", "SmolVLMQModel"),
    ],
)
def test_vl_definitions_pad_batched_conversations(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    qmodel_cls = getattr(module, class_name)
    processor = _PaddingProcessor()

    qmodel_cls.prepare_inputs_for_conversations(processor, _batched_conversations())

    assert len(processor.calls) == 1
    assert processor.calls[0]["padding"] is True
