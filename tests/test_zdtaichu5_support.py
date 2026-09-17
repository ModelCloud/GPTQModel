# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.qwen3_5_text import Qwen3_5TextQModel
from gptqmodel.models.definitions.zdtaichu5 import ZDTaichu5QModel
from gptqmodel.utils.model import MODALITY


def _tiny_qwen35_config():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
        dtype="float32",
    )
    config._attn_implementation = "eager"
    return config


class _TinyZDTaichuWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = Qwen3_5ForCausalLM(_tiny_qwen35_config())
        self.vision_model = nn.Linear(3, 8, bias=False)
        self.mlp1 = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 32))

    def forward(self, **kwargs):
        return self.language_model(**kwargs)


def _definition_instance(model=None):
    instance = object.__new__(ZDTaichu5QModel)
    nn.Module.__init__(instance)
    instance.model = model or _TinyZDTaichuWrapper()
    return instance


def test_zdtaichu5_model_type_registry_and_trust(monkeypatch):
    config = SimpleNamespace(model_type="zdtaichu5_0")
    calls = []

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: calls.append((args, kwargs)) or config,
    )

    assert auto.check_and_get_model_definition("/tmp/zdtaichu5", trust_remote_code=False) is ZDTaichu5QModel
    assert calls[-1][1]["trust_remote_code"] is True
    assert auto.MODEL_MAP["zdtaichu5_0"] is ZDTaichu5QModel
    assert ZDTaichu5QModel.loader is AutoModelForCausalLM
    assert ZDTaichu5QModel.require_trust_remote_code is True
    assert ZDTaichu5QModel.require_load_processor is True


def test_zdtaichu5_decoder_tree_matches_tiny_qwen35_hybrid_model():
    wrapper = _TinyZDTaichuWrapper()
    config = wrapper.language_model.config

    assert ZDTaichu5QModel.module_tree[1:] == Qwen3_5TextQModel.module_tree
    assert ZDTaichu5QModel.extract_layers_node() == ["language_model.model.layers"]
    assert ZDTaichu5QModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT, MODALITY.VIDEO_TO_TEXT]
    assert ZDTaichu5QModel.out_of_model_tensors == {"prefixes": ["mtp"]}

    for layer in wrapper.language_model.model.layers:
        if getattr(layer, "linear_attn", None) is not None:
            assert all(hasattr(layer.linear_attn, name) for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))
        else:
            assert all(hasattr(layer.self_attn, name) for name in ("q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj"))

    modules = ZDTaichu5QModel.full_layer_modules(config)
    flattened = {name for block in modules for name in block}
    assert "linear_attn.in_proj_qkv" in flattened
    assert "linear_attn.in_proj_z" in flattened
    assert "linear_attn.norm:!" in flattened
    assert "self_attn.q_norm:!" in flattened
    assert "self_attn.k_norm:!" in flattened


def test_zdtaichu5_base_modules_cover_wrapper_roots():
    base_modules = set(ZDTaichu5QModel.get_base_modules(_TinyZDTaichuWrapper()))

    assert {
        "language_model.lm_head",
        "language_model.model.embed_tokens",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "vision_model",
        "mlp1",
    }.issubset(base_modules)
    assert "language_model.model.layers" not in base_modules


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def from_messages(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 4, 4),
            "pixel_values_videos": torch.randn(1, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 2]]),
        }


def test_zdtaichu5_processor_routes_conversations_and_delegates_text(monkeypatch):
    processor = _FakeProcessor()
    monkeypatch.setattr(ZDTaichu5QModel, "load_processor", lambda self: processor)
    delegated = []

    def fake_base_prepare(self, dataset, **kwargs):
        delegated.append((dataset, kwargs))
        return ["base"]

    monkeypatch.setattr(BaseQModel, "prepare_dataset", fake_base_prepare)
    instance = _definition_instance()
    conversation = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]

    prepared = instance.prepare_dataset(conversation, batch_size=4)
    assert len(prepared) == 1
    assert processor.calls[0][0] == conversation[0]
    assert processor.calls[0][1] == {"return_tensors": "pt"}

    image_video = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "video", "video": "video.mp4"},
                {"type": "text", "text": "compare"},
            ],
        }
    ]]
    multimodal = instance.prepare_dataset((sample for sample in image_video))[0]
    assert processor.calls[1][0] == image_video[0]
    assert {"pixel_values", "pixel_values_videos", "image_grid_thw",
            "video_grid_thw", "mm_token_type_ids"} <= multimodal.keys()
    assert torch.equal(multimodal["mm_token_type_ids"], torch.tensor([[0, 1, 2]]))

    text = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}]
    assert instance.prepare_dataset(text, batch_size=2, calibration_data_min_length=7) == ["base"]
    assert delegated[0][0] == text
    assert delegated[0][1] == {"batch_size": 2, "calibration_data_min_length": 7}


def test_zdtaichu5_hooks_propagate_lazy_paths_and_keep_pixel_dtype(monkeypatch):
    model = _TinyZDTaichuWrapper()
    model.vision_model = model.vision_model.to(dtype=torch.bfloat16)
    instance = _definition_instance(model)
    instance.quantize_config = SimpleNamespace(device=torch.device("cpu"), offload_to_disk=False)
    paths = []

    def materialize(module, device, module_path=None, **kwargs):
        del device, kwargs
        paths.append(module_path)
        return module

    instance.shell_module_materialize = materialize
    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert paths == [
        "language_model.model.embed_tokens",
        "language_model.model.rotary_emb",
        "vision_model",
        "mlp1",
    ]
    example = {
        "pixel_values": torch.randn(1, 3, 4, 4, dtype=torch.float32),
        "pixel_values_videos": torch.randn(1, 3, 4, 4, dtype=torch.float32),
        "input_ids": torch.ones((1, 3), dtype=torch.long),
    }
    moved = instance.move_input_capture_example(example, torch.device("cpu"))
    assert moved["pixel_values"].dtype == next(model.vision_model.parameters()).dtype
    assert moved["pixel_values_videos"].dtype == next(model.vision_model.parameters()).dtype


def test_zdtaichu5_offload_hook_covers_vision_and_projector(monkeypatch):
    from gptqmodel.models.definitions import zdtaichu5

    model = _TinyZDTaichuWrapper()
    instance = _definition_instance(model)
    instance.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        offload_to_disk=True,
        offload_to_disk_path="/tmp/zdtaichu5-offload",
    )
    calls = []
    monkeypatch.setattr(
        zdtaichu5,
        "offload_to_disk",
        lambda **kwargs: calls.append(kwargs),
    )
    instance.shell_module_materialize = lambda module, device, **kwargs: module
    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert [call["module"] for call in calls] == [
        model.language_model.model.embed_tokens,
        model.language_model.model.rotary_emb,
        model.vision_model,
        model.mlp1,
    ]


@pytest.mark.parametrize("padding", [torch.ones((1, 5), dtype=torch.bool), torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)])
def test_zdtaichu5_hybrid_replay_masks_match_dense_layers(padding):
    wrapper = _TinyZDTaichuWrapper().eval()
    instance = _definition_instance(wrapper)
    layers = wrapper.language_model.model.layers
    captured = {}
    expected_outputs = []

    def capture_first(module, args, kwargs):
        captured["hidden"] = args[0].detach().clone()
        captured["kwargs"] = instance.capture_first_layer_input_kwargs(
            args, kwargs, torch.device("cpu"),
            {key: value for key, value in kwargs.items() if key != "attention_mask"},
        )

    def capture_output(module, args, output):
        expected_outputs.append(output.detach().clone())

    handles = [layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)]
    handles.extend(layer.register_forward_hook(capture_output) for layer in layers)
    # Keep text positions separate from the three visual RoPE channels.
    positions = torch.arange(5).view(1, -1)
    position_ids = torch.stack([positions, positions.flip(-1), positions * 2, positions * 0])
    try:
        with torch.inference_mode():
            native = instance.run_input_capture(
                {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                 "attention_mask": padding, "position_ids": position_ids},
                use_cache=False, data_device=torch.device("cpu"),
            )
    finally:
        for handle in handles:
            handle.remove()
    assert instance.__dict__["_zdtaichu5_capture_padding_mask"] is None

    hidden = captured["hidden"]
    for layer, expected in zip(layers, expected_outputs):
        replay_kwargs = dict(captured["kwargs"])
        # The shared executor drops the first-layer mask for unbatched models.
        replay_kwargs["attention_mask"] = None
        replay_kwargs["use_cache"] = False
        actual_kwargs = instance.prepare_layer_replay_kwargs(
            layer,
            [hidden],
            replay_kwargs,
            torch.device("cpu"),
        )
        assert "_zdtaichu5_padding_mask" not in actual_kwargs
        with torch.inference_mode():
            hidden = layer(hidden, **actual_kwargs)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-5)
    with torch.inference_mode():
        logits = wrapper.language_model.lm_head(wrapper.language_model.model.norm(hidden))
    torch.testing.assert_close(logits, native.logits, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mask", [torch.ones((1, 4), dtype=torch.bool), torch.tensor([[0, 1, 1, 1]], dtype=torch.bool)])
def test_zdtaichu5_capture_preserves_raw_padding_mask(mask):
    instance = _definition_instance()
    instance.__dict__["_zdtaichu5_capture_padding_mask"] = mask
    hidden = torch.zeros((1, 4, 32))
    captured = instance.capture_first_layer_input_kwargs(
        args=(hidden,),
        kwargs={"attention_mask": torch.ones((1, 1, 4, 4))},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )
    assert torch.equal(captured["_zdtaichu5_padding_mask"], mask)
