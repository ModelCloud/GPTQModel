# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import BatchEncoding

from gptqmodel.models import auto
from gptqmodel.models.definitions import gemma3 as gemma3_def
from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ, Gemma3QModel
from gptqmodel.models.writer import _convert_checkpoint_state_dict_keys
from gptqmodel.utils.model import (
    MODALITY,
    TensorSource,
    _cleanup_remapped_checkpoint,
    _remap_safetensors_checkpoint_keys,
    load_checkpoint_in_model_then_tie_weights,
)


class _FakeProcessor:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.template_calls.append((conversations, kwargs))
        return {"input_ids": [[index] for index, _ in enumerate(conversations)]}


def _vl_qmodel():
    model = object.__new__(Gemma3ForConditionalGenerationGPTQ)
    nn.Module.__init__(model)
    model.model_local_path = "google/gemma-3-4b-it"
    return model


def _conversation():
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "http://images.cocodataset.org/val2017/000000037777.jpg",
                },
                {"type": "text", "text": "generate a caption for this image"},
            ],
        },
        {"role": "assistant", "content": "a kitchen with wooden cabinets"},
    ]


def _fake_vl_model(offload_to_disk):
    language_model = nn.Module()
    language_model.embed_tokens = nn.Embedding(4, 4)
    language_model.rotary_emb = nn.Identity()
    language_model.norm = nn.LayerNorm(4)

    core_model = nn.Module()
    core_model.language_model = language_model
    core_model.vision_tower = nn.Identity()
    core_model.multi_modal_projector = nn.Linear(4, 4)

    wrapper = nn.Module()
    wrapper.model = core_model

    model = _vl_qmodel()
    model.model = wrapper
    model.quantize_config = SimpleNamespace(
        device="cuda:0",
        offload_to_disk=offload_to_disk,
        offload_to_disk_path="/tmp/gemma3-offload",
    )
    return model


def test_gemma3_vl_definition_declares_multimodal_contract():
    assert auto.MODEL_MAP["gemma3"] is Gemma3ForConditionalGenerationGPTQ
    assert Gemma3ForConditionalGenerationGPTQ.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert Gemma3ForConditionalGenerationGPTQ.require_load_processor is True
    assert Gemma3ForConditionalGenerationGPTQ.pre_lm_head_norm_module == "model.language_model.norm"


def test_gemma3_text_definition_stays_text_only():
    assert auto.MODEL_MAP["gemma3_text"] is Gemma3QModel
    assert Gemma3QModel.modality == [MODALITY.TEXT]
    assert Gemma3QModel.require_load_processor is False


def test_gemma3_vl_load_processor_uses_model_path(monkeypatch):
    calls = []
    processor = object()

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return processor

    monkeypatch.setattr(gemma3_def.AutoProcessor, "from_pretrained", fake_from_pretrained)
    model = _vl_qmodel()

    assert model.load_processor() is processor
    assert calls == [(("google/gemma-3-4b-it",), {"trust_remote_code": False})]


def test_gemma3_vl_prepare_dataset_renders_conversations_with_processor(monkeypatch):
    processor = _FakeProcessor()
    monkeypatch.setattr(Gemma3ForConditionalGenerationGPTQ, "load_processor", lambda self: processor)
    model = _vl_qmodel()

    prepared = model.prepare_dataset([_conversation(), _conversation()], batch_size=1)

    assert len(prepared) == 2
    assert [len(conversations) for conversations, _ in processor.template_calls] == [1, 1]
    for _, kwargs in processor.template_calls:
        assert kwargs == {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "tokenize": True,
        }


def test_gemma3_vl_prepare_dataset_honors_batch_size(monkeypatch):
    processor = _FakeProcessor()
    monkeypatch.setattr(Gemma3ForConditionalGenerationGPTQ, "load_processor", lambda self: processor)
    model = _vl_qmodel()

    prepared = model.prepare_dataset([_conversation(), _conversation()], batch_size=2)

    assert len(prepared) == 1
    assert [len(conversations) for conversations, _ in processor.template_calls] == [2]


def test_gemma3_vl_prepare_dataset_accepts_pretokenized_text_without_processor(monkeypatch):
    rows = [
        BatchEncoding({"input_ids": list(range(12)), "attention_mask": [0] + [1] * 11}),
        BatchEncoding({"input_ids": list(range(13)), "attention_mask": [0] + [1] * 12}),
    ]
    model = _vl_qmodel()

    def fail_load_processor(self):
        raise AssertionError("Pre-tokenized text calibration must not use the multimodal processor")

    monkeypatch.setattr(Gemma3ForConditionalGenerationGPTQ, "load_processor", fail_load_processor)

    prepared = model.prepare_dataset(rows, batch_size=1)

    assert len(prepared) == 2
    assert [row["input_ids"].shape for row in prepared] == [(1, 12), (1, 13)]
    assert prepared[0]["input_ids"].tolist() == [list(range(12))]
    assert prepared[1]["input_ids"].tolist() == [list(range(13))]


def test_gemma3_checkpoint_keys_are_saved_in_huggingface_layout():
    convert = Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_save

    assert convert("model.language_model.layers.0.self_attn.q_proj.qweight") == (
        "language_model.model.layers.0.self_attn.q_proj.qweight"
    )
    assert convert("model.language_model.norm.weight") == "language_model.model.norm.weight"
    assert Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_load(
        "language_model.model.norm.weight"
    ) == "model.language_model.norm.weight"
    assert convert("model.vision_tower.encoder.layers.0.self_attn.q_proj.weight") == (
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    )
    assert convert("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight") == (
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    )
    assert convert("model.multi_modal_projector.mm_input_projection_weight") == (
        "multi_modal_projector.mm_input_projection_weight"
    )


def test_checkpoint_key_conversion_updates_tensor_source_names():
    source = TensorSource(
        name="model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
        torch_dtype=torch.float16,
        shape=(2, 2),
        source=torch.zeros((2, 2), dtype=torch.float16),
    )

    converted = _convert_checkpoint_state_dict_keys(
        {source.name: source},
        Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_save,
    )

    assert list(converted) == ["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"]
    assert source.name == list(converted)[0]


def test_gemma3_checkpoint_load_remaps_huggingface_names_for_transformers5(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"vision_tower.vision_model.encoder.weight": torch.ones((2, 2))}, str(checkpoint))

    converted_path = _remap_safetensors_checkpoint_keys(
        checkpoint,
        Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_load,
    )
    try:
        assert converted_path is not None
        with safe_open(converted_path, framework="pt", device="cpu") as checkpoint_file:
            assert list(checkpoint_file.keys()) == ["model.vision_tower.encoder.weight"]
            assert torch.equal(checkpoint_file.get_tensor("model.vision_tower.encoder.weight"), torch.ones((2, 2)))
    finally:
        _cleanup_remapped_checkpoint(converted_path)


def test_gemma3_checkpoint_load_remaps_sharded_huggingface_names(tmp_path):
    shard = tmp_path / "part.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    save_file({"vision_tower.vision_model.encoder.weight": torch.ones((2, 2))}, str(shard))
    index.write_text(
        json.dumps({"weight_map": {"vision_tower.vision_model.encoder.weight": shard.name}}),
        encoding="utf-8",
    )

    converted_path = _remap_safetensors_checkpoint_keys(
        index,
        Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_load,
    )
    try:
        assert converted_path is not None
        converted_dir = os.path.dirname(converted_path)
        with open(converted_path, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        assert list(weight_map) == ["model.vision_tower.encoder.weight"]
        with safe_open(
            os.path.join(converted_dir, weight_map["model.vision_tower.encoder.weight"]),
            framework="pt",
            device="cpu",
        ) as checkpoint_file:
            assert list(checkpoint_file.keys()) == ["model.vision_tower.encoder.weight"]
            assert torch.equal(
                checkpoint_file.get_tensor("model.vision_tower.encoder.weight"),
                torch.ones((2, 2)),
            )
    finally:
        _cleanup_remapped_checkpoint(converted_path)


def test_gemma3_sharded_checkpoint_load_restores_vision_weights(tmp_path):
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.vision_tower = torch.nn.Module()
            self.model.vision_tower.encoder = torch.nn.Linear(2, 2, bias=False)
            self.model.vision_tower.encoder.weight.data.zero_()

        def tie_weights(self, **kwargs):
            pass

    shard = tmp_path / "part.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    save_key = Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_save(
        "model.vision_tower.encoder.weight"
    )
    save_file({save_key: torch.ones((2, 2))}, str(shard))
    index.write_text(json.dumps({"weight_map": {save_key: shard.name}}), encoding="utf-8")

    for checkpoint in (shard, index):
        model = Fixture()
        load_checkpoint_in_model_then_tie_weights(
            model,
            checkpoint=str(checkpoint),
            device_map={"": "cpu"},
            key_converter=Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_load,
        )
        assert torch.equal(model.model.vision_tower.encoder.weight, torch.ones((2, 2)))


def test_gemma3_tiny_checkpoint_roundtrip_restores_language_and_vision(tmp_path):
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 16,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
        },
        mm_tokens_per_image=4,
    )
    source = Gemma3ForConditionalGeneration(config)
    for parameter in source.parameters():
        parameter.data.fill_(1)

    state = {
        Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_save(name): tensor.clone()
        for name, tensor in source.state_dict().items()
    }

    single_file = tmp_path / "single"
    single_file.mkdir()
    save_file(state, str(single_file / "model.safetensors"))

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    keys = sorted(state)
    midpoint = len(keys) // 2
    shards = {
        "model-00001-of-00002.safetensors": keys[:midpoint],
        "model-00002-of-00002.safetensors": keys[midpoint:],
    }
    weight_map = {}
    for filename, shard_keys in shards.items():
        save_file({key: state[key] for key in shard_keys}, str(sharded / filename))
        weight_map.update(dict.fromkeys(shard_keys, filename))
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    for checkpoint in (
        single_file / "model.safetensors",
        sharded / "model.safetensors.index.json",
    ):
        target = Gemma3ForConditionalGeneration(config)
        for parameter in target.parameters():
            parameter.data.fill_(0)
        load_checkpoint_in_model_then_tie_weights(
            target,
            checkpoint=str(checkpoint),
            device_map={"": "cpu"},
            key_converter=Gemma3ForConditionalGenerationGPTQ.checkpoint_key_for_load,
        )
        assert target.model.language_model.norm.weight.sum().item() == 8.0
        assert target.model.vision_tower.encoder.layers[0].layer_norm1.weight.sum().item() == 8.0


def test_gemma3_save_config_excludes_unquantized_multimodal_modules():
    quantize_config = SimpleNamespace(dynamic={r"+:model.language_model.layers.*": {"bits": 4}})

    Gemma3ForConditionalGenerationGPTQ.prepare_quantize_config_for_save(None, quantize_config)

    assert quantize_config.dynamic == {
        r"+:model.language_model.layers.*": {"bits": 4},
        r"-:vision_tower\..*": {},
        r"-:multi_modal_projector\..*": {},
    }


def test_gemma3_save_copies_legacy_processor_assets(tmp_path):
    import os

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "preprocessor_config.json").write_text("{}")
    (source / "processor_config.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    (source / "tokenizer.model").write_bytes(b"unused")

    model = SimpleNamespace(model_local_path=str(source))
    Gemma3ForConditionalGenerationGPTQ.copy_checkpoint_auxiliary_files_for_save(model, str(target))

    assert (target / "preprocessor_config.json").read_text() == "{}"
    assert (target / "processor_config.json").read_text() == "{}"
    assert (target / "tokenizer_config.json").read_text() == "{}"
    assert not os.path.exists(target / "tokenizer.model")
    assert not os.path.exists(target / "chat_template.json")


def test_gemma3_vl_calibration_dataset_resolves_to_image_conversations(monkeypatch):
    from ovis import image_to_test_dataset

    captured = {}

    def fake_prepare_dataset(format_func, n_sample=20):
        captured["format_func"] = format_func
        captured["n_sample"] = n_sample
        return [format_func("http://images.cocodataset.org/val2017/000000037777.jpg", "a kitchen")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)
    model = object.__new__(Gemma3ForConditionalGenerationGPTQ)

    samples = image_to_test_dataset.get_calib_dataset(model)

    assert captured["format_func"] is image_to_test_dataset.format_qwen2_vl_dataset
    assert captured["n_sample"] == 20
    assert samples[0][0]["content"][0] == {
        "type": "image",
        "image": "http://images.cocodataset.org/val2017/000000037777.jpg",
    }
    assert samples[0][1] == {"role": "assistant", "content": "a kitchen"}


def test_gemma3_vl_hooks_materialize_multimodal_modules(monkeypatch):
    materialized = []
    monkeypatch.setattr(
        Gemma3ForConditionalGenerationGPTQ,
        "shell_module_materialize",
        lambda self, target_submodule, device: materialized.append((target_submodule, device)),
    )
    model = _fake_vl_model(offload_to_disk=False)
    core_model = model.model.model
    language_model = core_model.language_model

    model.pre_quantize_generate_hook_start()

    assert [module for module, _ in materialized] == [
        language_model.embed_tokens,
        language_model.rotary_emb,
        language_model.norm,
        core_model.vision_tower,
        core_model.multi_modal_projector,
    ]
    assert [device for _, device in materialized] == ["cuda:0"] * 5


def test_gemma3_vl_hooks_offload_multimodal_modules(monkeypatch):
    offloaded = []
    monkeypatch.setattr(
        gemma3_def,
        "offload_to_disk",
        lambda module, model, disk_path: offloaded.append((module, model, disk_path)),
    )
    model = _fake_vl_model(offload_to_disk=True)
    core_model = model.model.model
    language_model = core_model.language_model

    model.pre_quantize_generate_hook_end()

    assert offloaded == [
        (language_model.embed_tokens, language_model, "/tmp/gemma3-offload"),
        (language_model.rotary_emb, language_model, "/tmp/gemma3-offload"),
        (language_model.norm, language_model, "/tmp/gemma3-offload"),
        (core_model.vision_tower, core_model, "/tmp/gemma3-offload"),
        (core_model.multi_modal_projector, core_model, "/tmp/gemma3-offload"),
    ]


def test_gemma3_vl_hooks_release_multimodal_modules_to_cpu(monkeypatch):
    moved = []
    monkeypatch.setattr(
        gemma3_def,
        "move_to",
        lambda module, device: moved.append((module, device)) or module,
    )
    model = _fake_vl_model(offload_to_disk=False)
    core_model = model.model.model
    language_model = core_model.language_model

    model.pre_quantize_generate_hook_end()

    assert [module for module, _ in moved] == [
        language_model.embed_tokens,
        language_model.rotary_emb,
        language_model.norm,
        core_model.vision_tower,
        core_model.multi_modal_projector,
    ]
    assert [device for _, device in moved] == [gemma3_def.CPU] * 5
