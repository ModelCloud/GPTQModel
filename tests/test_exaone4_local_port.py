# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native EXAONE 4/4.5 identity, loading and checkpoint transitions."""

import json
from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoConfig,
    AutoProcessor,
    Exaone4_5_Config,
    Exaone4_5_ForConditionalGeneration,
    Exaone4_5_Processor,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
)

from transformers.models.exaone4.configuration_exaone4 import Exaone4Config as NativeExaone4Config
from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM as NativeExaone4ForCausalLM

from gptqmodel.models.auto import MODEL_MAP, check_and_get_model_definition
from gptqmodel.models.ports.native.exaone4.configuration_exaone4 import Exaone4Config
from gptqmodel.models.ports.native.exaone4.modeling_exaone4 import Exaone4ForCausalLM
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_text_config(**overrides):
    values = {
        "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
    }
    values.update(overrides)
    return Exaone4Config(**values)


def tiny_vision_config(text_type="exaone4_5_text"):
    text = tiny_text_config().to_dict()
    text["model_type"] = text_type
    return Exaone4_5_Config(
        text_config=text,
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "num_key_value_heads": 2, "out_hidden_size": 32, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2, "window_size": 8,
            "fullatt_block_indexes": [1],
        },
        image_token_id=67, video_token_id=68,
    )


def tiny_model(model_type, **text_overrides):
    config = tiny_vision_config() if model_type == "exaone4_5" else tiny_text_config(**text_overrides)
    if model_type == "exaone4_5":
        for name, value in text_overrides.items():
            setattr(config.text_config, name, value)
    return MODEL_MAP[model_type].loader.from_config(config, attn_implementation="eager").eval()


def tiny_tokenizer():
    vocab = {f"t{i}": i for i in range(128)}
    for index, name in {0: "[PAD]", 1: "[BOS]", 2: "[EOS]", 3: "[UNK]", 67: "<|image_pad|>",
                        68: "<|video_pad|>"}.items():
        del vocab[f"t{index}"]
        vocab[name] = index
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>"],
    )


def tiny_processor():
    settings = {"patch_size": 2, "temporal_patch_size": 2, "merge_size": 2,
                    "size": {"shortest_edge": 16, "longest_edge": 64}}
    return Exaone4_5_Processor(
        tokenizer=tiny_tokenizer(), image_processor=Qwen2VLImageProcessor(**settings),
        video_processor=Qwen2VLVideoProcessor(**settings, do_sample_frames=False),
    )


def hostile_config(path, model_type):
    payload = tiny_text_config().to_dict()
    payload["model_type"] = model_type
    payload["architectures"] = ["Exaone4ForCausalLM"]
    marker = path / "EXECUTED"
    (path / "remote.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('checkpoint code executed')\n"
    )
    payload["auto_map"] = {
        "AutoConfig": "remote.Config", "AutoModelForCausalLM": "remote.Model",
    }
    (path / "config.json").write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize(
    "model_type,trust,bypass", [("exaone4", False, False), ("exaone4_5_text", True, True), ("exaone4_5_text", False, True)]
)
def test_text_alias_uses_native_config_and_model_without_checkpoint_code(tmp_path, monkeypatch, model_type, trust, bypass):
    marker = hostile_config(tmp_path, model_type)
    # This is the pre-existing security bypass, not a model behavior switch.
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is Exaone4Config
    assert config.model_type == "exaone4"
    definition = check_and_get_model_definition(str(tmp_path), trust_remote_code=trust)
    assert definition is MODEL_MAP[model_type]
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    model = definition.loader.from_config(config, trust_remote_code=trust, attn_implementation="eager")
    assert type(model) is Exaone4ForCausalLM
    assert model(torch.tensor([[1, 5, 9]]), use_cache=False).logits.shape == (1, 3, 128)
    assert not marker.exists()


def test_native_multimodal_config_already_normalizes_legacy_text_identity():
    config = tiny_vision_config()
    # EXAONE 4.5 is still a Transformers-native definition; its nested text config is Transformers' own.
    assert type(config.text_config) is NativeExaone4Config
    assert config.text_config.model_type == "exaone4"
    assert config.to_dict()["text_config"]["model_type"] == "exaone4"


@pytest.mark.parametrize("layer_types", [["sliding_attention", "full_attention"], ["full_attention", "full_attention"]])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@torch.inference_mode()
def test_vendored_port_matches_transformers_oracle(layer_types, implementation):
    torch.manual_seed(7)
    config = tiny_text_config(layer_types=layer_types)
    model = MODEL_MAP["exaone4"].loader.from_config(deepcopy(config), attn_implementation=implementation).eval()
    oracle = NativeExaone4ForCausalLM._from_config(
        NativeExaone4Config(**config.to_dict()), attn_implementation=implementation
    ).eval()
    assert set(model.state_dict()) == set(oracle.state_dict())
    oracle.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 7, 11, 13, 19, 23, 29]])
    assert torch.equal(model(ids, use_cache=False).logits, oracle(ids, use_cache=False).logits)
    assert torch.equal(
        model.generate(ids[:, :3], max_new_tokens=4, do_sample=False, eos_token_id=None),
        oracle.generate(ids[:, :3], max_new_tokens=4, do_sample=False, eos_token_id=None),
    )


@pytest.mark.parametrize("model_type", ["exaone4", "exaone4_5_text", "exaone4_5"])
@pytest.mark.parametrize("kv_heads,layer_types", [(2, ["sliding_attention", "full_attention"]),
                                                (4, ["full_attention", "full_attention"])])
@torch.inference_mode()
def test_native_cache_generation_and_safetensors_roundtrip(tmp_path, model_type, kv_heads, layer_types):
    model = tiny_model(model_type, num_key_value_heads=kv_heads, layer_types=layer_types)
    expected_type = Exaone4_5_ForConditionalGeneration if model_type == "exaone4_5" else Exaone4ForCausalLM
    assert type(model) is expected_type
    ids = torch.tensor([[1, 7, 11, 13, 19, 23, 29]])
    expected = model(ids, use_cache=False).logits
    first = model(ids[:, :5], use_cache=True)
    assert first.past_key_values.get_seq_length() == 5
    continued = model(ids[:, 5:], past_key_values=first.past_key_values, use_cache=True)
    assert continued.past_key_values.get_seq_length() == 7
    torch.testing.assert_close(continued.logits, expected[:, 5:], atol=1e-6, rtol=1e-5)
    generated = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
    no_cache = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
    assert torch.equal(generated, no_cache)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("*.bin"))
    serialized = json.loads((tmp_path / "config.json").read_text())
    text = serialized.get("text_config", serialized)
    assert text["model_type"] == "exaone4"
    if model_type == "exaone4_5_text":
        serialized["model_type"] = model_type
        (tmp_path / "config.json").write_text(json.dumps(serialized))
    config = load_model_config(tmp_path)
    loaded = MODEL_MAP[model_type].loader.from_pretrained(tmp_path, config=config, attn_implementation="eager").eval()
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, atol=0, rtol=0)
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    second = tmp_path / "second"
    loaded.save_pretrained(second)
    # The legacy input identity is migrated, not perpetuated on the next save.
    assert json.loads((second / "config.json").read_text()).get("model_type") == model.config.model_type
    loaded_again = MODEL_MAP[model_type].loader.from_pretrained(second, attn_implementation="eager").eval()
    torch.testing.assert_close(loaded_again(ids, use_cache=False).logits, expected, atol=0, rtol=0)


def test_selected_subfolder_and_invalid_config_retry(tmp_path):
    from gptqmodel.models.ports.exaone4 import Exaone4TextConfig

    subfolder = tmp_path / "text"
    subfolder.mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    marker = hostile_config(subfolder, "exaone4_5_text")
    loaded = load_model_config(tmp_path, subfolder="text")
    assert type(loaded) is Exaone4Config
    assert loaded.model_type == "exaone4"
    with pytest.raises(ValueError, match="requires exaone4 or exaone4_5_text"):
        Exaone4TextConfig.from_pretrained(tmp_path)
    assert type(Exaone4TextConfig.from_pretrained(tmp_path, subfolder="text")) is Exaone4Config
    assert not marker.exists()
    with pytest.raises(ValueError):
        AutoConfig.from_pretrained(subfolder, trust_remote_code=False)


@pytest.mark.parametrize("modality", ["image", "video"])
@torch.inference_mode()
def test_native_media_processor_save_reload_and_failure_retry(tmp_path, modality):
    import numpy as np
    from PIL import Image

    processor = tiny_processor()
    processor.save_pretrained(tmp_path)
    tiny_vision_config().save_pretrained(tmp_path)
    loaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    assert type(loaded) is Exaone4_5_Processor
    model = tiny_model("exaone4_5")
    if modality == "image":
        values = {"text": ["t5 <|image_pad|> t9"], "images": [Image.new("RGB", (4, 4), (30, 90, 150))]}
        key = "pixel_values"
    else:
        video = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        values = {"text": ["t5 <|video_pad|> t9"], "videos": [video]}
        key = "pixel_values_videos"
    batch = loaded(**values, return_tensors="pt")
    expected_batch = processor(**values, return_tensors="pt")
    for name in batch:
        torch.testing.assert_close(batch[name], expected_batch[name], atol=0, rtol=0)
    assert batch[key].shape == (4, 24)
    expected = model(**batch, use_cache=False).logits
    assert torch.isfinite(expected).all()
    changed = deepcopy(batch)
    changed[key] = changed[key] + torch.linspace(-0.8, 1.1, changed[key].numel()).reshape_as(changed[key])
    assert not torch.allclose(model(**changed, use_cache=False).logits, expected, atol=1e-6, rtol=1e-5)
    prefill = model(**batch, use_cache=True)
    next_token = torch.tensor([[17]])
    cached = model(next_token, past_key_values=prefill.past_key_values, use_cache=True).logits
    full = dict(batch, input_ids=torch.cat((batch["input_ids"], next_token), dim=1),
                attention_mask=torch.ones(1, batch["input_ids"].shape[1] + 1, dtype=torch.long))
    torch.testing.assert_close(cached, model(**full, use_cache=False).logits[:, -1:], atol=1e-6, rtol=1e-5)
    broken = deepcopy(batch)
    broken["input_ids"] = torch.full_like(batch["input_ids"], 5)
    with pytest.raises((ValueError, RuntimeError), match="features|tokens"):
        model(**broken, use_cache=False)
    torch.testing.assert_close(model(**batch, use_cache=False).logits, expected, atol=0, rtol=0)
    model.save_pretrained(tmp_path)
    reloaded = MODEL_MAP["exaone4_5"].loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
    # Weights and buffers reload bit-exactly; reloaded storage may take a different kernel path.
    torch.testing.assert_close(reloaded(**batch, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "model_type,trust,bypass", [("exaone4", False, False), ("exaone4_5_text", True, True), ("exaone4_5", True, False)]
)
@torch.inference_mode()
def test_public_load_ignores_hostile_model_config(tmp_path, monkeypatch, model_type, trust, bypass):
    from gptqmodel import GPTQModel

    model = tiny_model(model_type)
    ids = torch.tensor([[1, 5, 9]])
    expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path)
    if model_type == "exaone4_5":
        tiny_processor().save_pretrained(tmp_path)
    else:
        tiny_tokenizer().save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["model_type"] = model_type
    marker = tmp_path / "EXECUTED"
    (tmp_path / "remote.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('checkpoint code executed')\n"
    )
    payload["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModelForCausalLM", "AutoModelForImageTextToText"), "remote.Model")
    if model_type == "exaone4_5":
        payload["text_config"]["model_type"] = "exaone4_5_text"
        payload["text_config"]["auto_map"] = {"AutoConfig": "remote.Config", "AutoModel": "remote.Model"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32,
                           attn_implementation="eager")
    assert type(loaded) is MODEL_MAP[model_type]
    assert type(loaded.model) is type(model)
    torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    assert not marker.exists()
