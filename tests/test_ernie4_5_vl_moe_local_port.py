# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native ERNIE VL, including the original flat Baidu checkpoint schema."""

import json
import re
import shutil
from pathlib import Path

import pytest
import torch
from transformers import Ernie4_5_VLMoeConfig, Ernie4_5_VLMoeForConditionalGeneration

from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel
from gptqmodel.utils.hf import load_model_config
from gptqmodel.utils.security import BYPASS_ENV


ALIASES = ("ernie4_5_moe_vl", "ernie4_5_vl_moe")
REPO = "baidu/ERNIE-4.5-VL-28B-A3B-PT"
REVISION = "e3815e65c607ea211bfe21b46ab0cd264b76731c"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**overrides):
    values = {
        "text_config": {
            "vocab_size": 48, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "moe_num_experts": 2,
            "moe_k": 1, "moe_intermediate_size": [8, 4], "moe_num_shared_experts": 1,
            "max_position_embeddings": 64, "pad_token_id": 0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0, "mrope_section": [1, 1, 2]},
        },
        "vision_config": {
            "depth": 1, "hidden_size": 16, "intermediate_size": 24, "num_heads": 2, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_merge_size": 2,
        },
        "image_token_id": 40, "image_start_token_id": 41, "image_end_token_id": 42,
        "video_token_id": 43, "video_start_token_id": 44, "video_end_token_id": 45,
    }
    values.update(overrides)
    return Ernie4_5_VLMoeConfig(**values)


def legacy_config(config, alias):
    text = config.text_config.to_dict()
    vision = config.vision_config.to_dict()
    text.update(
        model_type=alias, architectures=["Ernie4_5_VLMoeForConditionalGeneration"],
        moe_num_experts=[text["moe_num_experts"]] * 2,
        moe_layer_start_index=[1, 1], moe_layer_end_index=[config.text_config.num_hidden_layers] * 2,
        moe_layer_interval=1, pixel_hidden_size=vision["hidden_size"],
        spatial_conv_size=vision["spatial_merge_size"], temporal_conv_size=vision["temporal_merge_size"],
        im_patch_id=config.image_token_id, vision_config=vision,
    )
    for name in ("image_start_token_id", "image_end_token_id", "video_token_id", "video_start_token_id", "video_end_token_id"):
        text[name] = getattr(config, name)
    return text


def test_before_model_load_never_imports_checkpoint(monkeypatch, tmp_path):
    calls = []

    def checkpoint_import(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("checkpoint code must not be queried")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", checkpoint_import)
    wrapper = Ernie4_5_VLMoeQModel.__new__(Ernie4_5_VLMoeQModel)
    wrapper.before_model_load(str(tmp_path), False)
    assert calls == [], "the native model must not probe or patch checkpoint decoder classes"


@pytest.mark.parametrize("alias", ALIASES)
def test_flat_config_preserves_architecture_without_remote_code(tmp_path, alias):
    (tmp_path / "config.json").write_text(json.dumps(legacy_config(tiny_config(), alias)))
    config = load_model_config(tmp_path, trust_remote_code=False)
    assert type(config) is Ernie4_5_VLMoeConfig
    assert config.text_config.hidden_size == 16
    assert config.text_config.moe_num_experts == 2
    assert config.text_config.mlp_layer_types == ["dense", "sparse"]
    assert config.vision_config.intermediate_size == 24
    assert config.image_token_id == 40
    with torch.device("meta"):
        model = Ernie4_5_VLMoeQModel.loader.from_config(config)
    assert type(model) is Ernie4_5_VLMoeForConditionalGeneration
    assert model.model.language_model.layers[1].mlp.text_moe.experts.gate_up_proj.shape == (2, 16, 16)


def poison(path, alias):
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    values = legacy_config(tiny_config(), alias)
    values["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModelForImageTextToText", "AutoProcessor"), "hostile.Model")
    (path / "config.json").write_text(json.dumps(values))
    return marker


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_native_loading_ignores_hostile_automap(tmp_path, monkeypatch, alias, trust, bypass):
    model = Ernie4_5_VLMoeQModel.loader.from_config(tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    marker = poison(tmp_path, alias)
    monkeypatch.setenv(BYPASS_ENV, "1") if bypass else monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(str(tmp_path), trust_remote_code=trust)
    loaded, info = Ernie4_5_VLMoeQModel.loader.from_pretrained(
        str(tmp_path), config=config, trust_remote_code=trust, code_revision="untrusted", output_loading_info=True,
    )
    assert type(loaded) is Ernie4_5_VLMoeForConditionalGeneration
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert not marker.exists()
    inputs = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(inputs).logits, model(inputs).logits, rtol=0, atol=0)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_native_forward_cache_generation_and_reload(tmp_path, attention):
    torch.manual_seed(19)
    config = tiny_config()
    config._attn_implementation = attention
    model = Ernie4_5_VLMoeQModel.loader.from_config(config).eval()
    ids = torch.tensor([[3, 5, 8, 12, 7]])
    with torch.no_grad():
        complete = model(ids, use_cache=False).logits
        prefix = model(ids[:, :3], use_cache=True)
        suffix = model(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
        torch.testing.assert_close(suffix.logits, complete[:, 3:], atol=2e-6, rtol=2e-5)
        assert suffix.past_key_values.get_seq_length() == ids.shape[1]
        tokens = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False)
        assert tokens.shape == (1, 6)
        assert torch.isfinite(complete).all()
        model.save_pretrained(tmp_path, safe_serialization=True)
        loaded = Ernie4_5_VLMoeQModel.loader.from_pretrained(str(tmp_path), attn_implementation=attention)
        torch.testing.assert_close(loaded(ids).logits, complete, atol=0, rtol=0)
        assert torch.equal(loaded.generate(ids[:, :3], max_new_tokens=3, do_sample=False), tokens)


@pytest.fixture(scope="module")
def public_bundle(tmp_path_factory):
    from huggingface_hub import hf_hub_download

    path = tmp_path_factory.mktemp("ernie-public-metadata")
    for name in ("config.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json",
                 "special_tokens_map.json", "chat_template.json", "preprocessor_config.json",
                 "model.safetensors.index.json"):
        shutil.copyfile(hf_hub_download(REPO, name, revision=REVISION), path / name)
    return path


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_tokenizer_processor_with_hostile_metadata(public_bundle, tmp_path, monkeypatch, trust, bypass):
    import numpy as np
    from PIL import Image

    from gptqmodel.utils.hf import load_hf_tokenizer

    shutil.copytree(public_bundle, tmp_path, dirs_exist_ok=True)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        values = json.loads((tmp_path / name).read_text())
        values["auto_map"] = dict.fromkeys(("AutoConfig", "AutoProcessor", "AutoImageProcessor"), "hostile.Model")
        values["auto_map"]["AutoTokenizer"] = ["hostile.Tokenizer", None]
        (tmp_path / name).write_text(json.dumps(values))
    monkeypatch.setenv(BYPASS_ENV, "1") if bypass else monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(str(tmp_path), trust_remote_code=trust)
    tokenizer = load_hf_tokenizer(str(tmp_path), model_config=config, trust_remote_code=trust)
    processor = Ernie4_5_VLMoeQModel.loader.load_processor(str(tmp_path), config=config, tokenizer=tokenizer,
                                                       trust_remote_code=trust)
    assert processor.tokenizer is tokenizer
    assert tokenizer.encode("Hello world!  中文\nX") == [16276, 3135, 94018, 269, 11422, 23, 94026]
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                           {"type": "text", "text": "Describe."}]}], tokenize=False)
    assert "Picture 1:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Describe." in prompt
    inputs = processor(text=prompt, images=Image.new("RGB", (56, 56)), return_tensors="pt")
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert (inputs["input_ids"] == config.image_token_id).sum() == 4
    assert inputs["moe_mm_token_type_ids"].max() == 1
    assert Path(processor.video_processor.font).is_file()
    video_prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": "Describe."}]}], tokenize=False)
    video_inputs = processor(text=video_prompt, videos=np.zeros((2, 56, 56, 3), dtype=np.uint8),
                             size={"shortest_edge": 3136, "longest_edge": 3136},
                             do_sample_frames=False, return_tensors="pt")
    assert video_inputs["video_grid_thw"].tolist() == [[2, 4, 4]]
    assert (video_inputs["input_ids"] == config.video_token_id).sum() == 4
    assert video_inputs["moe_mm_token_type_ids"].max() == 2
    saved = tmp_path / "saved-processor"
    processor.save_pretrained(saved)
    restored = Ernie4_5_VLMoeQModel.loader.load_processor(str(saved), config=config)
    assert Path(restored.video_processor.font).is_file()
    assert restored.tokenizer.image_token_id == 100295
    assert restored.tokenizer.video_token_id == 103367
    restored_inputs = restored(text=prompt, images=Image.new("RGB", (56, 56)), return_tensors="pt")
    torch.testing.assert_close(restored_inputs["pixel_values"], inputs["pixel_values"], atol=0, rtol=0)
    assert torch.equal(restored_inputs["input_ids"], inputs["input_ids"])
    assert not marker.exists()


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_gptqmodel_dispatch_and_processor_reuse(public_bundle, tmp_path, monkeypatch, alias, trust, bypass):
    from gptqmodel import GPTQModel

    config = tiny_config()
    original = Ernie4_5_VLMoeQModel.loader.from_config(config).eval()
    original.save_pretrained(tmp_path, safe_serialization=True)
    for source in public_bundle.iterdir():
        if source.name not in {"config.json", "model.safetensors.index.json"}:
            shutil.copyfile(source, tmp_path / source.name)
    marker = poison(tmp_path, alias)
    monkeypatch.setenv(BYPASS_ENV, "1") if bypass else monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32,
                           attn_implementation="eager")
    assert type(loaded) is Ernie4_5_VLMoeQModel
    assert type(loaded.model) is Ernie4_5_VLMoeForConditionalGeneration
    assert loaded.trust_remote_code is False
    assert loaded.load_processor() is loaded.processor
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    assert not marker.exists()
    with torch.no_grad():
        ids = torch.tensor([[3, 5, 7]])
        torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("bad", [{"moe_num_experts": [2, 3]}, {"moe_layer_start_index": [0, 1]},
                                 {"moe_layer_interval": 0}])
def test_invalid_legacy_layout_rejected_then_retried(tmp_path, bad):
    values = legacy_config(tiny_config(), ALIASES[0])
    values.update(bad)
    (tmp_path / "config.json").write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_model_config(str(tmp_path), trust_remote_code=True)
    (tmp_path / "config.json").write_text(json.dumps(legacy_config(tiny_config(), ALIASES[0])))
    assert load_model_config(str(tmp_path), trust_remote_code=True).text_config.moe_num_experts == 2


def test_pinned_checkpoint_key_inventory_and_real_meta_layout(public_bundle):
    from transformers.core_model_loading import revert_weight_conversion

    config = load_model_config(str(public_bundle), trust_remote_code=False)
    assert config.text_config.num_hidden_layers == 28
    assert config.text_config.hidden_size == 2560
    assert config.text_config.moe_intermediate_size == [1536, 512]
    assert config.vision_config.hidden_size == 1280
    assert config.vision_config.intermediate_size == 5120
    assert config.text_config.rope_parameters["mrope_section"] == [22, 22, 20]
    with torch.device("meta"):
        full = Ernie4_5_VLMoeQModel.loader.from_config(config)
        tiny = Ernie4_5_VLMoeQModel.loader.from_config(tiny_config())
    assert len(full.model.language_model.layers) == 28
    assert full.lm_head.weight is full.model.language_model.embed_tokens.weight
    source = revert_weight_conversion(tiny, tiny.state_dict())
    checkpoint = json.loads((public_bundle / "model.safetensors.index.json").read_text())["weight_map"]
    def normalize(key):
        return re.sub(r"(?<=experts\.)\d+", "#", key)
    for index in (0, 1):
        prefix = f"model.layers.{index}."
        actual = {normalize(key) for key in source if key.startswith(prefix)}
        expected = {normalize(key) for key in checkpoint if key.startswith(prefix)}
        assert actual == expected
    assert "model.layers.1.mlp.gate.weight_1" in source
    assert "model.layers.1.mlp.experts.3.gate_proj.weight" in source


@pytest.mark.parametrize("kind", ["image", "video"])
def test_native_real_media_forward_and_cache(kind):
    torch.manual_seed(79)
    model = Ernie4_5_VLMoeQModel.loader.from_config(tiny_config()).eval()
    if kind == "image":
        ids = torch.tensor([[41, 40, 42, 3]])
        media = {"pixel_values": torch.randn(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])}
        type_id = 1
    else:
        ids = torch.tensor([[44, 43, 45, 3]])
        media = {"pixel_values_videos": torch.randn(8, 12), "video_grid_thw": torch.tensor([[2, 2, 2]])}
        type_id = 2
    with torch.no_grad():
        output = model(ids, moe_mm_token_type_ids=torch.tensor([[type_id] * 3 + [0]]),
                       mm_token_type_ids=torch.tensor([[0, type_id, 0, 0]]), use_cache=True, **media)
        assert output.logits.shape == (1, 4, 48)
        assert torch.isfinite(output.logits).all()
        continued = model(torch.tensor([[7]]), past_key_values=output.past_key_values, use_cache=True)
        assert continued.past_key_values.get_seq_length() == 5
        assert torch.isfinite(continued.logits).all()
