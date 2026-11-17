# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native LFM3-VL qualification; no checkpoint Python or trained weights."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForImageTextToText,
    Lfm3Config,
    Lfm3VlConfig,
    Lfm3VlForConditionalGeneration,
    Lfm3VlProcessor,
    Lfm3VlVisionConfig,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
)

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.lfm3_vl import LFM3VLQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


AUDIT = Path(__file__).parents[1] / "gptqmodel/models/ports/lfm3_vl"


def tiny_config(kv_heads=2, bias=False, gate=True, sandwich=True):
    text = Lfm3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=8,
        layer_types=["conv", "full_attention", "sliding_attention"], sliding_window=4,
        conv_L_cache=3, conv_bias=bias, attention_bias=bias,
        use_attention_gate=gate, use_sandwich_norm=sandwich,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    vision = Lfm3VlVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=kv_heads, layer_types=["conv", "full_attention"],
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
    )
    return Lfm3VlConfig(
        text_config=text, vision_config=vision, image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=63, projector_hidden_size=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )


def native_model(**kwargs):
    torch.manual_seed(571)
    return AutoModelForImageTextToText.from_config(tiny_config(**kwargs), attn_implementation="eager").eval()


def tiny_processor():
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "describe": 4}
    vocab.update({f"t{i}": i for i in range(5, 60)})
    vocab.update({"<|image_pad|>": 60, "<|video_pad|>": 61, "<|vision_start|>": 62, "<|vision_end|>": 63})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]",
        unk_token="[UNK]", additional_special_tokens=list(vocab)[-4:],
    )
    return Lfm3VlProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2,
            size={"shortest_edge": 16, "longest_edge": 64},
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2,
            size={"shortest_edge": 16, "longest_edge": 64}, do_sample_frames=False,
        ),
        tokenizer=tokenizer,
        chat_template="{% for message in messages %}{% for item in message['content'] %}"
                      "{% if item['type'] == 'text' %}{{ item['text'] }} "
                      "{% elif item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                      "{% endif %}{% endfor %}{% endfor %}",
    )


def media_inputs(processor, modality="image"):
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    if modality == "image":
        return processor(text="describe <|image_pad|>", images=Image.fromarray(pixels), return_tensors="pt")
    return processor(
        text="describe <|video_pad|>", videos=np.stack([pixels, pixels[::-1].copy()] * 2),
        video_metadata=[{"fps": 2, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}],
        return_tensors="pt",
    )


def hostile_metadata(path):
    """Keep pinned checkpoint processor spellings and poison every discovery map."""
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code')\n")
    for filename in ("config.json", "tokenizer_config.json", "processor_config.json"):
        target = path / filename
        data = json.loads(target.read_text())
        data["auto_map"] = dict.fromkeys((
            "AutoConfig", "AutoModel", "AutoModelForImageTextToText", "AutoProcessor",
        ), "hostile.Bad")
        if filename == "tokenizer_config.json":
            data["auto_map"]["AutoTokenizer"] = ["hostile.Bad", "hostile.Bad"]
        if filename == "processor_config.json":
            for media in ("image", "video"):
                config = data[f"{media}_processor"]
                config[f"{media}_processor_type"] = f"Lfm3Vl{media.title()}Processor"
                config["auto_map"] = {f"Auto{media.title()}Processor": "hostile.Bad"}
        target.write_text(json.dumps(data))
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_native_load_ignores_hostile_checkpoint_code(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original = native_model()
    original.save_pretrained(tmp_path, safe_serialization=True)
    tiny_processor().save_pretrained(tmp_path)
    marker = hostile_metadata(tmp_path)
    assert not LFM3VLQModel.require_trust_remote_code
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is Lfm3VlConfig
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                           attn_implementation="eager")
    assert type(loaded.model) is Lfm3VlForConditionalGeneration
    assert type(loaded.processor) is Lfm3VlProcessor
    inputs = media_inputs(loaded.processor)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**inputs).logits, original(**inputs).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_media_cache_generation_and_safetensors(tmp_path, modality, dtype):
    initial = tmp_path / "initial"
    native_model().save_pretrained(initial, safe_serialization=True)
    model = Lfm3VlForConditionalGeneration.from_pretrained(initial, dtype=dtype,
                                                         attn_implementation="eager").eval()
    processor = tiny_processor()
    inputs = media_inputs(processor, modality).to(dtype=dtype)
    with torch.inference_mode():
        expected = model(**inputs, use_cache=False).logits
        cached = model(**inputs, use_cache=True)
        continuation = torch.tensor([[7, 8]])
        full_inputs = dict(inputs)
        full_inputs["input_ids"] = torch.cat([inputs.input_ids, continuation], dim=1)
        full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])
        full = model(**full_inputs, use_cache=False).logits[:, -2:]
        tail = model(input_ids=continuation, attention_mask=full_inputs["attention_mask"],
                     past_key_values=cached.past_key_values, use_cache=True).logits
        torch.testing.assert_close(tail, full, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                                   atol=0.002 if dtype == torch.bfloat16 else 1e-6)
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        assert generated.shape[1] == inputs.input_ids.shape[1] + 2
        changed = dict(inputs)
        key = "pixel_values" if modality == "image" else "pixel_values_videos"
        changed[key] = inputs[key] + 2.0
        assert not torch.equal(model(**changed, use_cache=False).logits, expected)
    model.save_pretrained(tmp_path, safe_serialization=True)
    processor.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert not list(tmp_path.glob("*.bin"))
    restored = Lfm3VlForConditionalGeneration.from_pretrained(tmp_path, dtype=dtype,
                                                            attn_implementation="eager").eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(**inputs, use_cache=False).logits, expected, rtol=0, atol=0)


def test_legacy_checkpoint_conversion_is_complete_and_exact(tmp_path):
    model = native_model()
    model.config.save_pretrained(tmp_path)
    state = {name.replace(".self_attn.g_proj.", ".self_attn.gate_proj.")
             .replace(".patch_embed.", ".patchifier.")
             .replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
             .replace(".self_attn.k_norm.", ".self_attn.k_layernorm."): value.clone()
             for name, value in model.state_dict().items()
             if name != "lm_head.weight"}
    # Only the vision checkpoint uses q/k_layernorm; text keeps q/k_norm.
    state = {name.replace(".q_layernorm.", ".q_norm.").replace(".k_layernorm.", ".k_norm.")
             if name.startswith("model.language_model.") else name: value for name, value in state.items()}
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    restored, info = Lfm3VlForConditionalGeneration.from_pretrained(
        tmp_path, dtype=torch.float32, attn_implementation="eager", output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    inputs = media_inputs(tiny_processor())
    with torch.inference_mode():
        torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing_tokenizer", "audio_component", "malformed_json"])
def test_processor_failure_then_clean_retry(tmp_path, failure):
    processor = tiny_processor()
    processor.save_pretrained(tmp_path)
    metadata_path = tmp_path / "processor_config.json"
    metadata = metadata_path.read_text()
    kwargs = {"tokenizer": processor.tokenizer}
    if failure == "missing_tokenizer":
        kwargs["tokenizer"] = None
        error = "requires the tokenizer"
    elif failure == "audio_component":
        data = json.loads(metadata)
        data["audio_tokenizer"] = {"audio_tokenizer_class": "AutoModel", "audio_tokenizer_name_or_path": "hostile"}
        metadata_path.write_text(json.dumps(data))
        error = "does not accept checkpoint-selected"
    else:
        metadata_path.write_text("{")
        error = "not a valid JSON"
    with pytest.raises((ValueError, OSError), match=error):
        LFM3VLQModel.loader.load_processor(str(tmp_path), **kwargs)
    metadata_path.write_text(metadata)
    restored = LFM3VLQModel.loader.load_processor(str(tmp_path), tokenizer=processor.tokenizer)
    assert restored.tokenizer is processor.tokenizer
    assert restored.chat_template == processor.chat_template
    for modality in ("image", "video"):
        expected = media_inputs(processor, modality)
        actual = media_inputs(restored, modality)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def test_pinned_native_media_metadata_against_pixel_and_grid_oracles(tmp_path):
    (tmp_path / "processor_config.json").write_bytes((AUDIT / "checkpoint_processor_config.json").read_bytes())
    tokenizer = tiny_processor().tokenizer
    processor = LFM3VLQModel.loader.load_processor(str(tmp_path), tokenizer=tokenizer)
    assert isinstance(processor.image_processor, Qwen2VLImageProcessor)
    assert isinstance(processor.video_processor, Qwen3VLVideoProcessor)
    image = processor(text=processor.image_token, images=Image.new("RGB", (256, 256), (100, 40, 20)),
                      return_tensors="pt")
    assert image.image_grid_thw.tolist() == [[1, 16, 16]]
    assert int((image.input_ids == processor.image_token_id).sum()) == 64
    expected = ((torch.tensor([100., 40., 20.]) / 255 - 0.5) / 0.5)
    expected = expected.view(1, 3, 1, 1, 1).expand(256, 3, 2, 16, 16).reshape(256, 1536)
    torch.testing.assert_close(image.pixel_values, expected, rtol=0, atol=1e-7)
    video = processor(text=processor.video_token, videos=np.full((4, 64, 64, 3), 80, dtype=np.uint8),
                      video_metadata=[{"fps": 2, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}],
                      return_tensors="pt")
    assert video.video_grid_thw.tolist() == [[2, 4, 4]]
    assert int((video.input_ids == processor.video_token_id).sum()) == 8
    torch.testing.assert_close(video.pixel_values_videos, torch.full((32, 1536), 80 / 255 * 2 - 1),
                               rtol=0, atol=1e-7)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_public_awq_plain_quantization_packs_all_decoder_linears(tmp_path, kv_heads, bias):
    source = tmp_path / "source"
    native_model(kv_heads=kv_heads, bias=bias).save_pretrained(source, safe_serialization=True)
    tiny_processor().save_pretrained(source)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=False)
    wrapper = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH,
                            dtype=torch.float16, attn_implementation="eager")
    expected_names = {f"model.language_model.layers.{i}.{name}" for i in range(3)
                      for name in (["conv.in_proj", "conv.out_proj"] if i == 0 else
                                   [f"self_attn.{p}_proj" for p in ("q", "k", "v", "o")])}
    expected_names.update(f"model.language_model.layers.{i}.feed_forward.w{p}"
                          for i in range(3) for p in (1, 2, 3))
    before = {name: value.clone() for name, value in wrapper.model.state_dict().items()
              if not any(name.startswith(prefix + ".") for prefix in expected_names)}
    assert all(wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
               for layer in wrapper.model.model.language_model.layers)
    samples = [[{"role": "user", "content": [
        {"type": "text", "text": "t5 t6 t7 t8 t9 t10 t11 t12"},
        {"type": "image", "image": Image.new("RGB", (4, 4), (100, 40, 20))},
    ]}] for _ in range(2)]
    wrapper.quantize(samples, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert set(packed) == expected_names
    assert len(packed) == 19
    for name, value in before.items():
        torch.testing.assert_close(wrapper.model.state_dict()[name], value, rtol=0, atol=0)
    inputs = media_inputs(wrapper.processor).to(dtype=torch.float16)
    with torch.inference_mode():
        expected = wrapper.model(**inputs, use_cache=False).logits
    assert torch.isfinite(expected).all()
    destination = tmp_path / "quantized"
    wrapper.save(str(destination))
    assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
    restored = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                             dtype=torch.float16, attn_implementation="eager")
    assert {name for name, module in restored.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected_names
    with torch.inference_mode():
        torch.testing.assert_close(restored.model(**inputs, use_cache=False).logits, expected, rtol=0, atol=0)
    for name, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0)
