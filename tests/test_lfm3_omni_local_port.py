# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real LFM3 Omni wrapper qualification; no checkpoint Python or trained weights."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.lfm3_omni import LFM3OmniQModel
from gptqmodel.models.ports.lfm3_omni.configuration_lfm3_omni import Lfm3OmniConfig
from gptqmodel.models.ports.lfm3_omni.feature_extraction_lfm3_omni import Lfm3OmniFeatureExtractor
from gptqmodel.models.ports.lfm3_omni.image_processing_lfm3_vl import Lfm3VlImageProcessor
from gptqmodel.models.ports.lfm3_omni.modeling_lfm3_omni import Lfm3OmniForConditionalGeneration
from gptqmodel.models.ports.lfm3_omni.processing_lfm3_omni import Lfm3OmniProcessor
from gptqmodel.models.ports.lfm3_omni.video_processing_lfm3_vl import Lfm3VlVideoProcessor


PORT = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/lfm3_omni"


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(kv_heads=2, bias=False, sandwich=True, layer_types=None):
    layer_types = layer_types or ["conv", "full_attention", "sliding_attention"]
    text = {
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4, "num_key_value_heads": kv_heads, "head_dim": 8, "layer_types": layer_types,
        "sliding_window": 4, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "conv_bias": bias,
        "use_sandwich_norm": sandwich, "attn_implementation": "eager",
    }
    encoder = {
        "num_mel_bins": 8, "conv_channels": 8, "hidden_size": 16, "sliding_window": 4,
        "text_config": {"vocab_size": 2, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                         "num_attention_heads": 2, "num_key_value_heads": 1, "layer_types": ["conv", "full_attention"],
                         "conv_L_cache": 3, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": None,
                         "block_auto_adjust_ff_dim": False, "attn_implementation": "eager"},
    }
    depth = {**text, "num_hidden_layers": 1, "layer_types": ["full_attention"],
             "use_sandwich_norm": False, "use_cache": False}
    vision = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                  "num_attention_heads": 2, "num_key_value_heads": 1, "layer_types": ["conv", "full_attention"],
                  "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "attn_implementation": "eager"}
    return Lfm3OmniConfig(
        text_config=text, audio_encoder_config=encoder, depth_config=depth, vision_config=vision,
        num_codebooks=2, codebook_size=8, audio_eos_code=7, maskgit_schedule=[1, 1],
        audio_token_id=50, audio_out_token_id=51, audio_start_token_id=52, text_end_token_id=53,
        mixed_start_token_id=54, image_token_id=55, video_token_id=56, projector_hidden_size=32,
        attn_implementation="eager",
    )


def tiny_model(**kwargs):
    torch.manual_seed(971)
    return Lfm3OmniForConditionalGeneration(tiny_config(**kwargs)).eval()


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {f"token{i}": i for i in range(64)}
    for index, token in {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 50: "<|audio|>",
                         51: "<|audio_out|>", 52: "<|audio_start|>", 53: "<|text_end|>",
                         54: "<|mixed_start|>", 55: "<|image|>", 56: "<|video|>",
                         57: "<|media_start|>", 58: "<|media_end|>"}.items():
        del vocab[f"token{index}"]
        vocab[token] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", additional_special_tokens=[token for token in vocab if token.startswith("<|")],
    )


def tiny_processor():
    return Lfm3OmniProcessor(
        image_processor=Lfm3VlImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                             size={"shortest_edge": 16, "longest_edge": 64}),
        feature_extractor=Lfm3OmniFeatureExtractor(feature_size=8, n_fft=128, win_length=128, hop_length=8),
        tokenizer=tiny_tokenizer(),
        video_processor=Lfm3VlVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                             size={"shortest_edge": 16, "longest_edge": 64},
                                             min_frames=2, max_frames=4, do_sample_frames=False),
    )


def write_checkpoint(path, model=None, hostile=False):
    model = model or tiny_model()
    model.save_pretrained(path)
    tiny_processor().save_pretrained(path)
    marker = path / "CHECKPOINT_PYTHON_EXECUTED"
    if hostile:
        for filename, entries in (
            ("config.json", {"AutoConfig": "poison.Config", "AutoModel": "poison.Model",
                             "AutoModelForImageTextToText": "poison.Model"}),
            ("tokenizer_config.json", {"AutoTokenizer": ["poison.Tokenizer", "poison.Tokenizer"]}),
            ("processor_config.json", {"AutoProcessor": "poison.Processor"}),
        ):
            config = json.loads((path / filename).read_text())
            config["auto_map"] = entries
            if filename == "processor_config.json":
                config["processor_class"] = "PoisonProcessor"
                for name in ("image_processor", "feature_extractor", "video_processor"):
                    config[name]["auto_map"] = {"AutoProcessor": "poison.Component"}
                    config[name][f"{name}_type"] = "PoisonComponent"
            (path / filename).write_text(json.dumps(config))
        (path / "poison.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint executed')\n"
        )
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_load_owns_config_model_processor_despite_hostile_metadata(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = write_checkpoint(tmp_path, hostile=True)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(wrapper) is LFM3OmniQModel
    assert type(wrapper.model) is Lfm3OmniForConditionalGeneration
    assert type(wrapper.model.config) is Lfm3OmniConfig
    assert type(wrapper.processor) is Lfm3OmniProcessor
    assert wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
    assert wrapper.model(torch.tensor([[1, 4, 5]]), use_cache=False).logits.shape == (1, 3, 64)
    assert not marker.exists()


@pytest.mark.parametrize("component", ["text_config", "depth_config", "vision_config", "audio_encoder_config",
                                       "audio_encoder_config.text_config"])
def test_public_rejects_foreign_nested_architecture_then_retries(tmp_path, component):
    write_checkpoint(tmp_path)
    original = (tmp_path / "config.json").read_text()
    values = json.loads(original)
    target = values
    for part in component.split("."):
        target = target[part]
    target["model_type"] = "llama"
    (tmp_path / "config.json").write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Expected .* component"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    (tmp_path / "config.json").write_text(original)
    assert type(GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32).model) is Lfm3OmniForConditionalGeneration


def reference_text_forward(model, ids, positions, keep):
    """Independent tensor equations of pinned upstream text, no port mask/cache/RoPE helpers."""
    weights, config = model.state_dict(), model.config.text_config
    hidden = F.embedding(ids, weights["model.language_model.embed_tokens.weight"])

    def norm(x, name):
        return weights[name + ".weight"] * (x.float() * (x.float().square().mean(-1, keepdim=True)
                                                       + config.norm_eps).rsqrt()).to(x.dtype)

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    batch, length = ids.shape
    dim, heads, kv_heads = config.head_dim, config.num_attention_heads, config.num_key_value_heads
    for index, kind in enumerate(config.layer_types):
        prefix = f"model.language_model.layers.{index}."
        normalized = norm(hidden, prefix + "operator_norm")
        if kind == "conv":
            normalized = normalized * keep[..., None]
            b, c, x = linear(normalized, prefix + "conv.in_proj").chunk(3, -1)
            gated = b * x * keep[..., None]
            kernel = weights[prefix + "conv.conv.weight"][:, 0]
            padded = F.pad(gated, (0, 0, kernel.shape[-1] - 1, 0))
            convolved = torch.stack([
                (padded[:, time:time + kernel.shape[-1]].float() * kernel.T.float()).sum(1)
                for time in range(length)
            ], 1)
            convolved = convolved + weights.get(prefix + "conv.conv.bias", 0)
            operated = linear(c * convolved.to(c.dtype), prefix + "conv.out_proj")
        else:
            q = norm(linear(normalized, prefix + "self_attn.q_proj").view(batch, length, heads, dim),
                     prefix + "self_attn.q_norm")
            k = norm(linear(normalized, prefix + "self_attn.k_proj").view(batch, length, kv_heads, dim),
                     prefix + "self_attn.k_norm")
            v = linear(normalized, prefix + "self_attn.v_proj").view(batch, length, kv_heads, dim)
            rope = config.rope_parameters[kind]
            rotated = int(rope.get("partial_rotary_factor", 1.0) * dim) // 2
            inverse = torch.zeros(dim // 2)
            inverse[:rotated] = rope["rope_theta"] ** (-torch.arange(rotated).float() * 2 / dim)
            theta = positions[..., None].float() * inverse
            cosine, sine = theta.cos()[:, :, None].to(q.dtype).float(), theta.sin()[:, :, None].to(q.dtype).float()
            def rotate(tensor):
                a, b = tensor.float().chunk(2, -1)
                return torch.cat((a * cosine - b * sine, b * cosine + a * sine), -1).to(tensor.dtype)
            q, k = rotate(q).transpose(1, 2), rotate(k).repeat_interleave(heads // kv_heads, 2).transpose(1, 2)
            v = v.repeat_interleave(heads // kv_heads, 2).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) * dim ** -0.5
            row = torch.arange(length)
            allowed = row[None, :] <= row[:, None]
            if kind == "sliding_attention":
                allowed &= row[None, :] > row[:, None] - config.sliding_window
            allowed = allowed[None, None] & keep[:, None, None].bool()
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            attended = (scores.float().softmax(-1).to(q.dtype) @ v).transpose(1, 2)
            gates = linear(normalized, prefix + "self_attn.gate_proj").sigmoid()
            operated = linear((attended * gates[..., None]).flatten(2), prefix + "self_attn.o_proj")
        if config.use_sandwich_norm:
            operated = norm(operated, prefix + "post_operator_norm")
        hidden = hidden + operated
        normalized = norm(hidden, prefix + "ffn_norm")
        ffn = linear(F.silu(linear(normalized, prefix + "feed_forward.w1"))
                     * linear(normalized, prefix + "feed_forward.w3"), prefix + "feed_forward.w2")
        if config.use_sandwich_norm:
            ffn = norm(ffn, prefix + "post_ffn_norm")
        hidden = hidden + ffn
    hidden = norm(hidden, "model.language_model.embedding_norm")
    return linear(hidden, "lm_head")


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("bias,sandwich", [(False, True), (True, False), (False, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_text_equations_cache_and_causality(kv_heads, bias, sandwich, dtype):
    model = tiny_model(kv_heads=kv_heads, bias=bias, sandwich=sandwich).to(dtype)
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10], [0, 0, 1, 11, 12, 13, 14, 15]])
    keep = (ids != 0).long()
    positions = (keep.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        expected = reference_text_forward(model, ids, positions, keep)
        full = model(ids, attention_mask=keep, position_ids=positions, use_cache=False).logits
        atol = {torch.float32: 3e-6, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
        torch.testing.assert_close(full[keep.bool()], expected[keep.bool()], atol=atol, rtol=atol)
        prefix = model(ids[:, :3], attention_mask=keep[:, :3], position_ids=positions[:, :3], use_cache=True)
        for start, end in ((3, 4), (4, 8)):
            suffix = model(ids[:, start:end], attention_mask=keep[:, :end], position_ids=positions[:, start:end],
                           past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(suffix.logits, full[:, start:end], atol=atol, rtol=atol)
        altered = ids.clone()
        altered[:, -1] = 23
        changed = model(altered, attention_mask=keep, position_ids=positions, use_cache=False).logits[:, :-1]
        torch.testing.assert_close(changed[keep[:, :-1].bool()], full[:, :-1][keep[:, :-1].bool()], atol=0, rtol=0)


def test_multimodal_forward_and_safetensors_roundtrip(tmp_path):
    model = tiny_model()
    # Three audio tokens, one output frame, one image and two video frames.
    ids = torch.tensor([[1, 50, 50, 50, 51, 55, 56, 56, 4]])
    inputs = {"input_ids": ids, "input_features": torch.randn(1, 8, 17),
                  "feature_attention_mask": torch.ones(1, 17, dtype=torch.long),
                  "audio_out_codes": torch.tensor([[1, 2]]), "pixel_values": torch.randn(4, 24),
                  "image_grid_thw": torch.tensor([[1, 2, 2]]), "pixel_values_videos": torch.randn(8, 24),
                  "video_grid_thw": torch.tensor([[2, 2, 2]]), "use_cache": False}
    with torch.no_grad():
        baseline = model(**inputs).logits
        changed = {**inputs, "input_features": inputs["input_features"] + 2}
        assert not torch.equal(model(**changed).logits, baseline)
        model.save_pretrained(tmp_path)
        reloaded, info = LFM3OmniQModel.loader.from_pretrained(
            tmp_path, output_loading_info=True, attn_implementation="eager")
        assert not any(info.values()), info
        torch.testing.assert_close(reloaded(**inputs).logits, baseline, atol=0, rtol=0)
        assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
        assert set(reloaded.state_dict()) == set(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], value, atol=0, rtol=0)


def test_text_generation_audio_depth_and_rejected_generation_retry():
    model = tiny_model()
    ids = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        expected = ids
        forbidden = [50, 51, 52, 54, 55, 56]
        for _ in range(3):
            scores = model(expected, use_cache=False).logits[:, -1].clone()
            scores[:, forbidden] = -float("inf")
            expected = torch.cat((expected, scores.argmax(-1, keepdim=True)), -1)
        assert torch.equal(model.generate(ids, generation_mode="text", max_new_tokens=3,
                                          do_sample=False).sequences, expected)
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=model.config.text_config)
        with pytest.raises(ValueError, match="one greedy or sampled continuation"):
            model.generate(ids, past_key_values=cache, max_new_tokens=2, num_beams=2)
        assert cache.get_seq_length() == 0
        resumed = model.generate(ids, past_key_values=cache, max_new_tokens=2, generation_mode="text", do_sample=False)
        assert torch.equal(resumed.sequences, expected[:, :5])
        assert cache.get_seq_length() == 5
        frames = model.sample_audio_frame(torch.randn(2, 32), temperature=0)
        assert frames.shape == (2, 2) and ((frames >= 0) & (frames < 8)).all()
        output = model.generate(torch.tensor([[1, 52]]), generation_mode="audio", max_new_tokens=2,
                                audio_temperature=0, return_audio_eos=True)
        assert output.audio_codes.shape[1] == 2
        model.config.interleaved_n_text = 1
        model.config.interleaved_n_audio = 1
        mixed = model.generate(torch.tensor([[1, 54]]), generation_mode="interleaved", max_new_tokens=4,
                               audio_temperature=0, do_sample=False)
        assert mixed.sequences.shape[-1] >= 2 and mixed.audio_codes.shape[1] == 2


def test_feature_stream_audio_encoder_stream_and_failure_retry():
    extractor = tiny_processor().feature_extractor
    waveform = torch.sin(torch.arange(331) * 0.19)
    offline = extractor(waveform, sampling_rate=16000, return_tensors="pt")["input_features"]
    state, chunks = None, []
    for start, end in ((0, 11), (11, 129), (129, 331)):
        output, state = extractor.stream(waveform[start:end], state, is_final=end == 331, sampling_rate=16000)
        chunks.append(output["input_features"])
    torch.testing.assert_close(torch.cat(chunks, -1), offline, atol=0, rtol=0)
    with pytest.raises(ValueError, match="finalized"):
        extractor.stream(waveform[:1], state)
    fresh, cache = extractor.stream(waveform[:11], sampling_rate=16000)
    snapshot = copy.deepcopy(cache)
    with pytest.raises(ValueError, match="16000"):
        extractor.stream(waveform[11:], cache, sampling_rate=8000)
    assert cache.num_samples == snapshot.num_samples and cache.num_frames == snapshot.num_frames
    torch.testing.assert_close(cache.samples, snapshot.samples, atol=0, rtol=0)
    retried, _ = extractor.stream(waveform[11:], cache, is_final=True, sampling_rate=16000)
    torch.testing.assert_close(torch.cat((fresh["input_features"], retried["input_features"]), -1), offline,
                               atol=0, rtol=0)
    encoder = tiny_model().model.audio_encoder
    features = torch.randn(1, 8, 41)
    with torch.no_grad():
        complete, _ = encoder(features, torch.tensor([41]))
        cache, chunks = None, []
        for start, end in ((0, 7), (7, 24), (24, 41)):
            out, cache = encoder.stream(features[..., start:end], cache, is_final=end == 41)
            chunks.append(out)
        torch.testing.assert_close(torch.cat(chunks, 1), complete, atol=2e-6, rtol=2e-5)
        with pytest.raises(ValueError, match="finalized"):
            encoder.stream(features[..., :1], cache)


def test_processor_real_image_video_audio_and_roundtrip(tmp_path):
    from PIL import Image

    processor = tiny_processor()
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    image = Image.fromarray(pixels)
    video = np.stack([pixels, 255 - pixels, pixels, 255 - pixels])
    audio = np.sin(np.arange(200, dtype=np.float32) * 0.17)
    inputs = processor(text="token4 <|image|> <|audio|>", images=image, audio=audio,
                       audio_kwargs={"sampling_rate": 16000}, return_tensors="pt")
    assert inputs["pixel_values"].shape == (4, 24)
    assert inputs["input_features"].shape[1] == 8
    assert (inputs["input_ids"] == 55).sum() == 1
    assert (inputs["input_ids"] == 50).sum() == 2
    with torch.no_grad():
        assert torch.isfinite(tiny_model()(**inputs).logits).all()
    with pytest.raises(ValueError, match="metadata|clock|fps"):
        processor(text="<|video|>", videos=video, videos_kwargs={"input_data_format": "channels_last"},
                  return_tensors="pt")
    video_inputs = processor(text="<|video|>", videos=video,
                             videos_kwargs={"input_data_format": "channels_last"},
                             video_metadata={"fps": 2, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]},
                             return_tensors="pt")
    assert (video_inputs["input_ids"] == 56).sum() == 2
    with torch.no_grad():
        assert torch.isfinite(tiny_model()(**video_inputs).logits).all()
    processor.save_pretrained(tmp_path)
    restored = Lfm3OmniProcessor.from_pretrained(tmp_path, tokenizer=processor.tokenizer)
    repeated = restored(text="token4 <|image|> <|audio|>", images=image, audio=audio,
                        audio_kwargs={"sampling_rate": 16000}, return_tensors="pt")
    assert set(repeated) == set(inputs)
    for key in inputs:
        torch.testing.assert_close(repeated[key], inputs[key], atol=0, rtol=0)


def test_codec_real_encode_decode_stream_legacy_safetensors_and_processor_reload(tmp_path):
    from safetensors.torch import save_file

    from gptqmodel.models.ports.lfm3_omni.configuration_liquid_codec import LiquidCodecConfig
    from gptqmodel.models.ports.lfm3_omni.modeling_liquid_codec import LiquidCodecModel, LiquidCodecStreamingCache

    torch.manual_seed(142)
    config = LiquidCodecConfig(encoder_dim=16, encoder_conv_dim=8, stft_n_fft=16, stft_win_length=16,
                              stft_hop_length=8, stft_frame_stack=2, n_q=2, quantizer_dim=8,
                              fsq_levels=[2, 4], decoder_window=4, attn_implementation="eager")
    codec = LiquidCodecModel(config).eval()
    waveform = torch.sin(torch.arange(61).float() * 0.27)[None] * 0.1
    with torch.no_grad():
        codes = codec.encode(waveform).audio_codes
        assert codes.shape == (1, 2, 4) and ((codes >= 0) & (codes < 8)).all()
        audio = codec.decode(codes).audio_values
        assert audio.shape == (1, 64) and torch.isfinite(audio).all()
        assert codec(waveform).audio_values.shape == waveform.shape
        cache = LiquidCodecStreamingCache(config)
        chunks = []
        for start, end in ((0, 1), (1, 3), (3, 4)):
            chunks.append(codec.decode(codes[..., start:end], streaming_cache=cache,
                                       finalize=end == 4).audio_values)
        torch.testing.assert_close(torch.cat(chunks, -1), audio, atol=3e-6, rtol=3e-5)
        with pytest.raises(ValueError, match="finalized"):
            codec.decode(codes[..., :1], streaming_cache=cache)
        with pytest.raises((RuntimeError, ValueError), match="finite"):
            codec.encode(torch.full((1, 61), float("nan")))
        assert torch.equal(codec.encode(waveform).audio_codes, codes)
    processor = tiny_processor()
    processor.audio_tokenizer = codec
    processor.save_pretrained(tmp_path)
    # The real released codec header uses these older owner names. Conversion
    # is per load; it must not alter Transformers' global conversion registry.
    codec_dir = tmp_path / "audio_tokenizer"
    saved = codec.state_dict()
    legacy = {(key.replace("frontend.", "_frontend.", 1) if key.startswith("frontend.")
               else key.replace("quantizer.", "quantizer.fsq.", 1) if key.startswith("quantizer.") else key):
              value.contiguous() for key, value in saved.items()}
    save_file(legacy, codec_dir / "model.safetensors", metadata={"format": "pt"})
    values = json.loads((codec_dir / "config.json").read_text())
    values["auto_map"] = {"AutoConfig": "poison.Config", "AutoModel": "poison.Model"}
    (codec_dir / "config.json").write_text(json.dumps(values))
    (codec_dir / "poison.py").write_text("raise RuntimeError('codec checkpoint Python executed')")
    restored = Lfm3OmniProcessor.from_pretrained(tmp_path, tokenizer=processor.tokenizer)
    assert type(restored.audio_tokenizer) is LiquidCodecModel
    for key, value in saved.items():
        torch.testing.assert_close(restored.audio_tokenizer.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        assert torch.equal(restored.audio_tokenizer.encode(waveform).audio_codes, codes)
        torch.testing.assert_close(restored.audio_tokenizer.decode(codes).audio_values, audio, atol=3e-6, rtol=3e-5)
    bad = json.loads((tmp_path / "processor_config.json").read_text())
    bad["audio_tokenizer_subfolder"] = "../untrusted_codec"
    (tmp_path / "processor_config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="audio_tokenizer subfolder"):
        Lfm3OmniProcessor.from_pretrained(tmp_path, tokenizer=processor.tokenizer)
    bad["audio_tokenizer_subfolder"] = "audio_tokenizer"
    (tmp_path / "processor_config.json").write_text(json.dumps(bad))
    assert type(Lfm3OmniProcessor.from_pretrained(tmp_path, tokenizer=processor.tokenizer).audio_tokenizer) is LiquidCodecModel


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("bias", [False, True])
def test_public_plain_awq_quantizes_only_text_linears_and_reloads(tmp_path, kv_heads, bias):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = tiny_model(kv_heads=kv_heads, bias=bias).half()
    write_checkpoint(tmp_path, source)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                            backend=BACKEND.TORCH, dtype=torch.float16)
    before = {key: value.clone() for key, value in source.state_dict().items()
              if not key.startswith("model.language_model.layers.")
              or any(tag in key for tag in ("norm", "gate_proj", "conv.conv"))}
    for layer in wrapper.model.model.language_model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 19
    assert all(name.startswith("model.language_model.layers.") for name in packed)
    for key, value in before.items():
        torch.testing.assert_close(wrapper.model.state_dict()[key], value, atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.rglob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 19
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected,
                                   atol=2e-4, rtol=2e-3)
    for key, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(reloaded.model.state_dict()[key], value, atol=0, rtol=0)
