# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Mage-VL model/processor qualification; no checkpoint Python execution."""

import copy
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.mage_vl import MageVLQModel
from gptqmodel.models.ports.mage_vl.configuration_mage_vl import MageVLConfig
from gptqmodel.models.ports.mage_vl.modeling_mage_vl import MageVLForConditionalGeneration
from gptqmodel.models.ports.mage_vl.processing_mage_vl import MageVLProcessor
from gptqmodel.models.ports.mage_vl.video_processing_mage_vl import build_patch_positions


def _config(**overrides):
    text = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "num_hidden_layers": 2, "max_position_embeddings": 128,
                "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "attn_implementation": "eager"}
    vision = {"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 1,
                  "patch_size": 2, "image_size": 4, "out_hidden_size": 32, "max_position_embeddings": 32,
                  "attn_implementation": "eager"}
    text.update(overrides.pop("text", {}))
    vision.update(overrides.pop("vision", {}))
    return MageVLConfig(text_config=text, vision_config=vision, image_token_id=60, video_token_id=61,
                        vision_start_token_id=62, vision_end_token_id=63, **overrides)


def _model(**kwargs):
    torch.manual_seed(706)
    return MageVLForConditionalGeneration(_config(**kwargs)).eval()


def _processor_assets(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessorPil

    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             **{str(i): i for i in range(4, 60)}, "<|image_pad|>": 60, "<|video_pad|>": 61,
             "<|vision_start|>": 62, "<|vision_end|>": 63}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                       bos_token="[BOS]", unk_token="[UNK]",
                                       additional_special_tokens=list(vocab)[-4:])
    tokenizer.chat_template = ("{% for message in messages %}{% if message['content'] is string %}"
                               "{{ message['content'] }}{% else %}{% for item in message['content'] %}"
                               "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                               "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
                               "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}")
    tokenizer.save_pretrained(path)
    image = Qwen2VLImageProcessorPil(patch_size=2, temporal_patch_size=1, merge_size=2,
                                    min_pixels=16, max_pixels=64)
    image.save_pretrained(path)
    (path / "video_preprocessor_config.json").write_text(json.dumps({"max_frames": 7}))
    return tokenizer


def _image_inputs(model, frames=1, video=False):
    grid = torch.tensor([[frames, 2, 2]])
    positions = build_patch_positions(grid)
    pixels = torch.linspace(-1, 1, frames * 4 * 12, dtype=model.dtype).reshape(frames * 4, 12)
    image_token = model.config.video_token_id if video else model.config.image_token_id
    ids = torch.tensor([[1, 62] + [image_token] * frames + [63, 9, 8]])
    return dict(input_ids=ids, patch_positions=positions,
                **({"pixel_values_videos": pixels, "video_grid_thw": grid} if video else
                   {"pixel_values": pixels, "image_grid_thw": grid}))


def _vision_oracle(model, pixels, grid, positions):
    """Pinned source equations, independent of port attention/rope/encoder/merger helpers."""
    w = model.state_dict()
    c = model.config.vision_config
    prefix = "model.visual."

    def norm(x, key):
        if c.layer_norm_type == "rms_norm":
            return F.rms_norm(x, (x.shape[-1],), w[prefix + key + ".weight"], c.layer_norm_eps)
        return F.layer_norm(x, (x.shape[-1],), w[prefix + key + ".weight"],
                            w[prefix + key + ".bias"], c.layer_norm_eps)

    def linear(x, key):
        return F.linear(x, w[prefix + key + ".weight"], w.get(prefix + key + ".bias"))

    x = F.conv2d(pixels.reshape(-1, 3, c.patch_size, c.patch_size),
                 w[prefix + "embeddings.patch_embedding.weight"], stride=c.patch_size).flatten(1)[None]
    x = norm(x, "layernorm_pre")
    dim = c.hidden_size // c.num_attention_heads
    sizes = [dim // 8, dim * 3 // 16, dim * 3 // 16]
    freq = torch.cat([positions[:, axis:axis + 1].float() *
                      c.rope_theta ** (-torch.arange(size).float() / size)
                      for axis, size in enumerate(sizes)], -1)
    freq = torch.cat((freq, freq), -1)[None, None]
    lengths = []
    for t, h, width in grid.tolist():
        window = c.frame_windows_size
        while t:
            take = min(t, window) if window > 0 else t
            lengths.append(take * h * width)
            t -= take
    for i in range(c.num_hidden_layers):
        base = f"encoder.layers.{i}."
        h = norm(x, base + "layer_norm1")
        q, k, v = linear(h, base + "self_attn.qkv").reshape(1, -1, 3, c.num_attention_heads, dim).permute(2, 0, 3, 1, 4)

        def rope(tensor):
            original = tensor.dtype
            tensor = tensor.float()
            rotated = torch.empty_like(tensor)
            rotated[..., ::2] = -tensor[..., 1::2]
            rotated[..., 1::2] = tensor[..., ::2]
            return (tensor * freq.cos() + rotated * freq.sin()).to(original)

        q, k = rope(q), rope(k)
        outputs = []
        offset = 0
        for length in lengths:
            sl = slice(offset, offset + length)
            scores = q[:, :, sl] @ k[:, :, sl].transpose(-1, -2) / dim ** 0.5
            outputs.append(scores.float().softmax(-1).to(v.dtype) @ v[:, :, sl])
            offset += length
        h = torch.cat(outputs, 2).transpose(1, 2).reshape_as(x)
        x = x + linear(h, base + "self_attn.proj")
        h = norm(x, base + "layer_norm2")
        x = x + linear(F.gelu(linear(h, base + "mlp.fc1")), base + "mlp.fc2")
    if c.use_head:
        x = norm(x, "layernorm_post")
    h = F.layer_norm(x, (c.hidden_size,), w[prefix + "merger.ln_q.weight"],
                     w[prefix + "merger.ln_q.bias"], c.layer_norm_eps)
    h = h.reshape(-1, c.hidden_size * c.spatial_merge_size ** 2)
    h = linear(F.gelu(linear(h, "merger.mlp.0")), "merger.mlp.2")
    if c.use_patch_position_encoding:
        pp = positions.reshape(-1, c.spatial_merge_size ** 2, 3)[:, 0] // c.spatial_merge_size
        h = h + w[prefix + "merger.pos_emb_h.weight"][pp[:, 1]] + w[prefix + "merger.pos_emb_w.weight"][pp[:, 2]]
    return h


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_load_ignores_all_hostile_checkpoint_maps(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.models.local import LocalModelLoader

    model = _model()
    model.save_pretrained(tmp_path)
    _processor_assets(tmp_path)
    marker = tmp_path / "checkpoint_executed"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n")
    for filename in ("config.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        config = json.loads((tmp_path / filename).read_text())
        config["auto_map"] = {key: ("hostile.Evil", "hostile.Evil") if key == "AutoTokenizer" else "hostile.Evil"
                              for key in ("AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText",
                                          "AutoProcessor", "AutoImageProcessor", "AutoVideoProcessor", "AutoTokenizer")}
        if filename == "config.json":
            config["text_config"]["auto_map"] = {"AutoModel": "hostile.Evil"}
        (tmp_path / filename).write_text(json.dumps(config))
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass is not None:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    assert isinstance(MageVLQModel.loader, LocalModelLoader)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is MageVLQModel
    assert type(loaded.model) is MageVLForConditionalGeneration
    assert type(loaded.model.config) is MageVLConfig
    assert type(loaded.processor) is MageVLProcessor
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    assert loaded.processor.video_processor.max_frames == 7
    assert loaded.trust_remote_code is False
    assert not marker.exists()
    inputs = _image_inputs(model)
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(**inputs).logits, model(**inputs).logits)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,bias", [(4, False), (2, True), (1, False)])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_multimodal_native_decoder_cache_and_safetensors(tmp_path, dtype, heads, bias, attention):
    from transformers import Qwen3ForCausalLM

    model = _model(text={"num_key_value_heads": heads, "attention_bias": bias,
                         "attn_implementation": attention}, vision={"attn_implementation": attention}).to(dtype)
    inputs = _image_inputs(model, frames=2)
    ids = inputs["input_ids"]
    native = Qwen3ForCausalLM(copy.deepcopy(model.config.text_config)).eval().to(dtype)
    native.model.load_state_dict(model.model.language_model.state_dict())
    native.lm_head.load_state_dict(model.lm_head.state_dict())
    tolerance = 4e-6 if dtype == torch.float32 else .005
    with torch.inference_mode():
        vision = _vision_oracle(model, inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
        actual_vision = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
        torch.testing.assert_close(actual_vision[0], vision, atol=tolerance, rtol=tolerance)
        embeds = native.get_input_embeddings()(ids)
        embeds[ids == model.config.image_token_id] = vision
        expected = native(inputs_embeds=embeds, use_cache=False).logits
        actual = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        prefix = dict(inputs, input_ids=ids[:, :-2])
        cache = model(**prefix, use_cache=True).past_key_values
        tail = model(ids[:, -2:], past_key_values=cache, attention_mask=torch.ones_like(ids), use_cache=True)
        torch.testing.assert_close(tail.logits, actual[:, -2:], atol=tolerance, rtol=tolerance)
        first = model(**prefix, use_cache=True).past_key_values
        no_mask = model(ids[:, -2:], past_key_values=first, use_cache=True)
        torch.testing.assert_close(no_mask.logits, actual[:, -2:], atol=tolerance, rtol=tolerance)
        model.save_pretrained(tmp_path)
        assert list(tmp_path.glob("*.safetensors"))
        restored = MageVLQModel.loader.from_pretrained(tmp_path, dtype=dtype)
        torch.testing.assert_close(restored(**inputs).logits, actual, atol=tolerance, rtol=tolerance)
        assert restored.state_dict().keys() == model.state_dict().keys()
        for key, tensor in model.state_dict().items():
            assert torch.equal(restored.state_dict()[key], tensor), key


def test_image_video_generation_causality_and_bad_visual_retry():
    model = _model()
    with torch.inference_mode():
        for video in (False, True):
            inputs = _image_inputs(model, frames=3, video=video)
            ids = inputs["input_ids"]
            actual = model(**inputs, use_cache=False).logits
            changed = dict(inputs, input_ids=ids.clone())
            changed["input_ids"][0, -1] = 7
            torch.testing.assert_close(model(**changed).logits[:, :-1], actual[:, :-1])
            visits = []
            handle = model.model.visual.register_forward_hook(lambda *args: visits.append(True))
            try:
                generated = model.generate(**inputs, max_new_tokens=3, do_sample=False, eos_token_id=None)
            finally:
                handle.remove()
            assert generated.shape == (1, ids.shape[1] + 3)
            assert len(visits) == 1
            greedy = ids.clone()
            for _ in range(3):
                logits = model(**dict(inputs, input_ids=greedy), use_cache=False).logits
                greedy = torch.cat((greedy, logits[:, -1].argmax(-1, keepdim=True)), -1)
            assert torch.equal(generated, greedy)
            cache = model(ids[:, :1], use_cache=True).past_key_values
            broken = dict(inputs, input_ids=ids[:, 1:].clone(), past_key_values=cache)
            broken["input_ids"][broken["input_ids"] == (61 if video else 60)] = 9
            with pytest.raises(ValueError, match="features and .*tokens do not match"):
                model(**broken)
            assert cache.get_seq_length() == 1
            repaired = model(**dict(inputs, input_ids=ids[:, 1:], past_key_values=cache)).logits
            torch.testing.assert_close(repaired, actual[:, 1:], atol=4e-6, rtol=2e-5)


def test_ragged_image_batch_beam_generation_matches_native_decoder(tmp_path):
    from transformers import Qwen3ForCausalLM

    _processor_assets(tmp_path)
    processor = MageVLProcessor.from_pretrained(tmp_path)
    processor.tokenizer.padding_side = "left"
    images = [Image.fromarray(np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3))
              for h, w in ((4, 4), (4, 8))]
    text = ["4 <|vision_start|><|image_pad|><|vision_end|> 9", "<|vision_start|><|image_pad|><|vision_end|> 7"]
    inputs = processor(text=text, images=images, padding=True)
    model = _model()
    native = Qwen3ForCausalLM(copy.deepcopy(model.config.text_config)).eval()
    native.model.load_state_dict(model.model.language_model.state_dict())
    native.lm_head.load_state_dict(model.lm_head.state_dict())
    with torch.no_grad():
        embeds = native.get_input_embeddings()(inputs["input_ids"])
        vision = _vision_oracle(model, inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
        embeds[inputs["input_ids"] == 60] = vision
        expected = native.generate(input_ids=inputs["input_ids"], inputs_embeds=embeds,
                                   attention_mask=inputs["attention_mask"], num_beams=3,
                                   max_new_tokens=3, do_sample=False, eos_token_id=None)
        actual = model.generate(**inputs, num_beams=3, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert torch.equal(actual, expected)


def test_explicit_implicit_grid_rope_and_numpy_frame_retry(tmp_path):
    model = _model()
    inputs = _image_inputs(model, frames=5)
    with torch.no_grad():
        explicit = model(**inputs).logits
        implicit = model(**{k: v for k, v in inputs.items() if k != "patch_positions"}).logits
    torch.testing.assert_close(explicit, implicit)
    _processor_assets(tmp_path)
    p = MageVLProcessor.from_pretrained(tmp_path)
    with pytest.raises(TypeError, match="Expected file path"):
        p(text="<|vision_start|><|video_pad|><|vision_end|>", videos=[[]])
    frames = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
    fixed = p(text="<|vision_start|><|video_pad|><|vision_end|>", videos=[list(frames)])
    assert fixed["input_ids"].eq(60).sum() == 3
    assert fixed["patch_positions"][::4, 0].tolist() == [0, 1, 2]
    with torch.no_grad():
        assert torch.isfinite(model(**fixed).logits).all()


def test_vision_encoder_optional_outputs_keep_native_tuple_contract():
    model = _model()
    encoder = model.model.visual.encoder
    hidden = torch.randn(1, 4, model.config.vision_config.hidden_size)
    with torch.no_grad():
        baseline = encoder(hidden, return_dict=True)
        assert baseline.hidden_states is None
        assert baseline.attentions is None
        assert len(encoder(hidden, return_dict=False)) == 1
        output = encoder(hidden, output_attentions=True, output_hidden_states=True, return_dict=True)
        assert len(output.hidden_states) == 3
        assert len(output.attentions) == 2
        assert output.attentions == (None, None)  # The pinned encoder does not expose attention weights.
        torch.testing.assert_close(output.last_hidden_state, baseline.last_hidden_state)


@pytest.mark.parametrize("norm,head,position", [("layer_norm", False, False), ("rms_norm", True, True)])
def test_vision_variants_and_window_isolation(norm, head, position):
    model = _model(vision={"layer_norm_type": norm, "use_head": head, "use_patch_position_encoding": position,
                            "frame_windows_size": 2})
    inputs = _image_inputs(model, frames=5)
    with torch.inference_mode():
        expected = _vision_oracle(model, inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
        actual = model.model.visual(inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"]).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
        perturbed = inputs["pixel_values"].clone()
        perturbed[-4:] += 2
        other = model.model.visual(perturbed, inputs["image_grid_thw"], inputs["patch_positions"]).last_hidden_state
        torch.testing.assert_close(actual[:-1], other[:-1])
        assert not torch.allclose(actual[-1], other[-1])


def test_processor_images_frame_video_prompt_order_save_reload_and_retry(tmp_path):
    _processor_assets(tmp_path)
    processor = MageVLProcessor.from_pretrained(tmp_path)
    image = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    frames = [image, Image.fromarray(np.full((4, 4, 3), 210, dtype=np.uint8))]
    text = "<|vision_start|><|video_pad|><|vision_end|> 4 <|vision_start|><|image_pad|><|vision_end|>"
    inputs = processor(text=text, images=[image], videos=[frames], num_frames=2)
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
    assert inputs["patch_positions"].tolist() == [[t, h, w] for t in (0, 1, 0) for h in range(2) for w in range(2)]
    assert inputs["input_ids"].eq(60).sum() == 3
    expected_video = processor.video_processor(videos=[frames])
    torch.testing.assert_close(inputs["pixel_values"][:8], expected_video["pixel_values_videos"])
    assert processor.video_processor.fixed_num_frames is None
    with pytest.raises(ValueError, match="placeholders"):
        processor(text="4", images=[image])
    repaired = processor(text=text, images=[image], videos=[frames])
    torch.testing.assert_close(repaired["pixel_values"], inputs["pixel_values"])
    saved = tmp_path / "saved"
    processor.save_pretrained(saved)
    reloaded = MageVLProcessor.from_pretrained(saved)
    after = reloaded(text=text, images=[image], videos=[frames])
    for key in inputs:
        assert torch.equal(inputs[key], after[key]), key
    assert reloaded.video_processor.max_frames == 7
    model = _model()
    with torch.no_grad():
        assert torch.isfinite(model(**after).logits).all()


def test_neural_codec_rejects_before_subprocess_and_frame_retry(tmp_path, monkeypatch):
    import subprocess

    from gptqmodel.models.ports.mage_vl.codec_video_processing_mage_vl import CodecConfig, process_codec_video

    def forbidden(*args, **kwargs):
        pytest.fail("Neural DCVC may not execute checkpoint scripts")
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(NotImplementedError, match="No checkpoint Python"):
        process_codec_video("video.mp4", CodecConfig(engine="dcvc-rt", cache_root=tmp_path))
    assert not list(tmp_path.iterdir())
    _processor_assets(tmp_path)
    p = MageVLProcessor.from_pretrained(tmp_path)
    image = Image.new("RGB", (4, 4), color=(20, 100, 200))
    inputs = p(text="<|vision_start|><|video_pad|><|vision_end|>", videos=[[image]])
    assert inputs["input_ids"].eq(60).sum() == 1


def test_traditional_codec_missing_dependency_and_failed_job_retry(tmp_path, monkeypatch):
    from gptqmodel.models.ports.mage_vl import codec_video_processing_mage_vl as codec

    bindir = tmp_path / "bin"
    bindir.mkdir()
    cache = tmp_path / "cache"
    config = codec.CodecConfig(cache_root=str(cache))
    monkeypatch.setenv("PATH", str(bindir))
    with pytest.raises(RuntimeError, match="installed codec-video-prep.*cv-preinfer"):
        codec.process_codec_video("video.mp4", config)
    assert not cache.exists()
    # This is an executable ABI/lifecycle fixture, NOT a real codec oracle.
    binary = bindir / "cv-preinfer"
    binary.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('fixture failure')\nsys.exit(9)\n")
    binary.chmod(0o755)
    monkeypatch.setattr(codec, "_get_video_total_frames", lambda _: 256)
    def broken_metadata(_):
        raise ValueError("injected metadata failure")

    monkeypatch.setattr(codec, "_get_video_total_frames", broken_metadata)
    with pytest.raises(ValueError, match="injected metadata failure"):
        codec.process_codec_video("video.mp4", config)
    assert not list(cache.glob(".tmp_*")) and not list(cache.glob("*/meta.json"))
    monkeypatch.setattr(codec, "_get_video_total_frames", lambda _: 256)
    with pytest.raises(RuntimeError, match="rc=9.*fixture failure"):
        codec.process_codec_video("video.mp4", config)
    assert not list(cache.glob(".tmp_*")) and not list(cache.glob("*/meta.json"))
    binary.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nimport numpy as np\n"
        "p=Path(sys.argv[sys.argv.index('--out_dir')+1])\n"
        "np.save(p/'canvas_0.npy', np.full((4,4,3), 70, dtype=np.uint8))\n"
        "np.save(p/'src_patch_position.npy', np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]]))\n"
        "(p/'meta.json').write_text(json.dumps({'canvas_files':['canvas_0.npy'],'fps':2.0}))\n"
    )
    payload = codec.process_codec_video("video.mp4", config)
    assert np.asarray(payload["images"][0]).mean() == 70
    assert payload["src_positions"].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    binary.unlink()
    replay = codec.process_codec_video("video.mp4", config)
    assert replay["out_dir"] == payload["out_dir"]


def test_checkpoint_codec_settings_only_supply_processing_options(tmp_path):
    _processor_assets(tmp_path)
    config_path = tmp_path / "preprocessor_config.json"
    payload = json.loads(config_path.read_text())
    payload["codec"] = {
        "target_canvas": 24,
        "cache_root": str(tmp_path / "checkpoint-cache"),
        "timeout_seconds": 1,
    }
    config_path.write_text(json.dumps(payload))

    processor = MageVLProcessor.from_pretrained(tmp_path)
    assert processor._codec_config_defaults == {"target_canvas": 24}

    caller_cache = tmp_path / "caller-cache"
    processor = MageVLProcessor.from_pretrained(
        tmp_path, codec_config={"target_canvas": 20, "cache_root": caller_cache, "timeout_seconds": 15},
    )
    assert processor._codec_config_defaults["cache_root"] == caller_cache
    assert processor._codec_config_defaults["timeout_seconds"] == 15
    # Caller execution policy layers over, rather than discards, checkpoint geometry.
    processor = MageVLProcessor.from_pretrained(tmp_path, codec_config={"cache_root": caller_cache})
    assert processor._codec_config_defaults == {"target_canvas": 24, "cache_root": caller_cache}
    saved = tmp_path / "saved"
    processor.save_pretrained(saved)
    assert json.loads((saved / "preprocessor_config.json").read_text())["codec"] == {"target_canvas": 24}


def test_codec_defaults_ignore_environment_and_accept_explicit_api_paths(tmp_path, monkeypatch):
    from gptqmodel.models.ports.mage_vl import codec_video_processing_mage_vl as codec

    baseline = codec.CodecConfig()
    # These variables must never select a different runtime configuration.
    for key in ("DCVC_INTRA_TAR", "DCVC_INTER_TAR", "ONLINE_CODEC_CACHE_DIR", "HF_HOME"):
        monkeypatch.setenv(key, str(tmp_path / key))
    actual = codec.CodecConfig()
    assert actual == baseline
    explicit = codec.CodecConfig(
        cache_root=str(tmp_path / "explicit"),
        dcvc={"intra_ckpt": "explicit-intra", "inter_ckpt": "explicit-inter"},
    )
    assert explicit.cache_root == tmp_path / "explicit"
    assert explicit.dcvc.intra_ckpt == "explicit-intra"
    assert explicit.dcvc.inter_ckpt == "explicit-inter"


def test_streammind_missing_dependency_is_lazy_and_transactional(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    model = _model()
    inputs = _image_inputs(model)
    model.model._streammind_model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="local safetensors asset is missing"):
        model.streammind_gate_forward(inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
    assert model.model.streammind_gate is None
    save_file({"placeholder": torch.ones(1)}, tmp_path / "streammind_gate.safetensors")
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ImportError, match="installed mamba-ssm"):
        model.streammind_gate_forward(inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"])
    assert model.model.streammind_gate is None
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    with torch.no_grad():
        assert torch.isfinite(model(**inputs).logits).all()


@pytest.mark.cpu
@pytest.mark.parametrize("kv,bias", [(4, False), (2, True), (1, False)])
def test_plain_awq_all_decoder_linears_pack_and_two_safetensors_roundtrips(tmp_path, kv, bias):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = _model(text={"num_key_value_heads": kv, "attention_bias": bias})
    model.save_pretrained(tmp_path)
    _processor_assets(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                             backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.model.language_model.layers[0], {}, {}) == []
    dense = {k: v.to(torch.float16) for k, v in model.state_dict().items() if k.startswith("model.visual.")}
    image = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    calibration = [[{"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": "4 5 6 7 8 9 10 11"}]}]
                   for _ in range(2)]
    prepared = wrapper.prepare_dataset(calibration, batch_size=1, calibration_data_min_length=1)
    model = model.half()
    with torch.no_grad():
        first = prepared[0]
        vision = _vision_oracle(model, first["pixel_values"].half(), first["image_grid_thw"], first["patch_positions"])
        expected_capture = model.model.get_input_embeddings()(first["input_ids"])
        expected_capture[first["input_ids"] == 60] = vision
    captured = []

    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    handle = wrapper.model.model.language_model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        wrapper.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    finally:
        handle.remove()
    assert len(captured) >= 2
    for actual in captured[:2]:
        torch.testing.assert_close(actual, expected_capture, atol=.003, rtol=.003)
    assert sum(isinstance(m, AwqTorchLinear) for m in wrapper.model.modules()) == 14
    assert all(torch.equal(wrapper.model.state_dict()[k], v) for k, v in dense.items())
    inputs = _image_inputs(wrapper.model, frames=2)
    with torch.no_grad():
        expected = wrapper.model(**inputs, use_cache=False).logits
    packed = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
    for iteration in range(2):
        path = tmp_path / f"quantized-{iteration}"
        wrapper.save_quantized(str(path))
        wrapper = GPTQModel.load(str(path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        assert sum(isinstance(m, AwqTorchLinear) for m in wrapper.model.modules()) == 14
        for k, v in packed.items():
            assert torch.equal(wrapper.model.state_dict()[k], v), k
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(**inputs, use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
