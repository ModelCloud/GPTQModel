# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Behavioral qualification for the owned LocateAnything port."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.locateanything.configuration_locateanything import LocateAnythingConfig
from gptqmodel.models.ports.locateanything.generate_utils import decode_bbox_avg, handle_pattern
from gptqmodel.models.ports.locateanything.image_processing_locateanything import LocateAnythingImageProcessor
from gptqmodel.models.ports.locateanything.modeling_locateanything import LocateAnythingForConditionalGeneration
from gptqmodel.models.ports.locateanything.processing_locateanything import LocateAnythingProcessor


PORT = Path(__file__).parents[1] / "gptqmodel/models/ports/locateanything"


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(architecture="Qwen2ForCausalLM", kv_heads=2, implementation="sdpa", tied=False, bias=False):
    return LocateAnythingConfig(
        text_config={"architectures": [architecture], "model_type": "qwen3" if "3" in architecture else "qwen2",
                     "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": kv_heads, "head_dim": 8,
                     "max_position_embeddings": 128, "rope_theta": 1000000., "block_size": 6, "attention_bias": bias,
                     "text_mask_token_id": 60, "null_token_id": 61, "switch_token_id": 62,
                     "eos_token_id": 2, "pad_token_id": 0, "tie_word_embeddings": tied},
        vision_config={"model_type": "moonvit", "hidden_size": 16, "num_attention_heads": 2,
                       "num_hidden_layers": 1, "intermediate_size": 24, "patch_size": 2,
                       "init_pos_emb_height": 4, "init_pos_emb_width": 4},
        image_token_index=59, box_start_token_id=32, box_end_token_id=33, ref_start_token_id=34,
        ref_end_token_id=35, none_token_id=36, coord_start_token_id=16, coord_end_token_id=31,
        attn_implementation=implementation,
    )


def tiny_model(**kwargs):
    torch.manual_seed(732)
    return LocateAnythingForConditionalGeneration(tiny_config(**kwargs)).eval()


def write_tokenizer(path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{str(i): i for i in range(3, 64)}}
    vocab.pop("59")
    vocab["<IMG_CONTEXT>"] = 59
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
                                 pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tok.add_special_tokens({"additional_special_tokens": ["<IMG_CONTEXT>"]})
    tok.save_pretrained(path)
    return tok


def media():
    # Two independently packed images with different grids and interpolation.
    return torch.linspace(-1, 1, 20 * 3 * 2 * 2).reshape(20, 3, 2, 2), torch.tensor([[2, 2], [4, 4]])


def rms(x, weight, eps):
    fp = x.float()
    return weight * (fp * torch.rsqrt(fp.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def decoder_oracle(model, ids, visible, positions=None, visual=None):
    """Independent checkpoint equations: no model/helper forward calls."""
    config = model.config.text_config
    w = model.language_model.state_dict()
    x = F.embedding(ids, w["model.embed_tokens.weight"])
    if visual is not None:
        x[ids == model.image_token_index] = visual.to(x)
    batch, length, width = x.shape
    head_dim = width // config.num_attention_heads
    positions = torch.arange(length)[None].expand(batch, -1) if positions is None else positions
    inv = config.rope_parameters["rope_theta"] ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    phase = positions.float()[..., None] * inv
    phase = torch.cat((phase, phase), -1)[:, None]
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        h = rms(x, w[prefix + "input_layernorm.weight"], config.rms_norm_eps)
        q, k, v = [F.linear(h, w[prefix + f"self_attn.{name}_proj.weight"],
                            w.get(prefix + f"self_attn.{name}_proj.bias"))
                   for name in ("q", "k", "v")]
        q = q.reshape(batch, length, config.num_attention_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, length, config.num_key_value_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, length, config.num_key_value_heads, head_dim).transpose(1, 2)
        if config.model_type == "qwen3":
            q = rms(q, w[prefix + "self_attn.q_norm.weight"], config.rms_norm_eps)
            k = rms(k, w[prefix + "self_attn.k_norm.weight"], config.rms_norm_eps)
        q = q * phase.cos() + torch.cat((-q[..., head_dim // 2:], q[..., :head_dim // 2]), -1) * phase.sin()
        k = k * phase.cos() + torch.cat((-k[..., head_dim // 2:], k[..., :head_dim // 2]), -1) * phase.sin()
        k = k.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, 1)
        v = v.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, 1)
        scores = (q.double() @ k.double().transpose(-1, -2)) / head_dim ** .5
        scores.masked_fill_(~visible[:, None], -float("inf"))
        out = (scores.softmax(-1) @ v.double()).to(x.dtype).transpose(1, 2).reshape(batch, length, width)
        x = x + F.linear(out, w[prefix + "self_attn.o_proj.weight"], w.get(prefix + "self_attn.o_proj.bias"))
        h = rms(x, w[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps)
        gate = F.silu(F.linear(h, w[prefix + "mlp.gate_proj.weight"]))
        up = F.linear(h, w[prefix + "mlp.up_proj.weight"])
        x = x + F.linear(gate * up, w[prefix + "mlp.down_proj.weight"])
    return F.linear(rms(x, w["model.norm.weight"], config.rms_norm_eps), w["lm_head.weight"]).float()


def vision_oracle(model, pixels, grids):
    """Independent MoonViT equations with per-image attention, 2D complex RoPE and merging."""
    w = model.vision_model.state_dict()
    x = F.conv2d(pixels, w["patch_embed.proj.weight"], w["patch_embed.proj.bias"], stride=2).flatten(1)
    output, offset = [], 0
    for height, width in grids.tolist():
        length = height * width
        pos = F.interpolate(w["patch_embed.pos_emb.weight"].permute(2, 0, 1)[None],
                            size=(height, width), mode="bicubic")[0].permute(1, 2, 0).reshape(length, -1)
        image = x[offset:offset + length] + pos
        for layer in range(model.config.vision_config.num_hidden_layers):
            p = f"encoder.blocks.{layer}."
            h = F.layer_norm(image, (16,), w[p + "norm0.weight"], w[p + "norm0.bias"], 1e-5)
            q, k, v = F.linear(h, w[p + "wqkv.weight"], w[p + "wqkv.bias"]).reshape(length, 3, 2, 8).unbind(1)
            freqs = []
            for y in range(height):
                for col in range(width):
                    freqs.append([col, y, col / 100., y / 100.])
            angles = torch.tensor(freqs)
            rotation = torch.polar(torch.ones_like(angles), angles)[:, None]
            q = torch.view_as_real(torch.view_as_complex(q.float().reshape(length, 2, 4, 2)) * rotation).flatten(-2)
            k = torch.view_as_real(torch.view_as_complex(k.float().reshape(length, 2, 4, 2)) * rotation).flatten(-2)
            weights = (q.transpose(0, 1).double() @ k.transpose(0, 1).double().transpose(-1, -2) / 8 ** .5).softmax(-1)
            h = (weights @ v.transpose(0, 1).double()).to(x.dtype).transpose(0, 1).reshape(length, 16)
            image = image + F.linear(h, w[p + "wo.weight"], w[p + "wo.bias"])
            h = F.layer_norm(image, (16,), w[p + "norm1.weight"], w[p + "norm1.bias"], 1e-5)
            h = F.gelu(F.linear(h, w[p + "mlp.fc0.weight"], w[p + "mlp.fc0.bias"]), approximate="tanh")
            image = image + F.linear(h, w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"])
        image = F.layer_norm(image, (16,), w["encoder.final_layernorm.weight"], w["encoder.final_layernorm.bias"], 1e-5)
        image = image.reshape(height // 2, 2, width // 2, 2, 16).permute(0, 2, 1, 3, 4).reshape(-1, 64)
        output.append(image)
        offset += length
    return output


@pytest.mark.parametrize("implementation", ["sdpa", "eager"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_decoder_matches_independent_equations(implementation, kv_heads, mtp):
    model = tiny_model(kv_heads=kv_heads, implementation=implementation)
    ids = torch.tensor([[1, 4, 5, 6, 60, 60, 60, 60, 60]]) if mtp else torch.tensor([[1, 7, 9, 11, 4]])
    length = ids.shape[1]
    visible = torch.ones(1, length, length, dtype=torch.bool).tril()
    if mtp:
        visible[:, -6:, -6:] = True
        visible[:, -6:, -7] = False
    expected = decoder_oracle(model, ids, visible)
    actual = model.language_model(ids, use_cache=mtp).logits
    torch.testing.assert_close(actual, expected, atol=6e-7, rtol=2e-5)
    if mtp:
        causal = decoder_oracle(model, ids, torch.ones_like(visible).tril())
        assert (actual - causal).abs().max() > 1e-3


@pytest.mark.parametrize("implementation", ["sdpa", "eager"])
def test_vision_and_image_injection_match_independent_equations(implementation):
    model = tiny_model(implementation=implementation)
    pixels, grids = media()
    expected_features = vision_oracle(model, pixels, grids)
    actual_features = model.extract_feature(pixels, grids)
    for actual, expected in zip(actual_features, expected_features, strict=True):
        torch.testing.assert_close(actual, expected, atol=7e-7, rtol=2e-5)
    ids = torch.tensor([[1, 59, 59, 59, 59, 59, 8]])
    features = model.mlp1(torch.cat(expected_features))
    expected = decoder_oracle(model, ids, torch.ones(1, 7, 7, dtype=torch.bool).tril(), visual=features)
    actual = model(input_ids=ids, pixel_values=pixels, image_grid_hws=grids).logits
    torch.testing.assert_close(actual, expected, atol=8e-7, rtol=2e-5)
    assert (actual - model(input_ids=ids).logits).abs().max() > 1e-3


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_cache_continuation_generation_and_safetensors_retry(tmp_path, architecture, dtype, tied):
    model = tiny_model(architecture=architecture, tied=tied).to(dtype)
    ids = torch.tensor([[1, 5, 9, 11, 6]])
    prefix = model(input_ids=ids[:, :3], use_cache=True)
    assert isinstance(prefix.past_key_values, DynamicCache)
    actual = model(input_ids=ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    expected = model(input_ids=ids, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits[:, 3:], atol=.003, rtol=.01)
    assert prefix.past_key_values.get_seq_length() == 5
    greedy = model.language_model.generate(ids, max_new_tokens=3, do_sample=False)
    assert greedy.shape[1] > ids.shape[1]
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    # Fail after the config exists, then repair and retry the same path.
    config_path = tmp_path / "config.json"
    original = config_path.read_text()
    payload = json.loads(original)
    payload["text_config"]["architectures"] = ["ForeignDecoder"]
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported LocateAnything decoder"):
        LocateAnythingQModel.loader.from_pretrained(tmp_path)
    config_path.write_text(original)
    loaded = LocateAnythingQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    assert (loaded.language_model.lm_head.weight is loaded.language_model.model.embed_tokens.weight) is tied
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert loaded.state_dict()[key].dtype == value.dtype
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    tolerance = 0 if dtype == torch.float32 else .003
    torch.testing.assert_close(loaded(input_ids=ids).logits, expected.logits, atol=tolerance, rtol=tolerance)
    loaded.save_pretrained(tmp_path / "second")
    again = LocateAnythingQModel.loader.from_pretrained(tmp_path / "second", dtype=dtype).eval()
    torch.testing.assert_close(again(input_ids=ids).logits, expected.logits, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_owned_model_and_processor_ignore_hostile_maps(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = tiny_model()
    model.save_pretrained(tmp_path)
    tok = write_tokenizer(tmp_path)
    LocateAnythingProcessor(LocateAnythingImageProcessor(patch_size=2), tok).save_pretrained(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('executed')\n")
    for name in ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
        file = tmp_path / name
        payload = json.loads(file.read_text())
        payload["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModel": "hostile.Model",
                               "AutoModelForCausalLM": "hostile.Model", "AutoProcessor": "hostile.Processor",
                               "AutoImageProcessor": "hostile.ImageProcessor", "AutoTokenizer": ["hostile.Tokenizer", None]}
        file.write_text(json.dumps(payload))
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is LocateAnythingQModel
    assert type(loaded.model) is LocateAnythingForConditionalGeneration
    assert loaded.trust_remote_code is False
    processor = loaded.load_processor()
    assert type(processor) is LocateAnythingProcessor
    assert type(processor.image_processor) is LocateAnythingImageProcessor
    assert not marker.exists()
    ids = torch.tensor([[1, 7, 9]])
    torch.testing.assert_close(loaded(input_ids=ids).logits, model(input_ids=ids).logits, atol=0, rtol=0)


def test_processor_patches_video_and_reload(tmp_path):
    tok = write_tokenizer(tmp_path)
    proc = LocateAnythingProcessor(LocateAnythingImageProcessor(patch_size=2), tok)
    img = Image.new("RGB", (4, 4), (64, 128, 255))
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                 {"type": "video", "video": [img, img]}, {"type": "text", "text": "Find object"}]}]
    text = proc.py_apply_chat_template(messages, add_generation_prompt=True)
    assert "<image-1><video-1>Find object" in text
    images, videos = proc.process_vision_info(messages)
    batch = proc(text=[text], images=images, videos=videos, return_tensors="pt")
    assert batch["pixel_values"].shape == (12, 3, 2, 2)
    assert batch["image_grid_hws"].tolist() == [[2, 2]] * 3
    torch.testing.assert_close(batch["pixel_values"][0, :, 0, 0], torch.tensor([64 / 127.5 - 1, 128 / 127.5 - 1, 1.]))
    expanded, _, _, image_count, video_count = proc.replace_media_placeholder(
        "<image-1><video-1>", images, videos, None, None, images_kwargs={}, videos_kwargs={})
    assert expanded.count("<IMG_CONTEXT>") == 3 and image_count == video_count == 1
    assert "Frame-1:" in expanded and "Frame-2:" in expanded
    proc.save_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    reloaded = LocateAnythingQModel.loader.load_processor(tmp_path)
    actual = reloaded(text=[text], images=images, videos=videos, return_tensors="pt")
    for key in batch:
        torch.testing.assert_close(actual[key], batch[key])
    with pytest.raises((ValueError, IndexError)):
        reloaded(text=["<image-1>"], images=[])
    torch.testing.assert_close(reloaded(text=[text], images=images, videos=videos, return_tensors="pt")["pixel_values"],
                               batch["pixel_values"])


def test_localization_sampler_patterns_and_weighted_coordinate_selection():
    model = tiny_model()
    token_ids = model.token_ids
    logits = torch.full((6, 64), -20.)
    expected = [32, 17, 20, 25, 29, 33]
    logits[torch.arange(6), expected] = 20
    for mode in ("fast", "hybrid"):
        decoded = decode_bbox_avg(logits, logits.softmax(-1), token_ids, generation_mode=mode)
        assert decoded.tolist() == expected
        assert handle_pattern(decoded, token_ids, mode)["type"] == "coord_box"
    assert handle_pattern(torch.tensor([32, 17, 20, 33, 61, 61]), token_ids)["type"] == "point_box"
    assert handle_pattern(torch.tensor([32, 17, 8, 9, 10, 33]), token_ids)["type"] == "error_box"
    assert handle_pattern(torch.tensor([61] * 6), token_ids)["tokens"] == [2]
    assert handle_pattern(torch.tensor([32, 36, 33, 61, 61, 61]), token_ids)["tokens"] == [32, 36, 33]


@pytest.mark.parametrize("mode", ["slow", "fast", "hybrid"])
@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
def test_image_localization_generation_runs_real_decoder(mode, architecture):
    model = tiny_model(architecture=architecture)
    pixels, grids = media()
    ids = torch.tensor([[1, 59, 59, 59, 59, 59, 8]])
    tokenizer = SimpleNamespace(model_max_length=64,
                                batch_decode=lambda rows, **kw: [','.join(map(str, row.tolist())) for row in rows])
    response = model.generate(input_ids=ids, pixel_values=pixels, image_grid_hws=grids,
                              tokenizer=tokenizer, use_cache=True, max_new_tokens=8, generation_mode=mode)
    assert isinstance(response, str) and response


def test_visual_failure_does_not_mutate_cache_and_retry_matches_fresh():
    model = tiny_model()
    cache = DynamicCache()
    ids = torch.tensor([[1, 59, 4]])
    with pytest.raises(ValueError, match="placeholder count"):
        model.language_model(ids, visual_features=torch.ones(2, 32), image_token_index=59,
                             past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0
    features = torch.linspace(-1, 1, 32).reshape(1, 32)
    actual = model.language_model(ids, visual_features=features, image_token_index=59, past_key_values=cache,
                                  use_cache=True)
    expected = model.language_model(ids, visual_features=features, image_token_index=59, use_cache=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)


def test_dummy_images_and_flagged_packed_images_preserve_upstream_selection():
    model = tiny_model()
    pixels, grids = media()
    ids = torch.tensor([[1, 5, 7]])
    torch.testing.assert_close(model(input_ids=ids, pixel_values=pixels, image_grid_hws=grids,
                                    image_flags=torch.tensor([0, 0])).logits,
                               model(input_ids=ids).logits, atol=0, rtol=0)
    ids = torch.tensor([[1, 59, 59, 59, 59, 8]])
    actual = model(input_ids=ids, pixel_values=pixels, image_grid_hws=grids, image_flags=torch.tensor([0, 1])).logits
    expected = model(input_ids=ids, pixel_values=pixels[4:], image_grid_hws=grids[1:]).logits
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)


def test_explicit_attention_and_cache_contracts_fail_before_mutating_input():
    assert tiny_config(implementation="magi")._attn_implementation == "sdpa"
    with pytest.raises(ValueError, match="supports eager or sdpa"):
        tiny_model(implementation="flash_attention_2")
    model = tiny_model()
    with pytest.raises(ValueError, match="DynamicCache only"):
        model.language_model(torch.tensor([[1, 4]]), past_key_values=(), use_cache=True)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_qwen3_native_variant_matches_independent_qk_norm_equations(bias, kv_heads):
    model = tiny_model(architecture="Qwen3ForCausalLM", bias=bias, kv_heads=kv_heads)
    ids = torch.tensor([[1, 5, 8, 11, 7]])
    expected = decoder_oracle(model, ids, torch.ones(1, 5, 5, dtype=torch.bool).tril())
    torch.testing.assert_close(model(input_ids=ids).logits, expected, atol=7e-7, rtol=2e-5)


def test_mtp_cache_keeps_only_real_prefix_across_two_speculative_windows():
    model = tiny_model()
    accepted = torch.tensor([[1, 4, 6]])
    cache = model.language_model(accepted, use_cache=True).past_key_values
    for new_tokens in ([], [32, 17, 20, 25, 29, 33]):
        if new_tokens:
            accepted = torch.cat((accepted, torch.tensor([new_tokens])), 1)
        speculative = torch.cat((accepted, accepted[:, -1:], torch.full((1, 5), 60)), 1)
        positions = torch.arange(speculative.shape[1])[None]
        positions[:, -6:] -= 1
        length, seen = speculative.shape[1], cache.get_seq_length()
        visible = torch.ones(1, length, length, dtype=torch.bool).tril()
        visible[:, -6:, -6:] = True
        visible[:, -6:, -7] = False
        expected = decoder_oracle(model, speculative, visible, positions=positions)
        actual = model.language_model(speculative[:, seen:], past_key_values=cache, position_ids=positions[:, seen:],
                                       use_cache=True)
        torch.testing.assert_close(actual.logits, expected[:, seen:], atol=6e-7, rtol=2e-5)
        cache.crop(-6)
        assert cache.get_seq_length() == accepted.shape[1]


@pytest.mark.parametrize("causal", [False, True])
def test_training_block_visibility_and_logits_have_independent_scalar_oracle(causal):
    model = tiny_model().train()
    model.config.text_config.causal_attn = causal
    ids = torch.tensor([[1, 4, 6, 9, 11, 3] + [60] * 12])
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8]])
    visible = torch.zeros(1, 18, 18, dtype=torch.bool)
    for row in range(18):
        for col in range(18):
            if row < 6:
                visible[0, row, col] = col <= row
            elif col < 6:
                visible[0, row, col] = col < positions[0, 6 + ((row - 6) // 6) * 6]
            else:
                visible[0, row, col] = (row - 6) // 6 == (col - 6) // 6 and (not causal or col <= row)
    expected = decoder_oracle(model, ids, visible, positions=positions)
    actual = model(input_ids=ids, position_ids=positions, use_cache=False, labels=ids)
    torch.testing.assert_close(actual.logits, expected, atol=6e-7, rtol=2e-5)
    torch.testing.assert_close(actual.loss, F.cross_entropy(expected[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)))


@pytest.mark.parametrize("causal", [False, True])
def test_uncached_multiple_diffusion_windows_obey_visible_real_prefix(causal):
    model = tiny_model()
    model.config.text_config.causal_attn = causal
    ids = torch.tensor([[1, 4, 6, 9] + [60] * 5 + [8, 9, 10] + [60] * 5])
    visible = torch.ones(1, 17, 17, dtype=torch.bool).tril()
    # Independent sequence semantics: later real tokens cannot see speculative
    # tokens (including the repeated predecessor); a window sees its peers.
    for start, end in ((3, 9), (11, 17)):
        visible[:, end:, start:end] = False
        if not causal:
            visible[:, start:end, start:end] = True
    expected = decoder_oracle(model, ids, visible)
    torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, expected, atol=6e-7, rtol=2e-5)


def test_sampler_hybrid_rejects_ambiguous_coordinates_while_fast_uses_top_one():
    token_ids = {**tiny_model().token_ids, "coord_start_token_id": 48, "coord_end_token_id": 148}
    probs = torch.full((6, 160), 1e-10)
    probs[0, 32] = probs[5, 33] = 1
    probs[1:5, 60] = .55
    probs[1:5, 140] = .45
    assert decode_bbox_avg(probs.log(), probs, token_ids, generation_mode="fast").tolist() == [32, 60, 60, 60, 60, 33]
    assert decode_bbox_avg(probs.log(), probs, token_ids, generation_mode="hybrid").tolist() == [32, 0, 0, 0, 0, 33]


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
@pytest.mark.parametrize("mode", ["slow", "fast", "hybrid"])
def test_sampler_repeated_box_prefix_exercises_real_multiround_cache(architecture, mode):
    model = tiny_model(architecture=architecture)
    # A real deterministic checkpoint: zero residual paths, a nonzero embedding
    # coordinate, and a head selecting <box>. No forward or cache is mocked.
    with torch.no_grad():
        for name, parameter in model.language_model.named_parameters():
            if "norm" not in name:
                parameter.zero_()
        model.language_model.model.embed_tokens.weight[:, 0] = 1
        model.language_model.lm_head.weight[32, 0] = 1
    calls = []
    hook = model.language_model.register_forward_hook(
        lambda module, args, output: calls.append(output.past_key_values.get_seq_length()))
    tokenizer = SimpleNamespace(model_max_length=64,
                                batch_decode=lambda rows, **kw: [','.join(map(str, row.tolist())) for row in rows])
    try:
        result = model.generate(input_ids=torch.tensor([[1, 4, 6]]), tokenizer=tokenizer,
                                 use_cache=True, max_new_tokens=8, generation_mode=mode)
    finally:
        hook.remove()
    # Fast commits a whole 6-token block; hybrid falls back after the first
    # malformed box prefix, then exits AR when a non-coordinate is produced.
    assert result == ','.join(["32"] * {"slow": 8, "fast": 12, "hybrid": 2}[mode])
    assert len(calls) == {"slow": 8, "fast": 2, "hybrid": 2}[mode]


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
def test_public_plain_awq_media_replay_pack_infer_and_reload(tmp_path, architecture):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = tiny_model(architecture=architecture)
    model.save_pretrained(tmp_path)
    tokenizer = write_tokenizer(tmp_path)
    LocateAnythingProcessor(LocateAnythingImageProcessor(patch_size=2), tokenizer).save_pretrained(tmp_path)
    cfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=cfg, backend=BACKEND.TORCH, dtype=torch.float16)
    # Quantization shells are lazy/meta until capture. The saved source is the
    # independent oracle for every dense tensor after materialization.
    dense = {key: value.half().clone() for key, value in model.state_dict().items()
             if not (key.startswith("language_model.model.layers.") and "norm" not in key)}
    image = Image.new("RGB", (4, 4), (100, 160, 220))
    calibration = [[{"role": "user", "content": [{"type": "image", "image": image},
                     {"type": "text", "text": "Locate object"}]},
                    {"role": "assistant", "content": "<box>1 2 3 4</box>"}]] * 2
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 14
    for key, value in dense.items():
        torch.testing.assert_close(wrapper.model.state_dict()[key], value, atol=0, rtol=0)
    processor = wrapper.load_processor()
    example = processor(text=["<image-1>Locate"], images=[image], return_tensors="pt")
    example["pixel_values"] = example["pixel_values"].half()
    with torch.inference_mode():
        expected = wrapper.model(**example).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 14
    saved_processor = reloaded.load_processor()
    assert saved_processor.image_processor.patch_size == 2
    restored_media = saved_processor(text=["<image-1>Locate"], images=[image], return_tensors="pt")
    torch.testing.assert_close(restored_media["input_ids"], example["input_ids"], atol=0, rtol=0)
    for key, value in dense.items():
        torch.testing.assert_close(reloaded.model.state_dict()[key], value, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**example).logits, expected, atol=.002, rtol=.02)


def test_definition_selects_owned_loader():
    # Regression: the previous definition selected AutoModel and checkpoint Python.
    assert isinstance(LocateAnythingQModel.loader, LocalModelLoader)
    assert not LocateAnythingQModel.require_trust_remote_code
