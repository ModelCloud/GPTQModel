# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.phi4 import Phi4MMGPTQ
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.phi4.configuration_phi4 import Phi4MMConfig
from gptqmodel.models.ports.phi4.modeling_phi4 import Phi4MMForCausalLM
from tests.port_assets import iter_port_assets, read_port_asset


def _config(**overrides):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "vision_lora": {"r": 4, "lora_alpha": 6, "dp": 0.0},
        "speech_lora": {"r": 2, "lora_alpha": 5, "dp": 0.0},
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 16,
            "crop_size": 16,
            "patch_size": 4,
            "image_token_id": 60,
            "feature_layer": -2,
        },
        "audio_config": {
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_blocks": 2,
            "num_attention_heads": 2,
            "input_size": 8,
            "nemo_conv_channels": 4,
            "ext_pw_out_channel": 8,
            "depthwise_separable_out_channel": 8,
            "time_reduction": 8,
            "kernel_size": 3,
            "audio_token_id": 61,
            "feature_layer": -2,
            "bias_max_distance": 500,
        },
        "attn_implementation": "eager",
    }
    values.update(overrides)
    return Phi4MMConfig(**values)


def _model(**overrides):
    torch.manual_seed(501)
    model = Phi4MMForCausalLM(_config(**overrides)).eval()
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if ".lora_B." in name:
                weight.normal_(0, 0.12)
    return model


def _oracle(model, ids, mode=0, padding=None, positions=None, embeddings=None):
    """Independent pinned-source decoder equations; no port forward/cache/rope helpers."""
    weights, config = model.state_dict(), model.config
    weights = {key: value.double() for key, value in weights.items()}
    x = (
        F.embedding(ids.clamp_min(0), weights["model.embed_tokens.weight"])
        if embeddings is None
        else embeddings.double()
    )
    batch, length = ids.shape
    heads, kv, dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )

    def norm(x, path):
        return x * (x.square().mean(-1, keepdim=True) + config.rms_norm_eps).rsqrt() * weights[path + ".weight"]

    def linear(x, path):
        y = F.linear(x, weights[path + ".base_layer.weight"])
        if mode:
            adapter = "speech" if mode == 2 else "vision"
            spec = getattr(config, adapter + "_lora")
            y = (
                y
                + F.linear(
                    F.linear(x, weights[path + ".lora_A." + adapter + ".weight"]),
                    weights[path + ".lora_B." + adapter + ".weight"],
                )
                * spec["lora_alpha"]
                / spec["r"]
            )
        return y

    if positions is None:
        positions = torch.arange(length)[None]
    rotary_dim = int(dim * config.partial_rotary_factor)
    inv = 10000.0 ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    angles = positions.double()[..., None] * inv
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]

    def rotary(x):
        first, second = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine, x[..., rotary_dim:]), -1)

    cache = []
    for i in range(config.num_hidden_layers):
        root = f"model.layers.{i}."
        packed = linear(norm(x, root + "input_layernorm"), root + "self_attn.qkv_proj")
        q, k, v = packed.split((heads * dim, kv * dim, kv * dim), -1)
        q = rotary(q.reshape(batch, length, heads, dim).transpose(1, 2))
        k = rotary(k.reshape(batch, length, kv, dim).transpose(1, 2))
        v = v.reshape(batch, length, kv, dim).transpose(1, 2)
        cache.append((k, v))
        k, v = k.repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
        allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
        if padding is not None:
            allowed = allowed & padding[:, None, None].bool()
        score = (q @ k.transpose(-1, -2) / dim**0.5).masked_fill(~allowed, float("-inf"))
        out = (score.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, -1)
        x = x + linear(out, root + "self_attn.o_proj")
        gate, up = linear(norm(x, root + "post_attention_layernorm"), root + "mlp.gate_up_proj").chunk(2, -1)
        x = x + linear(F.silu(gate) * up, root + "mlp.down_proj")
    return F.linear(norm(x, "model.norm"), weights["lm_head.weight"]), cache


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_source_decoder_oracle_and_cache(heads, mode, attention):
    model = _model(num_key_value_heads=heads, attn_implementation=attention)
    ids = torch.tensor([[3, 7, 8, 12, 9], [4, 1, 6, 5, 8]])
    padding = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    positions = torch.tensor([[60, 61, 62, 63, 64], [70, 71, 72, 73, 74]])
    with torch.inference_mode():
        expected, cache = _oracle(model, ids, mode, padding, positions)
        actual = model(ids, input_mode=mode, attention_mask=padding, position_ids=positions, use_cache=True)
        torch.testing.assert_close(actual.logits.double(), expected, atol=3e-6, rtol=3e-6)
        for layer, (key, value) in zip(actual.past_key_values.layers, cache):
            torch.testing.assert_close(layer.keys.double(), key, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(layer.values.double(), value, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_generation_cache_safetensors_and_failure_retry(tmp_path, dtype, mode):
    model = _model().to(dtype)
    ids = torch.tensor([[3, 7, 8, 12, 9, 5]])
    with torch.inference_mode():
        expected = model(ids, input_mode=mode).logits
        first = model(ids[:, :1], input_mode=mode, use_cache=True)
        second = model(ids[:, 1:5], input_mode=mode, past_key_values=first.past_key_values, use_cache=True)
        final = model(ids[:, 5:], input_mode=mode, past_key_values=second.past_key_values, use_cache=True)
        tolerance = 3e-6 if dtype == torch.float32 else 0.006
        torch.testing.assert_close(
            torch.cat((first.logits, second.logits, final.logits), 1), expected, atol=tolerance, rtol=tolerance
        )
        for beams in (1, 3):
            generated = model.generate(
                ids[:, :3], input_mode=mode, num_beams=beams, max_new_tokens=3, do_sample=False, eos_token_id=None
            )
            assert generated.shape == (1, 6)
        with pytest.raises(ValueError, match="input_mode"):
            model(ids, input_mode=8)
        torch.testing.assert_close(model(ids, input_mode=mode).logits, expected, atol=0, rtol=0)
        for _ in range(2):
            model.save_pretrained(tmp_path, safe_serialization=True)
            assert (tmp_path / "model.safetensors").is_file()
            model = Phi4MMForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
            torch.testing.assert_close(model(ids, input_mode=mode).logits, expected, atol=0, rtol=0)


def test_definition_owns_legacy_phi4mm():
    assert isinstance(Phi4MMGPTQ.loader, LocalModelLoader)
    assert Phi4MMGPTQ.loader.model_class == "gptqmodel.models.ports.phi4.modeling_phi4.Phi4MMForCausalLM"
    assert not Phi4MMGPTQ.require_monkeypatch


def _vision_oracle(module, pixels):
    weights = {key: value.double() for key, value in module.state_dict().items()}
    config = module.config.vision_config

    def linear(x, path):
        return F.linear(x, weights[path + ".weight"], weights.get(path + ".bias"))

    def norm(x, path):
        return F.layer_norm(
            x, (config.hidden_size,), weights[path + ".weight"], weights[path + ".bias"], config.layer_norm_eps
        )

    root = "img_processor."
    x = (
        F.conv2d(
            pixels.flatten(0, 1).double(),
            weights[root + "embeddings.patch_embedding.weight"],
            weights[root + "embeddings.patch_embedding.bias"],
            stride=config.patch_size,
        )
        .flatten(2)
        .transpose(1, 2)
    )
    x = x + weights[root + "embeddings.position_embedding.weight"]
    states = [x]
    for i in range(config.num_hidden_layers):
        prefix = root + f"encoder.layers.{i}."
        z = norm(x, prefix + "layer_norm1")
        q, k, v = [
            linear(z, prefix + "self_attn." + p + "_proj").reshape(z.shape[0], -1, 2, 4).transpose(1, 2)
            for p in ("q", "k", "v")
        ]
        a = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(z.shape)
        x = x + linear(a, prefix + "self_attn.out_proj")
        z = norm(x, prefix + "layer_norm2")
        x = x + linear(F.gelu(linear(z, prefix + "mlp.fc1"), approximate="tanh"), prefix + "mlp.fc2")
        states.append(x)
    x = states[config.feature_layer].reshape(-1, 4, 4, 8).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, 2).permute(0, 2, 3, 1).reshape(-1, 2, 2, 8)
    newline = weights["sub_GN"].reshape(1, 1, 8).expand(2, 1, 8)
    global_features = torch.cat((x[0], newline), 1).reshape(-1, 8)
    sub_features = torch.cat((x[1], newline), 1).reshape(-1, 8)
    combined = torch.cat((sub_features, weights["glb_GN"].reshape(-1, 8), global_features), 0)
    return linear(F.gelu(linear(combined, "img_projection.0")), "img_projection.2")


def _audio_oracle(module, features, mask):
    config = module.config
    w = {key: value.double() for key, value in module.state_dict().items()}

    def linear(x, path):
        return F.linear(x, w[path + ".weight"], w.get(path + ".bias"))

    def norm(x, path):
        return F.layer_norm(x, (config.hidden_size,), w[path + ".weight"], w[path + ".bias"])

    def conv(x, path, groups=1, padding=0):
        return F.conv1d(x, w[path + ".weight"], w[path + ".bias"], padding=padding, groups=groups)

    def mlp(x, path):
        gate, up = linear(norm(x, path + ".layer_norm"), path + ".net.0.linear").chunk(2, -1)
        return linear(F.silu(gate) * up, path + ".net.2")

    x = (features.double() - w["encoder_embedding.global_mean"]) * w["encoder_embedding.global_invstd"]
    x = x[:, None]
    for index, stride, groups in [
        (0, 2, 1),
        (2, 2, config.nemo_conv_channels),
        (3, 1, 1),
        (5, 2, config.nemo_conv_channels),
        (6, 1, 1),
    ]:
        path = f"embed.conv.{index}"
        x = F.conv2d(
            x, w[path + ".weight"], w[path + ".bias"], stride=stride, padding=1 if stride == 2 else 0, groups=groups
        )
        if index in (0, 3, 6):
            x = F.relu(x)
    batch, _, frames, _ = x.shape
    x = linear(x.transpose(1, 2).reshape(batch, frames, -1), "embed.out")
    valid = (
        torch.arange(frames)[None] < ((mask.sum(-1) + 7) // 8)[:, None]
        if mask is not None
        else torch.ones(batch, frames, dtype=torch.bool)
    )
    if frames > 500:
        padding = (-frames) % 500
        x = F.pad(x, (0, 0, 0, padding)).reshape(-1, 500, config.hidden_size)
        valid = F.pad(valid, (0, padding)).reshape(-1, 500)
    length = x.shape[1]
    rel = (torch.arange(length)[None] - torch.arange(length)[:, None]).clamp(
        -config.bias_max_distance, config.bias_max_distance - 1
    )
    bias = w["relative_attention_bias_layer.bias_values.weight"][rel + config.bias_max_distance].permute(2, 0, 1)
    for index in range(config.num_blocks):
        root = f"encoders.{index}."
        x = x + 0.5 * mlp(x, root + "feed_forward_in")
        z = norm(x, root + "layer_norm_att")
        q, k, v = [
            linear(z, root + "self_attn.linear_" + p).reshape(z.shape[0], length, 2, 4).transpose(1, 2)
            for p in ("q", "k", "v")
        ]
        scores = q @ k.transpose(-1, -2) / 2 + bias
        scores = scores.masked_fill(~valid[:, None, None], -torch.inf)
        attention = scores.softmax(-1).masked_fill(~valid[:, None, None], 0)
        x = x + linear((attention @ v).transpose(1, 2).reshape(z.shape), root + "self_attn.linear_out")
        z = norm(x, root + "conv.layer_norm").transpose(1, 2)
        first, gate = conv(z, root + "conv.glu.ext_pw_conv_1d").chunk(2, 1)
        z = (first + w[root + "conv.glu.b1"]) * F.silu(gate + w[root + "conv.glu.b2"])
        z = conv(z, root + "conv.dw_sep_conv_1d.dw_conv", groups=config.hidden_size, padding=2)[..., :-2]
        z = conv(z, root + "conv.dw_sep_conv_1d.pw_conv")
        z = conv(F.silu(z), root + "conv.ext_pw_conv_1d").transpose(1, 2)
        x = x + z
        x = norm(x + 0.5 * mlp(x, root + "feed_forward_out"), root + "layer_norm")
    return x.reshape(batch, -1, config.hidden_size)[:, :frames]


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nonconstant_image_features_and_decoder_match_independent_source_equations(dtype, attention):
    model = _model(attn_implementation=attention).to(dtype)
    pixels = torch.linspace(-1.0, 1.0, 2 * 3 * 16 * 16).reshape(1, 2, 3, 16, 16).to(dtype)
    ids = torch.tensor([[3] + [-1] * 13 + [7, 8]])
    sizes = torch.tensor([[16, 16]])
    mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    image = model.model.embed_tokens_extend.image_embed
    with torch.inference_mode():
        expected_image = _vision_oracle(image, pixels)
        embeds = model.model.embed_tokens(ids.clamp_min(0)).double()
        embeds[:, 1:14] = expected_image
        expected, _ = _oracle(model, ids, 1, embeddings=embeds)
        actual = model(
            ids, input_mode=1, input_image_embeds=pixels, image_sizes=sizes, image_attention_mask=mask
        ).logits
        tolerance = 3e-6 if dtype == torch.float32 else 0.008
        torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
        changed = model(
            ids, input_mode=1, input_image_embeds=pixels.flip(-1), image_sizes=sizes, image_attention_mask=mask
        ).logits
        assert (actual - changed).abs().max() > 1e-5


@pytest.mark.parametrize("frames", [24, 4008])
@pytest.mark.parametrize("empty_second", [False, True])
def test_audio_subsampling_conformer_mask_and_500_frame_unfold_oracle(frames, empty_second):
    model = _model()
    encoder = model.model.embed_tokens_extend.audio_embed.encoder
    features = torch.sin(torch.arange(2 * frames * 8).reshape(2, frames, 8) * 0.013)
    mask = torch.arange(frames)[None] < torch.tensor([frames, 0 if empty_second else frames - 9])[:, None]
    with torch.inference_mode():
        expected = _audio_oracle(encoder, features, mask)
        actual = encoder(features, mask)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-6)


ASSETS = Path(__file__).parents[1] / "gptqmodel/models/ports/phi4"


def _assets(path):
    path.mkdir(parents=True, exist_ok=True)
    for asset in iter_port_assets(ASSETS / "released_assets", "*"):
        (path / asset.name).write_bytes(read_port_asset(asset))


def _checkpoint(path, **overrides):
    path.mkdir(parents=True, exist_ok=True)
    values = {"vocab_size": 200064, "bos_token_id": 199999, "eos_token_id": 199999, "pad_token_id": 199999}
    values.update(overrides)
    model = _model(**values)
    model.save_pretrained(path, safe_serialization=True)
    _assets(path)
    marker = path / "checkpoint_executed"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    for name, mapping in [
        ("config.json", {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}),
        ("tokenizer_config.json", {"AutoTokenizer": ["evil.Tokenizer", "evil.Tokenizer"]}),
        (
            "preprocessor_config.json",
            {
                "AutoProcessor": "evil.Processor",
                "AutoImageProcessor": "evil.Image",
                "AutoFeatureExtractor": "evil.Audio",
            },
        ),
        ("processor_config.json", {"AutoProcessor": "evil.Processor"}),
    ]:
        payload = json.loads((path / name).read_text())
        payload["auto_map"] = mapping
        (path / name).write_text(json.dumps(payload), encoding="utf-8")
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_owned_model_tokenizer_processor_and_asset_repair(tmp_path, monkeypatch, trust, bypass):
    from transformers import TokenizersBackend

    from gptqmodel import GPTQModel
    from gptqmodel.models.ports.phi4.processing_phi4 import Phi4MMProcessor

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    path = tmp_path / "source"
    source, marker = _checkpoint(path)
    wrapped = GPTQModel.load(
        str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    assert type(wrapped.model) is Phi4MMForCausalLM
    assert type(wrapped.model.config) is Phi4MMConfig
    assert type(wrapped.processor) is Phi4MMProcessor
    assert isinstance(wrapped.tokenizer, TokenizersBackend)
    assert wrapped.processor.tokenizer.encode("hello") == wrapped.tokenizer.encode("hello")
    cases = {
        "hello world": [24912, 2375],
        "你好，世界": [177519, 979, 28428],
        "Hello, GPT-QModel!": [13225, 11, 174803, 50456, 2239, 0],
    }
    for text, ids in cases.items():
        assert wrapped.tokenizer.encode(text) == ids
        assert wrapped.tokenizer.decode(ids) == text
    inputs = wrapped.processor("hello world")
    with torch.inference_mode():
        expected = source(**inputs).logits
        torch.testing.assert_close(wrapped.model(**inputs).logits, expected, atol=0, rtol=0)
        assert wrapped.model.generate(**inputs, max_new_tokens=2).shape[-1] == 4
    for i in range(2):
        destination = tmp_path / f"dense-{i}"
        wrapped.model.save_pretrained(destination, safe_serialization=True)
        wrapped.processor.save_pretrained(destination)
        wrapped = GPTQModel.load(
            str(destination), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
        )
        assert type(wrapped.processor) is Phi4MMProcessor
        for text, ids in cases.items():
            assert wrapped.tokenizer.encode(text) == ids
        with torch.inference_mode():
            torch.testing.assert_close(wrapped.model(**inputs).logits, expected, atol=0, rtol=0)
    # Failed processor construction may not poison a repaired subsequent load.
    asset = path / "preprocessor_config.json"
    original = asset.read_bytes()
    asset.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        Phi4MMGPTQ.loader.load_processor(path, tokenizer=wrapped.tokenizer, trust_remote_code=trust)
    with pytest.raises(json.JSONDecodeError):
        GPTQModel.load(
            str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
        )
    asset.write_bytes(original)
    restored = Phi4MMGPTQ.loader.load_processor(path, tokenizer=wrapped.tokenizer, trust_remote_code=trust)
    assert restored("hello world")["input_ids"].tolist() == [[24912, 2375]]
    retry = GPTQModel.load(
        str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    with torch.inference_mode():
        torch.testing.assert_close(retry.model(**inputs).logits, expected, atol=0, rtol=0)
    assert not marker.exists()


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_real_source_processor_pixels_audio_ids_and_save(tmp_path, sample_rate):
    import numpy as np
    from PIL import Image
    from transformers import TokenizersBackend

    from gptqmodel.models.ports.phi4.processing_phi4 import Phi4MMProcessor

    _assets(tmp_path)
    tokenizer = TokenizersBackend.from_pretrained(tmp_path)
    processor = Phi4MMGPTQ.loader.load_processor(tmp_path, tokenizer=tokenizer)
    image = Image.fromarray((np.arange(33 * 45 * 3).reshape(33, 45, 3) * 17).astype("uint8"))
    waveform = np.sin(np.arange(sample_rate // 5) * 0.041).astype("float32")
    inputs = processor("hello <|image_1|> <|audio_1|> world", images=image, audios=[(waveform, sample_rate)])
    assert inputs["input_mode"].tolist() == [3]
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    # 33:45 -> 336x448 resized content: 12x16 local pooled patches,
    # row newlines, one global separator and the full 16x16 global crop.
    assert (inputs["input_ids"] == 200010).sum() == 12 * 17 + 1 + 16 * 17
    assert (inputs["input_ids"] == 200011).sum() == inputs["audio_embed_sizes"].sum()
    assert inputs["input_image_embeds"].shape[2:] == (3, 448, 448)
    assert inputs["input_audio_embeds"].shape[-1] == 80
    # Independent SpeechLib Hamming/pre-emphasis/RFFT equations, including 8 kHz zero-fill.
    win, hop, fft = (200, 80, 256) if sample_rate == 8000 else (400, 160, 512)
    frames = np.stack([waveform[i : i + win] for i in range(0, (len(waveform) - win) // hop * hop + 1, hop)])
    previous = np.concatenate((frames[:, :1], frames[:, :-1]), -1)
    spectrum = np.fft.rfft((frames - 0.97 * previous) * 32768 * np.hamming(win), n=fft).astype("complex64")
    if sample_rate == 8000:
        spectrum = np.concatenate((spectrum[:, :-1], np.zeros_like(spectrum)), -1)
    expected = np.log(
        np.maximum((np.abs(spectrum).astype("float32") ** 2) @ processor.audio_processor._mel, 1)
    ).astype("float32")
    np.testing.assert_allclose(inputs["input_audio_embeds"][0].numpy(), expected, atol=2e-5, rtol=2e-5)
    processor.save_pretrained(tmp_path / "saved")
    restored = Phi4MMProcessor.from_pretrained(tmp_path / "saved")
    repeat = restored("hello <|image_1|> <|audio_1|> world", images=image, audios=[(waveform, sample_rate)])
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(repeat[name], value, atol=0, rtol=0)


def _media(mode, dtype=torch.float32):
    ids = [3]
    inputs = {"input_mode": mode}
    if mode in (1, 3):
        ids += [60] * 13
        inputs.update(
            input_image_embeds=torch.cos(torch.arange(1536).reshape(1, 2, 3, 16, 16) * 0.03).to(dtype),
            image_sizes=torch.tensor([[16, 16]]),
            image_attention_mask=torch.ones(1, 2, 4, 4, dtype=torch.bool),
        )
    if mode in (2, 3):
        ids += [61] * 3
        inputs.update(
            input_audio_embeds=torch.sin(torch.arange(24 * 8).reshape(1, 24, 8) * 0.09).to(dtype),
            audio_embed_sizes=torch.tensor([3]),
            audio_attention_mask=torch.ones(1, 24, dtype=torch.bool),
        )
    ids += [7, 8]
    inputs["input_ids"] = torch.tensor([ids])
    return inputs


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_media_generation_cache_and_two_safetensors_cycles(tmp_path, mode, dtype):
    model = _model().to(dtype)
    inputs = _media(mode, dtype)
    with torch.inference_mode():
        expected = model(**inputs, use_cache=True)
        assert torch.isfinite(expected.logits).all()
        token = torch.tensor([[11]])
        continued = model(token, input_mode=mode, past_key_values=expected.past_key_values).logits
        extended = dict(inputs, input_ids=torch.cat((inputs["input_ids"], token), -1))
        reference = model(**extended).logits[:, -1:]
        tolerance = 3e-6 if dtype == torch.float32 else 0.008
        torch.testing.assert_close(continued, reference, atol=tolerance, rtol=tolerance)
        for beams in (1, 3):
            cached = model.generate(**inputs, max_new_tokens=3, num_beams=beams, eos_token_id=None)
            uncached = model.generate(**inputs, max_new_tokens=3, num_beams=beams, eos_token_id=None, use_cache=False)
            assert torch.equal(cached, uncached)
        baseline = expected.logits.clone()
        for _ in range(2):
            model.save_pretrained(tmp_path, safe_serialization=True)
            model = Phi4MMForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
            torch.testing.assert_close(model(**inputs).logits, baseline, atol=0, rtol=0)


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_public_plain_awq_pack_infer_two_reloads_preserve_loras_and_media(tmp_path, offload, mode):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    vision = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": 448,
        "crop_size": 448,
        "patch_size": 14,
        "image_token_id": 200010,
        "feature_layer": -2,
    }
    audio = {
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_blocks": 2,
        "num_attention_heads": 2,
        "input_size": 80,
        "nemo_conv_channels": 4,
        "ext_pw_out_channel": 8,
        "depthwise_separable_out_channel": 8,
        "time_reduction": 8,
        "kernel_size": 3,
        "audio_token_id": 200011,
        "feature_layer": -2,
        "bias_max_distance": 500,
    }
    source, marker = _checkpoint(
        tmp_path / "source",
        intermediate_size=64,
        max_position_embeddings=2048,
        vision_config=vision,
        audio_config=audio,
    )
    qcfg = AWQConfig(
        bits=4,
        group_size=16,
        device="cpu",
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
        fallback=None,
    )
    model = GPTQModel.load(
        str(tmp_path / "source"),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    import numpy as np
    from PIL import Image

    image = Image.fromarray((np.arange(16 * 16 * 3).reshape(16, 16, 3) * 17).astype("uint8"))
    waveform = np.sin(np.arange(1600) * 0.09).astype("float32")
    inputs = model.processor(
        "hello " + ("<|image_1|> " if mode in (1, 3) else "") + ("<|audio_1|> " if mode in (2, 3) else "") + "world",
        images=image if mode in (1, 3) else None,
        audios=[(waveform, 16000)] if mode in (2, 3) else None,
    )
    assert inputs["input_mode"].item() == mode
    roots = {
        f"model.layers.{i}.{name}.base_layer"
        for i in range(2)
        for name in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    }
    dense = {
        name: value.half().clone()
        for name, value in source.state_dict().items()
        if name.removesuffix(".weight") not in roots
    }
    expected_inputs, captured = [], []

    def record(destination):
        def hook(module, args, kwargs):
            destination.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

        return hook

    source.half()
    handle = source.model.layers[0].register_forward_pre_hook(record(expected_inputs), with_kwargs=True)
    with torch.inference_mode():
        source(**inputs, logits_to_keep=1)
    handle.remove()
    handle = model.model.model.layers[0].register_forward_pre_hook(record(captured), with_kwargs=True)
    model.quantize(
        CalibrationConfig(dataset=[dict(inputs), dict(inputs)], batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    handle.remove()
    assert captured, "Actual first-decoder calibration input must be captured"
    torch.testing.assert_close(captured[0], expected_inputs[0], atol=0, rtol=0)
    packed = {name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == roots
    model.pre_quantize_generate_hook_start()
    model.shell_module_materialize(model.model.model.norm, "cpu")
    model.shell_module_materialize(model.model.lm_head, "cpu")
    for name, value in dense.items():
        torch.testing.assert_close(model.model.state_dict()[name], value, atol=0, rtol=0, msg=name)
    with torch.inference_mode():
        logits = model.model(**inputs, logits_to_keep=1).logits
        generated = model.model.generate(**inputs, max_new_tokens=2)
        assert torch.isfinite(logits).all()
    for i in range(2):
        destination = tmp_path / f"quantized-{i}"
        model.save(destination)
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        restored = GPTQModel.load(
            str(destination), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16, attn_implementation="eager"
        )
        assert type(restored.model) is Phi4MMForCausalLM
        for name, value in model.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value, atol=0, rtol=0, msg=name)
        with torch.inference_mode():
            torch.testing.assert_close(restored.model(**inputs, logits_to_keep=1).logits, logits, atol=0, rtol=0)
            assert torch.equal(restored.model.generate(**inputs, max_new_tokens=2), generated)
        model = restored
    assert not marker.exists()


@pytest.mark.parametrize("policy", ["default", "longrope", "su", "yarn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_source_rotary_short_long_threshold_and_frequency_precision(policy, dtype):
    spec = {"rope_type": policy, "short_factor": [1.25, 1.5], "long_factor": [2.0, 4.0], "factor": 4.0}
    config = _config(
        max_position_embeddings=128, original_max_position_embeddings=32, partial_rotary_factor=0.5, rope_scaling=spec
    )
    model = Phi4MMForCausalLM(config).eval().to(dtype)
    inv_base = 10000.0 ** (-torch.arange(0, 4, 2, dtype=torch.float64) / 4)
    for positions in ([0, 11, 31], [31, 32, 127]):
        position_ids = torch.tensor([positions])
        cos, sin = model.model.rotary_emb(torch.zeros(1, 3, 32, dtype=dtype), position_ids)
        scale = torch.ones(2, dtype=torch.float64)
        amplitude = 1.0
        if policy != "default":
            scale = torch.tensor(spec["long_factor"] if max(positions) + 1 > 32 else spec["short_factor"]).double()
            amplitude = (1 + torch.log(torch.tensor(4.0)).item() / torch.log(torch.tensor(32.0)).item()) ** 0.5
            if policy == "yarn":
                amplitude = 1 + 0.1 * torch.log(torch.tensor(4.0)).item()
        phases = position_ids.double()[..., None] * inv_base / scale
        phases = torch.cat((phases, phases), -1)
        torch.testing.assert_close(cos.double(), (phases.cos() * amplitude).to(dtype).double(), atol=0.008, rtol=0.008)
        torch.testing.assert_close(sin.double(), (phases.sin() * amplitude).to(dtype).double(), atol=0.008, rtol=0.008)


def test_ragged_multiple_images_audio_grouped_beam_expansion():
    model = _model()
    a = _media(3)
    b = _media(3)
    a["input_audio_embeds"] = a["input_audio_embeds"].repeat(2, 1, 1)
    a["input_audio_embeds"][1].mul_(-0.4)
    a["audio_embed_sizes"] = torch.tensor([3, 3])
    a["audio_attention_mask"] = a["audio_attention_mask"].repeat(2, 1)
    a["input_ids"] = torch.tensor([[3] + [60] * 13 + [61] * 6 + [7, 8]])
    b["input_image_embeds"] = b["input_image_embeds"].repeat(2, 1, 1, 1, 1)
    b["input_image_embeds"][1].mul_(-0.3)
    b["image_sizes"] = b["image_sizes"].repeat(2, 1)
    b["image_attention_mask"] = b["image_attention_mask"].repeat(2, 1, 1, 1)
    b["input_ids"] = torch.tensor([[3] + [60] * 26 + [61] * 3 + [7, 8]])
    length = b["input_ids"].shape[-1]
    batch = {
        "input_ids": torch.cat((F.pad(a["input_ids"], (length - a["input_ids"].shape[-1], 0)), b["input_ids"])),
        "input_mode": 3,
    }
    batch["attention_mask"] = (batch["input_ids"] != 0).long()
    for key in (
        "input_image_embeds",
        "image_sizes",
        "image_attention_mask",
        "input_audio_embeds",
        "audio_embed_sizes",
        "audio_attention_mask",
    ):
        batch[key] = torch.cat((a[key], b[key]))
    with torch.inference_mode():
        position_ids = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        expected_logits = model(**batch, position_ids=position_ids, logits_to_keep=1).logits
        expanded_ids, expanded = model._expand_inputs_for_generation(
            expand_size=3,
            input_ids=batch["input_ids"],
            **{k: v for k, v in batch.items() if k != "input_ids"},
            position_ids=position_ids,
        )
        expanded_logits = model(expanded_ids, **expanded, logits_to_keep=1).logits
        torch.testing.assert_close(expanded_logits, expected_logits.repeat_interleave(3, 0), atol=1e-6, rtol=1e-6)
        reference = [
            model.generate(**item, max_new_tokens=3, num_beams=3, eos_token_id=None)[:, -3:] for item in (a, b)
        ]
        actual = model.generate(**batch, max_new_tokens=3, num_beams=3, eos_token_id=None)
    assert torch.equal(actual[:, -3:], torch.cat(reference))
